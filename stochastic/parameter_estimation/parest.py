"""
This is a heuristic parameter estimator.
"""
from ipdb import set_trace as debug  # NOQA
from collections import defaultdict
from multiprocessing import Pool
#import pathos.multiprocessing as mp
#import pprocess
from pandas import DataFrame as frame
from pandas import concat, read_csv
# For 'fixing' pickle
import copy_reg
import types
import time
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "parameter_estimation"))


## Multiprocessing is a broken in that it cannot work with class methods.
## So here is some fancy redefinition of pickle that fixes it.
def _pickle_method(m):
    if m.im_self is None:
        return getattr, (m.im_class, m.im_func.func_name)
    else:
        return getattr, (m.im_self, m.im_func.func_name)
copy_reg.pickle(types.MethodType, _pickle_method)


# Directory for saving results
save_dir = 'param_est_storage'
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)


class _Controller(object):
    """
    Handle the minutia of multiprocessing. And the non-minutia of
    single-processing which become minutia when combined with
    multiprocessing.
    """
    def __init__(self, processes, function):
        self._procs = processes
        if self._procs > 1:
            self._multiprocessing = True
        else:
            self._multiprocessing = False

        self._f = function
        self._sim_nrs = []
        self._parameters = []
        self._results = []

    def addSimulation(self, sim_nr, params):
        self._sim_nrs.append(sim_nr)
        self._parameters.append(params)

    def runSimulations(self):
        """
        With multiprocessing, add each simulation to a pool before extracting
        them. map_async preserves order! apply_async apparently does not :S
        """
        if self._multiprocessing:
            P = Pool(self._procs)
            sims = P.map_async(self._f, self._parameters)
            P.close()
            P.join()
            self._results = sims.get()
        else:
            self._results = [self._f(params) for params in self._parameters]

    def getSimulations(self):
        """
        Relies on the order with which arguments and simulation numbers were
        added in addSimulation
        """
        return zip(self._sim_nrs, self._results, self._parameters)


class Parest(object):
    """
    This class implements a heuristic parameter estimation method that targets
    models that are expensive to run and have relatively easy-to-reach
    optimums.
    """

    def __init__(self, setup, samples, name, processes, batch_size=5):

        self.parameters = setup.parameters
        self.simulation = setup.simulation
        self.observed = setup.observation()
        self.evaluate = setup.evaluation
        self.name = name

        # Nr of samples to make
        self.samples = samples

        # Nr of samples in each batch
        # Used for iterative sampling of parameter space
        self.bsize = batch_size

        # Number of processes to use for multiprocessing
        self.procs = processes

        self.sim_nr = 0
        self.batch_nr = 1
        self.batch_mode = False

        # Initialize variames that will hold dataframes
        self.parameter_frame = -1
        self.timeseries_frame = -1

        # Track time
        self._init_time = time.time()

    def _search(self, ctrl, iterated_parameters, nr_samples):

        for sim in range(nr_samples):
            self.sim_nr +=1
            if iterated_parameters is False:
                params = self.parameters()  # initial parameters
            else:
                params = iterated_parameters

            ctrl.addSimulation(self.sim_nr, params)

        ctrl.runSimulations()

    def _evaluate_search(self, ctrl):

        par_data = defaultdict(list)
        ts_data, par_index = {}, []
        for sim_nr, sim_result, params_used in ctrl.getSimulations():
            # New requirement: sim_result should return a dataframe
            ts_data[sim_nr] = sim_result['model']
            par_index.append(sim_nr)
            scores = self.evaluate(sim_result, self.observed)
            for score_name, score in scores.items():
                par_data[score_name].append(score)
            for name, val in params_used.items():
                par_data[name].append(val)

        # Generate dataframes
        # should be as general as possible inside the parest routine. It
        # should know as little as possible about the outside world.
        par_frame = frame(data=par_data, index=par_index)
        ts_frame = frame(data=ts_data)

        return par_frame, ts_frame

    def search(self, iterated_parameters=False):
        """
        Search for optimal parameters. Store outcome in two dataframes.

        1) Parameters tested with their score.
        2) Simulation results

        The two are linked by a simulation number (sim_nr)

        Challenge when doing multi_proc: ensure that sim_nr gets added up
        correctly.
        """

        nr_samples = self._get_nr_samples()
        print('beg nr_samples', nr_samples)

        # Launch parameter searches
        controller = _Controller(self.procs, self.simulation)
        self._search(controller, iterated_parameters, nr_samples)

        # Extract results and compare
        par_frame, ts_frame = self._evaluate_search(controller)

        # Add results to previous
        self._concatenate_results(par_frame, ts_frame)

        if self._continue_searching():
            parameter_range = self._new_parameter_ranges(par_frame)
            self.search(parameter_range)

    def _continue_searching(self):
        # If we still have more simulations to perform
        if self.sim_nr < self.samples:
            return True
        else:
            duration = time.time() - self._init_time
            print('End of parameter search. Duration: {0}'.format(duration))
            self._store_results()

    def _store_results(self):
        """
        Save results
        """

        # parameters
        param_path = _get_param_path(self.name)
        self.parameter_frame.to_csv(param_path, sep='\t')

        # Simulation results
        results_path = _get_result_path(self.name)
        self.timeseries_frame.to_csv(results_path, sep='\t')

    def _get_nr_samples(self):
        """
        Find how many rounds of sampling should be performed for this instance
        """
        # Disable this while debugging
        #if self.samples < 2 * self.bsize:
        if self.samples < 1 * self.bsize:
            print("Single run: sample size too small for batches")
            nr_samples = self.samples - self.sim_nr
        else:
            self.batch_mode = True
            print("Running batch nr {0}".format(self.batch_nr))
            nr_samples = min(self.bsize, self.samples - self.sim_nr)
            self.batch_nr +=1

        assert 0 <= nr_samples <= self.samples

        return nr_samples

    def _concatenate_results(self, par_frame, ts_frame):
        """
        Add results to previous results if they exist. Join parameters by the
        index (simulation number) and timeseries
        """
        if par_frame.index[0] == 1:
            self.parameter_frame = par_frame
            self.timeseries_frame = ts_frame
        else:
            self.parameter_frame = concat([self.parameter_frame, par_frame], axis=0)
            self.timeseries_frame = concat([self.timeseries_frame, ts_frame], axis=1)

            debug()

    def _new_parameter_ranges(self, par_frame):
        """
        Use results from the previous batch to find parameter ranges for the
        next batch.
        """
        #debug()
        return False

    def get_results(self):
        return {'parameters': self.parameter_frame,
                'time_series': self.timeseries_frame}


def load_results(name):
    """
    Loads results given a name ID
    """
    # parameters
    param_path = _get_param_path(name)
    parameter_frame = read_csv(param_path, sep='\t', index_col=0)

    # simulation results
    results_path = _get_result_path(name)
    timeseries_frame = read_csv(results_path, sep='\t', index_col=0)

    # You want the columns to be ints
    timeseries_frame.columns = [int(i) for i in timeseries_frame.columns]

    return {'parameters': parameter_frame,
            'time_series': timeseries_frame}


def _get_param_path(name):
    file_name = '_'.join([name, 'parameters.csv'])
    file_path = os.path.join(save_dir, file_name)
    return file_path


def _get_result_path(name):
    file_name = '_'.join([name, 'sim_results.csv'])
    file_path = os.path.join(save_dir, file_name)
    return file_path


