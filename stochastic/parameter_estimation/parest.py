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
    Handle the minutia of multiprocessing.

    Execpt now it's gone down the drain because pickle can't deal with
    functions defined in a class. What used to be a nice design is now a
    spaghetti mess.
    """
    def __init__(self, processes, function):
        self._simulations = []
        self._pool = Pool(processes)
        self._f = function
        self._sim_nrs = []
        self._parameters_used = []

    def addSimulation(self, sim_nr, params):
        self._sim_nrs.append(sim_nr)
        self._parameters_used.append(params)
        s = self._pool.apply_async(self._f, (params,))
        self._simulations.append(s)

    def runSimulations(self):
        self._pool.close()
        self._pool.join()

    def getSimulations(self):
        """
        Relies on the order with which arguments and simulation numbers were
        added in addSimulation
        """
        results = [s.get() for s in self._simulations]
        return zip(self._sim_nrs, results, self._parameters_used)


class Parest(object):
    """
    This class implements a heuristic parameter estimation method that targets
    models that are expensive to run and have relatively easy-to-reach
    optimums.
    """

    def __init__(self, setup, samples, name, processes, batch_size=5):

        self.par = setup.parameters
        self.sim = setup.simulation
        self.obs = setup.observation()
        self.eva = setup.evaluation
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
        self.result_frame = -1

        # Track time
        self._init_time = time.time()

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
        if nr_samples is 0:
            print("End of simulation")
            return

        # Do parallell runs even if you only run with 1 processor; your runs
        # are so long that the overhead does not matter
        ctrl = _Controller(self.procs, self.sim)

        for sim in range(nr_samples):
            self.sim_nr +=1
            # Get parameters (bad to test every time in here; but avoid perfection for now)
            if iterated_parameters is False:
                params = self.par()  # initial parameters
            else:
                params = iterated_parameters
            print params

            # Find new abortive probability with updated RNA fraction
            ctrl.addSimulation(self.sim_nr, params)

        ctrl.runSimulations()

        # Extract results from multicore processing
        par_data = defaultdict(list)
        res_data, index = {}, []
        for sim_nr, sim_result, params_used in ctrl.getSimulations():
            #print results
            index.append(sim_nr)
            score = self.eva(sim_result, self.obs)
            # Add results to datastructures
            res_data[sim_nr] = sim_result
            par_data['score'].append(score)
            for name, val in params_used.items():
                par_data[name].append(val)

        # Generate dataframes
        par_frame = frame(data=par_data, index=index)
        res_frame = frame(data=res_data)

        # Add new results to previous, if any.
        self._concatenate_results(par_frame, res_frame)

        # If we still have more simulations to perform
        if self.sim_nr < self.samples:
            #updated_parameter_range = self._new_parameter_ranges(par_frame)
            updated_parameter_range = False
            # Recursive call
            self.search(updated_parameter_range)
        else:
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
        self.result_frame.to_csv(results_path, sep='\t')

    def _run_multicore(self, sim_nr, f, par):
        """
        Keep track of simulation number for multicore runs.
        """
        return sim_nr, f(par)

    def _get_nr_samples(self):
        """
        Find how many rounds of sampling should be performed for this instance
        """
        if self.samples < 2 * self.bsize:
            print("Single run: sample size too small for batches")
            nr_samples = self.samples - self.sim_nr
        else:
            self.batch_mode = True
            print("Running batch nr {0}".format(self.batch_nr))
            nr_samples = min(self.bsize, self.samples - self.sim_nr)
            self.batch_nr +=1

        assert 0 <= nr_samples <= self.samples

        return nr_samples

    def _concatenate_results(self, par_frame, res_frame):
        """
        Add results to previous results if they exist.
        """
        if par_frame.index[0] == 1:
            self.parameter_frame = par_frame
            self.result_frame = res_frame
        else:
            self.parameter_frame = concat(self.parameter_frame, par_frame)
            self.result_frame = concat(self.result_frame, res_frame)

    def _new_parameter_ranges(self, par_frame):
        """
        Use results from the previous batch to find parameter ranges for the
        next batch.
        """
        return False

    def get_results(self):
        duration = time.time() - self._init_time
        print('Parameter search duration: {0}'.format(duration))
        return {'parameters': self.parameter_frame,
                'time_series': self.result_frame}


def load_results(name):
    """
    Loads results given a name ID
    """
    # parameters
    param_path = _get_param_path(name)
    parameter_frame = read_csv(param_path, sep='\t')

    # simulation results
    results_path = _get_result_path(name)
    result_frame = read_csv(results_path, sep='\t')

    return {'parameters': parameter_frame,
            'time_series': result_frame}


def _get_param_path(name):
    file_name = '_'.join([name, 'parameters.csv'])
    file_path = os.path.join(save_dir, file_name)
    return file_path


def _get_result_path(name):
    file_name = '_'.join([name, 'sim_results.csv'])
    file_path = os.path.join(save_dir, file_name)
    return file_path


