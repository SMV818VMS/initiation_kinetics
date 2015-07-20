"""
This is a heuristic parameter estimator.
"""
from collections import defaultdict
from pandas import DataFrame as frame


class Parest(object):

    def __init__(self, setup, samples, batch_size=5, processors=2):

        # Set up shortcuts
        self.par = setup.parameters
        self.sim = setup.simulation
        self.obs = setup.observation()
        self.eva = setup.evaluation

        # Nr of samples to make
        self.samples = samples

        # Nr of samples in each batch
        self.bsize = batch_size

        # Number of processors to use for multiprocessing
        self.procs = processors

    def search(self):
        """
        Search for optimal parameters. Store outcome in two dataframes.

        1) Parameters tested with their score.
        2) Simulation results

        The two are linked by a simulation number (sim_nr)
        """

        #if self.samples < 2 * self.bsize:
            #print("Single run: sample size too small for batches")
            #usebatch = False
        #else:
            #usebatch = True
        pdata = defaultdict(list)
        pindex = []
        rdata = {}
        for sim_nr in range(1, self.samples+1):
            # Get parameters, run model, compare results
            params = self.par()
            sim_result = self.sim(params['random'])
            score = self.eva(sim_result, self.obs)

            # Build index and values of varied parameters
            pindex.append(score)
            for name, val in zip(params['name'], params['random']):
                pdata[name].append(val)
            pdata['sim nr'].append(sim_nr)

            # Store simulation results
            rdata['sim nr {0}'.format(sim_nr)] = sim_result

        self.parameter_frame = frame(data=pdata, index=pindex)
        self.result_frame = frame(data=rdata)

    def get_o

        return parameter_frame, result_frame


