import os
from math import floor
import sys
sys.path.append('/home/jorgsk/Dropbox/phdproject/transcription_initiation/kinetic/model')
import kinetic_transcription_models as ktm
from ipdb import set_trace as debug  # NOQA

from collections import OrderedDict
from copy import deepcopy

import stochpy
import numpy as np
import pandas as pd

# Location of psc directory
psc_dir = '/home/jorgsk/Dropbox/phdproject/transcription_initiation/kinetic/input/psc_files'


class ITSimulationSetup(object):
    """
    Setup for stochastic simulation.
    """

    def __init__(self, name, R, stoi_setup, initial_RNAP, sim_end,
                 unproductive_frac=0., nr_trajectories=1,
                 unproductive_only=False):

        # Basic input
        self.nr_traj = nr_trajectories
        self.sim_end = sim_end
        self.unproc_frac = unproductive_frac
        self.init_RNAP = initial_RNAP

        # Find #productive and #unproductive RNAPs to simulate
        self.nr_prod_RNAP, self.nr_unprod_RNAP = self.calc_nr_RNAPs(unproductive_only)

        # Create the graph for the model
        sim_graph = ktm.CreateModelGraph(R, name, stoi_setup)

        # Extract values from the graph
        reactions, initial_values, parameters = \
            ktm.GenerateStochasticInput(sim_graph, self.nr_prod_RNAP, self.nr_unprod_RNAP)

        # Write the model input psc file
        psc_path = ktm.write_psc(reactions, initial_values, parameters, name, psc_dir)
        self.model_psc_input = psc_path

    def calc_nr_RNAPs(self, unproductive_only):

        if unproductive_only:
            nr_RNAP_unproductive = self.init_RNAP
            nr_RNAP_productive = 0
        elif self.unproc_frac > 0:
            assert 0 < self.unproc_frac < 1
            nr_RNAP_unproductive = floor(self.init_RNAP * self.unproc_frac)
            nr_RNAP_productive = self.init_RNAP - nr_RNAP_unproductive
        else:
            nr_RNAP_unproductive = 0
            nr_RNAP_productive = self.init_RNAP

        return nr_RNAP_productive, nr_RNAP_unproductive


class ITModel(object):
    """
    Initial transcription model
    """

    def __init__(self, setup):

        self.setup = setup
        self.nr_traj = setup.nr_traj
        self.duration = setup.sim_end
        self.psc_file = setup.model_psc_input
        self.sp_2_aborted_modname = self._species_2_aborted_model_name()
        self.sp_2_unreleased_modname = self._species_2_unreleased_model_name()

    def calc_setup_hash(self):

        import cPickle as pickle
        import hashlib
        # Remove reference to .psc file (varying)
        setupcopy = deepcopy(self.setup)
        del setupcopy.model_psc_input
        # Read the content of the .psc file instead! (unique!)
        setupcopy.psc_content = open(self.psc_file, 'r').readlines()
        setup_serialized = pickle.dumps(setupcopy)
        return hashlib.md5(setup_serialized).hexdigest()

    def _clean_up(self):
        """
        Remove psc file with input data.
        """

        os.remove(self.psc_file)

    def run(self, *args, **kwargs):
        """
        Run and pass on any parameters to the timeseries calculation method
        """

        sim = self._runStochPy(self.nr_traj, self.duration, self.psc_file)

        ts = self._calc_timeseries(sim, *args, **kwargs)

        # clean up: remove temporary file
        self._clean_up()

        return ts

    def _runStochPy(self, nr_traj, duration, psc_file):
        """
         Only for StochPy:
         - *method* [default='Direct'] stochastic algorithm (Direct, FRM, NRM, TauLeaping)
        """

        mod = stochpy.SSA(IsInteractive=False)
        model_dir, model_filename = os.path.split(psc_file)
        mod.Model(model_filename, dir=model_dir)

        # When you run this one, you get in return each single timestep when
        # something happend. From this it is easy to calculate #RNA
        mod.DoStochSim(end=duration, mode='time', trajectories=nr_traj)

        return mod

    def _calc_timeseries(self, sim, include_elongation=False, kind='aborted'):
        """
        Returns a Pandas dataframe with a timeseries of RNA species.

        if kind == 'aborted' (default), then the timeseries consists of the
        total amount of aborted/released RNA.

        if kind == 'unreleased', then the timeseries consists of the total
        amount of unreleased RNA.
        """

        species_names = ['rna_{0}'.format(i) for i in range(1, 21)] + ['FL'] + ['RNAPpoc']

        #species_names += ['productive open complex']
        #species_names += ['unproductive open complex']
        if include_elongation:
            species_names += ['elongating complex']

        model_names = sim.data_stochsim.species_labels

        # Convert data array to int32
        all_ts = np.array(sim.data_stochsim.species, dtype=np.int32)

        # In 2.3, you get a proper time-array here
        time = sim.data_stochsim.time
        # Fix for v 2.2
        time = [t[0] for t in sim.data_stochsim.time]

        data = OrderedDict()

        if kind == 'aborted':
            func = self._get_aborted_species
            mapping = self.sp_2_aborted_modname

        elif kind == 'unreleased':
            func = self._get_unreleased_species
            mapping = self.sp_2_unreleased_modname

        else:
            print('Invalid option')
            1 / 0

        # Slightly overly complex solution for dealing with getting two
        # different kinds of timeseries: for aborted ones we have to
        # accumulate the number of entries into the abortive state, while for
        # unreleased species we can just use the mode's native timeseries.
        for species in species_names:
            model_name = mapping[species]
            # Silently ignore species that were not modelled
            if model_name in model_names:
                sp_index = model_names.index(model_name)
                ts = func(all_ts, sp_index, model_name)

                if type(ts) is np.ndarray:
                    data[species] = ts
                else:
                    print('What?')
                    1 / 0

        df = pd.DataFrame(data=data, index=time)

        # Testing what happens when dropping duplicates. This is good for long
        # simulations: you can get a 10-fold decrease in size.
        # XXX but you discovered that this has unintended consequences when
        # not accumulating, since the row pattern may repeat and then pandas
        # thinks that the rows are duplicates!
        if kind == 'aborted':
            df = df.drop_duplicates()

        return df

    def _get_unreleased_species(self, all_ts, sp_index, model_name):
        """
        Get the timeseries of # unreleased RNA for the given species. This
        corresponds to native timeseries in the model.
        """

        ts = all_ts[:, sp_index]

        return ts

    def _get_aborted_species(self, all_ts, sp_index, model_name):
        """
        Get the timeseries of # aborted RNA for the given species.
        """

        # Some states are sinks, no need to accumulate
        sinks = ['productive open complex', 'unproductive open complex',
                 'elongating complex', 'FL']

        # Sinks are easy
        if model_name in sinks:
            ts = all_ts[:, sp_index]
        else:
            # For backtracked states, accumulate to find #aborted RNA of reach length
            #change = all_ts[:, sp_index]

            change = all_ts[1:, sp_index] - all_ts[:-1, sp_index]

            # ignore departures from abortive states
            change[change == -1] = 0

            # accumulate
            cumul = np.cumsum(change)
            # We lose the first timestep. It's anyway impossive to produce an
            # abortive RNA in the first timestep. So pad a 0 value to the
            # array to make it fit with the time index.
            #compensate for first timstep
            ts = np.insert(cumul, 0, 0)

        return ts

    def _species_2_unreleased_model_name(self):

        d = {}
        for nt in range(1, 21):
            key = 'rna_{0}'.format(nt)
            if nt < 10:
                species = 'RNAP{0}__'.format(nt)
            else:
                species = 'RNAP{0}_'.format(nt)

            d[key] = species

        d['RNAPpoc'] = 'RNAPpoc'
        d['unproductive open complex'] = 'RNAPuoc'
        d['elongating complex'] = 'RNAPelc'
        d['RNAPoc'] = 'RNAPoc'
        d['FL'] = 'RNAPflt'

        return d

    def _species_2_aborted_model_name(self):

        d = {}
        for nt in range(1, 21):
            key = 'rna_{0}'.format(nt)
            if nt < 10:
                species = 'RNAP{0}_b'.format(nt)
            else:
                species = 'RNAP{0}b'.format(nt)

            d[key] = species

        d['productive open complex'] = 'RNAPpoc'
        d['unproductive open complex'] = 'RNAPuoc'
        d['RNAPoc'] = 'RNAPoc'
        d['RNAPpoc'] = 'RNAPpoc'
        d['elongating complex'] = 'RNAPelc'
        d['FL'] = 'RNAPflt'

        return d

    def _species_2_model_name(self):
        """
        Mapping between species identifiers (rna_2) and species names in model (RNAP2__)
        """

        d = {}
        for nt in range(2, 21):
            key = 'rna_{0}'.format(nt)
            if nt < 10:
                species = ['RNAP{0}_b'.format(nt), 'RNAP{0}__'.format(nt)]
            else:
                species = ['RNAP{0}b'.format(nt), 'RNAP{0}_'.format(nt)]

            d[key] = species

        d['productive open complex'] = ['RNAPpoc']
        d['unproductive open complex'] = ['RNAPuoc']
        d['elongating complex'] = ['RNAPelc']
        d['FL'] = ['RNAPflt']

        return d
