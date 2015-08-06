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

    def run(self, *args, **kwargs):
        """
        Run and pass on any parameters to the timeseries calculation method
        """

        sim = self._runStochPy(self.nr_traj, self.duration, self.psc_file)

        return self._calc_timeseries(sim, *args, **kwargs)

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

    def _calc_timeseries(self, sim, include_elongation=False):
        """
        Returns a Pandas dataframe for all RNA species.
        """

        species_names = ['rna_{0}'.format(i) for i in range(2,21)] + ['FL']

        #species_names += ['productive open complex']
        #species_names += ['unproductive open complex']
        if include_elongation:
            species_names += ['elongating complex']

        model_names = sim.data_stochsim.species_labels

        # Convert data array to int32
        all_ts = np.array(sim.data_stochsim.species, dtype=np.int32)
        # ehy, the time array is weird; maybe a bug?
        #time = sim.data_stochsim.time
        time = [t[0] for t in sim.data_stochsim.time]

        data = OrderedDict()
        for species in species_names:

            ts = self._parse_model_data(species, all_ts, model_names)
            # I know type testing is bad, but how bad is it?
            # Try/except yeah ...
            #data[species] = ts
            if type(ts) is np.ndarray:
                data[species] = ts
            elif type(ts) is int:
                data[species] = np.zeros(len(time), dtype=np.int16)
            else:
                print('What?')
                1/0

        df = pd.DataFrame(data=data, index=time)

        # Testing what happens when dropping duplicates. This is good for long
        # simulations: you can get a 10-fold decrease in size.
        df = df.drop_duplicates()

        return df

    def _parse_model_data(self, species, all_ts, model_names):
        """
        Get the timeseries for the given plot name. This should work regardless of
        the simulation is with RNAPs that are productive, unproductive, or a combination.
        """

        sp2model_name = self._species_2_model_name()

        # The complexes and FL are associated with only 1 name in th model
        nr_names_in_model = len(sp2model_name[species])

        species_simulated = False

        if nr_names_in_model == 1:
            model_name = sp2model_name[species][0]
            if model_name in model_names:
                species_simulated = True
                sp_index = model_names.index(model_name)
                ts = all_ts[:, sp_index]

        # If Xnt, there are 2 possible names, both productive and unproductive
        elif nr_names_in_model == 2:
            ts = np.zeros(all_ts.shape[0], dtype=np.int32)
            for model_name in sp2model_name[species]:

                if model_name in model_names:
                    species_simulated = True
                    sp_index = model_names.index(model_name)
                    # Find -change in #species each timestep (+1=aborted RNA, -1=backtracked)
                    change = all_ts[:-1,sp_index] - all_ts[1:,sp_index]
                    # Set to zero those values that represent entry into the
                    # backstepped state
                    change[change==-1] = 0
                    # Get timeseries of # abortive RNA
                    cumul = np.cumsum(change)
                    # We lose the first timestep. It's anyway impossive to produce an
                    # abortive RNA in the first timestep. So pad a 0 value to the
                    # array to make it fit with the time index.
                    cumul = np.insert(cumul, 0, 0)

                    ts += cumul

        if species_simulated:
            return ts
        else:
            return -1

    def _species_2_model_name(self):
        """
        Mapping between species identifiers (rna_2) and species names in model (RNAP2__)
        """

        organizer = {}
        for nt in range(2,21):
            key = 'rna_{0}'.format(nt)
            if nt < 10:
                species = ['RNAP{0}_b'.format(nt), 'RNAP{0}_f'.format(nt)]
            else:
                species = ['RNAP{0}b'.format(nt), 'RNAP{0}f'.format(nt)]

            organizer[key] = species

        organizer['productive open complex'] = ['RNAPpoc']
        organizer['unproductive open complex'] = ['RNAPuoc']
        organizer['elongating complex'] = ['RNAPelc']
        organizer['FL'] = ['RNAPflt']

        return organizer


