"""
This is a parameter estimation file for estimating the FL transcript
distribution for N25 according to kinetic data in Vo 2003.

One issue with this data is the lack of sampling in the beginning.

Hey, maybe you should try the other method with a halt in the middle? It has
more samples in the beginning.
"""
import sys
sys.path.append('/home/jorgsk/Dropbox/phdproject/transcription_initiation/data')
from initial_transcription_model import ITModel, ITSimulationSetup
from metrics import rmse
from KineticRateConstants import ITRates
import kinetic_transcription_models as ktm

import numpy as np
import pandas as pd
from numpy.random import uniform

from ipdb import set_trace as debug  # NOQA


class estimator_setup(object):

    def __init__(self, its_variant, init_RNAP, sim_end, observation_data):

        self.its_variant = its_variant
        self.init_RNAP = init_RNAP
        self.duration = sim_end
        self.observation_data = observation_data

    def parameters(self, parameter_ranges=False):
        """
        Genrate sampled parameter values. If boundaries are not provided
        (should be the case for first iteration) boundariers are set here.
        """

        # Initial boundaries of parameter space
        if parameter_ranges is False:
            parameter_ranges = {'nac':         {'low': 5.0,  'high': 12.0},
                                'unscrunch':   {'low': 1.5,  'high': 6.5},
                                'escape':      {'low': 0.5,  'high': 9}}

        # Generate random values
        parameters = {}
        for name, ranges in parameter_ranges.items():
            parameters[name] = uniform(low=ranges['low'], high=ranges['high'])

        return parameters

    def simulation(self, parameters):

        stoi_setup = ktm.ITStoichiometricSetup(escape_RNA_length=12)
        nac = parameters['nac']
        unscrunch = parameters['unscrunch']
        escape = parameters['escape']
        R = ITRates(self.its_variant, nac=nac, unscrunch=unscrunch, escape=escape)
        # Create simulation setup
        sim_setup = ITSimulationSetup(self.its_variant.name, R, stoi_setup,
                                      self.init_RNAP, self.duration)

        model = ITModel(sim_setup)

        ts = model.run()

        # Return model data only for those time points that there
        # exists measurable data.
        simulation_result = self._extract_timeseries(ts)

        return simulation_result

    def observation(self):
        """
        Returns timeseries of abortive and FL in one array, abortive first
        I think this method should be called observation not evaluation;
        evalution happens in the likelihood function.
        """
        obs = self.observation_data['FL experiment']

        # Always return normalized values for comparison

        return obs / obs.max()

    def evaluation(self, sim, obs):

        measure_of_fit = rmse(sim, obs)

        return measure_of_fit

    def _extract_timeseries(self, model):

        # First, make a copy of the experiment dataframe so you don't alter it for
        # future comparisons. Make a dataframe from FL only.
        experiment = pd.DataFrame(self.observation_data['FL experiment'])

        # Normalize to the largest value (FL RNA)
        experiment = experiment / experiment.max().max()

        # Add up all abortive rna and add as separate column
        model = pd.DataFrame(model['FL'])  # get only FL product
        model.rename(columns={'FL': 'FL model'}, inplace=True)
        # Normalize the model RNA to max FL
        model = model/model.max().max()

        # Get the values in the model closest to those in the experiment using
        # linear interpolation
        joined = pd.concat([model, experiment]).sort_index().\
                   interpolate().reindex(experiment.index)

        return np.asarray(joined['FL model'])

