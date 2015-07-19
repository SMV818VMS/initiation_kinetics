"""
This is a parameter estimation file for estimating the transcript distribution
for N25 according to kinetic data in Vo 2003.
"""
import sys
sys.path.append('/home/jorgsk/Dropbox/phdproject/transcription_initiation/data')
from ITSframework import calc_abortive_probability
from initial_transcription_model import ITModel, ITSimulationSetup

import numpy as np
import pandas as pd

from ipdb import set_trace as debug  # NOQA


class estimator_setup(object):

    def __init__(self, initial_fraction, stoi_setup, R, name, init_RNAP,
                 sim_end, observation_data):

        self.init_fraction = initial_fraction
        self.R = R
        self.name = name
        self.init_RNAP = init_RNAP
        self.duration = sim_end
        self.stoi_setup = stoi_setup
        self.observation_data = observation_data

        # Make a single array that joins the two timeseries for comparison
        # with results
        self.sea_truth = np.append(observation_data['Abortive experiment'],
                                   observation_data['FL experiment'])

    def parameters(self):
        """
        This method will be sampled n times to get n different parameter
        values. Two approaches. One is to increase FL% and deduct from 2nt%,
        the other is to deduct from 2nt_frac and add to FL%. We know that FL%
        should be between 10 and 30 % for ONLY productive complexes.

        What happens when you simply move a fraction from 2nt to FL. You're
        going to change the APs of the other positions you know. Yeah but what
        can I do?
        """

        # From 1% to max 2nt fraction
        init_2nt = self.init_fraction[0]
        new_2nt = np.random.uniform(0.01, init_2nt)
        new_2nt = 0.01

        diff = init_2nt - new_2nt

        # update fl fraction with the difference
        new_fl_fraction = self.init_fraction[-1] + diff

        new_fraction = np.copy(self.init_fraction)
        new_fraction[0] = new_2nt
        new_fraction[-1] = new_fl_fraction

        assert new_fraction.sum() <= 1

        # calculate AP based on this new fraction
        new_ap = calc_abortive_probability(new_fraction)

        # vary the % of unproductive complexes from 1% to 20%
        unproductive_complex = np.random.uniform(0.01, 0.2)
        unproductive_complex = 0.01

        params = {'ap': new_ap, 'unprod_pct': unproductive_complex}
        self.new_fraction = new_fraction

        return params

    def simulation(self, params):

        aps = params['ap']
        unprod_pct = params['unprod_pct']

        # Update with new APs
        self.R.set_custom_AP(aps)

        # Create simulation setup (important: with updated % unproductive
        # sequences)
        sim_setup = ITSimulationSetup(self.name, self.R, self.stoi_setup,
                                      self.init_RNAP, self.duration,
                                      unproductive_pct=unprod_pct)

        model = ITModel(sim_setup)
        ts = model.run()

        # Here, return model data only for those time points that there
        # exists measurable data.

        for_optimization = self._extract_timeseries(self.observation_data, ts)

        simulation = [s/fr-p+1]

        return simulation
     
    def observation(self):
        """
        Returns timeseries of abortive and fl in one array, abortive first
        """
        return self.sea_truth
    
    def likelihood(self, simulation, observation):
        likelihood = -spotpy.likelihoods.rmse(simulation, observation)

        return likelihood

    def _extract_timeseries(self, exper, model):

        # First, make a copy of the experiment dataframe so you don't alter it for
        # future comparisons.
        experiment = exper.copy(deep=True)

        # Add up all abortive rna and add as separate column
        cols = model.columns.tolist()
        cols.remove('FL')
        model['Abortive model'] = model[cols].sum(axis=1)
        # Then remove the individual rna species columns
        model.drop(cols, inplace=True, axis=1)

        # Normalize the model RNA to max abortive RNA
        model = model/model.max().max()

        # Reindex the model-dataset to the times that are available from the
        # experimental data.

        # Get the values in the model closest to those in the experiment using
        # linear interpolation
        #combined_index = pd.concat([model, experiment]).sort_index().interpolate()
        joined = pd.concat([model, experiment]).sort_index().\
                   interpolate().reindex(experiment.index)

        # XXX turns out you can actually inspect joined to get a nice visual
        # feel for the data (good thing you normalized the results!)
        # But XXX there is something weird. With just 0.02 2nt fraction and
        # 0.35 fl fraction and just 0.01 abortive (1/100) there is still just
        # 0.017 (1.7%) FL transcript. That does not add up well. I think some
        # values are not updated properly or something. Definitely not. Even
        # when simulatin for 90 seconds you don't reach the expected
        # percentage of FL transcripts. OK that's a task for tomorrow to find
        # out!

        model_data_at_obs_times = np.append(joined['Abortive model'], joined['FL'])

        debug()

