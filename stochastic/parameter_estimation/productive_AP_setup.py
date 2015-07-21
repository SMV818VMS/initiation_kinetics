"""
This is a parameter estimation file for estimating the transcript distribution
for N25 according to kinetic data in Vo 2003.
"""
import sys
sys.path.append('/home/jorgsk/Dropbox/phdproject/transcription_initiation/data')
from ITSframework import calc_abortive_probability
from initial_transcription_model import ITModel, ITSimulationSetup
from metrics import rmse
from KineticRateConstants import ITRates

import numpy as np
import pandas as pd
from numpy.random import uniform

from ipdb import set_trace as debug  # NOQA


class estimator_setup(object):

    def __init__(self, initial_fraction, stoi_setup, name, init_RNAP,
                 sim_end, observation_data):

        self.init_fraction = initial_fraction
        self.name = name
        self.init_RNAP = init_RNAP
        self.duration = sim_end
        self.stoi_setup = stoi_setup
        self.observation_data = observation_data

    def parameters(self, parameter_ranges=False):
        """
        This method will be sampled n times to get n different parameter
        values. Two approaches. One is to increase FL% and deduct from 2nt%,
        the other is to deduct from 2nt_frac and add to FL%. We know that FL%
        should be between 10 and 30 % for ONLY productive complexes.

        What happens when you simply move a fraction from 2nt to FL. You're
        going to change the APs of the other positions you know. Yeah but what
        can I do?

        The result is a bit crazy: 1% abortive at 2nt for FL and only 1%
        unproductive complexes. And still we don't produce abortive product
        fast enough. Which indicates that productive complexes should have
        higher abortive probabilities earlier on... :S. If anything, N25
        should have lower AP.

        TODO: make plots for frac and APs for this whole process so you can
        compare with results of modeling. Build intuation in this way.
        """

        # Initial boundaries of parameter space
        if parameter_ranges is False:
            parameter_ranges = {'frac_2nt':   {'low': 0.05, 'high': 0.40},
                                'unprod_frac': {'low': 0.01, 'high': 0.10}}

        # Generate random values
        parameters = {}
        for name, ranges in parameter_ranges.items():
            parameters[name] = uniform(low=ranges['low'], high=ranges['high'])

        return parameters

    def simulation(self, parameters):

        frac_2nt = parameters['frac_2nt']
        unprod_frac = parameters['unprod_frac']

        # Find new abortive probability with updated RNA fraction
        new_aps = self._calc_new_aps(frac_2nt)

        # Important: make R in here
        R = ITRates(self.name, nac=9.2, unscrunch=1.6, escape=5,
                    custom_AP=new_aps)

        # Create simulation setup
        sim_setup = ITSimulationSetup(self.name, R, self.stoi_setup,
                                      self.init_RNAP, self.duration,
                                      unproductive_frac=unprod_frac)

        model = ITModel(sim_setup)

        ts = model.run()

        # Return model data only for those time points that there
        # exists measurable data.
        simulation_result = self._extract_timeseries(ts)

        return simulation_result

    def observation(self):
        """
        Returns timeseries of abortive and fl in one array, abortive first
        I think this method should be called observation not evaluation;
        evalution happens in the likelihood function.
        """
        # Make a single array that joins the two timeseries for comparison
        # with results
        observation = np.append(self.observation_data['Abortive experiment'],
                                self.observation_data['FL experiment'])
        return observation

    def evaluation(self, simulation, observation):

        measure_of_fit = rmse(simulation, observation)

        return measure_of_fit

    def _calc_new_aps(self, frac_2nt):

        # From 1% to max 2nt fraction
        init_2nt = self.init_fraction[0]
        new_2nt = uniform(0.20, 0.3)

        diff = init_2nt - new_2nt

        # update fl fraction with the difference
        new_fl_fraction = self.init_fraction[-1] + diff

        new_fraction = np.copy(self.init_fraction)
        new_fraction[0] = new_2nt
        new_fraction[-1] = new_fl_fraction

        assert new_fraction.sum() <= 1

        # calculate AP based on this new fraction
        new_ap = calc_abortive_probability(new_fraction)

        return new_ap

    def _extract_timeseries(self, model):

        # First, make a copy of the experiment dataframe so you don't alter it for
        # future comparisons.
        experiment = self.observation_data

        # Add up all abortive rna and add as separate column
        cols = model.columns.tolist()
        cols.remove('FL')
        model['Abortive model'] = model[cols].sum(axis=1)
        # Then remove the individual rna species columns
        model.drop(cols, inplace=True, axis=1)

        # Normalize the model RNA to max abortive RNA
        model = model/model.max().max()

        # Get the values in the model closest to those in the experiment using
        # linear interpolation
        joined = pd.concat([model, experiment]).sort_index().\
                   interpolate().reindex(experiment.index)

        # Turns out it's tricky to reproduce the curves from the experiments.
        # You end up with 1% nt2 and 1% nonproductive complex. And even then
        # the abortive synthesis is not fast enough to keep up.
        # Dang, what does the profile have to look like for N25 for this to
        # add up? You need more control: separate out the estimator of
        # productive % and productive AP and plot it together with the
        # unproductive FL and AP. That will give you a better feeling for
        # things. Remember to take time into the equation whatever you do.

        # XXX ALSO: important point. In single molecule studies of elongation
        # there have been reported these complexes that are inactive or doing
        # nothing or whatev. Look up some references and see if it provides
        # any clues. You got some now that talk about transition from fast to
        # slow complexes: Artsimovitch and Landic 2000; John 2000; the latter
        # citing Eire 1993. What about later studies? Yes! See Kareeva and
        # Kashlev 2009, very relevant, read after siesta! And Weixlbaumer. You
        # may have read them before: read your summary. Proposing that a
        # kinked bridge helix can be spelling trouble; if so, the kink
        # persists after at leat 2nt abortive release.

        # could this be the same? Indicates a slower rate of
        # initiation. Furthermore, hey, can aborted 2nt products be re-used as
        # as primers for initiation? Will be less efficient and will increase
        # propensity for unproductive complex formation.
        # Interesting is that 2nt products do not require translocation, so if
        # something is amiss with the translocation machinery (such as the
        # trigger loop, producing 2nt products will not be impaired, but 3nt
        # and further will!

        # XXX OK assume that the above explains the mechanism for unproductive
        # complexes, what gives? Well, you may assume a NAC that is slower,
        # but then it seems like AP should be higher: there is a non-trivial
        # tangle there. Even if you reduce NAC for AP, you are still having
        # too rapid production of FL transcript for the productive complex.
        # What should you do? Back down from the estimate you did before?

        # XXX What is different between steady state transcription and single
        # round transcription? If you knew that it would perhaps be easier to
        # say something about hwo to interpret the APs. Although you have
        # alreay used them for Revyakin's data...

        # XXX explanation for why [NTP] helps reduce unproductive: less chance
        # of backstepping/elementalpausing.

        # XXX a lot of thinking: time to act. Use what you have and at least
        # start estimating; you'll need that anyway.
        # XXX and you need to make some plots to see if your thoughts make
        # sense! Too much thinking!! For example, what if you take away from 2
        # and 3 nt aborts, and give not everything to FL but some to the
        # middle? Even though there may be around 35% FL in the experiment,
        # there should still be some to be spread around in the middle,
        # ensuring high initial abortive synthesis.

        model_data_at_obs_times = np.append(joined['Abortive model'], joined['FL'])

        return model_data_at_obs_times

