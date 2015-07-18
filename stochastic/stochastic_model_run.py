import os
import stochpy
stochpy.plt.switch_backend('Agg')
import sys
from math import floor
sys.path.append('/home/jorgsk/Dropbox/phdproject/transcription_initiation/kinetic/model')
import kinetic_transcription_models as ktm
from KineticRateConstants import RateConstants
from ipdb import set_trace as debug  # NOQA

import numpy as np
from scipy.linalg import solve
import pandas as pd
import matplotlib.pyplot as plt
import data_handler
from multiprocessing import Pool
import seaborn  # NOQA
import itertools

from scipy.optimize import curve_fit
import scipy.stats as stats


class SimSetup(object):
    """
    Setup for stochastic simulation.
    """

    def __init__(self, initial_RNAP=100, nr_trajectories=1, sim_end=60,
                 unproductive_pct=0., debug_mode=False):

        self.init_RNAP = initial_RNAP
        self.nr_traj = nr_trajectories
        self.sim_end = sim_end
        self.unprod_pct = unproductive_pct  # if more than 0, simulate unproductive complexes
        self.debug_mode = debug_mode  # use this to produce especially verbose output


def runStochPy(model_path, end, trajectories=1, method='direct'):
    """
     Only for StochPy:
     - *method* [default='Direct'] stochastic algorithm (Direct, FRM, NRM, TauLeaping)
    """

    mod = stochpy.SSA(IsInteractive=False)
    model_dir, model_filename = os.path.split(model_path)
    mod.Model(model_filename, dir=model_dir)

    # When you run this one, you get in return each single timestep when
    # something happend. From this it is easy to calculate #RNA
    mod.DoStochSim(end=end, mode='time', trajectories=trajectories)

    # This is fast, but no way to get the #RNA produced.
    #mod.DoStochKitStochSim(trajectories=1, IsTrackPropensities=True,
                           #customized_reactions=None, solver=None, keep_stats=True,
                           #keep_histograms=True, endtime=70)

    return mod


def Run(its_variant, R, reaction_setup, sim_setup, return_scrunch=False,
        return_model_rna=False, return_timeseries_FL_total_abortive=False):
    """
    Run stochastic model and return #AbortedRNA and #FL. Is this general
    enough? Should you also return timeseries? Or perhaps you should just
    return the model object and let the receiving end decide what to do with
    it.
    """

    # Make a graph for the system
    model_graph = ktm.CreateModel_3_Graph(R, its_variant, reaction_setup,
                                          sim_setup.debug_mode)

    # From the setup, find how many productive and how many unproductive RNAPs
    # should be simulated
    nr_prod_RNAP, nr_unprod_RNAP = calc_nr_RNAPs(reaction_setup, sim_setup)

    # Set initial values and extract values from the graph for simulation
    reactions, initial_values, parameters = \
    ktm.GenerateStochasticInput(model_graph, nr_prod_RNAP, nr_unprod_RNAP)

    # Write a .psc file for the system
    psc_dir = '/home/jorgsk/Dropbox/phdproject/transcription_initiation/kinetic/input/psc_files'
    ktm.write_psc(reactions, initial_values, parameters, its_variant, psc_dir)

    model_input = os.path.join(psc_dir, its_variant + '.psc')

    model = runStochPy(model_input, trajectories=sim_setup.nr_traj, end=sim_setup.sim_end)

    if return_scrunch:
        return calc_scrunch_distribution(model, sim_setup.init_RNAP)
    elif return_model_rna:
        nr_rna = get_nr_RNA(model)
        model_ts = calc_timeseries('all_rna', model)
        return nr_rna, model_ts
    else:
        return model


def calc_nr_RNAPs(reaction_setup, sim_setup):

    nr_RNAP = sim_setup.init_RNAP

    if reaction_setup['unproductive']:
        unproductive_pct = sim_setup.unprod_pct
        assert 0 < unproductive_pct < 1
        nr_RNAP_unproductive = floor(nr_RNAP * unproductive_pct)
        nr_RNAP_productive = nr_RNAP - nr_RNAP_unproductive

    elif reaction_setup['unproductive_only']:
        nr_RNAP_unproductive = nr_RNAP
        nr_RNAP_productive = 0

    else:
        nr_RNAP_unproductive = 0
        nr_RNAP_productive = nr_RNAP

    return nr_RNAP_productive, nr_RNAP_unproductive


def species_2_model_name():
    """
    
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


def calc_timeseries(species_names, sim):
    """
    New and improved. Provide the species you want:
    "productive open complex"
    "unproductive open complex"
    "xnt", where x is from 2 to 20
    "elongating complex"
    "FL"

    You can pass a single species or a list of species.
    Pass 'all_rna' to get all rna species.

    TODO reimplement an accepted return code when the requested species names
    were not part of the simulation.
    """

    if isinstance(species_names, str):
        if species_names == 'all_rna':
            species_names = ['rna_{0}'.format(i) for i in range(2,21)] + ['FL']
        else:
            species_names = [species_names]

    model_names = sim.data_stochsim.species_labels

    # Convert data array to int32
    all_ts = np.array(sim.data_stochsim.species, dtype=np.int32)
    # ehy, the time array is weird; maybe a bug?
    #time = sim.data_stochsim.time
    time = [t[0] for t in sim.data_stochsim.time]

    data = {}
    for species in species_names:

        ts = parse_model_data(species, all_ts, model_names)
        data[species] = ts

    df = pd.DataFrame(data=data, index=time)

    return df


def parse_model_data(species, all_ts, model_names):
    """
    Get the timeseries for the given plot name. This should work regardless of
    the simulation is with RNAPs that are productive, unproductive, or a combination.
    """

    sp2model_name = species_2_model_name()

    # The complexes and FL are associated with only 1 name in th model
    nr_names_in_model = len(sp2model_name[species])

    if nr_names_in_model == 1:
        model_name = sp2model_name[species][0]
        if model_name in model_names:
            sp_index = model_names.index(model_name)
            ts = all_ts[:, sp_index]

    # If Xnt, there are 2 possible names, both productive and unproductive
    elif nr_names_in_model == 2:
        ts = np.zeros(all_ts.shape[0], dtype=np.int32)
        for model_name in sp2model_name[species]:

            if model_name in model_names:
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

    return ts


def plot_timeseries_copy(species, sim, sim_name):
    """
    Plot timeseries of different molecular species from initial
    transcription modelling.
    """

    df = calc_timeseries(species, sim)

    f, ax = plt.subplots()
    df.plot(ax=ax)

    ax.set_ylabel('# of species')
    ax.set_xlabel('Time (seconds)')

    file_name = 'Timeseries_{0}.pdf'.format(sim_name)

    f.suptitle('Stochastic simulation of initial transcription for {0}'.format(sim_name))
    f.tight_layout()
    f.savefig(file_name, format='pdf')

    plt.close(f)


def bai_rates(nuc, NTP, EC):
    """
    Implements the method of Bai et. al to find the rate of NAC for a given
    [NTP].

    For your model, it is important to use the same NAC for all values, since
    any varition of NAC may be baked into the APs.

    NTP should be given in micro molar.

    Damn, your EC values are too high. Could be because of special conditions
    for RNA/DNA hybrid and scrunched bubbule. The Bai model does not apply for
    initial transcription. Can we just assume that it's going to be as for
    elongation? The average DeltaDeltaG value is 1.6, but the forward rates
    you get way too big.

    Anyway, there is no correlation with purines here C > A > G > U is the
    speed order.
    """

    dset = data_handler.ReadData('dg100-new')
    its = [i for i in dset if i.name == 'N25'][0]

    k_max = {'A': 50, 'U':18, 'G': 36, 'C': 33, 'T': 18}
    K_d = {'A': 38, 'U':24, 'G': 62, 'C': 7, 'T': 24}

    # Find EC for this sequence (only DNADNA and RNADNA, no DG3D since that
    # was not used by Bai et al)
    c1 = 1
    c2 = 2
    c3 = 3
    #rna_len = its.calculated_msat
    rna_len = 20
    its.calc_keq(c1, c2, c3, msat_normalization=False, rna_len=rna_len)

    # Old Bai
    nac2 = (24.7 * NTP) / (15.6*(1 + EC) + NTP)
    # New Bai
    #nac1 = (k_max[nuc] * NTP) / (K_d[nuc]*(1 + EC) + NTP)
    EC = 1.6
    for nuc in list("GATC"):
        nac1 = (k_max[nuc] * NTP) / (K_d[nuc]*(1 + EC) + NTP)
        print nuc, nac1

    #print nac2/nac1

    return nac2


def uniform_sample(a, b, nr_samples):
    return np.random.sample(nr_samples) * (b - a) + a


def get_estim_parameters(method='manual'):
    """
    Ideally you'd like for this to be iteratively. For now, just test simply.
    method: manual, uniform, normal
    If method == normal, must provide center of distribution and variance

    For N25 with escape at 14, with 0 ap the last 3 steps, and with GreB+ APs,
    the best fit seems to be around Nac=9.1, abort = 1.6, and Escape= ca 5.
    """

    params = []
    if method == 'manual':
        for nac in np.linspace(2.5, 10, 5):
            for abortive in [nac/i for i in [2, 4, 8]]:
                for escape in [nac, nac/2, nac/10.]:
                    params.append((nac, abortive, escape))

    elif method == 'uniform':
        for nac in uniform_sample(10, 40, 5):
            for abortive in uniform_sample(10, 50, 5):
                for escape in uniform_sample(10, 50, 5):
                    params.append((nac, abortive, escape))

    #  This was narrowed in
    #elif method == 'uniform':
        #for nac in uniform_sample(8, 10, 4):
            #for abortive in uniform_sample(1.5, 2.5, 4):
                #for escape in uniform_sample(3.5, 7.5, 4):
                    #params.append((nac, abortive, escape))

    return params


def revyakin_datafit():
    """
    Q: First, read the % transcript from Revyakin and see if it differs from Hsu?
    A: It's similar, but those were for the non-single-molecule studies.

    Result: You get a similar distribution to Revyakin using GreB APs, but you have to set
    nac/to_fl to 4 or 5.

    Using Hsu AP you also get a good match, but you have to use nac/to_fl of
    20, which is arguably too high.

    Now the question of forward rate comes into play. So far, you have used
    10b/s. Now is when you need to implement the results of Bai to get
    sequence specific rates? How does that improve/deteriorate your result?
    A: After some fiddling, you've found that 9.5 is a good fit to 100 uM NTP.

    Q: What exactly are the scrunch lifetimes? Are they only considering
    events without abortive release? If not, wouldn't they have called it
    abortive cycling lifetime? Fuck, if you can't use this either, then you
    have to move on :)

    A: They count a scrunch as midpoint of transition 2 to midpoint of transition
    3. So look at figure S10B. Here you see that the scrunching phase includes
    movement up and down... but other figures show multiple recursions to the
    'initial state' -- yes, after full length product has been produced!

    AHA: in their main paper: complexes engaged in abortive cycling were
    observed to be present in the scrunched state: transitions between the
    scrunched and unscrunched state havin the extent of unwinding in the open
    complex were not observed. They infer that forward translocation is fast
    relative to abortive-product release and unscrunching, and also fast
    relative to second-scale temporal resolution. At the same time,
    abortive-product release and unscrunching cannot be very slow either,
    otherwise it would supposedly have been detected with the 1s temporal
    resolution?

    The elongation speed seems to be around 10 bp/s. (see last Supplementary
    figure; Lilian's fragments are neutrally coiled: does it place it
    somewhere in the middle?)

    You can apply some logic here to estimate the speed of initial
    transcription. This would be an important result!!! Really, it is.
    Shortest scrunch-time is measured to be 2.5s (they say 100n, but only show
    15 points...). If 2.5 is the shortest they measure, we may assume that
    this corresponds to an abort-free scrunch. This corresponds to a speed of
    4.5 nt/s. This fits well with the standard model.

    Further, it seems to me that their experiment is not able to distinguish
    between a +10 and +5 state, so how can they say that unscrunching is slow?
    It is at least faster than their resolution. So we'll assume that backstep
    from 10nt RNA and abortive release happens in under 1/s. Also, Margeat
    cannot identify this even with 400ms resolution. So we are at a conundrum
    here. That's always good uh?

    Hey, most abortive initiation happens at the +2 position. But, this should
    look identical to open complex for Revyakin. Then it's kind of weird that
    no open complex is detected? If AP is 35% for 2nt product, then, it would
    be expected that even if it goes on at 10/s, that you would occasionally
    get 10 2nt synthesises in a row, so that one actually spends 1s in the
    open complex position? I would think so. Nevermind.

    If translocation rate is 5/s, it would take 2 seconds to reach promoter
    escape. What should abortive release rate be? Half of NAC, so 2.5/s?

    NOW the length-dependent abortive release rate might come into play ... :D

    OR NOT!! With NAC = 10 and abortive =2 you get a pretty damn good match.
    9/9/1.8 is very good.

    XXX Show that using GreB - that the optimal rates do not make sense, and
    that they make sense with GreB +

    Can you fit the experimental data to a gamma distribution? And to an
    exponential distribution?
    """

    # Setup
    backtrack_method = 'backstep'
    its_variant = 'N25'
    initial_RNAP = 500
    reaction_setup = ktm.ReactionSystemSetup(backtrack_method, escape_start=14,
                                            final_escape_RNA_length=14, abortive_end=14,
                                            abortive_beg=2)

    # We know: NAC should not be faster than 10nt/s and is likely faster than
    # 2.5 nt/s, since the fastest scrunch involving 11 NACs was reported in
    # about 2.5 seconds.
    # We know: abortive should be slower than nac: so less than 10/s, but not
    # much less than 1/s since it has not been captured by Revyakin.
    # Fl: less certain: Patel showed less than for nac, but this may be
    # different for T7, core promoter region, etc.

    # XXX HEY!! You can calculate what NAC needs to be if 20% of scrunches
    # should be less than 1 second: it will be quite high indeed.

    params = get_estim_parameters(method='uniform')
    #params = [(9.2, 1.6, 5)]  # the golden params

    log_file = 'simple_log.log'
    log_handle = open(log_file, 'w')
    log_handle.write('Fit\tFit_extrap\tNac\tAbort\tEscape\n')

    plot = False
    #plot = True

    multiproc = True
    #multiproc = False
    if multiproc:
        pool = Pool(processes=2)
        results = []
    for nac, abrt, escape in params:

        R = RateConstants(variant=its_variant, use_AP=True, nac=nac,
                          abortive=abrt, escape=escape, to_fl=0.2, GreB_AP=True)

        sim_name = get_sim_name(reaction_setup, its_variant)

        # Now just making sure that sim end is large enough; potentially you'd
        # want to ignore simulations that take longer than sim_end
        sim_setup = SimSetup(initial_RNAP=initial_RNAP, sim_end=180)

        kwargs ={'return_scrunch': True}
        args = (sim_name, R, reaction_setup, sim_setup)
        if multiproc:
            #r = apply_async(pool, Run, args)
            r = pool.apply_async(Run, args, kwargs)
            results.append(r)
        else:
            model_scrunches = Run(*args, **kwargs)
            plot_and_log_scrunch_comparison(plot, log_handle, model_scrunches,
                                            initial_RNAP, nac, abrt, escape)

    if multiproc:
        pool.close()
        pool.join()
        model_scrunches = [r.get() for r in results]
        for ms, pars in zip(model_scrunches, params):
            nac, abrt, escape = pars
            plot_and_log_scrunch_comparison(plot, log_handle, ms,
                                            initial_RNAP, nac, abrt, escape)

    log_handle.close()


def plot_and_log_scrunch_comparison(plot, log_handle, model_scrunches, initial_RNAP,
                                    nac, abrt, escape):
    """
    Convenience function for running in parallel
    """
    experiment, extrapolated = get_experiment_scrunch_distribution()
    val, val_extrap = compare_scrunch_data(model_scrunches, experiment, extrapolated)
    entry = '{0}\t{1}\t{2}\t{3}\t{4}'.format(val, val_extrap, nac, abrt, escape)
    log_handle.write(entry + '\n')

    if plot:
        plot_scrunch_distributions(model_scrunches, experiment, extrapolated, entry)


def get_experiment_scrunch_distribution():
    """
    Read the experimental data from file and fit the data. Then use that
    function (together with fit data) to compare with your results.

    Note: not sure why the exponential and gamma distributions don't give a
    good fit. Perhaps you are doing something wrong. The scipy fit stuff is
    very uniform but not very understandble.
    """

    datapath = '/home/jorgsk/Dropbox/phdproject/transcription_initiation/data/Revyakin 2006/Scrunch times.csv'

    df_measurements = pd.read_csv(datapath, index_col='time')
    x = np.asarray(df_measurements.index)
    y = df_measurements['inverse_cumul_experiment']

    #df_measurements['inverse_cumul_experiment'] = df_measurements['inverse_cumul_experiment'][::-1]
    #df_measurements.index = df_measurements.index[::-1]

    #ff, axx = plt.subplots()
    #df_measurements.plot(style='.', ax=axx)

    #ff.savefig('othewaay.pdf')
    #return

    def neg_exp(x, a):
        return np.exp(-a*x)

    popt, pcov = curve_fit(neg_exp, x, y)

    # I think this is a simple negative exponential
    # Does that point to an exponential distibution? Yes, but experiments
    # might not capture the very fast scrunches, so it's missing the beginning
    # of the distribution. Show that the actual distribution is something
    # else.

    sample_size = 1000

    # XXX Fit to gamma distribution and plot
    fit_a, fit_loc, fit_b = stats.gamma.fit(x)
    gamma_data = stats.gamma.rvs(fit_a, loc=fit_loc, scale=fit_b, size=sample_size)
    gamma_data.sort()

    # XXX Fit to exponential distribution and plot
    fit_l, fit_eloc = stats.expon.fit(x)
    expon_data = stats.expon.rvs(scale=fit_l, loc=fit_eloc, size=sample_size)
    expon_data.sort()

    #integrated_prob = np.linspace(1, 1/float(sample_size), sample_size)
    integrated_prob = np.linspace(1/float(sample_size), 1, sample_size)

    df_gamma = pd.DataFrame(data={'gamma':integrated_prob}, index=gamma_data)
    df_expon = pd.DataFrame(data={'exponential':integrated_prob}, index=expon_data)

    y_model = neg_exp(x, popt[0])

    # Make some preductions from this simple model
    xx = np.linspace(0, 40, 1000)
    y_extrapolated = neg_exp(xx, popt[0])
    df_extrapolated = pd.DataFrame(data={'extrap_inv_cumul':y_extrapolated}, index=xx)

    f, ax = plt.subplots()

    df_measurements.plot(style='.', ax=ax, logy=True)

    df_extrapolated.plot(ax=ax, logy=True)

    df_gamma.plot(ax=ax, logy=True)
    df_expon.plot(ax=ax, logy=True)

    ax.set_xlim(0, 20)
    ax.set_ylim(1e-2,1)
    ax.set_xticks(range(0,21))
    f.savefig('scrunch_times_experiment.pdf')

    plt.close(f)

    return df_measurements, df_extrapolated


def compare_scrunch_data(model, experiment, experiment_extrap):
    """
    What should you do with a nan? Indicates that the result was not within
    the proper range.
    """
    # Get the values in the model closest to those in the experiment using
    # linear interpolation
    compare = pd.concat([model, experiment]).sort_index().\
              interpolate().reindex(experiment.index)

    diff = np.abs(compare['inverse_cumul_experiment'] - compare['inverse_cumul_model'])
    distance = np.linalg.norm(diff)

    c_extr = pd.concat([model, experiment_extrap]).sort_index().\
              interpolate().reindex(experiment_extrap.index)

    # Ignore stuff before 0.5 seconds, it's impossible to go that fast,
    # so there will always be nans in the beginning
    c_extr = c_extr[0.5:]

    diff_extr = np.abs(c_extr['extrap_inv_cumul'] - c_extr['inverse_cumul_model'])
    distance_extr = np.linalg.norm(diff_extr)

    return distance, distance_extr


def calc_scrunch_distribution(sim, initial_RNAP):
    """
    Will probably not get same distribution in time as Revyakin, but we might
    get the same shape. They use a different salt to the other experiments,
    and the concentration of that salt matters a lot for the result (see last
    fig in supplementry of Revyakin).
    """

    df = calc_timeseries('FL', sim)

    # Starts with timestep 1!
    escapes = df['FL'][1:] - df['FL'][:-1]

    # Skip first timestep
    time = df.index[1:,0]
    escape_times = time[escapes == 1]

    integrated_probability = np.linspace(1, 1/float(initial_RNAP), initial_RNAP)

    data = {'inverse_cumul_model': integrated_probability}

    scrunch_dist = pd.DataFrame(data=data, index=escape_times)

    return scrunch_dist


def plot_scrunch_distributions(model_scrunches, expt, expt_extrapolated,
                               title=''):
    """
    Now you are plotting the cumulative distribution function, but can you
    also plot the probability density function? Start off with a histogram
    """

    plot_expt = True
    fig_dir = 'figures'

    # XXX take a break here. You need to figure out how to turn the image
    # upside down to make a cumulative distribution function.
    expt['cumul_expt'] = 1 - expt['inverse_cumul_experiment']
    expt_extrapolated['cumul_expt_extrap'] = 1 - expt_extrapolated['extrap_inv_cumul']
    model_scrunches['cumul_model'] = 1 - model_scrunches['inverse_cumul_model']

    # For the model data, insert a 0 to indicate that there is 0 probability
    # of 0 scrunch time ?
    dta = {'inverse_cumul_model': 1., 'cumul_model': 0.}
    index = [0.]
    extraDF = pd.DataFrame(data=dta, index=index)

    model_scrunches = extraDF.append(model_scrunches)

    #for logy in [True, False]:
    for logy in [False]:
        f, ax = plt.subplots()

        ax.set_xlabel('Time (seconds)', size=20)
        ax.set_ylabel('Probability', size=20)

        model_scrunches.plot(y='cumul_model', ax=ax, logy=logy, fontsize=20,
                             label='Model')

        if plot_expt:
            # Plot experiment as scatter
            ax.scatter(np.asarray(expt.index), expt['cumul_expt'],
                       label='Measurements')
            # Plot extrapolated experiment
            ax.plot(np.asarray(expt_extrapolated.index),
                    expt_extrapolated['cumul_expt_extrap'], label='Fit to measurements', ls='--')

        f.set_size_inches(9, 9)

        #f.suptitle(title)

        ax.set_xlim(0, 30)
        ax.set_ylim(-0.02, 1.02)

        if logy:
            ax.set_xlim(0, 30)
            ax.set_xticks(range(0,21))
            filename = 'scrunch_times_cumulative_logy_{0}.pdf'.format(title)
        else:
            filename = 'scrunch_times_cumulative_{0}.pdf'.format(title)

        ax.legend(loc='lower right', fontsize=20)

        # Indicate where experiments cannot measure
        ax.axvspan(0, 1, ymin=0, ymax=1, facecolor='g', alpha=0.1)

        f.savefig(os.path.join(fig_dir, filename))

        plt.close(f)

    hist_plot = False
    if hist_plot:
        just_data = np.asarray(model_scrunches.index)
        df = pd.DataFrame({'Model': just_data})

        # Hey that won't work, it's just a line for the cumulated probability. You
        # need to resample the distribution to get the underlying data.
        #if plot_expt:
            #df['Experiment_fit'] = np.asarray(expt_extrapolated.index)

        f, ax = plt.subplots()
        df.plot(ax=ax, kind='hist', fontsize=20, label=False, bins=50)
        f.set_size_inches(9, 9)
        f.savefig('scrunch_times_histogram.pdf')
        plt.close(f)


def main():

    #revyakin_datafit()

    # XXX OK you are running simulations of estimating AP and you're logging
    # the result. Now you need to get some numbers rolling! You will have to
    # simulate up to maybe 5 minutes?
    # XXX OK, to get a good match for the abortive, you seem to need a high AP
    # for +2, but then you get a poor match for FL.
    # Things to consider:
    #   1) add the % of unproductive complexes to the parameter estimation
    #   process
    #   2) Consider that the productive APs may be quite different from the
    #   nonproductive. Consider that for N25 after 0.5 minute, when
    #   FL_1/2 has been reached, PY is around 35%. I think that when you
    #   adjust AP, you also have to adjust the fraction of transcripts so that
    #   full length ends up with more.

    # After getting a bit closer with estimation of the signature of
    # productive transcripts, take stock of your situation. Count up the
    # results you've got and what, if else, more you need.

    productive_AP_estimate()

    #results = 'simple_log.log'

    #df = pd.read_csv(results, sep='\t', header=0)

    #df.sort('Fit', inplace=True)

    #debug()

    # Plot normalized plots of the 2003 data
    #plot_2003_kinetics()

    #debug()


def plot_2003_kinetics():
    """
    It seems that with the competitive promoter thre is much less abortive
    recycling.

    I think the kinetics of the competitive promoter looks best.
    """

    data ='/home/jorgsk/Dropbox/phdproject/transcription_initiation/data/'
    m1 = os.path.join(data, 'vo_2003/Fraction_FL_and_abortive_timeseries_method_1.csv')
    m2 = os.path.join(data, 'vo_2003/Fraction_FL_and_abortive_timeseries_method_2.csv')

    #df1 = pd.read_csv(m1, sep='\t', index_col='x')
    #df2 = pd.read_csv(m2, sep='\t', index_col='x')
    # Read up to 10 minutes
    df1 = pd.read_csv(m1, index_col=0)[:10]
    df2 = pd.read_csv(m2, index_col=0)[:10]

    # find max values for both
    max1 = df1.max().max()
    max2 = df2.max().max()

    df1_norm = df1/max1
    df2_norm = df2/max2

    # XXX the data is already interpolated by Eugene during export. I'm
    # guessing linear interpoltation.

    f, ax = plt.subplots()

    df1_norm.plot(ax=ax)
    df2_norm.plot(ax=ax)

    ax.set_xlabel('time (minutes)')
    ax.set_ylabel('relative to maximum abortive signal')

    ax.set_xticks(range(0,11))
    ax.set_yticks(np.arange(0,1.1, 0.1))

    ax.set_ylim(0,1.01)

    # find correlation between the two curves
    corr1 = df1_norm.corr().min().min()
    corr2 = df2_norm.corr().min().min()

    title = 'Abortive/FL curve correlations: competitive promoter: {0:.2f}; halted elongation: {1:.2f}'.format(corr2, corr1)

    f.suptitle(title)

    f.savefig('2003_kinetics_normalized.pdf')


def make_ap_variation(its_variant, nt_range, adjust_2003_FL):
    """
    Use the original AP distributions and modify them accordingly
    """
    initial_vals = get_starting_APs(its_variant, adjust_2003_FL)

    # Set up a dictionary of variation: (min, max) and steps
    vardict = {2: (0.5, 0.9),
               3: (0.05, 0.30),
               4: (0.10, 0.30),
               5: (0.02, 0.15),
               6: (0.10, 0.25),
               7: (0.02, 0.15),
               8: (0.30, 0.50),
               9: (0.20, 0.40),
               10: (0.20, 0.40),
               11: (0.01, 0.10),
               12: (0.01, 0.10),
               13: (0.01, 0.10),
               14: (0.01, 0.10),
               }
    # How many different values do you want to span?
    n = 2

    variations = [uniform_sample(vardict[nt][0], vardict[nt][1], n) for nt in nt_range]
    APs = []
    # Go though all combinations and modify the existing abortive fractions
    combos = itertools.product(*variations)
    for var in combos:
        ap = [i for i in initial_vals]  # make a copy
        for nt in nt_range:
            ap[nt-2] = var[nt-2]

        APs.append(ap)

    return APs


def productive_AP_estimate():

    # Setup
    backtrack_method = 'backstep'
    its_variant = 'N25'
    unproductive_pct = 0.10
    initial_RNAP = 100
    sim_end = 60. * 5

    #reaction_setup = ktm.ReactionSystemSetup(backtrack_method, escape_start=14,
                                            #final_escape_RNA_length=14, abortive_end=14,
                                            #abortive_beg=2, unproductive_only=True)

    reaction_setup = ktm.ReactionSystemSetup(backtrack_method, escape_start=14,
                                            final_escape_RNA_length=14, abortive_end=14,
                                            abortive_beg=2, unproductive=True)

    #reaction_setup = ktm.ReactionSystemSetup(backtrack_method, escape_start=14,
                                            #final_escape_RNA_length=14, abortive_end=14,
                                            #abortive_beg=2)

    # Hey, why are you comparing with the 2003 data? Well, it is to try to use
    # 10% of nonproductive complexes and see that they continue to produce
    # abortive transcript in the amount that is observed.
    # OK, so first plot the 2003 data kthnx.

    # So there are two fits: one for the APs, and another for the timeseries.

    # Only change a few parameters in the beginning.
    nt_range = range(2, 4)
    APs = make_ap_variation(its_variant, nt_range, adjust_2003_FL=True)

    log_file = 'productive_AP_estimator.log'
    log_handle = open(log_file, 'w')
    fits = ['Fit_fraction', 'Fit_FL', 'Fit_ab']
    header = '\t'.join(fits + ['{0}nt'.format(nt) for nt in nt_range])
    log_handle.write(header + '\n')

    # Precalculate this value for comparison with model results
    experiment_RNA_fraction = get_transcript_fraction(its_variant)

    # Fetch kinetic data for N25 for comparison
    data ='/home/jorgsk/Dropbox/phdproject/transcription_initiation/data/'
    m2 = os.path.join(data, 'vo_2003/Fraction_FL_and_abortive_timeseries_method_2.csv')
    N25_kinetic_ts = pd.read_csv(m2, index_col=0)[:10]

    multiproc = True
    #multiproc = False
    if multiproc:
        pool = Pool(processes=2)
        results = []

    for ap in APs:
        R = RateConstants(variant=its_variant, use_AP=True, nac=9.2, abortive=1.6,
                          to_fl=5, custom_AP=ap)

        sim_name = get_sim_name(reaction_setup, its_variant)

        sim_setup = SimSetup(initial_RNAP=initial_RNAP, sim_end=sim_end,
                             unproductive_pct=unproductive_pct)

        args = sim_name, R, reaction_setup, sim_setup
        kwargs = {'return_model_rna': True,
                  'return_timeseries_FL_total_abortive': True}
        if multiproc:
            r = pool.apply_async(Run, args, kwargs)
            results.append(r)
        else:
            model_RNA_nr, model_ts = Run(*args, **kwargs)
            # Compare the % of final species with the experimentally measured values
            val1 = compare_fraction(model_RNA_nr, experiment_RNA_fraction)
            val2, val3 = compare_timeseries(N25_kinetic_ts, model_ts)

            write_AP_estimator_log(log_handle, ap, val1, val2, val3, nt_range)

    if multiproc:
        mRfs = [r.get() for r in results]
        for (model_RNA_nr, model_ts), ap in zip(mRfs, APs):
            val1 = compare_fraction(model_RNA_nr, experiment_RNA_fraction)
            val2, val3 = compare_timeseries(N25_kinetic_ts, model_ts)

            write_AP_estimator_log(log_handle, ap, val1, val2, val3, nt_range)

    log_handle.close()


def compare_timeseries(exper, model):
    """
    Use the method with a competitive promoter as kinetic info on N25.

    You can compare shape with spearman.
    But you must also compare the distance, to match in time, not just shape.
    """
    # First, make a copy of the experiment dataframe so you don't alter it for
    # future comparisons.
    experiment = exper.copy(deep=True)

    # Add up all abortive rna and add as separate column
    cols = model.columns.tolist()
    cols.remove('FL')
    model['Abortive model'] = model[cols].sum(axis=1)
    # Then remove the individual rna species columns
    model.drop(cols, inplace=True, axis=1)

    # Then rename 'FL' to 'FL model'
    model.rename(columns={'FL': 'FL model'}, inplace=True)

    ren ={'Abortive competitive promoter': 'Abortive experiment',
          'FL competitive promoter': 'FL experiment'}

    experiment.rename(columns=ren, inplace=True)

    # Convert N25 timeseries to seconds
    experiment.index = experiment.index * 60.

    # Keep experimental data close to model data
    experiment = experiment[:model.index[-2]]

    # Normalize N25 timeseries to max abortive RNA
    experiment = experiment/experiment.max().max()

    # Normalize the model RNA to max abortive RNA
    model = model/model.max().max()

    # Awesome! You've got them both normalized now. Perfect time to start some
    # comparisons!!
    # But, one issue. For the model you've got a lot more data very early.
    # This is probably not usable.

    # Reindex the model-dataset to the times that are available from the
    # experimental data.

    # Get the values in the model closest to those in the experiment using
    # linear interpolation
    #combined_index = pd.concat([model, experiment]).sort_index().interpolate()
    joined = pd.concat([model, experiment]).sort_index().\
              interpolate().reindex(experiment.index)

    f, ax = plt.subplots()
    joined.plot(ax=ax)
    f.savefig('justlookatid.pdf')

    distance_FL = np.linalg.norm(abs(joined['FL model'] -
                                     joined['FL experiment']))
    distance_ab = np.linalg.norm(abs(joined['Abortive model'] -
                                     joined['Abortive experiment']))

    return distance_FL, distance_ab


def write_AP_estimator_log(handle, ap, fit_fraction, fit_ts_FL, fit_ts_ab, nt_range):
    """
    You got three measures. The fit in terms of final fraction of species; the
    fit in terms of timeseries of FL product; the fit in terms of timeseries
    of abortive product.

    I'm sure there are issues with these measures. For example, investigate if
    there is some time-dependent bias.
    """
    measures = [str(fit_fraction), str(fit_ts_FL), str(fit_ts_ab)]
    line = '\t'.join(measures + [str(ap[i]) for i in range(len(nt_range))])
    handle.write(line + '\n')


def get_starting_APs(its_variant, adjust_2003_FL=False):
    """
    """

    dset = data_handler.ReadData('dg100-new')
    its = [i for i in dset if i.name == its_variant][0]

    if adjust_2003_FL:
        #make fl equal to 12 % Subtract % according to the distribution of
        #abortive values
        new_fl = 0.12
        diff_fl = new_fl - its.fraction[-1]
        abortive_diff = its.abortive_fraction_normal * diff_fl
        total_diff = np.append(-abortive_diff, diff_fl)

        rna_frac_adj = its.fraction + total_diff

        APs = its.calc_abortive_probability(rna_frac_adj)

    else:
        APs = its.abortive_probability

    return APs


def compare_fraction(model_RNA_nr, experiment_RNA_fraction):
    """
    You probably need to improve this measure.
    """

    # Model fraction from #RNA
    model_RNA_fraction = model_RNA_nr / model_RNA_nr.sum()

    diff = experiment_RNA_fraction - model_RNA_fraction

    return np.linalg.norm(diff)


def get_transcript_fraction(its_name):
    """
    Provide the experimental fraction of RNA including FL. This is a value
    that includes productive and nonproductive complexes.
    """

    dset = data_handler.ReadData('dg100-new')
    its = [i for i in dset if i.name == its_name][0]

    return its.fraction


def get_nr_RNA(sim):
    """
    Get the nr of RNA from 2nt->20nt and FL (size = 20)
    """

    species = ['rna_{0}'.format(i) for i in range(2,21)] + ['FL']

    df = calc_timeseries(species, sim)

    # Get finalvalue and organize in an array
    nr_rna = np.asarray([df[s].tail().tolist()[-1] for s in species])

    return nr_rna


def productive_species(nr_unproductive_RNA, transcript_fraction):
    """
    We have X_i = (a_i + b_i) / (A + B),
    where:
    X_i = fraction of transcripts of length i
    a_i = # unproductive rna of species i
    b_i = # productive rna of species i
    A   = sum(a_i) = total # of unproductive species
    B   = sum(b_i) = total # of productive speices

    If we substitute B with sum(b_i) and solve for b_i, we end up with a
    system of linear equations: Ab = a on the form

     [(X_2-1)/X_2 1 1 .... 1]b2 = [a_2/Y_2 - A]
     [1 (X_3-1)/X_3 1 .... 1]b2 = [a_3/Y_3 - A]
     [......................]bk = [a_k/Y_k - A]
     [1 ........(X_n-1)/X_n]]bn = [a_n/Y_n - A]

    Since we know X_i (experiment) and a_i and A (model), we can calculate b_i
    and therefore B.

    From b_i we can find % of and AP for productive species.

    It seems not. And really, without knowing the distribution between A and
    B, it's as if something is missing.
    """

    A = sum(nr_unproductive_RNA)
    Y = transcript_fraction
    a = nr_unproductive_RNA

    nr_rnas = a.size

    # in the equation Hb = k, build H matrix and k vector
    matrix2b = []
    vector2b = []
    # Subtract 1 for FL: assumes all experiments have FL (that's true for DG100 and DG400)
    #nr_nonzero_transcript_lengths = sum(Y>0) - 1
    nr_nonzero_transcript_lengths = sum(Y>0)
    # iterate over the indices of Y that are non-zero (will be 1,2,3..14 and
    # 21) for FL at the end
    for i in range(nr_rnas):
        row = np.ones(nr_nonzero_transcript_lengths) * -Y[i]
        if Y[i] > 0:
            row[i] = 1 - Y[i]
            v = A * Y[i] - a[i]
        else:
            # No experimental data, no model data.
            continue

        matrix2b.append(row)
        vector2b.append(v)

    # finally, add entry for FL
    row = np.ones(nr_nonzero_transcript_lengths) * - Y[-1]
    row[-1] = 1 - Y[-1]

    matrix2b.append(row)
    vector2b.append(A * Y[-1])

    H = np.asarray(matrix2b)
    k = np.asarray(vector2b)

    b = solve(H, k)
    # DAMNIT! the results are almost good. But not quite.
    # And the least squares solution seems ALMOST good, except for a negative
    # number first.
    # Determinant of matrix is almost zero.
    # Also, the sum of A is bigger than the sum of B, perhaps expected since
    # there are more shorter transcripts? But, it's only supposed to be 10% of
    # total RNAP ... hard to figure out.

    # Ack, I think tis a dead end. You would need some information about the
    # total amount of RNA of species A or B: you need an extra information you
    # now lack. So. Go back to estimating the backtracking probabilities of
    # productive species via a parameter estimation process.

    return b


def get_sim_name(reaction_setup, its_variant):

    if reaction_setup['unproductive']:
        sim_name = '{0}_also_unproductive'.format(its_variant)
    elif reaction_setup['unproductive_only']:
        sim_name = '{0}_unproductive_only'.format(its_variant)
    else:
        sim_name = '{0}_productive_only'.format(its_variant)

    return sim_name


if __name__ == '__main__':
    main()
