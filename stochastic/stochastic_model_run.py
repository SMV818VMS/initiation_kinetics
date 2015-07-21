import os
import stochpy
stochpy.plt.switch_backend('Agg')
import sys
sys.path.append('/home/jorgsk/Dropbox/phdproject/transcription_initiation/kinetic/model')
sys.path.append('/home/jorgsk/Dropbox/phdproject/transcription_initiation/data')
sys.path.append(os.path.join(os.path.dirname(__file__), "parameter_estimation"))
import parest
import kinetic_transcription_models as ktm
from KineticRateConstants import ITRates
from initial_transcription_model import ITModel, ITSimulationSetup
from ITSframework import calc_abortive_probability

from ipdb import set_trace as debug  # NOQA
import numpy as np
from scipy.linalg import solve
import pandas as pd
import matplotlib.pyplot as plt
import data_handler
from multiprocessing import Pool
import seaborn  # NOQA

from scipy.optimize import curve_fit
import scipy.stats as stats


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
    9/9/1.8 is very good.

    XXX Show that using GreB - that the optimal rates do not make sense, and
    that they make sense with GreB +
    """

    # Setup
    variant_name = 'N25'
    ITSs = data_handler.ReadData('dg100-new')
    its_variant = [i for i in ITSs if i.name == variant_name][0]
    initial_RNAP = 500
    sim_end = 180
    #params = get_estim_parameters(method='uniform')
    params = [(9.2, 1.6, 5)]  # the golden params
    stoi_setup = ktm.ITStoichiometricSetup(escape_RNA_length=12)

    log_file = 'simple_log.log'
    log_handle = open(log_file, 'w')
    log_handle.write('Fit\tFit_extrap\tNac\tAbort\tEscape\n')

    #plot = False
    plot = True

    #multiproc = True
    multiproc = False
    if multiproc:
        pool = Pool(processes=2)
        results = []
    for nac, abrt, escape in params:

        R = ITRates(its_variant, nac=nac, unscrunch=abrt, escape=escape, GreB_AP=True)

        sim_name = get_sim_name(stoi_setup, its_variant)

        sim_setup = ITSimulationSetup(sim_name, R, stoi_setup, initial_RNAP, sim_end)

        model = ITModel(sim_setup)

        if multiproc:
            #r = apply_async(pool, Run, args)
            r = pool.apply_async(model.run)
            results.append(r)
        else:
            model_ts = model.run()
            model_scrunches = calc_scrunch_distribution(model_ts, initial_RNAP)
            plot_and_log_scrunch_comparison(plot, log_handle, model_scrunches,
                                            initial_RNAP, nac, abrt, escape)

    if multiproc:
        pool.close()
        pool.join()
        model_timeseries = [r.get() for r in results]
        for ts, pars in zip(model_timeseries, params):
            nac, abrt, escape = pars
            ms = calc_scrunch_distribution(ts, initial_RNAP)
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

    #y_model = neg_exp(x, popt[0])

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


def calc_scrunch_distribution(df, initial_RNAP):
    """
    Will probably not get same distribution in time as Revyakin, but we might
    get the same shape. They use a different salt to the other experiments,
    and the concentration of that salt matters a lot for the result (see last
    fig in supplementry of Revyakin).
    """

    # Starts with timestep 1!
    # BUT! Your index is now in time; you wish to work on row numbers. Turn
    # into numpy arrays to avoid index isues.
    escapes = np.asarray(df['FL'])[1:] - np.asarray(df['FL'])[:-1]

    # Skip first timestep
    time = df.index[1:]
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
    """
    ON parameter estimation. You found a good-looking package called SPOTPY
    which has built-in several methods for parameter estimation, including
    evolutionary approaches. But in just testing the tutorials, you run into
    errors. However, it can be a good foundation for you to build your own
    approach. They are consistently using root mean squared error (RMSE).
    And they have a way of calculating the 95% confidence interval. If you
    could do that too it would be awesome.

    So. What to do first? You can refactor the model run. You can make it into
    a class model_instance = IT(Rates, Blehs, Blahs).
    """

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

    I think the kinetics of the competitive promoter looks best. It has
    stronger correlation, and the rate of abortive synthesis is lower after
    all the FL has been reached, which indicates that there is less
    reinitiation.
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


def start_logger(path, header):

    log_handle = open(path, 'w')
    log_handle.write(header + '\n')

    return log_handle


def productive_AP_estimate():
    """
    You got it working basically. Now, take a SPOTPY-like approach to wrap
    parameter sampling and model running and model fitting and parameter
    analysis so it will be easier to get back to.
    """
    #from productive_AP_setup_noclass import estimator_setup
    from productive_AP_setup import estimator_setup

    # Setup
    variant_name = 'N25'
    ITSs = data_handler.ReadData('dg100-new')
    its_variant = [i for i in ITSs if i.name == variant_name][0]
    initial_RNAP = 400
    sim_end = 60. * 3

    stoi_setup = ktm.ITStoichiometricSetup(escape_RNA_length=12,
                                           part_unproductive=True,
                                           custom_AP=True)

    R_ref = ITRates
    # Precalculate this value for comparison with model results
    experiment_RNA_fraction = its_variant.fraction

    # Get experimental data -- only up to the duration of the simulation
    N25_kinetic_ts = get_N25_kinetic_data(sim_end)

    estim_setup = estimator_setup(experiment_RNA_fraction, stoi_setup, R_ref,
                                  variant_name, initial_RNAP, sim_end,
                                  N25_kinetic_ts)

    name = 'N25_AP_est_test'
    rerun = True
    #rerun = False
    if rerun:
        sampler = parest.Parest(estim_setup, samples=4, name=name,
                                processes=2, batch_size=11)
        sampler.search()
        results = sampler.get_results()
    else:
        results = parest.load_results(name)

    params = results['parameters']
    tss = results['time_series']

    model_abortives = tss[:37]
    model_FLs = tss[37:]

    obs_data = estim_setup.observation()
    obs_abortives = obs_data[:37]
    obs_FL = obs_data[37:]

    # TODO: plot results. Plot AP and fractions too for the best results.

    debug()


def get_N25_kinetic_data(sim_end):

    # Fetch kinetic data for N25 for comparison
    data ='/home/jorgsk/Dropbox/phdproject/transcription_initiation/data/'
    path = os.path.join(data, 'vo_2003/Fraction_FL_and_abortive_timeseries_method_2.csv')
    ts = pd.read_csv(path, index_col=0)
    ts.index = ts.index * 60.  # min -> sec

    # rename columns for later
    ren ={'Abortive competitive promoter': 'Abortive experiment',
          'FL competitive promoter': 'FL experiment'}
    ts.rename(columns=ren, inplace=True)

    # let there be 1 timestep more in model than in simulation
    t = ts[:sim_end-1]

    # Normalize to the largest value
    t = t / t.max().max()

    return t


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

        APs = calc_abortive_probability(rna_frac_adj)

    else:
        APs = its.abortive_probability

    return APs


def compare_fraction(df, experiment_RNA_fraction):
    """
    You probably need to improve this measure.
    """

    species = ['rna_{0}'.format(i) for i in range(2,21)] + ['FL']

    nr_final_rna = np.asarray([df[s].tail().tolist()[-1] for s in species])

    # Model fraction from #RNA
    model_RNA_fraction = nr_final_rna / nr_final_rna.sum()

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


def get_sim_name(stoi_setup, its_name):

    if stoi_setup.part_unproductive:
        sim_name = '{0}_also_unproductive'.format(its_name)
    elif stoi_setup.unproductive_only:
        sim_name = '{0}_unproductive_only'.format(its_name)
    else:
        sim_name = '{0}_productive_only'.format(its_name)

    return sim_name


if __name__ == '__main__':
    main()
