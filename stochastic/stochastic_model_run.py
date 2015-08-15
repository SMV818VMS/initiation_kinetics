import os
import stochpy
stochpy.plt.switch_backend('Agg')
import sys
sys.path.append('/home/jorgsk/Dropbox/phdproject/transcription_initiation/kinetic/model')
sys.path.append('/home/jorgsk/Dropbox/phdproject/transcription_initiation/data')
sys.path.append(os.path.join(os.path.dirname(__file__), "parameter_estimation"))
import parest
from kinetic_transcription_models import ITStoichiometricSetup
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
#import shelve

import scipy.stats as stats


log_dir = 'Log'
scrunch_db = os.path.join(log_dir, 'scrunch_fit_df_storage')


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

    k_max = {'A': 50, 'U': 18, 'G': 36, 'C': 33, 'T': 18}
    K_d = {'A': 38, 'U': 24, 'G': 62, 'C': 7, 'T': 24}

    # Find EC for this sequence (only DNADNA and RNADNA, no DG3D since that
    # was not used by Bai et al)
    c1 = 1
    c2 = 2
    c3 = 3
    #rna_len = its.calculated_msat
    rna_len = 20
    its.calc_keq(c1, c2, c3, msat_normalization=False, rna_len=rna_len)

    # Old Bai
    nac2 = (24.7 * NTP) / (15.6 * (1 + EC) + NTP)
    # New Bai
    #nac1 = (k_max[nuc] * NTP) / (K_d[nuc]*(1 + EC) + NTP)
    EC = 1.6
    for nuc in list("GATC"):
        nac1 = (k_max[nuc] * NTP) / (K_d[nuc] * (1 + EC) + NTP)
        print nuc, nac1

    #print nac2/nac1

    return nac2


def get_parameters(samples=10, GreB=True, boundaries=False):
    """
    For Revyakin data you got a similar result for stepwise and uniform
    parameter sampling
    """
    from numpy.random import uniform

    # Initial run
    if boundaries is False:
        nac_low = 1
        nac_high = 25

        abort_low = 1
        abort_high = 25

        escape_low = 1
        escape_high = 25
    else:
        nac_low = boundaries['nac_low']
        nac_high = boundaries['nac_high']

        abort_low = boundaries['abort_low']
        abort_high = boundaries['abort_high']

        escape_low = boundaries['escape_low']
        escape_high = boundaries['escape_high']

    params = []
    for sample_nr in range(samples):
        nac = uniform(nac_low, nac_high)
        abortive = uniform(abort_low, abort_high)
        escape = uniform(escape_low, escape_high)

        params.append((nac, abortive, escape))

    return params


def plot_scrunch(obs, results, param_names):
    """
    Shit! I don't know what it is, but there is something not right in this
    mor complex code you have made. I believe the scores, but I don't beleive
    the parameters associated with them. Thre
    """

    plot_top = 10

    # normalize for plotting
    # make into a pandas dataframe
    obs = pd.DataFrame(obs)
    obs = obs / obs.max()
    pars = results['parameters']

    pars_sorted = pars.sort('euclid')
    pars_sorted_nofit = pars_sorted[param_names]

    # Pick the top 3 results according to one of the metrics
    top_params = pars.sort('euclid')[:plot_top]

    # Extract corresponding top timeseries
    ts = results['time_series']
    top_params_sim_nr = top_params.index
    top_ts = ts[top_params_sim_nr]

    print pars_sorted_nofit
    print top_ts

    # For this in that, plot it.
    debug()

    #plot_scrunch_distributions(model_scrunches, experiment, extrapolated, title)


def fit_to_distributions():

    datapath = '/home/jorgsk/Dropbox/phdproject/transcription_initiation/data/Revyakin 2006/Scrunch times.csv'

    df_measurements = pd.read_csv(datapath, index_col='time')
    x = np.asarray(df_measurements.index)
    y = df_measurements['inverse_cumul_experiment']

    def neg_exp(x, a):
        return np.exp(-a * x)

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
    integrated_prob = np.linspace(1 / float(sample_size), 1, sample_size)

    df_gamma = pd.DataFrame(data={'gamma': integrated_prob}, index=gamma_data)
    df_expon = pd.DataFrame(data={'exponential': integrated_prob}, index=expon_data)

    #y_model = neg_exp(x, popt[0])

    # Make some preductions from this simple model
    xx = np.linspace(0, 40, 1000)
    y_extrapolated = neg_exp(xx, popt[0])
    df_extrapolated = pd.DataFrame(data={'extrap_inv_cumul': y_extrapolated}, index=xx)

    f, ax = plt.subplots()

    df_measurements.plot(style='.', ax=ax, logy=True)

    df_extrapolated.plot(ax=ax, logy=True)

    df_gamma.plot(ax=ax, logy=True)
    df_expon.plot(ax=ax, logy=True)

    ax.set_xlim(0, 20)
    ax.set_ylim(1e-2, 1)
    ax.set_xticks(range(0, 21))
    f.savefig('scrunch_times_experiment.pdf')

    plt.close(f)


def get_kinetics(variant, GreB=False, nac=10.1, unscrunch=1.8, escape=11.8,
                 escape_RNA_length=False, sim_end=1000, initial_RNAP=100):
    """
    Return a kinetic model ready to run.
    """

    ITSs = data_handler.ReadData('dg100-new')
    its = [i for i in ITSs if i.name == variant][0]

    if escape_RNA_length is False:
        escape_RNA_length = its.msat_calculated

    stoi_setup = ITStoichiometricSetup(escape_RNA_length=escape_RNA_length)

    R = ITRates(its, nac=nac, unscrunch=unscrunch, escape=escape, GreB_AP=GreB)

    sim_name = get_sim_name(stoi_setup, its)

    sim_setup = ITSimulationSetup(sim_name, R, stoi_setup, initial_RNAP, sim_end)

    model = ITModel(sim_setup)

    return model


def main():
    """
    """
    pass

    # StreZ TeZt
    p = Pool()
    for _ in range(4):
        model = get_kinetics('N25', initial_RNAP=10000)
        p.apply_async(model.run)
    p.close()
    p.join()

    # XXX not good :(
    #revyakin_datafit_new()

    # XXX try again with more simple method?
    # XXX this would be a great result if it works, but have you the time?
    #productive_AP_estimate()

    # Plot normalized plots of the 2003 data
    #plot_2003_kinetics()

    #debug()

    #multiproc_test()


def justaminute(i):
    import time
    from numpy.random import uniform
    randval = uniform()  # random number between 0 and 1
    print randval
    time.sleep(randval)
    return i


def multiproc_test():
    """
    Can you trust that the simulations you throw out with apply_async or amap
    return in the order you think? :S

    The answer seems to be yes. I think it is because I'm keeping the results
    in a list for apply_async. That list ensures proper order of input and
    output to multipricessing. Also, by using join, we wait until all
    processes have finished.
    """

    i_vals = range(1, 10)

    # First test apply async
    P1 = Pool(processes=1)
    results = []
    for i in i_vals:
        r = P1.apply_async(justaminute, (i,))
        results.append(r)
    P1.close()
    P1.join()
    print([res.get() for res in results])

    # Then test amap
    P2 = Pool(processes=2)
    results = P2.map_async(justaminute, i_vals)
    P2.close()
    P2.join()
    print([res for res in results.get()])


def plot_2003_kinetics():
    """
    It seems that with the competitive promoter thre is much less abortive
    recycling.

    I think the kinetics of the competitive promoter looks best. It has
    stronger correlation, and the rate of abortive synthesis is lower after
    all the FL has been reached, which indicates that there is less
    reinitiation.
    """

    data = '/home/jorgsk/Dropbox/phdproject/transcription_initiation/data/'
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

    df1_norm = df1 / max1
    df2_norm = df2 / max2

    # XXX the data is already interpolated by Eugene during export. I'm
    # guessing linear interpoltation.

    f, ax = plt.subplots()

    df1_norm.plot(ax=ax)
    df2_norm.plot(ax=ax)

    ax.set_xlabel('time (minutes)')
    ax.set_ylabel('relative to maximum abortive signal')

    ax.set_xticks(range(0, 11))
    ax.set_yticks(np.arange(0, 1.1, 0.1))

    ax.set_ylim(0, 1.01)

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
    #from productive_AP_setup import estimator_setup
    from FL_2003_setup import estimator_setup

    # Setup
    variant_name = 'N25'
    ITSs = data_handler.ReadData('dg100-new')
    its_variant = [i for i in ITSs if i.name == variant_name][0]
    initial_RNAP = 500
    sim_end = 60. * 1.5
    samples = 31
    processes = 2
    #batch_size = 15
    batch_size = samples - 1
    # XXX with my 8-core I could be 8 times as fast: 15 seconds instead of 2
    # minutes wait. Yes plz.

    # Get observation data up to the duration of the simulation
    N25_kinetic_ts = get_N25_kinetic_data(sim_end, method_nr=2)

    estim_setup = estimator_setup(its_variant, initial_RNAP, sim_end,
                                  N25_kinetic_ts, escape_length=14)

    # why are all simulations suddenly having an end time of 90s?
    name = 'N25_AP_est'
    #rerun = True
    rerun = False
    if rerun:
        sampler = parest.Parest(estim_setup, samples=samples, name=name,
                                processes=processes, batch_size=batch_size)
        sampler.search()
        results = sampler.get_results()
    else:
        results = parest.load_results(name)

    plot_APFL_results(estim_setup.observation(), results, estim_setup.parameters().keys())


def plot_APFL_results(obs, results, varied_parameters):
    """
    When comparing FL only, I think you need to normalize with max FL and not
    abortive. There are a lot of differences popping up for FL and NON FL. You
    should make two different setups.
    """

    plot_top = 5

    # normalize for plotting
    # make into a pandas dataframe
    obs = pd.DataFrame(obs)
    obs = obs / obs.max()
    pars = results['parameters']

    # Pick the top 3 results
    #top_params = pars.sort('score')[:plot_top]
    top_params = pars.sort('median_ae')[:plot_top]
    # Extract timeseries
    ts = results['time_series']

    top_params_sim_nr = top_params.index
    #top_params_col_names = [str(s) for s in top_params_sim_nr]

    #top_ts = ts[top_params_col_names]
    top_ts = ts[top_params_sim_nr]

    #  Interesting. Saving and loading to csv makes so that some indices lose
    #  precision so that they are not the same number any more. Must re-index
    #  to be sure. A shame!
    #top_ts.index = obs.index
    #joined = pd.concat([obs, top_ts], axis=1)
    # Do you need to join? can't you just plot both?

    # Build a list of labels
    labels = []
    #noscore = top_params.drop('score', axis=1)
    noscore = top_params[varied_parameters + ['median_ae']]
    for sim_nr in top_params_sim_nr:
        par_vals = noscore.loc[sim_nr].to_dict()
        lab = ' '.join(['{0}: {1:.2f}'.format(k, v) for k, v in par_vals.items()])
        labels.append(lab)
    # Restore observation
    labels.insert(0, 'Experiment')

    f, ax = plt.subplots()
    obs.plot(ax=ax)
    top_ts.plot(ax=ax)
    ax.legend(labels, loc='lower right')
    f.savefig('quicktest.pdf')


def get_N25_kinetic_data(sim_end, method_nr=1):

    # Fetch kinetic data for N25 for comparison
    data = '/home/jorgsk/Dropbox/phdproject/transcription_initiation/data/'
    if method_nr == 1:
        path = os.path.join(data, 'vo_2003/Fraction_FL_and_abortive_timeseries_method_1.csv')
    elif method_nr == 2:
        path = os.path.join(data, 'vo_2003/Fraction_FL_and_abortive_timeseries_method_2.csv')

    ts = pd.read_csv(path, index_col=0)
    ts.index = ts.index * 60.  # min -> sec

    # rename columns
    if method_nr == 1:
        ren = {'Abortive halted elongation': 'Abortive experiment',
               'FL halted elongation': 'FL experiment'}
    elif method_nr == 2:
        ren = {'Abortive competitive promoter': 'Abortive experiment',
               'FL competitive promoter': 'FL experiment'}

    ts.rename(columns=ren, inplace=True)

    # let there be 1 timestep more in model than in simulation
    t = ts[:sim_end - 1]

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

    ren = {'Abortive competitive promoter': 'Abortive experiment',
           'FL competitive promoter': 'FL experiment'}

    experiment.rename(columns=ren, inplace=True)

    # Convert N25 timeseries to seconds
    experiment.index = experiment.index * 60.

    # Keep experimental data close to model data
    experiment = experiment[:model.index[-2]]

    # Normalize N25 timeseries to max abortive RNA
    experiment = experiment / experiment.max().max()

    # Normalize the model RNA to max abortive RNA
    model = model / model.max().max()

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
        APs = its.abortive_prob

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
    nr_nonzero_transcript_lengths = sum(Y > 0)
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
