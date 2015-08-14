import numpy as np
from scipy.stats import spearmanr, pearsonr
import os
import sys
import pandas as pd
import shelve
sys.path.append('/home/jorgsk/Dropbox/phdproject/transcription_initiation/data/')
sys.path.append('/home/jorgsk/Dropbox/phdproject/transcription_initiation/kinetic/model')
sys.path.append(os.path.join(os.path.dirname(__file__), "parameter_estimation"))
from ITSframework import calc_abortive_probability
from kinetic_transcription_models import ITStoichiometricSetup
from KineticRateConstants import ITRates
from initial_transcription_model import ITModel, ITSimulationSetup
#from ITSframework import calc_abortive_probability
from metrics import weighted_avg_and_std
import data_handler
import stochastic_model_run as smr
import matplotlib.pyplot as plt
from multiprocessing import Pool
from collections import OrderedDict, defaultdict

plt.ioff()

import seaborn as sns
from ipdb import set_trace as debug  # NOQA
from scipy.interpolate import InterpolatedUnivariateSpline as interpolate
from scipy.optimize import curve_fit

# Global place for where you store your figures
here = os.path.abspath(os.path.dirname(__file__))  # where this script is located
fig_dir = os.path.join(here, 'figures')

log_dir = 'Log'
scrunch_db = os.path.join(log_dir, 'scrunch_fit_df_storage')

# Database for storing simulation results
simulation_storage = os.path.join(here, 'storage', 'shelve_storage_sim')

# Global setting for max processors
max_proc = 4

# Figure sizes in centimeter. Height is something you might set maually but
# width should always be the maximum
nar_width = 8.7
biochemistry_width = 8.5
current_journal_width = biochemistry_width


def set_fig_size(f, x_cm, y_cm):

    def cm2inch(cm):
        return float(cm) / 2.54

    x_inches = cm2inch(x_cm)
    y_inches = cm2inch(y_cm)

    f.set_size_inches(x_inches, y_inches)


def tau_variation_plot():
    """
    Plot taus for N25 (simulation), N25 (experiment) and N25 variants (anti,
    A1, A1-anti). Can also make it for all.
    """
    initial_RNAP = 1000
    time_sim = 60 * 30
    #variants = ['N25', 'N25anti', 'N25-A1', 'N25-A1anti']
    variants = ['N25', 'N25anti', 'N25-A1anti']

    tss = get_multiple_dg100_timeseries(variants, initial_RNAP, time_sim)

    # Extract only the FL part from the timeseires
    fls = {k: v['FL'] for k, v in tss.items()}

    half_lives(fls, initial_RNAP, include_experiment=True)


def get_multiple_dg100_timeseries(variant_names, initial_RNAP, sim_end):
    """
    Get variant names for dg100 with (-GreB) and return their timeseries.
    variant names can be a list of names: ['N25', 'DG136a'], or can be 'all'.

    Return them in the order they came! No safety checks for uniqueness, be
    responsible.
    """
    all_ITSs = data_handler.ReadData('dg100-new')

    if variant_names == 'all':
         ITSs = all_ITSs
    else:
        # Can make it better: go through all its once, when you find a
        # matching name, insert at position determined by index of name
        ITSs = [''] * len(variant_names)
        n2idx = {name: variant_names.index(name) for name in variant_names}
        for its in all_ITSs:
            if its.name in variant_names:
                ITSs[n2idx[its.name]] = its

    use_sim_db = True

    p = Pool(max_proc)
    results = []
    sim_ids = []

    # Then we can rely on this order later
    assert variant_names == [i.name for i in ITSs]

    shelve_database = shelve.open(simulation_storage)
    for its in ITSs:

        model = smr.get_kinetics(variant=its.name, initial_RNAP=initial_RNAP, sim_end=sim_end)
        sim_id = model.calc_setup_hash()
        sim_ids.append(sim_id)

        if use_sim_db and sim_id in shelve_database:
            results.append(None)
        else:
            results.append(p.apply_async(model.run))

    p.close()
    p.join()

    timeseries = OrderedDict()
    for sim_id, r, name in zip(sim_ids, results, variant_names):
        if use_sim_db and sim_id in shelve_database:
            ts = shelve_database[sim_id]
        else:
            ts = r.get()
            shelve_database[sim_id] = ts
        timeseries[name] = ts
    shelve_database.close()

    return timeseries


def initial_transcription_stats():
    """
    Run stochastic simulations on all the DG100 library variants
    """

    initial_RNAP = 1000

    tss = get_multiple_dg100_timeseries('all', initial_RNAP, 5000)

    # Keep all rnas to make a box plot after.
    all_rnas = {}
    all_FL_timeseries = {}
    for its_name, ts in tss.items():

        all_rnas[its_name] = ts.iloc[-1].values
        all_FL_timeseries[its_name] = ts['FL']

    # XXX Make the basic bar plots
    # These plots now serve only to show complete compliance between model
    # and experiment
    #for its, nr_rna in all_rnas.items():
        #write_bar_plot(nr_rna, its)

    # XXX deprecated: serve only to show indentity between model and experiment
    #deprecated_comparison_plots(all_rnas, ITSs)

    #XXX NEW AP XXX
    # Calculate AP anew!
    # This perhaps best shows the effect of the model?
    #AP_recalculated(ITSs, all_rnas)
    # This can be used as a metric. Just take the euclidian norm of the 19
    # dimensional abortive space.

    #XXX t_1/2 distributions for DG100 XXX
    half_lives(all_FL_timeseries, initial_RNAP)

    # XXX show kinetics for some representative promoter variants? N25, anti,
    # and some others?


def deprecated_comparison_plots(all_rnas, ITSs):

    dsetmean = np.mean([i.PY for i in ITSs])
    for partition in ['low PY', 'high PY', 'all']:

        if partition == 'low PY':
            low_ITSs = [i for i in ITSs if i.PY < dsetmean]
            write_box_plot(low_ITSs, all_rnas, title=partition)

        if partition == 'high PY':
            high_ITSs = [i for i in ITSs if i.PY >= dsetmean]
            write_box_plot(high_ITSs, all_rnas, title=partition)

        if partition == 'all':
            write_box_plot(ITSs, all_rnas, title=partition)


def calc_AP_anew(transcripts):
    """
    Calculate abortive probablity (duplicate of what's in ITSframework)
    """

    fl_transcript = transcripts['FL']
    total_abortive_transcript = sum(transcripts.values()) - fl_transcript
    total_rna = fl_transcript + total_abortive_transcript

    # Get abortive rna in a list with values ordered from 2nt to 20nt
    abortive_rna = []
    nt_range = range(2, 21)
    for pos in nt_range:
        key = str(pos)
        if key in transcripts:
            abortive_rna.append(float(transcripts[key]))
        else:
            abortive_rna.append(0.)

    index_range = range(len(nt_range))  # +1 for FL
    # percentage abortive yield at each x-mer
    pct_yield = [(ay / total_rna) * 100. for ay in abortive_rna]

    # amount of RNA produced from here and out
    rna_from_here = [np.sum(abortive_rna[i:]) + fl_transcript for i in index_range]

    # percent of RNAP reaching this position
    pct_rnap = [(rfh / total_rna) * 100. for rfh in rna_from_here]

    # probability to abort at this position
    # More accurately: probability that an abortive RNA of length i is
    # produced (it may backtrack and be slow to collapse, producing
    # little abortive product and reducing productive yield)
    ap = [pct_yield[i] / pct_rnap[i] for i in index_range]

    return ap


def AP_recalculated(ITSs, all_rnas):
    """
    Calculate AP from the kinetic model. How does this match the AP from the
    experiments?

    Since FL is higher for model, all APs are lower. But is that it? Would it
    be a perfect match if FL was reduced? Is it sufficient to reduce FL?
    """

    its_dict = {i.name: i for i in ITSs}

    for its_name, transcripts in all_rnas.items():

        data = {}
        index = []

        new_AP = calc_abortive_probability(transcripts)
        old_AP = its_dict[its_name].abortive_prob

        data['AP experiment'] = np.asarray(old_AP) * 100.
        data['AP model'] = np.asarray(new_AP) * 100.

        corrs, pvals = spearmanr(data['AP experiment'], data['AP model'])
        corrp, pvalp = pearsonr(data['AP experiment'], data['AP model'])

        title = 'AP: model vs experiment for {2}\n Spearman correlation: {0}\n Pearson correlation: {1}'.format(corrs, corrp, its_name)

        index = range(2, len(new_AP) + 2)

        fig, ax = plt.subplots()
        mydf = pd.DataFrame(data=data, index=index)
        mydf.plot(kind='bar', ax=ax, rot=0)

        ax.set_xlabel('ITS position')
        ax.set_ylabel('AP (%)')
        ax.set_ylim(0, 80)

        file_name = its_name + '_AP_experiment_vs_model.pdf'
        file_path = os.path.join(fig_dir, file_name)
        fig.suptitle(title)
        fig.savefig(file_path, format='pdf')
        plt.close(fig)


def half_lives(FL_timeseries, nr_RNAP, include_experiment=False):
    """
    For each variant, calculate the half life of FL. That means you got to
    calculate until the end of the simulation. Use just 100 NTPs and hope it's
    accurate enough.

    Terminology: your values are actually increasing. What is the value for
    50% of an increasing quantity? When Q(t) = Q(inf)/2.
    Yeah that seems to be what Lilian is using also.
    """
    data = {}
    # Hey, timeseries information is not stored in tses. What to do? Start
    # storing time series info, at least for FL. Then save that to another
    # file? Bleh, you're probably going to be re-running simulations for a
    # while.

    for its_name, timeseries in FL_timeseries.items():

        hl_index = timeseries[timeseries == nr_RNAP / 2].index.tolist()
        # For the cases where you don't simulate long enough to reach a half
        # life (or you are not using an even number of RNAP ... :S)
        if hl_index == []:
            half_life = np.nan
        else:
            half_life = hl_index[0]

        data[its_name] = half_life

    if include_experiment:
        data['N25\nexperiment'] = 13.7  # Vo et. al 2003 (my value after interpolation)

    df = pd.DataFrame(data, index=['t_1/2']).T
    ren = {'N25': 'N25\nsimulation',
           'N25anti': 'N25-anti\nsimulation',
           'N25-A1anti': 'N25-A1-anti\nsimulation'}
    df.rename(ren, inplace=True)
    df = df.sort('t_1/2')

    #if include_experiment:
        #colors = ['Gray'] * (len(FL_timeseries) + 1)
        #variants = df.index.tolist()
        #expt = variants.index('N25e')
        #sim = variants.index('N25')
        #colors[expt] = 'DarkBlue'
        #colors[sim] = 'Blue'
    #else:
        #colors = ['Gray'] * len(FL_timeseries)

    f, ax = plt.subplots()
    if include_experiment:
        rot = 0
    else:
        rot = 80
    df.plot(kind='bar', ax=ax, rot=rot, legend=False, fontsize=10)

    ax.set_ylabel('$\\tau$ (s)')

    file_name = 'Time_to_reach_fifty_percent_of_FL.pdf'

    file_path = os.path.join(fig_dir, file_name)

    f.tight_layout()

    #set_fig_size(f, current_journal_width, 8)

    f.savefig(file_path, format='pdf')

    # Explicitly close figure to save memory
    plt.close(f)


def write_box_plot(ITSs, tses, title=False):
    """
    Box plot comparing % model transcript and % gel-signal.

    Both absolute comparison and relative fold comparison.

    """
    # Make one big dictionary
    data_range = range(2, 21)  # look at AP from 2-nt to 20-nt

    data = {}
    data_frac = {}  # fraction difference between pct values
    for its in ITSs:
        # Experiment
        exp_key = '_'.join([its.name, 'experiment'])
        pct_values_experiment = pct_experiment(its)
        data[exp_key] = pct_values_experiment.tolist() + ['experiment']

        # Model
        mod_key = '_'.join([its.name, 'model'])
        ts = tses[its.name]
        pct_values_model = calc_pct_model(ts, data_range)
        data[mod_key] = pct_values_model.tolist() + ['model']

        # Fractional difference
        frac_diff = calc_frac_diff(np.array(pct_values_model), np.array(pct_values_experiment))
        data_frac[its.name] = frac_diff

    box_plot_model_exp(data, data_range, title)

    # XXX NICE!! You see the trend :) It is also more pronounced for low-PY
    # positions. You can choose one (N25?) to show this trend for a single
    # variant, then show this graph. For publication: green/blue for
    # model/experiment and positive negative axes.
    # And it is more pronounced for hi
    # XXX TODO: try to ignore the data point with MSAT. I think it's heavily
    # biasing your results, especially for low-py.
    box_plot_model_exp_frac(data_frac, data_range, title)

    # XXX THE optimal coolest question is: can you make a model that includes
    # some of the known suspects (long pauses, longer backtrack time)? That
    # brings the result closer? If you introduce stuff that increases the
    # backtracking time for late complexes (you'll need longer running time
    # ...) but then you SHOULD see a decrease in the % of these values
    # compared to others.
    # WOW this is just so amazingly cool :D:D:D:D:D:D

    # XXX Make a correlation plot between PY and fold difference between [0-6]
    # and [7-20], and between PY and fold difference betweern PY and FL
    #fold_diff_PY_correlations(ITSs, data_frac)

    # There are so many things you could check! You have to rein yourself in a
    # bit. God bless Pandas for speeding up my plot-work 100-fold!
    # (almost) never more needy tiddly messing around with matplotlib


def fold_diff_PY_correlations(ITSs, data_frac):
    """
    You have the fold difference (+ and -).

    Make a difference metric by sum(fold_diff[7:21]) - sum(fold_diff[2:6]).
    Then plot that metric against PY.

    For FL: just plot the fold difference for FL against PY.
    """
    PYs = {i.name: i.PY for i in ITSs}
    fold_diff_metric = []
    prod_yield = []
    fl_fold = []
    for its, folds in data_frac.items():
        early = np.nansum(folds[:5])
        late = np.nansum(folds[5:])

        flf = folds[-1]
        fl_fold.append(flf)

        measure = late - early
        PY = PYs[its]

        fold_diff_metric.append(measure)
        prod_yield.append(PY)

    prod_yield = np.asarray(prod_yield)

    # Fold diff metric
    fold_diff_metric = np.asarray(fold_diff_metric)
    corrs, pvals = spearmanr(prod_yield, fold_diff_metric)
    corrp, pvalp = pearsonr(prod_yield, fold_diff_metric)
    f, ax = plt.subplots()
    ax.scatter(prod_yield * 100., fold_diff_metric)

    ax.set_xlabel('PY')
    ax.set_ylabel('Sum(fold difference after +6) - Sum(fold difference before +6)')

    title = 'Low PY variants have greater overestimates of late abortive release in model\n Spearman correlation: {0}\n Pearson correlation: {1}'.format(corrs, corrp)
    f.suptitle(title)

    file_name = 'Fold_difference_PY_correlation.pdf'

    file_path = os.path.join(fig_dir, file_name)
    f.savefig(file_path, format='pdf')

    # Explicitly close figure to save memory
    plt.close(f)

    ## FL
    fl = np.asarray(fl_fold)
    corrs, pvals = spearmanr(prod_yield, fl)
    f, ax = plt.subplots()
    ax.scatter(prod_yield * 100., fl)

    ax.set_xlabel('PY')
    ax.set_ylabel('Fold difference model/experiment of percent FL transcript')

    title = 'Model with AP as backtracking probability greatly overestimates FL for low PY variants\n Spearman correlation: {0}'.format(corrs, corrp)
    f.suptitle(title)

    file_name = 'FL_fold_difference_PY_correlation.pdf'

    file_path = os.path.join(fig_dir, file_name)
    f.savefig(file_path, format='pdf')

    plt.close(f)


def box_plot_model_exp_frac(data, data_range, title):

    index = data_range + ['FL']
    mydf = pd.DataFrame(data=data, index=index)
    mydf_t = mydf.T

    f, ax = plt.subplots()
    #mydf_t.plot(kind='box', sym='')
    #mydf_t.boxplot(sym='')
     #Get mean and std for bar plot
    std = mydf_t.std()
    mean = mydf_t.mean()
    colors = ['green' if v > 0 else 'blue' for v in mean]
    mean.plot(kind='bar', yerr=std, color=colors, rot=0)

    # Get colors the way y0 want
    import matplotlib.patches as mpatches

    EU = mpatches.Patch(color='blue', label='More QI signal than model RNA')
    NA = mpatches.Patch(color='green', label='More model RNA than QI signal')
    ax.legend(handles=[NA, EU], loc=2)

    ax.set_ylabel("Fold difference of % QI units and % model RNA")

    ax.set_ylim(-6, 6.5)
    ax.set_yticks(range(-6, 7))
    ax.set_yticklabels([abs(v) for v in range(-6, 7)])

    ax.set_xlabel('ITS position')

    f.suptitle('Fold difference between model and experiment at each position for DG100')
    file_name = 'Model_and_experiment_fractions_{0}.pdf'.format(title)

    file_path = os.path.join(fig_dir, file_name)
    f.savefig(file_path, format='pdf')

    plt.close(f)


def box_plot_model_exp(data, data_range, title):

    index = data_range + ['FL', 'kind']
    mydf = pd.DataFrame(data=data, index=index)
    mydf_t = mydf.T

    # Prepare for seaborn!
    varname = 'ITS position'
    valname = '% of total'
    separator = 'kind'
    df_long = pd.melt(mydf_t, separator, var_name=varname, value_name=valname)
    f, ax = plt.subplots()
    sns.factorplot(varname, hue=separator, y=valname, data=df_long,
                   kind="box", ax=ax, sym='')  # sym='' to get rid of outliers

    ax.set_ylim(0, 50)
    if title:
        f.suptitle(title)
        file_name = 'Model_and_experiment_{0}.pdf'.format(title)
    else:
        file_name = 'Model_and_experiment.pdf'

    file_path = os.path.join(fig_dir, file_name)
    f.savefig(file_path, format='pdf')
    plt.close(f)


def calc_frac_diff(arr1, arr2):
    """
    Get fractional difference.

    arr1 = [20, 50, 60]
    arr2 = [35, 60, 20]

    result: [(-35./20)*100 (-60./50)*100 (60./20)*100]
    Gives percentage difference, but a negative sign if value in arr2 is
    bigger than in arr1.
    """

    assert arr1.size == arr2.size

    frac_diff = []

    for i in range(len(arr1)):
        val1 = arr1[i]
        val2 = arr2[i]

        if val1 == 0. or val2 == 0:
            diff = np.nan
        else:
            if val1 > val2:
                diff = float(val1) / val2
            else:
                diff = -float(val2) / val1

        frac_diff.append(diff)

    return np.array(frac_diff)


def pct_experiment(its):

    return its.fraction * 100.


def calc_pct_model(ts, data_range):
    """
    Ts must be an array with abortives from 2 to 20 and the FL
    """

    return 100. * ts / ts.sum()


def write_bar_plot(nr_rna, its):
    """
    There is a clear trend where the amount of abortive RNA is lower in the
    model < +9/10 and slightly higher in the model after +9/10 (except for +6).
    That could mean that in the experiment backtracking that occurs after
    +10 does not always end up in an aborted RNA, or ends up very slowly.
    Partially it can also be that escape begins before MSAT, which would
    reduce subsequent AP.

    The very last abortive signal / RNA matches poorly, probably because of
    the escape rate.

    FL is generally much higher in the model compared to experiments.
    One (probably small) reason is that you go directly from f.ex +18 to +64.
    You should add a helper-state +18 -> elongation -> FL, where the last rate
    (assuming 10bp/s) will be something like 4 or 5, so quite fast still.

    Also, the kinetics is way to fast for slow-PY variants. It fits pretty
    well for N25, but there is something slowing down those low-PY variants
    that is not being captured in the model.
    """

    data_range = range(2, 21)

    pct_values_experiment = 100. * its.fraction
    pct_values_model = 100. * nr_rna / nr_rna.sum()

    # Prepare for making a dataframe
    forDF = {'Experiment': pct_values_experiment,
             'Model': pct_values_model}
    index = data_range + ['FL']

    fig, ax = plt.subplots()
    mydf = pd.DataFrame(data=forDF, index=index)
    mydf.plot(kind='bar', ax=ax)

    ax.set_xlabel('ITS position')
    ax.set_ylabel('% of total')
    ax.set_ylim(0, 50)

    file_name = its.name + '_abortive_RNA_FL_model_comparison.pdf'
    file_path = os.path.join(fig_dir, file_name)
    fig.suptitle('Percentage of IQ units and modelled #RNA')
    fig.savefig(file_path, format='pdf')
    plt.close(fig)


def AP_and_pct_values_distributions_DG100():
    """
    Plot distributions of AP and %yield for high and low PY variants

    GOOD: differences between methods are not big on average, but there will
    probably be some big differences for individual promoters.
    """
    from operator import attrgetter

    ITSs = data_handler.ReadData('dg100-new')

    ITSs = sorted(ITSs, key=attrgetter('PY'))

    pymean = np.median(np.asarray([i.PY for i in ITSs]))
    low_PY = [i for i in ITSs if i.PY < pymean]
    high_PY = [i for i in ITSs if i.PY > pymean]

    # Make this plot with three different ways of calculating AP
    abortive_methods = ['abortiveProb_old', 'abortive_prob', 'abortive_prob_first_mean']

    # AP plots
    for ab_meth in abortive_methods:

        data_AP_mean = {}
        data_AP_std = {}

        for name, dset in [('Low PY', low_PY), ('High PY', high_PY)]:

            # AP stats
            f = attrgetter(ab_meth)
            aps = np.asarray([f(i) for i in dset])
            aps[aps <= 0.0] = np.nan

            ap_mean = np.nanmean(aps, axis=0)
            ap_std = np.nanstd(aps, axis=0)

            # Old method separated between abortive and FL
            if ab_meth == 'abortiveProb_old':
                ap_mean = np.append(ap_mean, 1.)
                ap_std = np.append(ap_std, 0.)

            data_AP_mean[name] = ap_mean * 100.
            data_AP_std[name] = ap_std * 100.

        f, ax = plt.subplots()
        df = pd.DataFrame(data=data_AP_mean, index=range(2, 21) + ['FL'])
        df_err = pd.DataFrame(data=data_AP_std, index=range(2, 21) + ['FL'])
        df.plot(kind='bar', yerr=df_err, ax=ax)

        ax.set_xlabel('ITS position')
        ax.set_ylabel('AP')
        ax.set_ylim(0, 60)

        file_name = 'AP_comparison_high_low_PY_{0}.pdf'.format(ab_meth)
        file_path = os.path.join(fig_dir, file_name)
        f.suptitle('Abortive probability in high and low PY variants\n{0}'.format(ab_meth))
        f.savefig(file_path, format='pdf')
        plt.close(f)

    # Signal Plots
    data_pct_mean = {}
    data_pct_std = {}

    for name, dset in [('Low PY', low_PY), ('High PY', high_PY)]:

        # Remember to get nans where there is zero!!
        ar = np.asarray([i.fraction * 100. for i in dset])
        ar[ar <= 0.0] = np.nan
        data_mean = np.nanmean(ar, axis=0)
        data_std = np.nanstd(ar, axis=0)

        data_pct_mean[name] = data_mean
        data_pct_std[name] = data_std

    f, ax = plt.subplots()
    df = pd.DataFrame(data=data_pct_mean, index=range(2, 21) + ['FL'])
    df_err = pd.DataFrame(data=data_pct_std, index=range(2, 21) + ['FL'])
    df.plot(kind='bar', yerr=df_err, ax=ax)

    ax.set_xlabel('ITS position')
    ax.set_ylabel('% of IQ signal')
    ax.set_ylim(0, 50)

    file_name = 'Percentage_comparison_high_low_PY.pdf'
    file_path = os.path.join(fig_dir, file_name)
    f.suptitle('Percentage of radioactive intensity for high and low PY variants')
    f.savefig(file_path, format='pdf')
    plt.close(f)


def get_2003_FL_kinetics(method=1):

    data = '/home/jorgsk/Dropbox/phdproject/transcription_initiation/data/vo_2003'
    m1 = os.path.join(data, 'Fraction_FL_and_abortive_timeseries_method_1_FL_only.csv')
    m2 = os.path.join(data, 'Fraction_FL_and_abortive_timeseries_method_2_FL_only.csv')

    # Read up to 10 minutes
    df1 = pd.read_csv(m1, index_col=0)[:10]
    df2 = pd.read_csv(m2, index_col=0)[:10]

    # Multiply by 60 to get seconds
    df1.index = df1.index * 60
    df2.index = df2.index * 60

    # Normalize
    df1_norm = df1 / df1.max()
    df2_norm = df2 / df2.max()

    # Add initial 0 value
    df1_first = pd.DataFrame(data={'FL halted elongation': 0.0}, index=[0.0])
    df1_final = df1_first.append(df1_norm)

    df2_first = pd.DataFrame(data={'FL competitive promoter': 0.0}, index=[0.0])
    df2_final = df2_first.append(df2_norm)

    #debug()
    if method == 1:
        return df1_final['FL halted elongation']
    elif method == 2:
        return df2_final['FL competitive promoter']
    else:
        print('No more methods in that paper!')


def fit_FL(x, y):
    """
    Assumes FL growth can be fitted to y = a + b(1-exp(-cx))
    """

    def saturation(x, a, b, c):
        return a + b * (1 - np.exp(-c * x))

    popt, pcov = curve_fit(saturation, x, y)

    # Make some preductions from this simple model
    xx = np.linspace(x.min(), x.max(), 1000)
    yy = saturation(xx, *popt)

    return xx, yy


def simple_vo_2003_comparison():
    """
    Purpose is to plot the performance on N25 GreB- and compare with Vo 2003
    data from both quantitations. Show quantitations as dots but also
    interpolate to make them look nicer.
    """

    # FL kinetics for 1 and 2.
    m1 = get_2003_FL_kinetics(method=1)

    # FL kinetics for model with GreB+/-
    #greb_plus_model = smr.get_kinetics(variant='N25', GreB=True,
                                       #escape_RNA_length=14,
                                       #initial_RNAP=1000, sim_end=60*5.)
    #greb_plus = greb_plus_model.run()['FL']
    #gp = greb_plus / greb_plus.max()

    f1, ax = plt.subplots()
    # Get and plot model data
    greb_minus_model = smr.get_kinetics(variant='N25', GreB=False,
                                        escape_RNA_length=14,
                                        initial_RNAP=1000, sim_end=60.5)
    greb_minus = greb_minus_model.run()['FL']
    gm = greb_minus / greb_minus.max()
    gm.plot(ax=ax, color='c')

    # Get half-life of FL from model
    #thalf_grebminus = f_gm(0.5)
    idx_gm = gm.tolist().index(0.5)
    thalf_grebminus = gm.index[idx_gm]

    # Plot 2003 data
    m1.plot(ax=ax, style='.', color='g')

    # Fit 2003 data and calculate half life
    xnew, ynew = fit_FL(m1.index, m1.values)
    fit = pd.DataFrame(data={'Vo et. al': ynew}, index=xnew)
    fit.plot(ax=ax, color='b')

    # Get the halflife of full length product
    f_vo = interpolate(ynew, xnew, k=1)
    thalf_vo = float(f_vo(0.5))

    # XXX looks better without for now
    # Plot and get halflives of GreB+
    #gp.plot(ax=ax, color='k')
    #f_gp = interpolate(gp.values, gp.index, k=2)
    #thalf_grebplus = f_gp(0.5)
    #idx_gp = gp.tolist().index(0.5)
    #thalf_grebplus = gp.index[idx_gp]
    #labels = ['Vo et. al GreB- measurements',
              #r'Vo et. al GreB- fit $t_{{1/2}}={0:.2f}$'.format(thalf_vo),
              #r'Simulation GreB- $t_{{1/2}}={0:.2f}$'.format(thalf_grebminus),
              #r'Simulation GreB+ $t_{{1/2}}={0:.2f}$'.format(thalf_grebplus)]

    labels = ['Vo et. al measurements',
              r'Vo et. al fit $t_{{1/2}}={0:.2f}$'.format(thalf_vo),
              r'Simulation $t_{{1/2}}={0:.2f}$'.format(thalf_grebminus)]

    ax.legend(labels, loc='lower right', fontsize=15)

    ax.set_xlabel('Time (seconds)', size=15)
    ax.set_ylabel('Fraction of full length transcript', size=15)

    ax.set_xlim(0, 70)
    ax.set_ylim(0, 1.01)
    f1.savefig('GreB_plus_and_GreB_minus_kinetics.pdf')


def revyakin_fit_plot():
    """
    Plot data from fit to Revyakin data
    """

    # A bit lame: database with one entry
    # But it allows you to tweak the plot long after running the simulation
    shelve_database = shelve.open(simulation_storage)
    params_per_cycle = shelve_database['params']
    shelve_database.close()

    NAC_mean = [value['NAC'][0] for value in params_per_cycle.values()]
    NAC_std = [value['NAC'][1] for value in params_per_cycle.values()]

    escape_mean = [value['Escape'][0] for value in params_per_cycle.values()]
    escape_std = [value['Escape'][1] for value in params_per_cycle.values()]

    unscrunch_mean = [value['Unscrunch'][0] for value in params_per_cycle.values()]
    unscrunch_std = [value['Unscrunch'][1] for value in params_per_cycle.values()]

    df_mean = pd.DataFrame(data={'NAC': NAC_mean,
                                 'Escape': escape_mean,
                                 'Unscrunch': unscrunch_mean},
                           index=range(1, len(NAC_mean) + 1))

    dict_err = {'NAC': NAC_std, 'Escape': escape_std, 'Unscrunch': unscrunch_std}
    df_mean = df_mean[dict_err.keys()]  # get df in same order as dict

    f, axes = plt.subplots(3)
    df_mean.plot(subplots=True, ax=axes, yerr=dict_err,
                 sharex=True,
                 xlim=(0.5, len(NAC_mean) + 0.5),
                 xticks=range(1, len(NAC_mean) + 1),
                 ylim=(0, 26),
                 legend=False,
                 rot=0)

    axes[-1].set_xlabel('Parameter estimation cycle')

    for ax, rc in zip(axes, df_mean.columns):
        ax.set_ylabel('{0} (1/s)'.format(rc))

    add_letters(axes, ['A', 'B', 'C'], ['UL'] * 3)

    #set_fig_size(f, current_journal_width, 8)
    f.savefig(os.path.join(fig_dir, 'parameter_estimation_cycles.pdf'))

    # Print to screen
    print("Mean")
    print(df_mean)
    print('Std')
    print(pd.DataFrame(data=dict_err, index=df_mean.index))

    # Also print Tau from Vo to have it at hand
    tau = get_tau(1000, 2000, NAC_mean[-1], unscrunch_mean[-1], escape_mean[-1])
    print('Tau from Vo et al')
    print(tau)


def revyakin_fit_calc():
    """
    Interesting. Even if you run with just 50 samples, you still get
    convergence to small values. So to make it fair you'd need to iterate 100,
    500, 1000 samples to also see a convergence in the convergence of values.

    The method is very sensitive to the initial number of simulations. It's
    important to start with a very high number. Subsequent steps mostly reduce
    uncertainty.
    """
    # Setup
    variant_name = 'N25'
    ITSs = data_handler.ReadData('dg100-new')
    its_variant = [i for i in ITSs if i.name == variant_name][0]
    initial_RNAP = 100
    sim_end = 1000
    escape_RNA_length = 14
    #GreB = False
    GreB = True

    samples_per_cycle = 200
    cycles = 4
    boundaries = False

    plot = False
    #plot = True

    multiproc = True
    #multiproc = False

    # When unpacking you can rely on getting cycle-order
    params_per_cycle = OrderedDict()
    shelve_database = shelve.open(simulation_storage)
    for cycle in range(cycles):

        log_name = '{0}-RNAP_{1}-samples_{2}-GreB_scrunch_cycle_{3}'.format(initial_RNAP,
                                                                            samples_per_cycle,
                                                                            GreB, cycle)

        params = get_parameters(samples=samples_per_cycle, GreB=GreB,
                                boundaries=boundaries)
        stoi_setup = ITStoichiometricSetup(escape_RNA_length=escape_RNA_length)

        search_parameter_space(multiproc, its_variant, GreB, stoi_setup,
                               initial_RNAP, sim_end, params, plot, log_name,
                               False)

        boundaries, stats = get_weighted_parameter_distributions(log_name)

        params_per_cycle[cycle] = stats

    shelve_database['params'] = params_per_cycle
    shelve_database.close()


def add_letters(axes, letters, positions, shiftX=0, shiftY=0):
    """
    letters = ('A', 'B') will add A and B to the subplots at positions ('UL
    for example')
    """

    xy = {'x': {'UL': 0.01, 'UR': 0.85},
          'y': {'UL': 0.98, 'UR': 0.97}}

    for pos, label, ax in zip(positions, letters, axes):
        ax.text(xy['x'][pos] + shiftX,
                xy['y'][pos] + shiftY,
                label, transform=ax.transAxes, fontsize=12,
                fontweight='bold', va='top')


def get_weighted_parameter_distributions(log_name):

    # Use the top 20 results to decide on new parameter values
    top = 20
    #top = 15

    fitness_log_file = os.path.join(log_dir, log_name + '.log')
    df = pd.read_csv(fitness_log_file, sep='\t', header=0)

    df.sort('Fit', inplace=True)
    # write back the sorted df
    df.to_csv(fitness_log_file, sep='\t')

    df = df[:top]
    # But drop any rows with NaN values in the Fit column
    df = df.dropna(subset=['Fit'])

    nac = df['Nac']
    ab = df['Abort']
    es = df['Escape']

    weight = 1. / (df['Fit'] / df['Fit'].min())

    #step = 0.01

    f, axes = plt.subplots(3)
    #dff = pd.DataFrame(data=np.asarray(nac), index=np.asarray(weight))
    data = {'nac': np.asarray(nac), 'abortive': np.asarray(ab), 'Escape': np.asarray(es)}
    dff = pd.DataFrame(data=data, index=np.asarray(weight))
    dff = dff.sort_index()
    new_index = np.linspace(dff.index.min(), dff.index.max(), 200)
    dff = dff.reindex(new_index, method='nearest')
    dff.plot(kind='hist', ax=axes, use_index=True, bins=10, legend=False,
             normed=True, subplots=True)

    axes[1].set_xlabel('Unschrunching (1/s)')
    axes[1].set_ylabel('Weighted frequency')
    axes[2].set_xlabel('Nucleotide addition cycle (1/s)')
    axes[2].set_ylabel('Weighted frequency')
    axes[0].set_xlabel('Escape rate(1/s)')
    axes[0].set_ylabel('Weighted frequency')

    f.tight_layout()
    f.savefig(os.path.join('figures', 'rate_distribution_' + log_name + '.pdf'))

    # Print weighted mean and std of Nac and abort
    nac_wmean, nac_wstd = weighted_avg_and_std(df['Nac'], weight)
    abort_wmean, abort_wstd = weighted_avg_and_std(df['Abort'], weight)
    escape_wmean, escape_wstd = weighted_avg_and_std(df['Escape'], weight)
    print(log_name)
    print('abort', abort_wmean, abort_wstd)
    print('nac', nac_wmean, nac_wstd)
    print('escape', escape_wmean, escape_wstd)

    # Print top 5 fits
    print("Top 5 fits")
    print df['Fit'][:5]

    # Create new boundaries for new parameter sampling
    nac_low = nac_wmean - nac_wstd
    nac_high = nac_wmean + nac_wstd

    abort_low = abort_wmean - abort_wstd
    abort_high = abort_wmean + abort_wstd

    escape_low = escape_wmean - escape_wstd
    escape_high = escape_wmean + escape_wstd

    boundaries = {
        'nac_low': max(nac_low, 0),
        'nac_high': nac_high,

        'abort_low': max(abort_low, 0),
        'abort_high': abort_high,

        'escape_low': max(escape_low, 0),
        'escape_high': escape_high,
    }

    stats = {'NAC': (nac_wmean, nac_wstd),
             'Unscrunch': (abort_wmean, abort_wstd),
             'Escape': (escape_wmean, escape_wstd)}

    return boundaries, stats


def search_parameter_space(multiproc, its_variant, GreB, stoi_setup,
                           initial_RNAP, sim_end, params, plot, log_name,
                           adjust_ap_each_position):

    # So you can save dataframes to disk
    dataframes = {}

    # And use another weird way of logging the parameter results
    fitness_log_file = os.path.join(log_dir, log_name + '.log')
    log_handle = open(fitness_log_file, 'w')
    log_handle.write('Fit\tFit_extrap\tNac\tAbort\tEscape\n')

    if multiproc:
        pool = Pool(max_proc)
        results = []

    for nac, abrt, escape in params:

        R = ITRates(its_variant, nac=nac, unscrunch=abrt, escape=escape,
                    GreB_AP=GreB, adjust_ap_each_position=adjust_ap_each_position)

        sim_name = get_sim_name(stoi_setup, its_variant)

        sim_setup = ITSimulationSetup(sim_name, R, stoi_setup, initial_RNAP, sim_end)

        model = ITModel(sim_setup)

        # Is it true that results don't get back in the order I call them?
        if multiproc:
            args = {'include_elongation': True}
            r = pool.apply_async(model.run, args)
            results.append(r)
        else:
            model_ts = model.run(include_elongation=True)
            model_scrunches = calc_scrunch_distribution(model_ts, initial_RNAP)
            val, val_extrap = plot_and_log_scrunch_comparison(plot, log_handle,
                                                              model_scrunches,
                                                              initial_RNAP, nac,
                                                              abrt, escape)

            dataframes[(val, val_extrap)] = model_ts

    if multiproc:
        pool.close()
        pool.join()
        model_timeseries = [res.get() for res in results]
        for ts, pars in zip(model_timeseries, params):
            nac, abrt, escape = pars
            ms = calc_scrunch_distribution(ts, initial_RNAP)
            val, val_extrap = plot_and_log_scrunch_comparison(plot, log_handle,
                                                              ms, initial_RNAP,
                                                              nac, abrt, escape)
            dataframes[(val, val_extrap)] = ts

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

    return val, val_extrap


def plot_scrunch_distributions(model_scrunches, expt, expt_extrapolated,
                               title=''):
    """
    Now you are plotting the cumulative distribution function, but can you
    also plot the probability density function? Start off with a histogram.
    """

    plot_expt = True
    fig_dir = 'figures'

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
            ax.set_xticks(range(0, 21))
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


def compare_scrunch_data(model, experiment, experiment_extrap):
    """
    What should you do with a nan? Indicates that the result was not within
    the proper range.
    """
    # Get the values in the model closest to those in the experiment using
    # linear interpolation
    compare = pd.concat([model, experiment]).sort_index().\
              interpolate(method='index').reindex(experiment.index)

    diff = np.abs(compare['inverse_cumul_experiment'] - compare['inverse_cumul_model'])
    distance = np.linalg.norm(diff)

    c_extr = pd.concat([model, experiment_extrap]).sort_index().\
              interpolate(method='index').reindex(experiment_extrap.index)

    # Ignore stuff before 0.5 seconds, it's hard to go that fast,
    # so there will always be nans in the beginning
    c_extr = c_extr[0.5:]

    diff_extr = np.abs(c_extr['extrap_inv_cumul'] - c_extr['inverse_cumul_model'])
    distance_extr = np.linalg.norm(diff_extr)

    return distance, distance_extr


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

    def neg_exp(x, a):
        return np.exp(-a * x)

    popt, pcov = curve_fit(neg_exp, x, y)

    # Make some preductions from this simple model
    xx = np.linspace(0, 40, 1000)
    y_extrapolated = neg_exp(xx, popt[0])
    df_extrapolated = pd.DataFrame(data={'extrap_inv_cumul': y_extrapolated}, index=xx)

    return df_measurements, df_extrapolated


def calc_scrunch_distribution(df, initial_RNAP):
    """
    Will probably not get same distribution in time as Revyakin, but we might
    get the same shape. They use a different salt to the other experiments,
    and the concentration of that salt matters a lot for the result (see last
    fig in supplementry of Revyakin).
    """

    # when elongating gets +1 you have a promoter escape
    escapes = np.asarray(df['elongating complex'])[1:] - np.asarray(df['elongating complex'])[:-1]

    # Skip first timestep
    time = df.index[1:]
    escape_times = time[escapes == 1]

    integrated_probability = np.linspace(1, 1 / float(initial_RNAP), initial_RNAP)

    data = {'inverse_cumul_model': integrated_probability}

    scrunch_dist = pd.DataFrame(data=data, index=escape_times)

    return scrunch_dist


def get_sim_name(stoi_setup, its_name):

    if stoi_setup.part_unproductive:
        sim_name = '{0}_also_unproductive'.format(its_name)
    elif stoi_setup.unproductive_only:
        sim_name = '{0}_unproductive_only'.format(its_name)
    else:
        sim_name = '{0}_productive_only'.format(its_name)

    return sim_name


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


def ap_sensitivity_calc():
    """
    Interesting. Even if you run with just 50 samples, you still get
    convergence to small values. So to make it fair you'd need to iterate 100,
    500, 1000 samples to also see a convergence in the convergence of values.
    """
    # Setup
    variant_name = 'N25'
    ITSs = data_handler.ReadData('dg100-new')
    its_variant = [i for i in ITSs if i.name == variant_name][0]
    initial_RNAP = 100
    sim_end = 1000
    escape_RNA_length = 14
    #GreB = False
    GreB = True

    samples_per_cycle = 200
    cycles = 4

    plot = False
    #plot = True

    multiproc = True
    #multiproc = False

    ap_adjusts = range(-15, 16, 6)   # later for the 'full' show
    #ap_adjusts = range(-5, 6, 2)
    #ap_adjusts = range(-5, 6, 10)

    # You need to save the optimal NAC and the tau obtained with the optimal
    names = ['NAC', 'Escape', 'Unscrunch']

    # When unpacking you can rely on getting cycle-order
    ap_adjust_effect = OrderedDict()
    for ap_adjustment in ap_adjusts:
        optimal_values = OrderedDict()
        boundaries = False  # Important! Start off with fresh boundaries before each cycle
        for cycle in range(cycles):

            log_str = '{0}-RNAP_{1}-samples_{2}-GreB_scrunch_cycle_{3}_ap_adjust{4}'
            log_name = log_str.format(initial_RNAP, samples_per_cycle, GreB,
                                      cycle, ap_adjustment)

            params = get_parameters(samples=samples_per_cycle, GreB=GreB,
                                    boundaries=boundaries)
            stoi_setup = ITStoichiometricSetup(escape_RNA_length=escape_RNA_length)

            search_parameter_space(multiproc, its_variant, GreB, stoi_setup,
                                   initial_RNAP, sim_end, params, plot, log_name,
                                   ap_adjustment)

            boundaries, stats = get_weighted_parameter_distributions(log_name)

            optimal_values[cycle] = {n: stats[n][0] for n in names}

        # Extract the final optimal values from the last simulation
        final_optimal = {n: optimal_values[cycles - 1][n] for n in names}

        # Get half life of FL synthesis for -GreB N25
        #tau = get_tau(1000, sim_end, final_optimal['NAC'], final_optimal['Unscrunch'],
                      #final_optimal['Escape'])
        tau = get_tau(1000, sim_end, **final_optimal)

        ap_adjust_effect[ap_adjustment] = {'NAC': final_optimal['NAC'], 'Tau': tau}

    #return
    shelve_database = shelve.open(simulation_storage)
    shelve_database['ap_adjust'] = ap_adjust_effect
    shelve_database.close()


def get_tau(initial_RNAP, sim_end, NAC, Unscrunch, Escape):
    """
    Returns half-life of 100% FL synthesis
    """

    greb_minus_model = smr.get_kinetics(variant='N25',
                                        escape_RNA_length=14,
                                        initial_RNAP=initial_RNAP,
                                        sim_end=sim_end,
                                        nac=NAC,
                                        escape=Escape,
                                        unscrunch=Unscrunch)

    greb_minus = greb_minus_model.run()['FL']
    gm = greb_minus / greb_minus.max()

    # Get half-life of FL from model
    #thalf_grebminus = f_gm(0.5)
    idx_gm = gm.tolist().index(0.5)
    thalf_grebminus = gm.index[idx_gm]

    return thalf_grebminus


def ap_sensitivity_plot():

    shelve_db = shelve.open(simulation_storage)
    ap_adjust_effect = shelve_db['ap_adjust']
    shelve_db.close()
    ap_adjust_effect[0] = {'NAC': 10.4, 'Tau': 17.5}  # XXX get from global "correct" value?

    # The indices from -x to + x now including 0
    adjusts = sorted(ap_adjust_effect.keys())

    data = {}
    data['NAC'] = [ap_adjust_effect[i]['NAC'] for i in adjusts]
    #data['NAC_rel'] = [ap_adjust_effect[i]['NAC'] for i in adjusts]
    data['Tau'] = [ap_adjust_effect[i]['Tau'] for i in adjusts]

    df = pd.DataFrame(data=data, index=adjusts)
    # Normalize NAC rel relative to average (pos/neg supercoil) elongation speed from Revyakin: 11.65
    #df['NAC_rel'] = df['NAC_rel'] / 11.65

    f, axes = plt.subplots(2)

    df.plot(kind='bar', ax=axes, subplots=True, sharex=True, rot=0,
            legend=False, label=False, title=None)

    axes[0].set_ylabel('NAC (1/s)')
    #axes[1].set_ylabel('NAC relative to elongation')
    axes[1].set_ylabel('$\\tau$ (s)')
    axes[1].set_xlabel('Increase in AP at each ITS position (%)')

    # Pandas bug
    axes[0].set_title('')
    axes[1].set_title('')
    #axes[2].set_title('')

    add_letters(axes, ['A', 'B'], ['UL'] * 2)

    f.savefig(os.path.join(fig_dir, 'ap_adjustment.pdf'))


def fraction_scrunches_less_than_1_second_calc():

    # Setup
    initial_RNAP = 100
    sim_end = 1000
    escape_RNA_length = 14
    GreB = True

    #samples = range(1000)
    samples = range(1000)

    p = Pool(max_proc)
    results = []

    # When unpacking you can rely on getting cycle-order
    ts = {}
    for sample in samples:
        model = smr.get_kinetics('N25', initial_RNAP=initial_RNAP,
                                 sim_end=sim_end, GreB=GreB,
                                 escape_RNA_length=escape_RNA_length)
        args = {'include_elongation': True}
        r = p.apply_async(model.run, args)
        results.append(r)
    p.close()
    p.join()

    fracs = []

    for r, sample_nr in zip(results, samples):
        ts = r.get()
        model_scrunches = calc_scrunch_distribution(ts, initial_RNAP)
        up_to_one = model_scrunches[:1]
        if not up_to_one.empty:
            less_than_1_sec_frac = 1 - up_to_one.iloc[-1]
        fracs.append(less_than_1_sec_frac)

    print np.mean(fracs)
    print np.std(fracs)


def sensitivity_analysis():
    """
    Calculate the partial derivative of an output relative to each of the rate
    constants: delta(X)/X / delta(k)/k

    Let X be tau: the half-life to FL. In mathematical terms, we could
    formulate tau. If Y was the # of FL produced, they X = Y(t_1), where t_1
    is found by solving Y(t) = Z(0)/2, where Z is the # of complexes in open
    formation.

    Result: this is looking good! But increase the range to perhaps 30% and
    increase the number of RNAPs to 5000 to reduce noise. Parallelize the
    beast, and then do a simple plot
    """

    init_rcs = {'NAC': 10.5, 'Escape': 12.4, 'Unscrunch': 1.6}

    init_rnap = 1000

    perturbations = range(-10, 10, 1)
    init_tau = get_tau(init_rnap, 1000, **init_rcs)

    shelve_database = shelve.open(simulation_storage)

    # Perturbations from -10% to 10% in steps of 1.
    effects = defaultdict(list)
    for rc_name, init_rc in init_rcs.items():
        for pb in perturbations:
            # Perturb the rate constant and insert it in a dictionary
            pert_rc = init_rc * (1. + float(pb)/100.)
            pert_rcs = init_rcs.copy()
            pert_rcs[rc_name] = pert_rc

            pert_tau = get_tau(init_rnap, 1000, **pert_rcs)

            dtau = pert_tau / init_tau
            drc = pert_rc / init_rc
            sensitivity = dtau / drc

            effects[rc_name].append(sensitivity)

    shelve_database['sensitivities'] = (effects, perturbations)
    shelve_database.close()


def plot_sensitivity_analysis():

    shelve_database = shelve.open(simulation_storage)
    effects, perturbations = shelve_database['sensitivities']
    shelve_database.close()

    for key, vals in effects.items():
        print key
        print vals

    df = pd.DataFrame(data=effects, index=perturbations)

    f, axes = plt.subplots(3)

    df.plot(ax=axes, subplots=True, sharex=True)

    y_min = df.min().min()
    y_max = df.max().max()

    axes[-1].set_xlabel('Perturbation (%)')
    for ax in axes:
        ax.set_ylim(y_min, y_max)

    file_name = 'sensitivity_plot.pdf'
    file_path = os.path.join(fig_dir, file_name)
    f.savefig(file_path, format='pdf')


def main():

    # Distributions of % of IQ units and #RNA
    #initial_transcription_stats()

    # Remember, you are overestimating PY even if there is no recycling. Would
    # recycling lead to higher or lower PY? I'm guessing lower.
    #AP_and_pct_values_distributions_DG100()

    # Fitting to Revyakin: +/-GreB, iteratively; save the w/mean
    # and w/std to make a line plot that shows the convergence of the
    # iterative scheme.
    #revyakin_fit_calc()
    #revyakin_fit_plot()

    # Obtain the best parameters for N25 +Greb with extra percentages to AP.
    # +/- 1 to +/- 10 for example. Plot NAC and Tau (Vo).
    #ap_sensitivity_calc()
    #ap_sensitivity_plot()
    # would it be helpful to show a bar plot of the modified abortive
    # probabilities? Yes, and for Tau some kinetic plots as well just to drive
    # the point in. But for the nonce it will do. Now, you need to write :)
    # Actually, people will want to see the fit as well. With some mini-plots
    # I'm sure you can do it. Invent a measure: 1-std

    #tau_variation_plot()
    #fraction_scrunches_less_than_1_second_calc()

    # Fitting to ensemble studies
    #simple_vo_2003_comparison()

    # Local sensitivity analysis
    #sensitivity_analysis()
    plot_sensitivity_analysis()


if __name__ == '__main__':
    main()
