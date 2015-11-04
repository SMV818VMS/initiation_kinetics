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
from metrics import weighted_avg_and_std
sys.path.append('auxiliary')
from auxiliary import print_time
import data_handler
import stochastic_model_run as smr
import matplotlib.pyplot as plt
import matplotlib
matplotlib.style.use('ggplot')
from multiprocessing import Pool
from collections import OrderedDict, defaultdict
from pandas.tools.plotting import scatter_matrix

from ipdb import set_trace as debug  # NOQA
from scipy.interpolate import InterpolatedUnivariateSpline as interpolate
from scipy.optimize import curve_fit

# Global place for where you store your figures
here = os.path.abspath(os.path.dirname(__file__))  # where this script is located
#fig_dir = os.path.join(here, 'figures')
# XXX save directly to manuscript directory
fig_dir = '/home/jorgsk/Dropbox/The-Tome/my_papers/kinetic_initiation/figures'

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
    import seaborn as sns
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

    #def saturation2(x, a, b, c):
        ##return a + b * (1 - np.exp(-c * x))
        #return a + b * (1 - p(-c * x))

    popt, pcov = curve_fit(saturation, x, y)
    #popt2, pcov2 = curve_fit(saturation2, x, y)

    # Make some preductions from this simple model
    xx = np.linspace(x.min(), x.max(), 1000)
    yy = saturation(xx, *popt)
    #yy2 = saturation2(xx, *popt2)

    # Compare fits?

    #debug()

    return xx, yy


def thalf_vo_2003(method=1):

    # FL kinetics for 1 and 2.
    df = get_2003_FL_kinetics(method=1)

    # Fit 2003 data and calculate half life
    xnew, ynew = fit_FL(df.index, df.values)

    # Get the halflife of full length product
    f_vo = interpolate(ynew, xnew, k=1)
    thalf_vo = float(f_vo(0.5))

    return thalf_vo


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


def heatmap(df, fit_name='Fit', ax=False, max_value=False):
    """
    Plot heatmap of Nac and Abort from a dataframe
    """

    x = df['Nac']
    y = df['Abort']
    z = df[fit_name]

    # For filling the Z matrix with values from x and y (thanks unutbu)
    x_range, x_idx = np.unique(x, return_inverse=True)
    y_range, y_idx = np.unique(y, return_inverse=True)

    X, Y = np.meshgrid(x_range, y_range)
    Z = np.zeros((x_idx.max() + 1, y_idx.max() + 1))
    Z[x_idx, y_idx] = z

    # If max value specified, mask higher values
    if max_value is False:
        pass
    else:
        Z = np.ma.masked_where(Z > max_value, Z)

    return_fig = False
    if ax is False:
        f, ax = plt.subplots()
        return_fig = True

    # Absurd way of specifiying a color bar to a plot
    # And why the transpose?
    whatever_matplotlib = ax.pcolormesh(X, Y, Z.T, cmap=matplotlib.cm.rainbow_r)
    plt.colorbar(whatever_matplotlib, ax=ax)

    ax.set_xlabel('NAC 1/s')
    ax.set_ylabel('Unscrunch 1/s')

    if return_fig:
        return f, ax


def plot_heatmaps(log_name):
    """
    Plot a heat map of NAC vs abortive and unscrunch.
    """

    fitness_log_file = os.path.join(log_dir, log_name + '.log')
    df = pd.read_csv(fitness_log_file, sep='\t', header=0)

    params = get_log_params(log_name)
    if params['fitextrap']:
        fit = 'Fit_extrap'
    else:
        fit = 'Fit'

    f, ax = heatmap(df, fit_name=fit)

    file_path = os.path.join(fig_dir, 'nac_vs_unscrunch_final_grid.pdf')
    f.savefig(file_path, format='pdf')
    plt.close(f)

    #fig = plt.figure()
    #from mpl_toolkits.mplot3d import Axes3D  # NOQA
    #ax = fig.gca(projection='3d')
    #whatev = ax.plot_surface(X, Y, Z, cmap=matplotlib.cm.rainbow_r)
    #plt.colorbar(whatev)
    #file_path = os.path.join(fig_dir, 'nac_vs_unscrunch_final_grid_3d.pdf')
    #fig.savefig(file_path, format='pdf')
    #plt.close()


def get_new_boundaries(log_name, fit_extrap, nr_samples):
    """
    Look at top 5% (or 1%) of simulation. Choose 95pct interval of top 5%.
    """

    fitness_log_file = os.path.join(log_dir, log_name + '.log')
    df = pd.read_csv(fitness_log_file, sep='\t', header=0)

    # Determines which value we optimize for
    if fit_extrap:
        fit = 'Fit_extrap'
    else:
        fit = 'Fit'

    df.sort(fit, inplace=True)

    # Get values from top1
    top_5pct_idx = int(nr_samples * 0.01)
    df = df[:top_5pct_idx]

    # But drop any rows with NaN values in the Fit column
    df = df.dropna(subset=[fit])

    # Extract the weighted mean +/- 1 std for Nac and Abort
    weight = 1. / (df[fit] / df[fit].min())
    # Print weighted mean and std of Nac and abort
    nac_wmean, nac_wstd = weighted_avg_and_std(df['Nac'], weight)
    abort_wmean, abort_wstd = weighted_avg_and_std(df['Abort'], weight)
    escape_wmean, escape_wstd = weighted_avg_and_std(df['Escape'], weight)

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
             'Escape': (escape_wmean, escape_wstd),
             'Top fit': df['Fit'].iloc(0)}

    return boundaries, stats


def bool_or_float(v):
    if v == 'False':
        return False
    elif v == 'True':
        return True
    else:
        return float(v)


def get_log_params(log_name):
    log_split1 = log_name.split('_')

    params = {}
    for tt in log_split1:
        k, v = tt.split('-')
        params[k] = bool_or_float(v)

    return params


def plot_parameter_scatter_matrix(log_name):

    fitness_log_file = os.path.join(log_dir, log_name + '.log')
    df = pd.read_csv(fitness_log_file, sep='\t', header=0)

    params = get_log_params(log_name)
    if params['fitextrap']:
        fit = 'Fit_extrap'
    else:
        fit = 'Fit'

    df.sort(fit, inplace=True)

    # First plot all values
    subset_df = df[[fit, 'Nac', 'Abort', 'Escape']]
    axes = scatter_matrix(subset_df, diagonal='hist', grid=False)
    for ax in axes[:, 0]:
        ax.grid('off', axis='both')
    for ax in axes[-1, :]:
        ax.grid('off', axis='both')

    if int(params['round']) == 1:
        file_name = 'first_round_scatter_matrix_{0}_complete.pdf'.format(fit)
    elif int(params['round']) == 2:
        file_name = 'final_round_scatter_matrix_{0}_complete.pdf'.format(fit)

    file_path = os.path.join(fig_dir, file_name)
    plt.savefig(file_path, format='pdf')
    plt.close()

    # Then plot top 1 and 5 pct
    tops = [1, 5]
    for top in tops:

        top_idx = int(params['samples'] * top * 0.01)
        top_df = subset_df[:top_idx]
        if int(params['round']) == 2:
            top_df.drop('Escape', axis=1, inplace=True)
        axes = scatter_matrix(top_df, diagonal='hist', grid=False)

        for ax in axes[:, 0]:
            ax.grid('off', axis='both')
        for ax in axes[-1, :]:
            ax.grid('off', axis='both')

        if int(params['round']) == 1:
            file_name = 'first_round_scatter_matrix_{0}_top_{1}_pct.pdf'.format(fit, top)

        elif int(params['round']) == 2:
            file_name = 'final_round_scatter_matrix_{0}_top_{1}_pct.pdf'.format(fit, top)

        file_path = os.path.join(fig_dir, file_name)
        plt.savefig(file_path, format='pdf')
        plt.close()


def add_letters(axes, letters, positions, shiftX=0, shiftY=0, fontsize=12):
    """
    letters = ('A', 'B') will add A and B to the subplots at positions ('UL
    for example')
    """

    xy = {'x': {'UL': 0.01, 'UR': 0.85},
          'y': {'UL': 0.98, 'UR': 0.97}}

    for pos, label, ax in zip(positions, letters, axes):
        ax.text(xy['x'][pos] + shiftX,
                xy['y'][pos] + shiftY,
                label, transform=ax.transAxes, fontsize=fontsize,
                fontweight='bold', va='top')


def get_weighted_parameter_distributions(df, fit_extrap=False):

    # Use the top 20 results to decide on new parameter values
    top = 20
    #top = 15

    if fit_extrap:
        fit = 'Fit_extrap'
    else:
        fit = 'Fit'

    df.sort_values(fit, inplace=True)

    df = df[:top]
    # But drop any rows with NaN values in the Fit column
    df = df.dropna(subset=[fit])

    nac = df['Nac']
    ab = df['Abort']
    es = df['Escape']

    weight = 1. / (df[fit] / df[fit].min())

    data = {'nac': np.asarray(nac), 'abortive': np.asarray(ab), 'Escape': np.asarray(es)}
    dff = pd.DataFrame(data=data, index=np.asarray(weight))
    dff = dff.sort_index()
    new_index = np.linspace(dff.index.min(), dff.index.max(), 200)
    dff = dff.reindex(new_index, method='nearest')

    # Print weighted mean and std of Nac and abort
    nac_wmean, nac_wstd = weighted_avg_and_std(df['Nac'], weight)
    abort_wmean, abort_wstd = weighted_avg_and_std(df['Abort'], weight)
    escape_wmean, escape_wstd = weighted_avg_and_std(df['Escape'], weight)

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
             'Escape': (escape_wmean, escape_wstd),
             'Top fit': df['Fit'].iloc(0)}

    return boundaries, stats


def add_search_result(result, fit, fit_extrap, nac, abrt, escape):

    result['Fit'].append(fit)
    result['Fit_extrap'].append(fit_extrap)
    result['NAC'].append(nac)
    result['Abort'].append(abrt)
    result['Escape'].append(escape)


def search_parameter_space(multiproc, its_variant, GreB, stoi_setup,
                           initial_RNAP, sim_end, params, adjust_ap_each_position):
    """
    Run simulation with the given setup. Compare scrunch-distribution of model
    with experimental measurements. Return dataframe with:

    Fit: fit to measured data points
    Fit_extrap: fit to extrapolated scrunch distribution
    NAC: NAC rate constant
    Abort: unscrunching and abortive release rate constant
    Escape: promoter escape rate constant

    With a high number of simulations memory is becoming an issue. This is
    because with multiprocessing all model timeseries are being kept in
    memory. Solution is to move the scrunch distribution calculation inside.

    Or. Better solution is to split the calculation up in digestible pieces.
    So max 1000 simulations, and then build the dataframe.
    """

    # For comparison with experimental
    experiment, extrapolated = get_experiment_scrunch_distribution()

    # Store the results in dict for converting to DataFrame later
    result = defaultdict(list)

    if multiproc:
        result_df = _search_parameter_space_multi(its_variant, GreB,
                                                  stoi_setup, initial_RNAP,
                                                  sim_end, params,
                                                  adjust_ap_each_position)

    else:
        for nac, abrt, escape in params:

            R = ITRates(its_variant, nac=nac, unscrunch=abrt, escape=escape,
                        GreB_AP=GreB, adjust_ap_each_position=adjust_ap_each_position)

            sim_name = get_sim_name(stoi_setup, its_variant)
            sim_setup = ITSimulationSetup(sim_name, R, stoi_setup, initial_RNAP, sim_end)
            model = ITModel(sim_setup)

            model_ts = model.run(include_elongation=True)
            scrunch_dist = calc_scrunch_distribution(model_ts, initial_RNAP)
            fit, fit_extrap = compare_scrunch_data(scrunch_dist, experiment, extrapolated)

            # Add result
            add_search_result(result, fit, fit_extrap, nac, abrt, escape)

        result_df = pd.DataFrame(result)

    return result_df


def _search_parameter_space_multi(its_variant, GreB, stoi_setup,
                                  initial_RNAP, sim_end, params,
                                  adjust_ap_each_position):

    """
    Isolate out the multiproc variant. It would be best to have a class that
    handles this.
    """
    # Split into poolsizes of 100 simulations. Avoids keeping too much in
    # memory.
    stepsize = 100

    # For comparison with experimental
    experiment, extrapolated = get_experiment_scrunch_distribution()

    result = defaultdict(list)

    for start_index in range(0, len(params), stepsize):

        print(start_index)

        subparams = params[start_index:start_index + stepsize]

        results = []
        pool = Pool(max_proc)
        for nac, abrt, escape in subparams:
            R = ITRates(its_variant, nac=nac, unscrunch=abrt, escape=escape,
                        GreB_AP=GreB, adjust_ap_each_position=adjust_ap_each_position)

            sim_name = get_sim_name(stoi_setup, its_variant)
            sim_setup = ITSimulationSetup(sim_name, R, stoi_setup, initial_RNAP, sim_end)
            model = ITModel(sim_setup)

            args = {'include_elongation': True}
            r = pool.apply_async(model.run, args)
            results.append(r)

        pool.close()
        pool.join()
        model_timeseries = [res.get() for res in results]

        for ts, pars in zip(model_timeseries, subparams):
            nac, abrt, escape = pars
            scrunch_dist = calc_scrunch_distribution(ts, initial_RNAP)
            fit, fit_extrap = compare_scrunch_data(scrunch_dist, experiment, extrapolated)

            # Add result
            add_search_result(result, fit, fit_extrap, nac, abrt, escape)

    result_df = pd.DataFrame(result)

    return result_df


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
    Get distribution of time spent in abortive cycling before promoter escape
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


def get_parameters(samples=10, boundaries=False):
    """
    For Revyakin data you got a similar result for stepwise and uniform
    parameter sampling.

    Method can be uniform or stepwise.
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
        nac_low = max(boundaries['nac_low'], 2)
        nac_high = min(boundaries['nac_high'], 25)

        abort_low = max(boundaries['abort_low'], 1)
        abort_high = min(boundaries['abort_high'], 25)

        escape_low = max(boundaries['escape_low'], 2)
        escape_high = min(boundaries['escape_high'], 25)

    params = []
    # Uniform sampling
    for sample_nr in range(samples):
        nac = uniform(nac_low, nac_high)
        abortive = uniform(abort_low, abort_high)
        escape = uniform(escape_low, escape_high)

        params.append((nac, abortive, escape))

    return params


def get_variant(name):

    ITSs = data_handler.ReadData('dg100-new')
    its_variant = [i for i in ITSs if i.name == name][0]

    return its_variant


def ap_sensitivity_calc():
    """
    Interesting. Even if you run with just 50 samples, you still get
    convergence to small values. So to make it fair you'd need to iterate 100,
    500, 1000 samples to also see a convergence in the convergence of values.
    """
    # Setup
    variant_name = 'N25'
    its_variant = get_variant(variant_name)
    initial_RNAP = 100
    sim_end = 200000  # large number to ensure all simulations go until the end
    escape_RNA_length = 14
    GreB = True

    samples_per_cycle = 1000
    #cycles = 4
    cycles = 1

    plot = False
    multiproc = True

    ap_adjusts = range(-15, 16, 3)   # later for the 'full' show
    ap_adjusts.append(0)  # get the base case as well
    ap_adjusts.sort()

    # You need to save the optimal NAC and the tau obtained with the optimal
    names = ['NAC', 'Escape', 'Unscrunch', 'Top fit']
    rc_names = ['NAC', 'Escape', 'Unscrunch']

    # When unpacking you can rely on getting cycle-order
    ap_adjust_effect = OrderedDict()
    for ap_adjustment in ap_adjusts:
        optimal_values = OrderedDict()
        boundaries = False  # Important! Start off with fresh boundaries before each cycle
        for cycle in range(cycles):

            # Can this have an effect? It shouldn't :S
            if ap_adjustment == 0:
                ap_adjustment = False

            log_str = '{0}-RNAP_{1}-samples_{2}-GreB_scrunch_cycle_{3}_ap_adjust{4}'
            log_name = log_str.format(initial_RNAP, samples_per_cycle, GreB,
                                      cycle, ap_adjustment)

            params = get_parameters(samples=samples_per_cycle,
                                    boundaries=boundaries)
            stoi_setup = ITStoichiometricSetup(escape_RNA_length=escape_RNA_length)

            result = search_parameter_space(multiproc, its_variant, GreB,
                                            stoi_setup, initial_RNAP, sim_end,
                                            params, plot, log_name,
                                            ap_adjustment)

            boundaries, stats = get_weighted_parameter_distributions(result)

            optimal_values[cycle] = {n: stats[n][0] for n in names}

        # Extract the final optimal values from the last simulation
        final_optimal = {n: optimal_values[cycles - 1][n] for n in names}
        final_optimal_rcs = {n: optimal_values[cycles - 1][n] for n in rc_names}

        # Get half life of FL synthesis for -GreB N25
        tau = get_tau(5000, sim_end, **final_optimal_rcs)

        ap_adjust_effect[ap_adjustment] = final_optimal.copy()
        ap_adjust_effect[ap_adjustment]['Tau'] = tau

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


def get_ap_adjusted_tau(data, init_rnap):
    """
    Just a not about zip. Zip works like this: zip([1,2,3], [4,5,6]) is an
    iterator that yields (1,4), (2,5), (3,6).
    """

    shelve_db = shelve.open(simulation_storage)
    sim_end = 9000
    p = Pool()
    results = []
    for nac, top_fit, unscrunch, escape in zip(*data.values()):
        args = (init_rnap, sim_end, nac, unscrunch, escape)
        res = p.apply_async(get_tau, args)
        results.append(res)
    p.close()
    p.join()

    taus = [r.get() for r in results]

    shelve_db['ap_adjusted_tau_{0}'.format(init_rnap)] = taus
    shelve_db.close()

    return taus


def aps_adjusted_plot(adjusts):
    """
    Show the AP distribution for -15 ... + 15 at all positions.

    Unfortunately the GreB+ APs are only available for a single experiment. By
    contranst, GreB- has multiple replicas. The GreB+ AP for +2 is 60%, while
    it's only 32% for GreB-. In theory, they should be the same. It's
    tempting to set the APs for +2 to +5 equal between GreB+ and GreB-. But
    then you'd get a slower NAC, and a worse fit with Vo 2003.
    """

    its_variant = get_variant('N25')

    baseline_aps = its_variant.abortive_prob_GreB

    positives = [baseline_aps]
    negatives = [baseline_aps]

    # Adjusted APs are symmetrical, so go through the postive ones
    false_i = adjusts.index(False)
    abs_adjusts = adjusts[false_i + 1:]

    for adj in abs_adjusts:

        pos_aps = baseline_aps + adj / 100.
        pos_aps[pos_aps > 1] = 1  # in case adjustment leads to aps > 1

        neg_aps = baseline_aps - adj / 100.
        neg_aps[neg_aps < 0] = 0  # in case adjustment leads to aps < 0

        positives.append(pos_aps)
        negatives.append(neg_aps)

    positions = range(2, len(baseline_aps) + 2)
    pos_col_names = ['Baseline'] + ['+' + str(a) + '%' for a in abs_adjusts]
    pos_data = {n: v for n, v in zip(pos_col_names, positives)}
    pos_df = pd.DataFrame(data=pos_data, index=positions)
    pos_df = pos_df[pos_col_names]

    neg_col_names = ['Baseline'] + ['-' + str(a) + '%' for a in abs_adjusts]
    neg_data = {n: v for n, v in zip(neg_col_names, negatives)}
    neg_df = pd.DataFrame(data=neg_data, index=positions)
    neg_df = neg_df[neg_col_names]

    # How do you arrange the sublots the way you want them? You could make two
    # dataframes, and arrange the columns from baseline to most extrme case.
    # Then plot, but pass tot axes the two columns you made with subplots.

    f, axes = plt.subplots(6, 2)

    # restrict to +11 and get %
    pos_df = pos_df[:10] * 100
    neg_df = neg_df[:10] * 100
    ylim = (0, 80)
    yticks = [0, 40, 80]

    pos_df.plot(kind='bar', ax=axes[:, 0], subplots=True, sharex=True, rot=0,
                legend=False, label=False, title=None, fontsize=7, width=0.3,
                color='k', ylim=ylim, yticks=yticks)

    # Try to mask negative values
    neg_df.mask(neg_df == 0, inplace=True)
    neg_df.plot(kind='bar', ax=axes[:, 1], subplots=True, sharex=True, rot=0,
                legend=False, label=False, title=None, fontsize=7, width=0.3,
                color='k', ylim=ylim, yticks=yticks)

    f.set_size_inches(6, 8)
    f.tight_layout()
    f.savefig(os.path.join(fig_dir, 'aps_after_adjustment.pdf'))

    1 / 0


def ap_sensitivity_plot():

    shelve_db = shelve.open(simulation_storage)
    ap_adjust_effect = shelve_db['ap_adjust']
    shelve_db.close()

    # The indices from -x to + x now including 0
    adjusts = sorted(ap_adjust_effect.keys())

    # Make a plot that shows the effect of these adjusts on the AP
    # distribution
    aps_adjusted_plot(adjusts)

    data = {}
    data['NAC'] = [ap_adjust_effect[i]['NAC'] for i in adjusts]
    data['Escape'] = [ap_adjust_effect[i]['Escape'] for i in adjusts]
    data['Unscrunch'] = [ap_adjust_effect[i]['Unscrunch'] for i in adjusts]
    data['Top fit'] = [ap_adjust_effect[i]['Top fit'] for i in adjusts]
    data['Tau'] = [ap_adjust_effect[i]['Tau'] for i in adjusts]

    # Get tau value from vo et. al
    tau_vo = thalf_vo_2003(method=1)

    # Replace False with 0
    adjusts[adjusts.index(False)] = 0
    df = pd.DataFrame(data=data, index=adjusts)
    #df = df[['NAC', 'Escape', 'Unscrunch', 'Top fit', 'Tau']]  # reorder in the order you want
    df = df[['Top fit', 'NAC', 'Unscrunch', 'Escape', 'Tau']]  # reorder in the order you want

    # Normalize to experimentally obtained
    df['Tau'] = df['Tau'] / tau_vo

    nr_subplots = 5
    f, axes = plt.subplots(nr_subplots, figsize=(4, 8))

    df.plot(kind='bar', ax=axes, subplots=True, sharex=True, rot=0,
            legend=False, label=False, title=None, fontsize=7, width=0.3,
            color='k')

    font_size = 9

    axes[0].set_ylabel(r'Fit (RMSD)', fontsize=font_size)
    axes[1].set_ylabel(r'NAC ($s^{-1}$)', fontsize=font_size)
    axes[2].set_ylabel(r'Unscrunch ($s^{-1}$)', fontsize=font_size)
    axes[3].set_ylabel(r'Escape ($s^{-1}$)', fontsize=font_size)
    axes[4].set_ylabel(r'$\tau_r$', fontsize=font_size)
    axes[4].set_xlabel('Increase in AP at each ITS position (%)', fontsize=font_size)

    # Pandas bug; have to remove title
    for i in range(nr_subplots):
        axes[i].set_title('')

    # Add A, B, C... etc to the plot
    import string
    letters = list(string.uppercase)[:nr_subplots]
    add_letters(axes, letters, ['UL'] * nr_subplots, shiftY=0.15, shiftX=-0.14)

    f.tight_layout()
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

    This code was nice before multiprocessing. Storing and later matching
    results values turned out to make a mess.
    """

    init_rcs = {'NAC': 10.5, 'Escape': 12.4, 'Unscrunch': 1.6}

    init_rnap = 5000

    perturbations = range(-30, 30, 2)
    #perturbations = range(-10, 10, 2)
    init_tau = get_tau(init_rnap, 1000, **init_rcs)

    shelve_database = shelve.open(simulation_storage)

    p = Pool()
    results = []
    numbers = []

    # Store all the numbers you need after multiprocessing is done
    effects = defaultdict(dict)
    for rc_name, init_rc in init_rcs.items():
        for pb in perturbations:
            # Perturb the rate constant and insert it in a dictionary
            pert_rc = init_rc * (1. + float(pb) / 100.)
            pert_rcs = init_rcs.copy()
            pert_rcs[rc_name] = pert_rc

            first_args = (init_rnap, 1000)

            # apply async accepts keyword arguments in third position
            r = p.apply_async(get_tau, first_args, pert_rcs)
            results.append(r)
            drc = pert_rc / init_rc

            number_tuple = (rc_name, pb, drc)
            numbers.append(number_tuple)
    p.close()
    p.join()

    for r, number_tuple in zip(results, numbers):
        pert_tau = r.get()
        rc_name, pb, drc = number_tuple  # unpack

        dtau = pert_tau / init_tau
        sensitivity = dtau / drc

        effects[rc_name][pb] = sensitivity

    # Reconstructs the effects, in order
    effects2 = defaultdict(list)
    for rc_name, init_rc in init_rcs.items():
        for pb in perturbations:

            effects2[rc_name].append(effects[rc_name][pb])

    shelve_database['sensitivities'] = (effects2, perturbations)
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


def parameter_estimation_plot():
    """
    Use the two 50k simulations you've made (2 rounds) to make plots that show
    the parameter estimation process. You need to do this in two parts. First
    you need to show what happens after the random sampling. Show the
    distribution of the top 5% and the top 1% of results. This explains your
    boundaries. This is quite a big result on its own. Then do the same for
    the gridded results. I'm not sure if you should show the x vs y plot,
    unless it gets drastically better when considering the top 5% and top 1%.

    TODO:
    1) Make two figures with 3x3 histograms (well, last one without escape,
    since it's not sensitive).
    2) Make one figure (1x3) plotting NAC vs Unscrunch using some heat map,
    for 100, 5, and 1% of the best values. Hopefully we're left with a circle
    on the figure.

    Or make one 6x3 figure for the supplementary.

    NAC, U+A, Escape
    I1 100%
    I1   5%
    I1   1%
    I2 100%
    I2   5%
    I2   1%

    But for I2, don't have anything for escape, but instead have heatmap 2x2 plotting
    of NAC vs U+A. At least try that and see how it goes. That's a full page
    in supplementary materials, or indeed in the main article itself. Not
    unheard of. You're not shooting for the stars here.
    """

    # Old was when you used top 5% for setting new boundaries
    # From now on use top 1% for setting new boundaries.
    #round_1 = 'Log/RNAP-100_samples-50000_GreB-True_round-1_fitextrap-False.log'
    #round_2 = 'Log/RNAP-100_samples-50000_GreB-True_round-2_fitextrap-False.log'

    round_1 = 'Log/old_RNAP-100_samples-50000_GreB-True_round-1_fitextrap-False.log'
    #round_2 = 'Log/old_RNAP-100_samples-50000_GreB-True_round-2_fitextrap-False.log'
    #round_0 = 'Log/RNAP-100_samples-60000_GreB-True_round-stepwise_fitextrap-False_high_res.log'

    # Read from file and sort by Fit
    r1 = pd.read_csv(round_1, sep='\t').sort_values('Fit')
    #r2 = pd.read_csv(round_2, sep='\t').sort_values('Fit')
    #r0 = pd.read_csv(round_0, sep='\t').sort_values('Fit')

    nr_samples = r1.shape[0]

    top1 = int(nr_samples * 0.01)
    top01 = int(nr_samples * 0.001)

    #r1_top01 = r1[:top01]
    r1_top1 = r1[:top1]
    r1_top01 = r1[:top01]

    #r2_top1 = r2[:top1]
    #r2_top01 = r2[:top01]

    #r0_top1 = r0[:top1]
    #r0_top01 = r0[:top01]

    # Min/max values for the x axis
    xmin_nac = r1['Nac'].min()
    xmax_nac = r1['Nac'].max()

    xmin_abort = r1['Abort'].min()
    xmax_abort = r1['Abort'].max()

    xmin_escape = r1['Escape'].min()
    xmax_escape = r1['Escape'].max()

    #only_first_iteration_plot(xmin_nac, xmax_nac, xmin_abort, xmax_abort,
                              #xmin_escape, xmax_escape, r1, r1_top01, r1_top1,
                              #top1, top01)

    only_first_iteration_plot_reduced(xmin_nac, xmax_nac, xmin_abort,
                                      xmax_abort, xmin_escape, xmax_escape,
                                      r1, r1_top01, r1_top1, top1, top01)

    #one_figure_estimation_plot(xmin_nac, xmax_nac, xmin_abort, xmax_abort,
                               #xmin_escape, xmax_escape, r1, r1_top01, r1_top1,
                               #r2, r2_top01, r2_top1, top1, top01)

    #two_figures_estimation_plot()


# start adding plots one by one
def hist_plotter(df, ax, xmin, xmax, c, bins=10, normed=True, histtype='bar'):
    df.plot(kind='hist', bins=bins, ax=ax, normed=normed, color=c)

    plt.setp(ax.get_yticklabels(), visible=False)
    ax.set_ylabel('')
    ax.set_xlim((xmin, xmax))
    xticks = np.linspace(xmin, xmax, 5)
    ax.set_xticks(xticks)

    ax.grid(b='off')


def only_first_iteration_plot_reduced(xmin_nac, xmax_nac, xmin_abort,
                                      xmax_abort, xmin_escape, xmax_escape,
                                      r1, r1_top01, r1_top1, top1, top01):

    # About one A4 page in size for this plot
    f1, axes = plt.subplots(1, 3, figsize=(8.87, 3.6))

    plot_color = 'gray'

    hist_plotter(r1_top1['Nac'], axes[0], xmin_nac, xmax_nac, plot_color)
    hist_plotter(r1_top1['Abort'], axes[1], 1, 5, plot_color)
    hist_plotter(r1_top1['Escape'], axes[2], xmin_escape, xmax_escape, plot_color)

    # Set titles for the plots
    axes[0].set_xlabel('NAC (1/s)', fontsize=14)
    axes[1].set_xlabel('Unscrunching and\nabortive release (1/s)',
                       multialignment='center', fontsize=14)
    axes[2].set_xlabel('Escape (1/s)', fontsize=14)

    f1.tight_layout()

    file_name = 'reduced_parameter_estimation_distributions_monte_carlo.pdf'
    file_path = os.path.join(fig_dir, file_name)
    f1.savefig(file_path, format='pdf')
    plt.close(f1)


def only_first_iteration_plot(xmin_nac, xmax_nac, xmin_abort, xmax_abort,
                              xmin_escape, xmax_escape, r1, r1_top01, r1_top1,
                              top1, top01):

    # About one A4 page in size for this plot
    f1, axes1 = plt.subplots(3, 4, figsize=(8.27, 5.6))

    # First 3 rows (Iteration 1)
    hist_plotter(r1['Fit'], axes1[0, 0], 0, 2)
    hist_plotter(r1['Nac'], axes1[0, 1], xmin_nac, xmax_nac)
    hist_plotter(r1['Abort'], axes1[0, 2], xmin_abort, xmax_abort)
    hist_plotter(r1['Escape'], axes1[0, 3], xmin_escape, xmax_escape)

    hist_plotter(r1_top1['Fit'], axes1[1, 0], 0, 0.2)
    hist_plotter(r1_top1['Nac'], axes1[1, 1], xmin_nac, xmax_nac)
    hist_plotter(r1_top1['Abort'], axes1[1, 2], 1, 5)
    hist_plotter(r1_top1['Escape'], axes1[1, 3], xmin_escape, xmax_escape)

    hist_plotter(r1_top01['Fit'], axes1[2, 0], 0, 0.2)
    hist_plotter(r1_top01['Nac'], axes1[2, 1], xmin_nac, xmax_nac)
    hist_plotter(r1_top01['Abort'], axes1[2, 2], 1, 5)
    hist_plotter(r1_top01['Escape'], axes1[2, 3], xmin_escape, xmax_escape)

    # Set titles for the plots
    axes1[2, 0].set_xlabel('Root mean squared distance')
    axes1[2, 1].set_xlabel('NAC (1/s)')
    axes1[2, 2].set_xlabel('Unscrunching and\nabortive release (1/s)',
                           multialignment='center')
    axes1[2, 3].set_xlabel('Escape (1/s)')

    axes1[0, 0].set_ylabel('All samples')
    axes1[1, 0].set_ylabel('Top 1%', multialignment='center')
    axes1[2, 0].set_ylabel('Top 0.1%', multialignment='center')

    f1.tight_layout()

    file_name = 'parameter_estimation_distributions_monte_carlo.pdf'
    file_path = os.path.join(fig_dir, file_name)
    f1.savefig(file_path, format='pdf')
    plt.close(f1)


def two_figures_estimation_plot(xmin_nac, xmax_nac, xmin_abort, xmax_abort,
                                xmin_escape, xmax_escape, r1, r1_top01,
                                r1_top1, r2, r2_top01, r2_top1, top1, top01):

    # About one A4 page in size for this plot
    f1, axes1 = plt.subplots(3, 4, figsize=(8.27, 5.6))

    # First 3 rows (Iteration 1)
    hist_plotter(r1['Fit'], axes1[0, 0], 0, 2)
    hist_plotter(r1['Nac'], axes1[0, 1], xmin_nac, xmax_nac)
    hist_plotter(r1['Abort'], axes1[0, 2], xmin_abort, xmax_abort)
    hist_plotter(r1['Escape'], axes1[0, 3], xmin_escape, xmax_escape)

    hist_plotter(r1_top1['Fit'], axes1[1, 0], 0, 0.2)
    hist_plotter(r1_top1['Nac'], axes1[1, 1], xmin_nac, xmax_nac)
    hist_plotter(r1_top1['Abort'], axes1[1, 2], 1, 5)
    hist_plotter(r1_top1['Escape'], axes1[1, 3], xmin_escape, xmax_escape)

    hist_plotter(r1_top01['Fit'], axes1[2, 0], 0, 0.2)
    hist_plotter(r1_top01['Nac'], axes1[2, 1], xmin_nac, xmax_nac)
    hist_plotter(r1_top01['Abort'], axes1[2, 2], 1, 5)
    hist_plotter(r1_top01['Escape'], axes1[2, 3], xmin_escape, xmax_escape)

    # Set titles for the plots
    axes1[2, 0].set_xlabel('Root mean squared distance')
    axes1[2, 1].set_xlabel('NAC 1/s')
    axes1[2, 2].set_xlabel('Unscrunching and\nabortive release 1/s',
                           multialignment='center')
    axes1[2, 3].set_xlabel('Escape 1/s')

    axes1[0, 0].set_ylabel('All samples')
    axes1[1, 0].set_ylabel('Top 1%', multialignment='center')
    axes1[2, 0].set_ylabel('Top 0.1%', multialignment='center')

    f1.tight_layout()

    file_name = 'parameter_estimation_distributions_1.pdf'
    file_path = os.path.join(fig_dir, file_name)
    f1.savefig(file_path, format='pdf')
    plt.close(f1)

    #######################################################################

    # About one A4 page in size for this plot
    f2, axes2 = plt.subplots(3, 4, figsize=(8.27, 5.6))

    # Next 3 rows (Iteration 2)
    hist_plotter(r2['Fit'], axes2[0, 0], 0, 1)
    hist_plotter(r2['Nac'], axes2[0, 1], 5, 20)
    hist_plotter(r2['Abort'], axes2[0, 2], 1, 10)

    hist_plotter(r2_top01['Fit'], axes2[1, 0], 0, 0.15)
    hist_plotter(r2_top01['Nac'], axes2[1, 1], 5, 20)
    hist_plotter(r2_top01['Abort'], axes2[1, 2], 1, 10)

    hist_plotter(r2_top1['Fit'], axes2[2, 0], 0, 0.15)
    hist_plotter(r2_top1['Nac'], axes2[2, 1], 5, 20)
    hist_plotter(r2_top1['Abort'], axes2[2, 2], 1, 5)

    axes2[2, 0].set_xlabel('Root mean squared distance')
    axes2[2, 1].set_xlabel('NAC (1/s)')
    axes2[2, 2].set_xlabel('Unscrunching and\nabortive release (1/s)',
                           multialignment='center')

    axes2[0, 0].set_ylabel('All samples')
    axes2[1, 0].set_ylabel('Top 5%', multialignment='center')
    axes2[2, 0].set_ylabel('Top 1%', multialignment='center')

    # Heatmaps
    #heatmap(r2, ax=axes2[0, 3])
    #heatmap(r2, ax=axes2[1, 3], max_value=r2.iloc[top01]['Fit'])
    #heatmap(r2, ax=axes2[2, 3], max_value=r2.iloc[top1]['Fit'])

    axes2[0, 3].set_xlabel('NAC (1/s)', fontsize=10)
    axes2[0, 3].set_ylabel('UAR (1/s)', fontsize=10)
    set_ticks(axes2[0, 3], 3)

    axes2[1, 3].set_xlabel('NAC (1/s)', fontsize=10)
    axes2[1, 3].set_ylabel('UAR (1/s)', fontsize=10)
    set_ticks(axes2[1, 3], 3)

    axes2[2, 3].set_xlabel('NAC (1/s)', fontsize=10)
    axes2[2, 3].set_ylabel('UAR (1/s)', fontsize=10)
    set_ticks(axes2[2, 3], 3)

    add_letters((axes2[0, 0],), ('A',), ['UL'], shiftX=-0.1, shiftY=0.2, fontsize=14)

    add_letters((axes2[0, 3], axes2[1, 3], axes2[2, 3]), ('B', 'C', 'D'),
                ['UL', 'UL', 'UL'], shiftX=0, shiftY=0.2, fontsize=14)

    f2.tight_layout()

    file_name = 'parameter_estimation_distributions_2.pdf'
    file_path = os.path.join(fig_dir, file_name)
    f2.savefig(file_path, format='pdf')
    plt.close(f2)


def one_figure_estimation_plot(xmin_nac, xmax_nac, xmin_abort,
                               xmax_abort, xmin_escape, xmax_escape, r1,
                               r1_top01, r1_top1, r2, r2_top01, r2_top1, top1,
                               top01):

    # About one A4 page in size for this plot
    f, axes = plt.subplots(6, 4, figsize=(8.27, 11.19))

    # First 3 rows (Iteration 1)
    hist_plotter(r1['Fit'], axes[0, 0], 0, 2)
    hist_plotter(r1['Nac'], axes[0, 1], xmin_nac, xmax_nac)
    hist_plotter(r1['Abort'], axes[0, 2], xmin_abort, xmax_abort)
    hist_plotter(r1['Escape'], axes[0, 3], xmin_escape, xmax_escape)

    hist_plotter(r1_top1['Fit'], axes[2, 0], 0, 0.25)
    hist_plotter(r1_top1['Nac'], axes[2, 1], xmin_nac, xmax_nac)
    hist_plotter(r1_top1['Abort'], axes[2, 2], xmin_abort, xmax_abort)
    hist_plotter(r1['Escape'], axes[2, 3], xmin_escape, xmax_escape)

    hist_plotter(r1_top01['Fit'], axes[1, 0], 0, 0.25)
    hist_plotter(r1_top01['Nac'], axes[1, 1], xmin_nac, xmax_nac)
    hist_plotter(r1_top01['Abort'], axes[1, 2], xmin_abort, xmax_abort)
    hist_plotter(r1['Escape'], axes[1, 3], xmin_escape, xmax_escape)

    # Next 3 rows (Iteration 2)
    hist_plotter(r2['Fit'], axes[3, 0], 0, 1)
    hist_plotter(r2['Nac'], axes[3, 1], 5, 20)

    hist_plotter(r2_top1['Fit'], axes[5, 0], 0, 0.15)
    hist_plotter(r2_top1['Nac'], axes[5, 1], 5, 20)
    hist_plotter(r2_top1['Abort'], axes[5, 2], 1, 5)
    hist_plotter(r2['Abort'], axes[3, 2], 1, 10)

    hist_plotter(r2_top01['Fit'], axes[4, 0], 0, 0.15)
    hist_plotter(r2_top01['Nac'], axes[4, 1], 5, 20)
    hist_plotter(r2_top01['Abort'], axes[4, 2], 1, 10)

    # Heatmaps
    heatmap(r2, ax=axes[3, 3])
    heatmap(r2, ax=axes[4, 3], max_value=r2.iloc[top01]['Fit'])
    heatmap(r2, ax=axes[5, 3], max_value=r2.iloc[top1]['Fit'])

    # Set titles for the plots
    axes[0, 0].set_title('Fit')
    axes[0, 1].set_title('NAC')
    axes[0, 2].set_title('Unscrunching and\nabortive release',
                         multialignment='center')
    axes[0, 3].set_title('Escape')

    axes[0, 0].set_ylabel('First iteration\nall samples')
    axes[1, 0].set_ylabel('First iteration\ntop 5%', multialignment='center')
    axes[2, 0].set_ylabel('First iteration\ntop 1%', multialignment='center')

    axes[3, 0].set_ylabel('Second iteration\nall samples')
    axes[4, 0].set_ylabel('Second iteration\ntop 5%', multialignment='center')
    axes[5, 0].set_ylabel('Second iteration\ntop 1%', multialignment='center')

    #axes[3, 3].set_title('Parameter\nco-variation', multialignment='center')
    axes[3, 3].set_xlabel('NAC 1/s', fontsize=10)
    axes[3, 3].set_ylabel('UAR 1/s', fontsize=10)
    set_ticks(axes[3, 3], 3)

    axes[4, 3].set_xlabel('NAC 1/s', fontsize=10)
    axes[4, 3].set_ylabel('UAR 1/s', fontsize=10)
    set_ticks(axes[4, 3], 3)

    axes[5, 3].set_xlabel('NAC 1/s', fontsize=10)
    axes[5, 3].set_ylabel('UAR 1/s', fontsize=10)
    set_ticks(axes[5, 3], 3)

    add_letters((axes[0, 0], axes[3, 0]), ('A', 'B'), ['UL', 'UL'],
                shiftX=-0.3, shiftY=0.2, fontsize=14)

    add_letters((axes[3, 3], axes[4, 3], axes[5, 3]), ('C', 'D', 'E'), ['UL', 'UL', 'UL'],
                shiftX=0, shiftY=0.2, fontsize=14)

    f.tight_layout()

    file_name = 'parameter_estimation_distributions.pdf'
    file_path = os.path.join(fig_dir, file_name)
    f.savefig(file_path, format='pdf')
    plt.close(f)


def set_ticks(ax, nr_xticks, nr_yticks=False):

    if nr_yticks is False:
        nr_yticks = nr_xticks

    xbeg, xend = ax.get_xlim()
    vals = np.linspace(xbeg, xend, nr_xticks)
    ax.set_xticks(vals)

    ybeg, yend = ax.get_ylim()
    vals = np.linspace(ybeg, yend, nr_yticks)
    ax.set_yticks(vals)


def ap_to_nac_examples():
    """
    Using NAC=10 to take some example AP values for N25 to calculate
    backstepping rates. To be used for figure demonstrating the parameter
    estimation process.

            r = nac * ap / (1-ap)

    """

    AP = np.asarray([0.6, 0.4, 0.2, 0.4, 0.3, 0.6, 0.3, 0.2, 0.1, 0.4, 0.1])
    nac = 10.0
    backstep = nac * AP / (1 - AP)
    x_values = range(2, len(AP) + 2)

    fig1, ax1 = plt.subplots()

    ax1.bar(x_values, AP, align='center')
    ax1.set_ylabel('AP (%)', size=22)
    ax1.set_xlabel('ITS position', size=22)

    fig2, ax2 = plt.subplots()

    ax2.bar(x_values, backstep, align='center')

    ax2.set_ylabel('Rate constant of backstepping (1/s)', size=22)
    ax2.set_xlabel('ITS position', size=22)

    for ax in [ax1, ax2]:
        ax.set_xlim(1, 13)
        # This is where matplotlib does not shine
        ax.set_xticks(x_values)
        set_ticklabel_size(ax, 20)

    ax1.set_ylim(0, 0.7)

    fig1.tight_layout()
    fig2.tight_layout()

    fig1.savefig(os.path.join(fig_dir, 'APs.pdf'))
    fig2.savefig(os.path.join(fig_dir, 'Backsteps.pdf'))


def set_ticklabel_size(ax, size):
    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(size)

    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(size)


def finegrain_MC():
    """
    Boundaries have been found from coarse analysis. Now do more finegrained
    search.
    """

    GreB = True
    fit_extrap = False

    nr_samples = 100000

    import time
    beg = time.time()
    boundary_analysis = 'finer_grain'
    boundaries = get_boundaries(analysis=boundary_analysis)
    result = simulator(GreB, fit_extrap, nr_samples, boundaries)
    secs_elapsed = time.time() - beg
    print_time(secs_elapsed)

    db_string = 'finegrain_search-nr_samples_{0}_restricted_nac_analysis_{1}'.format(nr_samples,
                                                                                     boundary_analysis)

    shelve_database = shelve.open(simulation_storage)
    shelve_database[db_string] = result
    shelve_database.close()


def finegrain_MC_plot():
    """
    The good news: you have two parameter value peaks right where you have
    previously seen them. The bad, or strange, news is that the values at the
    two peaks do not correspond to each other: choosing NAC values centered around
    10, you get UAR values around 2. Likewise the other way around.

    So the peaks do not correspond to each other. What would the picture look
    like if we had disallowed NAC above 15? You get a peak around 1.5 for UAR,
    which corresponds to a peak around 12/13 for NAC. Paraphrase the other way
    around: if you have to choose those parameter values that were used
    together in the simulation, which peak should you choose?

    Also, this was looking at the top 1%. But zooming down in size you don't
    see much difference. It's all too stochastic at this point.
    """

    #db_string ='finegrained_mc_samples_10000_rounds_10'  # 1 to 25 for NAC
    db_string = 'finegrain_search-nr_samples_100000_restricted_nac_analysis_finer_grain'

    shelve_database = shelve.open(simulation_storage)
    df = shelve_database[db_string]
    shelve_database.close()

    df.sort_values('Fit', inplace=True)

    fig, axes = plt.subplots(2)
    axN, axA = axes

    #df = df[df['NAC'] < 15]
    #debug()

    df = df[:1000]

    #nacmin, nacmax = 6, 16
    #abmin, abmax = 1, 3

    #df['NAC'].plot(kind='hist', ax=axN, subplots=True, bins=100, normed=True, color='green')
    #df['Abort'].plot(kind='hist', ax=axA, subplots=True, bins=100, normed=True, color='blue')
    df['NAC'].plot(kind='hist', ax=axN, subplots=True, bins=27, normed=True, color='green')
    what = df['Abort'].plot(kind='hist', ax=axA, subplots=True, bins=23, normed=True, color='blue')
    #debug()

    #af = df[(3 < df['NAC']) & (df['NAC'] < 15)]
    #of = af[(1.2 < af['Abort']) & (af['Abort'] < 1.7)]
    #af = df[:100]
    #nr_bins_af = int(100 * af.shape[0] / df.shape[0])
    #af['NAC'].plot(kind='hist', ax=axN, subplots=True, bins=10, normed=True,
                   #color='yellow')
    #af['Abort'].plot(kind='hist', ax=axA, subplots=True, bins=10,
                     #normed=True, color='yellow')
    #nr_bins_of = int(100 * of.shape[0] / df.shape[0])
    #of['NAC'].plot(kind='hist', ax=axN, subplots=True, bins=nr_bins_of, normed=True,
                   #color='red')
    #of['Abort'].plot(kind='hist', ax=axA, subplots=True, bins=nr_bins_of,
                     #normed=True, color='red')

    # you need to procuce a some kind of mean value or something.

    for param in ['NAC', 'Abort']:
        #nr_bins = int(2 * (df[param].max() - df[param].min()))
        nr_bins = 100  # same as for histogram
        binning = pd.cut(df[param], nr_bins)
        bin_counts = pd.value_counts(binning)
        mode = mean_of_bin_index(bin_counts.index[0])
        print('Mode for {0}: {1}'.format(param, mode))

    spacingN = (16 - 6) * 0.1
    axN.set_xlim(6 - spacingN, 16 + spacingN)
    spacingA = (3 - 1) * 0.1
    axA.set_xlim(1 - spacingA, 3 + spacingA)

    for ax in axes:
        ax.set_ylabel('')
        ax.set_yticklabels([])

    axN.set_xlabel('Rate constant for NAC (1/s)')
    axA.set_xlabel('Rate constant for UAR (1/s)')

    add_letters((axN, axA), ('A', 'B'), ['UL', 'UL'], shiftY=-0.01, fontsize=14)

    fig.tight_layout()

    file_name = 'high_res_search_for_rate_constants.pdf'
    file_path = os.path.join(fig_dir, file_name)
    fig.savefig(file_path, format='pdf')

    plt.close(fig)


def mean_of_bin_index(bin_index):
    """
    Binning by cut and value_counts gives an index like this:

    ipdb> ga.index[0]
    '(8.895, 9.0843]'

    Return the mean of that.
    """

    one, two = bin_index.split(',')

    return np.mean((float(one[1:]), float(two[1:-1])))


def simulator(GreB, fit_extrap, nr_samples, boundaries):
    """
    The main simulator function.
    """
    # Setup
    variant_name = 'N25'
    ITSs = data_handler.ReadData('dg100-new')
    its_variant = [i for i in ITSs if i.name == variant_name][0]
    initial_RNAP = 100
    sim_end = 9000
    escape_RNA_length = 14

    stoi_setup = ITStoichiometricSetup(escape_RNA_length=escape_RNA_length)

    params = get_parameters(samples=nr_samples, boundaries=boundaries)

    # evaluate parameters and save to log file
    result = search_parameter_space(True, its_variant, GreB, stoi_setup,
                                    initial_RNAP, sim_end, params, False)

    return result


def get_boundaries(analysis):

    b = {}

    if analysis == 'finegrain':
        b['nac_low'] = 6
        b['nac_high'] = 25

        b['abort_low'] = 0.5
        b['abort_high'] = 3

        b['escape_low'] = 20
        b['escape_high'] = 20

    if analysis == 'finer_grain':
        b['nac_low'] = 6
        b['nac_high'] = 16

        b['abort_low'] = 0.5
        b['abort_high'] = 3

        b['escape_low'] = 20
        b['escape_high'] = 20

    return b


def get_top_sims(df, fit_extrap, nr_top_sims):
    """
    Get the top "nr_top_sims" from a series of simulations. Specify column
    which indicates what is 'top' and how many should be selected as top.
    """

    # Set which value which has been optimized for
    if fit_extrap:
        fit = 'Fit_extrap'
    else:
        fit = 'Fit'

    df.sort(fit, inplace=True)
    df = df[:nr_top_sims]

    # But drop any rows with NaN values in the Fit column
    df = df.dropna(subset=[fit])

    return df


def scrunch_time_comparison():
    """
    Using the obtained rate constants for plotting.
    """

    params = (10.0, 1.2, 20)
    variant_name = 'N25'
    ITSs = data_handler.ReadData('dg100-new')
    its_variant = [i for i in ITSs if i.name == variant_name][0]
    initial_RNAP = 100
    sim_end = 9000

    experiment, extrapolated = get_experiment_scrunch_distribution()
    stoi_setup = ITStoichiometricSetup(escape_RNA_length=14)

    R = ITRates(its_variant, nac=params[0], unscrunch=params[1],
                escape=params[2], GreB_AP=True)

    sim_name = get_sim_name(stoi_setup, its_variant)
    sim_setup = ITSimulationSetup(sim_name, R, stoi_setup, initial_RNAP, sim_end)
    model = ITModel(sim_setup)

    model_ts = model.run(include_elongation=True)
    scrunch_dist = calc_scrunch_distribution(model_ts, initial_RNAP)
    fit, fit_extrap = compare_scrunch_data(scrunch_dist, experiment, extrapolated)

    plot_scrunch_distributions(scrunch_dist, experiment, extrapolated, title='')


def fitness_bound_calc():
    """
    You obtain a limit for max fitness by running many many simulations. When
    rerunning say the top 5 or top 10, what fitnesses are you getting the
    second time around? Does that fitnessing vary for the different parameter
    ranges?

    Interesting: the top results rarely get a 'repeat'. Would be cool if you
    could quantify the likelihood of getting a repeat for the different
    parameter values.

    Result: tops are between 0.04 and 0.05, but when rerunning the median is
    more than 0.16, so quite a bit worse.
    """

    shelve_database = shelve.open(simulation_storage)
    df = shelve_database['finegrained_mc_samples_10000_rounds_10']
    shelve_database.close()
    df.sort_values('Fit', inplace=True)

    variant_name = 'N25'
    ITSs = data_handler.ReadData('dg100-new')
    its_variant = [i for i in ITSs if i.name == variant_name][0]
    initial_RNAP = 100
    sim_end = 9000
    GreB = True
    nr_samples = range(100)
    stoi_setup = ITStoichiometricSetup(escape_RNA_length=14)

    rerun = False
    if rerun:

        results = []
        for index, result in df[:10].iterrows():
            params = [(result['NAC'], result['Abort'], result['Escape']) for _ in nr_samples]

            result = search_parameter_space(True, its_variant, GreB, stoi_setup,
                                            initial_RNAP, sim_end, params, False,
                                            False)

            results.append(result)

        all_results = pd.concat(results, ignore_index=True)

        shelve_database = shelve.open(simulation_storage)
        shelve_database['rerun_top_results'] = all_results
        shelve_database.close()

    else:
        shelve_database = shelve.open(simulation_storage)
        all_results = shelve_database['rerun_top_results']
        shelve_database.close()

    all_results['Fit'].plot(kind='hist', normed=True, bins=20)
    plt.savefig('best_results_rerun.pdf')
    plt.close()


def coarse_search():
    """
    Coarse search with/without fit_extrap, with/without GreB
    """
    import time

    for GreB in [True, False]:
        for fit_extrap in [True, False]:

            # this combination is not interesting
            if not GreB and fit_extrap:
                continue

            # Already done
            if GreB and fit_extrap:
                continue

            samples = 100000
            boundaries = False  # defaults to the 1-25 search space

            info = 'Running GreB: {0} and Fit_extrap: {1}'.format(GreB, fit_extrap)
            print('\n' + info + '\n')
            beg = time.time()
            result = simulator(GreB, fit_extrap, samples, boundaries)
            secs_elapsed = time.time() - beg
            print_time(secs_elapsed)

            db_string = 'coarse_search_nr_samples_{0}-fitextrap_{1}-GreB_{2}'.format(samples,
                                                                                     fit_extrap,
                                                                                     GreB)
            shelve_database = shelve.open(simulation_storage)
            shelve_database[db_string] = result
            shelve_database.close()


def coarse_search_plot_correct():
    """
    Single 1x3 plot of the three rate constants.
    """

    db_string = 'coarse_search_nr_samples_100000-fitextrap_False-GreB_True'

    shelve_database = shelve.open(simulation_storage)

    df_all = shelve_database[db_string]
    shelve_database.close()

    nr_top_sims = 1000
    df = get_top_sims(df_all, False, nr_top_sims)

    fig, axes = plt.subplots(1, 3)

    c = 'gray'

    df['NAC'].plot(kind='hist', ax=axes[0], subplots=True, bins=15,
                   normed=False, color=c)
                   #histtype='stepfilled')

    df['Abort'].plot(kind='hist', ax=axes[1], subplots=True, bins=30,
                     normed=False, color=c)
                     #histtype='stepfilled')

    df['Escape'].plot(kind='hist', ax=axes[2], subplots=True, bins=15,
                      normed=False, color=c)
                      #histtype='stepfilled')

    for ax in axes:
        ax.set_ylabel('')
        ax.set_yticklabels([])

        ax.set_xlim(0, 25)

    axes[0].set_xlim(4, 25)
    axes[1].set_xlim(0, 6)

    axes[0].set_xlabel('Rate constant for NAC (1/s)')
    axes[1].set_xlabel('Rate constant for UAR (1/s)')
    axes[2].set_xlabel('Rate constant for Escape (1/s)')

    # Estimate the mode
    for param in ['NAC', 'Abort']:
        #nr_bins = int(2 * (df[param].max() - df[param].min()))
        nr_bins = 100  # same as for histogram
        binning = pd.cut(df[param], nr_bins)
        bin_counts = pd.value_counts(binning)
        mode = mean_of_bin_index(bin_counts.index[0])
        print('Mode for {0}: {1}'.format(param, mode))

    fig.tight_layout()

    file_name = 'coarse_search-fits_GreB_no_extrap.pdf'
    file_path = os.path.join(fig_dir, file_name)
    fig.savefig(file_path, format='pdf')

    plt.close(fig)


def coarse_search_plot_correct_and_finegrain():
    """
    Single 1x3 plot of the three rate constants.
    """

    db_string = 'coarse_search_nr_samples_100000-fitextrap_False-GreB_True'
    db_string_finegrain = 'finegrain_search-nr_samples_100000_restricted_nac_analysis_finer_grain'

    shelve_database = shelve.open(simulation_storage)

    df_all = shelve_database[db_string]
    df_all_fg = shelve_database[db_string_finegrain]
    shelve_database.close()

    nr_top_sims = 1000
    df = get_top_sims(df_all, False, nr_top_sims)
    df_fg = get_top_sims(df_all_fg, False, nr_top_sims)

    fig, axes = plt.subplots(2, 3)

    c = 'gray'

    # First row
    df['NAC'].plot(kind='hist', ax=axes[0, 0], subplots=True, bins=15,
                   normed=False, color=c)

    df['Abort'].plot(kind='hist', ax=axes[0, 1], subplots=True, bins=30,
                     normed=False, color=c)

    df['Escape'].plot(kind='hist', ax=axes[0, 2], subplots=True, bins=15,
                      normed=False, color=c)

    # Second row
    df_fg['NAC'].plot(kind='hist', ax=axes[1, 0], subplots=True, bins=15,
                      normed=False, color=c)

    df_fg['Abort'].plot(kind='hist', ax=axes[1, 1], subplots=True, bins=12,
                        normed=False, color=c)

    axes[-1, -1].axis('off')

    for ax in axes.flat:
        ax.set_ylabel('')
        ax.set_yticklabels([])

    axes[0, 0].set_xlim(0, 26)
    axes[0, 1].set_xlim(0, 6)
    axes[0, 2].set_xlim(0, 26)
    axes[1, 0].set_xlim(4, 17)
    axes[1, 1].set_xlim(0.5, 3)

    axes[0, 0].set_xlabel('Rate constant for NAC (1/s)')
    axes[1, 0].set_xlabel('Rate constant for NAC (1/s)')
    axes[0, 1].set_xlabel('Rate constant for UAR (1/s)')
    axes[1, 1].set_xlabel('Rate constant for UAR (1/s)')
    axes[0, 2].set_xlabel('Rate constant for Escape (1/s)')

    # Estimate the mode
    for d in [df, df_fg]:
        for param in ['NAC', 'Abort']:
            #nr_bins = int(2 * (df[param].max() - df[param].min()))
            nr_bins = 15  # same as for histogram
            binning = pd.cut(d[param], nr_bins)
            bin_counts = pd.value_counts(binning)
            mode = mean_of_bin_index(bin_counts.index[0])
            print('Mode for {0}: {1}'.format(param, mode))

    topleft = axes[0, 0]
    botleft = axes[1, 0]
    add_letters([topleft, botleft], ('A', 'B'), ['UL', 'UL'], fontsize=14,
                shiftX=-0.1, shiftY=0.03)

    fig.tight_layout()

    file_name = 'coarse_and_finegrain_search_GreB_no_extrap.pdf'
    file_path = os.path.join(fig_dir, file_name)
    fig.savefig(file_path, format='pdf')

    plt.close(fig)


def coarse_search_plot_tests():
    """
    Coarse search plots. Showing the effect of not using GreB, and of using
    the extrapolated fit.
    """

    # Plot these two after one another.
    db_string0 = 'coarse_search_nr_samples_100000-fitextrap_False-GreB_True'
    db_string1 = 'coarse_search_nr_samples_100000-fitextrap_True-GreB_True'
    db_string2 = 'coarse_search_nr_samples_100000-fitextrap_False-GreB_False'

    shelve_database = shelve.open(simulation_storage)

    df0 = shelve_database[db_string0]
    df1 = shelve_database[db_string1]
    df2 = shelve_database[db_string2]
    shelve_database.close()

    nr_top_sims = 1000
    df0 = get_top_sims(df0, False, nr_top_sims)
    df1 = get_top_sims(df1, True, nr_top_sims)
    df2 = get_top_sims(df2, False, nr_top_sims)

    fig, axes = plt.subplots(1, 3)

    df0['NAC'].plot(kind='hist', ax=axes[0], subplots=True, bins=20,
                    normed=False, color='blue', alpha=0.5, histtype='step')
    df0['Abort'].plot(kind='hist', ax=axes[1], subplots=True, bins=20,
                      normed=False, color='blue', alpha=0.5, histtype='step')
    df0['Escape'].plot(kind='hist', ax=axes[2], subplots=True, bins=20,
                       normed=False, color='blue', alpha=0.5, histtype='step')

    df1['NAC'].plot(kind='hist', ax=axes[0], subplots=True, bins=20,
                    normed=False, color='green', alpha=0.5, histtype='stepfilled')
    df1['Abort'].plot(kind='hist', ax=axes[1], subplots=True, bins=5,
                      normed=False, color='green', alpha=0.5, histtype='stepfilled')
    df1['Escape'].plot(kind='hist', ax=axes[2], subplots=True, bins=20,
                       normed=False, color='green', alpha=0.5, histtype='stepfilled')

    df2['NAC'].plot(kind='hist', ax=axes[0], subplots=True, bins=15,
                    normed=False, color='purple', alpha=0.5, histtype='stepfilled')
    df2['Abort'].plot(kind='hist', ax=axes[1], subplots=True, bins=10,
                      normed=False, color='purple', alpha=0.5, histtype='stepfilled')
    df2['Escape'].plot(kind='hist', ax=axes[2], subplots=True, bins=20,
                       normed=False, color='purple', alpha=0.5, histtype='stepfilled')

    for ax in axes:
        ax.set_ylabel('')
        ax.set_yticklabels([])
        ax.set_xlim(0, 25)

        tickpos = [int(_) for _ in np.linspace(0, 25, 3)]
        ax.set_xticks(tickpos)

    axes[0].set_xlim(5, 25)
    axes[0].set_xticks(np.linspace(5, 25, 3))

    axes[0].set_xlabel('Rate constant NAC (1/s)')
    axes[1].set_xlabel('Rate constant UAR (1/s)')
    axes[2].set_xlabel('Rate constant Escape (1/s)')

    for df, label in zip([df1, df2], ['Fitextrap', 'GreB False']):
        for param in ['NAC', 'Abort']:
            nr_bins = 20  # same as for histogram
            binning = pd.cut(df[param], nr_bins)
            bin_counts = pd.value_counts(binning)
            mode = mean_of_bin_index(bin_counts.index[0])
            print('Mode for {0} {2}: {1}'.format(param, mode, label))

    # Figure out how to make this small enough (one biochem-width -- or spend
    # a two-rows on this? Important figure.
    set_fig_size(fig, biochemistry_width * 2, 5.5)

    fig.tight_layout()

    file_name = 'coarse_search-fits.pdf'
    file_path = os.path.join(fig_dir, file_name)
    fig.savefig(file_path, format='pdf')

    plt.close(fig)


def kinetic_scheme(nac=9, uar=1.2, escape=20):
    """
    Use template SVG file to insert rate constants for the two cases, GreB+
    and GreB-.

    Then run inkscape via command line to generate pdf from svg.
    """
    variant_name = 'N25'
    its_variant = get_variant(variant_name)

    Rp = ITRates(its_variant, nac=nac, unscrunch=uar, escape=escape, GreB_AP=True)
    Rm = ITRates(its_variant, nac=nac, unscrunch=uar, escape=escape, GreB_AP=False)

    rates = {}
    rates['GreB_plus'] = [format(Rp.Backtrack(n), '.1f') for n in range(2, 12)]
    rates['GreB_minus'] = [format(Rm.Backtrack(n), '.1f') for n in range(2, 12)]

    new_svg = kinetic_svg_edit(rates)

    ildir, ilname, ilext = getdirnamext(new_svg)
    new_pdf = os.path.join(ildir, ilname + '.pdf')

    from subprocess import call
    cmd = 'inkscape -A {0} {1}'.format(new_pdf, new_svg)
    call(cmd, shell=True)


def getdirnamext(path):
    dirname = os.path.dirname(path)
    basename = os.path.basename(path)
    filename, extension = os.path.splitext(basename)

    return dirname, filename, extension


def kinetic_svg_edit(rates):
    """
    Open template svg figure and insert appropriate rate constants.
    """

    ildir = '/home/jorgsk/Dropbox/The-Tome/my_papers/kinetic_initiation/illustrations'

    template = os.path.join(ildir, 'estimated_parameters_template.svg')
    outfile = os.path.join(ildir, 'estimated_parameters_filled.svg')

    outhandle = open(outfile, 'w')

    for line in open(template, 'r'):

        backtrack_gBp_found = False
        backtrack_gBm_found = False
        nac_found = False
        uar_found = False

        # Update backtrack-rc, both GreB+ and GreB-
        for val in range(2, 12):
            placeholder_gBp = 'B{0} s'.format(val)
            if placeholder_gBp in line:
                if backtrack_gBp_found is True:
                    1 / 0  # should only happen once per line
                rate = rates['GreB_plus'][val - 2]
                outline = line.replace(placeholder_gBp, '{0} s'.format(rate))
                backtrack_gBp_found = True

            placeholder_gBm = 'C{0} s'.format(val)
            if placeholder_gBm in line:
                if backtrack_gBm_found is True:
                    1 / 0  # should only happen once per line
                rate = rates['GreB_minus'][val - 2]
                outline = line.replace(placeholder_gBm, '{0} s'.format(rate))
                backtrack_gBm_found = True

        # Update NAC-rc
        if 'NAC s' in line:
            outline = line.replace('NAC s', '9 s')
            nac_found = True

        if 'UAR s' in line:
            outline = line.replace('UAR s', '1.2 s')
            uar_found = True

        # Don't change line
        if not backtrack_gBp_found and not backtrack_gBm_found and not nac_found and not uar_found:
            outline = line

        outhandle.write(outline)
    outhandle.close()

    return outfile


def main():

    # Distributions of % of IQ units and #RNA
    #initial_transcription_stats()

    # Distributions
    #AP_and_pct_values_distributions_DG100()

    # Coarse search with/withot GreB/extrapolated
    #coarse_search()
    #coarse_search_plot_correct()  # GreB, not extrapolated
    coarse_search_plot_tests()    # Not GreB, and extrapolated
    #coarse_search_plot_correct_and_finegrain()  # both initial and finegrain search

    # Finegrain search
    #finegrain_MC()
    #finegrain_MC_plot()
    # XXX TODO: Make a (2,3) plot with [coarse, fine]. Discard the old 'red'
    # data, and use the new ones you got from coarse search.

    # Obtain the best parameters for N25 +Greb with extra percentages to AP.
    #ap_sensitivity_calc()
    #ap_sensitivity_plot()

    # Plot of model vs measured vs extrapolated scrunch times
    #scrunch_time_comparison()

    # Rerunning top 10 results to see variation in fitness
    #fitness_bound_calc()

    #tau_variation_plot()
    #fraction_scrunches_less_than_1_second_calc()

    # Fitting to ensemble studies
    #simple_vo_2003_comparison()

    # Local sensitivity analysis
    #sensitivity_analysis()
    #plot_sensitivity_analysis()

    # Parameter estimation plotting
    #parameter_estimation_plot()

    #ap_to_nac_examples()

    # kinetic scheme figure +/- GreB
    #kinetic_scheme(nac=9, uar=1.2, escape=20)

    pass


if __name__ == '__main__':
    main()
