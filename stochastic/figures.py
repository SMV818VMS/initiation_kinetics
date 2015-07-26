import numpy as np
from scipy.stats import spearmanr, pearsonr
import os
import sys
import pandas as pd
import shelve
sys.path.append('/home/jorgsk/Dropbox/phdproject/transcription_initiation/data/')
sys.path.append('/home/jorgsk/Dropbox/phdproject/transcription_initiation/kinetic/model')
import data_handler
import stochastic_model_run as smr
import matplotlib.pyplot as plt

plt.ioff()

import seaborn as sns
from operator import attrgetter
from ipdb import set_trace as debug  # NOQA
from scipy.interpolate import InterpolatedUnivariateSpline as interpolate
from scipy.optimize import curve_fit

# Global place for where you store your figures
here = os.path.abspath(os.path.dirname(__file__))  # where this script is located
fig_dir1 = os.path.join(here, 'figures')

# Global database for storing simulation results -- don't re-calculate when
# you are using the excact same input. Just make sure there is no implicit
# input changes that are not being serialized for the md5 sum
simulation_storage = os.path.join(here, 'storage', 'shelve_storage')


def CalculateTranscripts(model):
    """
    Get #RNA: aborted and FL. Calculate intensities and PY.
    """

    sp = np.array(model.data_stochsim.species, dtype=np.int32)
    lb = model.data_stochsim.species_labels

    # open complex index
    oc_indx = lb.index('RNAPpoc')

    # full length index (if it exists..?)
    fl_indx = lb.index('RNAPflt')

    #test=np.array(
            #[[10, 0, 0, 0],
              #[9, 1, 0, 0],
              #[9, 0, 1, 0],
              #[8, 1, 1, 0],
              #[7, 2, 1, 0],
              #[7, 2, 0, 1],
              #[8, 1, 0, 1],
              #[7, 2, 0, 1],
              #[7, 1, 1, 1],
              #[7, 0, 2, 1],
              #[8, 0, 1, 1],
              #[8, 0, 0, 2],
              #[8, 0, 0, 2],
              #[7, 1, 1, 2]])

    #diff = test[1:,0] - test[:-1,0]
    # diff : array([-1,  0, -1, -1,  1, -1,  0,  1,  0])
    #ch = diff == +1
    # The ch-indices are the indices before the change happens: so we need to
    # take the timestep in ch and subtract it from the timestep after, and then sum.
    #abort = sum(test[ch, :] - test[np.roll(ch,1), :])
    # XXX tried using shift from scipy: but there must be a bug! Some times
    # the array was simply not shifted. Maybe it's not meant to be used on
    # true/false arrays.

    # If we are using the backstep method, then the amount of abortive
    # hey hey, remember, the state is the # of species at each time step.

    oc_change = sp[1:,oc_indx] - sp[:-1,oc_indx]
    # Find those time indices where RNAP_000 grew with +1 (actually, it ends
    # up being the indices of the time step before the change)
    indx_before_abort = oc_change == +1
    indx_after_abort = np.roll(indx_before_abort, 1)
    abrt_rna = sum(sp[indx_before_abort,:] - sp[indx_after_abort, :])

    # XXX Something is funky here. The abrt_rna shows a value for RNAPflt.
    # RNAPflt should not change when OC changes, so I would have expected the
    # value for RNAPflt to be zero. XXX The problem disappeared ...
    # XXX and returned ... it seems to be ... wait for it... stochastic
    # XXX I can't reproduce it by just running several times. It might have
    # something to do with rerunning afer debugging. it's weird.
    #
    # XXX Clue: it seems to happen whenever I change the amount of initial
    # RNAP... something is not quite reset the way it should be. Fix is to
    # leave ipython and go back.

    fl_rna = sp[:, fl_indx][-1]

    # Tests. Check that the - number for the open complex corresponds to all
    # the other aborted ones
    assert sum(abs(abrt_rna)) == 2 * abs(abrt_rna[oc_indx])
    assert abrt_rna[fl_indx] == 0  # full length abort not possible
    assert fl_rna > 0  # if you get no FL RNA, something else is wrong

    # Summarize in a lookup table
    transcription_result = {'FL': fl_rna}

    for nr_hits, code in zip(abrt_rna, lb):

        # Skip non-abortive stuff
        if nr_hits == 0:
            continue

        # Skip open complex
        if code == 'RNAPpoc':
            continue

        # Bleh
        infos = code[-3:]
        if infos[1] == '_':
            rna_len = infos[0]
        else:
            rna_len = infos[:2]

        transcription_result[rna_len] = nr_hits

    return transcription_result


def initial_transcription_stats():
    """
    Run stochastic simulations on all the DG100 library variants
    """
    ITSs = data_handler.ReadData('dg100-new')
    ITSs = sorted(ITSs, key=attrgetter('PY'))

    use_sim_db = True
    #use_sim_db = False
    #plot_species = [....]

    screen = False
    #screen = True
    subset = ['N25', 'N25-A1anti']
    #subset = ['N25']
    #subset = ['N25-A1anti']

    # Keep all rnas to make a box plot after.
    all_rnas = {}
    all_FL_timeseries = {}

    for its in ITSs:
        ## Filter out some specific ones
        if screen and its.name not in subset:
            continue

        model = smr.get_kinetics(variant=its.name)
        sim_id = model.calc_setup_hash()

        shelve_database = shelve.open(simulation_storage)
        if use_sim_db and sim_id in shelve_database:
            ts = shelve_database[sim_id]
        else:
            ts = model.run()
            shelve_database[sim_id] = ts
        shelve_database.close()

        #if do_plot_timeseries:
            #plot_timeseries(its.name, ts, species)

        nr_rna = ts.iloc[-1].values
        FL_timeseries = ts['FL'].values

        all_rnas[its.name] = nr_rna
        all_FL_timeseries[its.name] = FL_timeseries

        # XXX Make the basic bar plots
        # XXX This is pretty useless now: the results are identical to the RNA
        # fractions (thankfully!).
        #write_bar_plot(nr_rna, its)

    #XXX BOX, BAR, AND PY-distribution PLOTs XXX
    # Divide itss into high and low and analyse
    #dsetmean = np.mean([i.PY for i in ITSs])
    for partition in ['low PY', 'high PY', 'all']:

        #if partition == 'low PY':
            #low_ITSs = [i for i in ITSs if i.PY < dsetmean]
            #write_box_plot(low_ITSs, all_rnas, title=partition)

        #if partition == 'high PY':
            #high_ITSs = [i for i in ITSs if i.PY >= dsetmean]
            #write_box_plot(high_ITSs, all_rnas, title=partition)

        if partition == 'all':
            write_box_plot(ITSs, all_rnas, title=partition)

    #XXX NEW AP XXX
    # Calculate AP anew!
    # This perhaps best shows the effect of the model?
    #AP_recalculated(ITSs, all_rnas)
    # This can be used as a metric. Just take the euclidian norm of the 19
    # dimensional abortive space.

    #XXX t_1/2 distributions for DG100 XXX
    #half_lives(all_FL_timeseries, sim_setup.init_RNAP)


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
    pct_yield = [(ay/total_rna)*100. for ay in abortive_rna]

    # amount of RNA produced from here and out
    rna_from_here = [np.sum(abortive_rna[i:]) + fl_transcript for i in index_range]

    # percent of RNAP reaching this position
    pct_rnap = [(rfh/total_rna)*100. for rfh in rna_from_here]

    # probability to abort at this position
    # More accurately: probability that an abortive RNA of length i is
    # produced (it may backtrack and be slow to collapse, producing
    # little abortive product and reducing productive yield)
    ap = [pct_yield[i]/pct_rnap[i] for i in index_range]

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

        new_AP = calc_AP_anew(transcripts)
        old_AP = its_dict[its_name].abortive_probability

        data['AP experiment'] = np.asarray(old_AP) * 100.
        data['AP model'] = np.asarray(new_AP) * 100.

        corrs, pvals = spearmanr(data['AP experiment'], data['AP model'])
        corrp, pvalp = pearsonr(data['AP experiment'], data['AP model'])

        title = 'AP: model vs experiment for {2}\n Spearman correlation: {0}\n Pearson correlation: {1}'.format(corrs, corrp, its_name)

        index = range(2, len(new_AP)+2)

        fig, ax = plt.subplots()
        mydf = pd.DataFrame(data=data, index=index)
        mydf.plot(kind='bar', ax=ax, rot=0)

        ax.set_xlabel('ITS position')
        ax.set_ylabel('AP (%)')
        ax.set_ylim(0, 80)

        file_name = its_name + '_AP_experiment_vs_model.pdf'
        file_path = os.path.join(fig_dir1, file_name)
        fig.suptitle(title)
        fig.savefig(file_path, format='pdf')
        plt.close(fig)


def plot_timeseries(name, ts, species):
    """
    Just plot timeseries data.
    """

    pass

    #sp_2_label = {'RNAPpoc': 'Productive open complex',
                  #'RNAPuoc': 'Unproductive open complex',
                  #'RNAPflt': 'Full length transcript',
                  #'RNAP2_b': '2nt abortive RNA',
                  #'RNAP3_b': '3nt abortive RNA',
                  #'RNAP4_b': '4nt abortive RNA',
                  #'RNAP6_b': '6nt abortive RNA',
                  #'RNAP7_b': '7nt abortive RNA',
                  #'RNAP8_b': '8nt abortive RNA'
                  #}

    # f, ax = plt.subplots()
    #ts[species].plot()

    #ax.set_ylabel('# of species')
    #ax.set_xlabel('Time (seconds)')

    #file_name = 'Timeseries_{0}.pdf'.format(name)

    #file_path = os.path.join(fig_dir1, file_name)

    #f.suptitle('Stochastic simulation of initial transcription for {0}'.format(name))
    #f.tight_layout()
    #f.savefig(file_path, format='pdf')

    ## Explicitly close figure to save memory
    #plt.close(f)


def GetFLTimeseries(sim):

    fl_index = sim.data_stochsim.species_labels.index('RNAPflt')

    data = sim.data_stochsim.species[:, fl_index]
    time = sim.data_stochsim.time

    df = pd.DataFrame({'FL': data}, index=time)

    # So you end up with a huge array of almost identical values. Trim so that
    # you only keep the values when there is a change.
    return df.drop_duplicates(cols='FL')


def half_lives(all_FL_timeseries, nr_RNAP):
    """
    For each variant, calculate the half life of FL. That means you got to
    calculate until the end of the simulation. Use just 100 NTPs and hope it's
    accurate enough.

    Terminology: your values are actually increasing. What is the value for
    50% of an increasing quantity? When Q(t) = Q(inf)/2.

    Lilian talks about escape half life. I'm not sure how to interpret it. Ask
    her!
    """
    data = {}
    # Hey, timeseries information is not stored in tses. What to do? Start
    # storing time series info, at least for FL. Then save that to another
    # file? Bleh, you're probably going to be re-running simulations for a
    # while.

    for its_name, timeseries in all_FL_timeseries.items():

        hl_index = timeseries.FL[timeseries.FL==nr_RNAP/2.].index.tolist()
        # For the cases where you don't simulate long enough to reach a half
        # life
        if hl_index == []:
            half_life = np.nan
        else:
            half_life = hl_index[0]

        data[its_name] = half_life

    df = pd.DataFrame(data, index=['t_1/2']).T
    # Sort by increasing t_1/2
    df = df.sort('t_1/2')

    f, ax = plt.subplots()
    df.plot(kind='bar', ax=ax, rot=80, legend=False)

    ax.set_ylabel('Time (seconds)')

    file_name = 'Time_to_reach_fifty_percent_of_FL.pdf'

    file_path = os.path.join(fig_dir1, file_name)

    f.suptitle('Time to reach 50% of FL product')
    f.tight_layout()
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

    #box_plot_model_exp(data, data_range, title)

    # XXX NICE!! You see the trend :) It is also more pronounced for low-PY
    # positions. You can choose one (N25?) to show this trend for a single
    # variant, then show this graph. For publication: green/blue for
    # model/experiment and positive negative axes.
    # And it is more pronounced for hi
    # XXX TODO: try to ignore the data point with MSAT. I think it's heavily
    # biasing your results, especially for low-py.
    #box_plot_model_exp_frac(data_frac, data_range, title)

    # XXX THE optimal coolest question is: can you make a model that includes
    # some of the known suspects (long pauses, longer backtrack time)? That
    # brings the result closer? If you introduce stuff that increases the
    # backtracking time for late complexes (you'll need longer running time
    # ...) but then you SHOULD see a decrease in the % of these values
    # compared to others.
    # WOW this is just so amazingly cool :D:D:D:D:D:D

    # XXX Make a correlation plot between PY and fold difference between [0-6]
    # and [7-20], and between PY and fold difference betweern PY and FL
    fold_diff_PY_correlations(ITSs, data_frac)

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
    ax.scatter(prod_yield*100., fold_diff_metric)

    ax.set_xlabel('PY')
    ax.set_ylabel('Sum(fold difference after +6) - Sum(fold difference before +6)')

    title = 'Low PY variants have greater overestimates of late abortive release in model\n Spearman correlation: {0}\n Pearson correlation: {1}'.format(corrs, corrp)
    f.suptitle(title)

    file_name = 'Fold_difference_PY_correlation.pdf'

    file_path = os.path.join(fig_dir1, file_name)
    f.savefig(file_path, format='pdf')

    # Explicitly close figure to save memory
    plt.close(f)

    ## FL
    fl = np.asarray(fl_fold)
    corrs, pvals = spearmanr(prod_yield, fl)
    f, ax = plt.subplots()
    ax.scatter(prod_yield*100., fl)

    ax.set_xlabel('PY')
    ax.set_ylabel('Fold difference model/experiment of percent FL transcript')

    title = 'Model with AP as backtracking probability greatly overestimates FL for low PY variants\n Spearman correlation: {0}'.format(corrs, corrp)
    f.suptitle(title)

    file_name = 'FL_fold_difference_PY_correlation.pdf'

    file_path = os.path.join(fig_dir1, file_name)
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
    ax.legend(handles=[NA,EU], loc=2)

    ax.set_ylabel("Fold difference of % QI units and % model RNA")

    ax.set_ylim(-6, 6.5)
    ax.set_yticks(range(-6, 7))
    ax.set_yticklabels([abs(v) for v in range(-6, 7)])

    ax.set_xlabel('ITS position')

    f.suptitle('Fold difference between model and experiment at each position for DG100')
    file_name = 'Model_and_experiment_fractions_{0}.pdf'.format(title)

    file_path = os.path.join(fig_dir1, file_name)
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

    ax.set_ylim(0,50)
    if title:
        f.suptitle(title)
        file_name = 'Model_and_experiment_{0}.pdf'.format(title)
    else:
        file_name = 'Model_and_experiment.pdf'

    file_path = os.path.join(fig_dir1, file_name)
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

        if val1 > val2:
            diff = float(val1)/val2
        else:
            diff = -float(val2)/val1

        # Override the above if one of the values is zero
        if val1 == 0. or val2 == 0:
            diff = np.nan

        frac_diff.append(diff)

    return np.array(frac_diff)


def pct_experiment(its):

    return its.fraction * 100


def calc_pct_model(ts, data_range):
    """
    Ts must be an array with abortives from 2 to 20 and the FL
    """

    return 100 * ts / ts.sum()


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
    file_path = os.path.join(fig_dir1, file_name)
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
            aps[aps<=0.0] = np.nan

            ap_mean = np.nanmean(aps, axis=0)
            ap_std = np.nanstd(aps, axis=0)

            # Old method separated between abortive and FL
            if ab_meth == 'abortiveProb_old':
                ap_mean = np.append(ap_mean, 1.)
                ap_std = np.append(ap_std, 0.)

            data_AP_mean[name] = ap_mean * 100.
            data_AP_std[name] = ap_std * 100.

        f, ax = plt.subplots()
        df = pd.DataFrame(data=data_AP_mean, index=range(2,21) + ['FL'])
        df_err = pd.DataFrame(data=data_AP_std, index=range(2,21) + ['FL'])
        df.plot(kind='bar', yerr=df_err, ax=ax)

        ax.set_xlabel('ITS position')
        ax.set_ylabel('AP')
        ax.set_ylim(0, 60)

        file_name = 'AP_comparison_high_low_PY_{0}.pdf'.format(ab_meth)
        file_path = os.path.join(fig_dir1, file_name)
        f.suptitle('Abortive probability in high and low PY variants\n{0}'.format(ab_meth))
        f.savefig(file_path, format='pdf')
        plt.close(f)

    # Signal Plots
    data_pct_mean = {}
    data_pct_std = {}

    for name, dset in [('Low PY', low_PY), ('High PY', high_PY)]:

        # Remember to get nans where there is zero!!
        ar = np.asarray([i.fraction * 100. for i in dset])
        ar[ar<=0.0] = np.nan
        data_mean = np.nanmean(ar, axis=0)
        data_std = np.nanstd(ar, axis=0)

        data_pct_mean[name] = data_mean
        data_pct_std[name] = data_std

    f, ax = plt.subplots()
    df = pd.DataFrame(data=data_pct_mean, index=range(2,21) + ['FL'])
    df_err = pd.DataFrame(data=data_pct_std, index=range(2,21) + ['FL'])
    df.plot(kind='bar', yerr=df_err, ax=ax)

    ax.set_xlabel('ITS position')
    ax.set_ylabel('% of IQ signal')
    ax.set_ylim(0, 50)

    file_name = 'Percentage_comparison_high_low_PY.pdf'
    file_path = os.path.join(fig_dir1, file_name)
    f.suptitle('Percentage of radioactive intensity for high and low PY variants')
    f.savefig(file_path, format='pdf')
    plt.close(f)


def get_2003_FL_kinetics(method=1):

    data ='/home/jorgsk/Dropbox/phdproject/transcription_initiation/data/vo_2003'
    m1 = os.path.join(data, 'Fraction_FL_and_abortive_timeseries_method_1_FL_only.csv')
    m2 = os.path.join(data, 'Fraction_FL_and_abortive_timeseries_method_2_FL_only.csv')

    # Read up to 10 minutes
    df1 = pd.read_csv(m1, index_col=0)[:10]
    df2 = pd.read_csv(m2, index_col=0)[:10]

    # Multiply by 60 to get seconds
    df1.index = df1.index * 60
    df2.index = df2.index * 60

    # Normalize
    df1_norm = df1/df1.max()
    df2_norm = df2/df2.max()

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
        return a + b*(1 - np.exp(-c*x))

    popt, pcov = curve_fit(saturation, x, y)

    # Make some preductions from this simple model
    xx = np.linspace(x.min(), x.max(), 1000)
    yy = saturation(xx, *popt)

    return xx, yy


def simple_vo_2003_comprison():
    """
    Purpose is to plot the performance on N25 GreB- and compare with Vo 2003
    data from both quantitations. Show quantitations as dots but also
    interpolate to make them look nicer.
    """

    # FL kinetics for 1 and 2.
    m1 = get_2003_FL_kinetics(method=1)

    # FL kinetics for model with GreB+/-
    greb_plus = smr.get_kinetics(variant='N25', GreB=True, escape_RNA_length=14)['FL']
    gp = greb_plus / greb_plus.max()

    greb_minus = smr.get_kinetics(variant='N25', GreB=False, escape_RNA_length=14)['FL']
    gm = greb_minus / greb_minus.max()

    # Plot 2003 data
    f1, ax = plt.subplots()
    m1.plot(ax=ax, style='.', color='g')

    # Fit 2003 data and calculate half life
    xnew, ynew = fit_FL(m1.index, m1.values)
    fit = pd.DataFrame(data={'Vo et. al': ynew}, index=xnew)
    fit.plot(ax=ax, color='b')

    # Get the halflife of full length product
    f_vo = interpolate(ynew, xnew, k=1)
    thalf_vo = float(f_vo(0.5))

    # Plot and get halflives of GreB-
    gm.plot(ax=ax, color='c')
    #f_gm = interpolate(gm.values, gm.index, k=2)
    #thalf_grebminus = f_gm(0.5)
    idx_gm = gm.tolist().index(0.5)
    thalf_grebminus = gm.index[idx_gm]

    # Plot and get halflives of GreB+
    gp.plot(ax=ax, color='k')
    #f_gp = interpolate(gp.values, gp.index, k=2)
    #thalf_grebplus = f_gp(0.5)
    idx_gp = gp.tolist().index(0.5)
    thalf_grebplus = gp.index[idx_gp]

    labels = ['Vo et. al GreB- measurements',
              r'Vo et. al GreB- fit $t_{{1/2}}={0:.2f}$'.format(thalf_vo),
              r'Simulation GreB- $t_{{1/2}}={0:.2f}$'.format(thalf_grebminus),
              r'Simulation GreB+ $t_{{1/2}}={0:.2f}$'.format(thalf_grebplus)]

    ax.legend(labels, loc='lower right', fontsize=15)

    ax.set_xlabel('Time (seconds)', size=15)
    ax.set_ylabel('Fraction of full length transcript', size=15)

    ax.set_xlim(0, 70)
    ax.set_ylim(0, 1.01)
    f1.savefig('GreB_plus_and_GreB_minus_kinetics.pdf')


def main():

    # Distributions of % of IQ units and #RNA
    initial_transcription_stats()

    # Remember, you are overestimating PY even if there is no recycling. Would
    # recycling lead to higher or lower PY? I'm guessing lower.
    #AP_and_pct_values_distributions_DG100()

    # Result
    #simple_vo_2003_comprison()


if __name__ == '__main__':
    main()
