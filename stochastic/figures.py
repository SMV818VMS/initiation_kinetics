"""
Figures for stochastic simulations.
"""
import numpy as np
from scipy.stats import spearmanr, pearsonr
import os
import sys
import pandas as pd
import shelve
import cPickle as pickle
import hashlib
sys.path.append('/home/jorgsk/Dropbox/phdproject/transcription_initiation/transcription_data/')
sys.path.append('/home/jorgsk/Dropbox/phdproject/transcription_initiation/kinetic_model/model')
import data_handler
import stochastic_model_run as smr
import kinetic_transcription_models as ktm
import matplotlib.pyplot as plt
plt.ioff()
import seaborn as sns
from operator import attrgetter
from ipdb import set_trace as debug  # NOQA
from KineticRateConstants import RateConstants

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

    Mmmh. A big challenge. The StochKit solvers are super fast, but do not
    save info about transitions (I need to know the flux of species).
    the timestep. Can you get it to record every timestep? Won't be easy.

    So OK. Estimated run time is about 20 seconds for each simulation (1000
    RNAPs) if you want to get the exact #RNA. But I don't think you need the
    exact #RNA for each of your intended use cases.
    """

    # XXX If you disbandon the direct_backtrack method, you will simplify the
    # steup and you can simplify the code fore calculating #abortive RNA since
    # you can rely on the backtracked state being present. see plot_timeseries
    # for how it can be done.

    sp = np.array(model.data_stochsim.species, dtype=np.int32)
    lb = model.data_stochsim.species_labels

    # open complex index
    oc_indx = lb.index('RNAP000')

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
        if code == 'RNAP000':
            continue

        # Bleh
        infos = code[-3:]
        if infos[1] == '_':
            rna_len = infos[0]
        else:
            rna_len = infos[:2]

        transcription_result[rna_len] = nr_hits

    return transcription_result


def naive_initial_transcription_values():
    """
    Run stochastic simulations on all the DG100 library variants

    Hey, it is of course possible to get the kind of kinetic plot Lilian is
    getting by simulating with let's say 1000 species and having a relatively
    slow transition from an 'ether' state to RNAP000; in this way one would
    see the abortive states generate more and more abortive product while FL
    would increase only extremely slowly after a while.

    But what I don't get from the Vo and unpublished kinetic data is why FL
    stops all together.
    """
    ITSs = data_handler.ReadData('dg100-new')
    ITSs = sorted(ITSs, key=attrgetter('PY'))

    backtrack_method = 'backstep'

    use_sim_db = True
    #use_sim_db = False
    #do_plot_timeseries = True
    do_plot_timeseries = False
    plotme = ['RNAP000', 'RNAP8_b', 'RNAPflt']

    #screen = False
    screen = True
    #subset = ['N25', 'N25-A1anti']
    subset = ['N25']

    # Keep all rnas to make a box plot after.
    all_rnas = {}
    all_FL_timeseries = {}

    # Test making the abortive step dependent on length. How about 21-nt?
    #length_dep_abortive_step = True

    for its in ITSs:

        ## Filter out some specific ones
        if screen and its.name not in subset:
            continue

        reaction_setup = ktm.ReactionSystemSetup(backtrack_method, escape_start=its.msat,
                                final_escape_RNA_length=its.msat, abortive_end=its.msat)

        rate_constants = RateConstants(variant=its.name, use_AP=True, nac=10,
                                       abortive=10, to_fl=5, dataset=ITSs)

        sim_setup = smr.SimSetup(initial_RNAP=500, nr_trajectories=1, sim_end=180)

        setup = {'reaction_setup': reaction_setup,
                 'rate_constants': rate_constants,
                 'simulation_setup': sim_setup}

        # serialize to a string and make a hash
        setup_serialized = pickle.dumps(setup)
        sim_id = hashlib.md5(setup_serialized).hexdigest()
        print(sim_id)

        # Shit, "cannot pickle code objects", so cannot store entire sim
        # (maybe you don't want to?).
        # For now, just save the post-processed results ts.
        shelve_database = shelve.open(simulation_storage)
        if use_sim_db and sim_id in shelve_database:
            db_entry = shelve_database[sim_id]
            nr_rna = db_entry['nr_rna']
            FL_timeseries = db_entry['FL_ts']
        else:
            sim = smr.Run(rate_constants.variant, rate_constants, reaction_setup, sim_setup)
            nr_rna = CalculateTranscripts(sim)
            FL_timeseries = GetFLTimeseries(sim)
            # Overwrite what's in the database by default
            # Bleh, you keep adding stuff here. You should perhaps consider
            # upgrading. Or making this a dictionary, with version. 'v1' has
            # only ts. 'v2' has ts and timeseris, 'v3' has ts,... but not now,
            # it's too early still. Just delete the database :)
            shelve_database[sim_id] = {'nr_rna': nr_rna,
                                       'FL_ts': FL_timeseries}

            if do_plot_timeseries:
                plot_timeseries(rate_constants.variant, sim, plotme)

        shelve_database.close()

        all_rnas[its.name] = nr_rna
        all_FL_timeseries[its.name] = FL_timeseries

        # XXX Make the basic bar plots
        write_bar_plot(nr_rna, its)

    #XXX BOX, BAR, AND PY-distribution PLOTs XXX
    # Divide itss into high and low and analyse
    #dsetmean = np.mean([i.PY for i in ITSs])
    #for partition in ['low PY', 'high PY', 'all']:

        #if partition == 'low PY':
            #low_ITSs = [i for i in ITSs if i.PY < dsetmean]
            #write_box_plot(low_ITSs, all_rnas, title=partition)

        #if partition == 'high PY':
            #high_ITSs = [i for i in ITSs if i.PY >= dsetmean]
            #write_box_plot(high_ITSs, all_rnas, title=partition)

        #if partition == 'all':
            #write_box_plot(ITSs, all_rnas, title=partition)

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
        old_AP = its_dict[its_name].abortiveProb

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


def plot_timeseries(name, sim, plotme):
    """
    Just plot timeseries data.
    """

    sp_2_label = {'RNAP000': 'Open complex',
                  'RNAP8_b': '8nt abortive RNA',
                  'RNAPflt': 'Full length transcript'}

    data = {}
    for sp_name in plotme:
        sp_index = sim.data_stochsim.species_labels.index(sp_name)
        sp = np.array(sim.data_stochsim.species, dtype=np.int32)

        # FL is additative so we keep the whole timeseries
        # Open complex should just reflect what is provided: number of species
        # at eacah time point
        if sp_name in ['RNAPflt', 'RNAP000']:
            ts = sp[:, sp_index]
        # But for #RNA, we need to take a the diff after in the backstepped
        # state - before we get instantaneous numbers
        else:

            ts = sp[:-1, sp_index] - sp[1:, sp_index]
            # Find -change in #species each timestep (+1 = aborted RNA, -1 = backtracked)
            change = sp[:-1,sp_index] - sp[1:,sp_index]
            # Set to zero those values that represent entry into the backstepped state
            change[change==-1] = 0
            # Get timeseries of # abortive RNA
            ts = np.cumsum(change)
            # We lose the first timestep. It's anyway impossive to produce an
            # abortive RNA in the first timestep. So pad a 0 value to the
            # array.
            ts = np.insert(ts, 0, 0)

        data[sp_2_label[sp_name]] = ts

    df = pd.DataFrame(data, index=sim.data_stochsim.time)
    # Drop duplicate lines?
    #df = df.drop_duplicates()
    # with 3 species dfd it's about 10 times smaller than df
    # Compared to full species array, dfd is 100 times smaller.

    # XXX Just noticed a mistake here. The amount of 7-nt RNA is not
    # increasing, since this is not an 'end station' in the model: you have to
    # re-obtain this. You can do that however, using a similar approach as you
    # do to get the final value.

    f, ax = plt.subplots()
    df.plot(ax=ax)

    ax.set_ylabel('# of species')
    ax.set_xlabel('Time (seconds)')

    file_name = 'Timeseries_{0}.pdf'.format(name)

    file_path = os.path.join(fig_dir1, file_name)

    f.suptitle('Stochastic simulation of initial transcription for {0}'.format(name))
    f.tight_layout()
    f.savefig(file_path, format='pdf')


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

        half_life = timeseries.FL[timeseries.FL==nr_RNAP/2.].index.tolist()[0]
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
        pct_values_experiment = calc_pct_experiment(its, max_ap_pos=data_range[-1])
        data[exp_key] = pct_values_experiment + ['experiment']

        # Model
        mod_key = '_'.join([its.name, 'model'])
        ts = tses[its.name]
        pct_values_model = calc_pct_model(ts, data_range)
        data[mod_key] = pct_values_model + ['model']

        # Fractional difference
        frac_diff = calc_frac_diff(np.array(pct_values_model), np.array(pct_values_experiment))
        data_frac[its.name] = frac_diff

    box_plot_model_exp(data, data_range, title)

    # XXX NICE!! You see the trend :) It is also more pronounced for low-PY
    # positions. You can choose one (N25?) to show this trend for a single
    # variant, then show this graph. For publication: green/blue for
    # model/experiment and positive negative axes.
    # And it is more pronounced for hi
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
    for its, folds in data_frac.items():
        early = np.nansum(folds[:5])
        late = np.nansum(folds[5:])

        measure = late - early
        PY = PYs[its]

        fold_diff_metric.append(measure)
        prod_yield.append(PY)

    prod_yield = np.asarray(prod_yield)
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
    mean.plot(kind='bar', yerr=std, colors=colors, rot=0)

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


def rel_diff(ar1, ar2):
    """
    Return the relative difference in % betwe
    """


def calc_pct_experiment(its, max_ap_pos=20):

    # We start with 2nt rna so subtract two; you want 19 values in return
    raw_abortive_data = its.rawDataMean[:max_ap_pos-1]
    raw_FL = its.fullLengthMean

    total_signal = sum(raw_abortive_data) + raw_FL

    pct_abortive_signal = raw_abortive_data * 100. / total_signal
    pct_FL_signal = raw_FL * 100 / total_signal

    return list(pct_abortive_signal) + [pct_FL_signal]


def calc_pct_model(ts, data_range):

    total_RNA_model = sum(ts.values()) + ts['FL']

    RNA_model = []
    for pos in data_range:
        k = str(pos)
        if k in ts:
            RNA_model.append(ts[k])
        else:
            RNA_model.append(0)

    pct_RNA_model = np.array(RNA_model) * 100. / total_RNA_model
    pct_FL_model = ts['FL'] * 100. / total_RNA_model

    return list(pct_RNA_model) + [pct_FL_model]


def write_bar_plot(ts, its):
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

    pct_values_experiment = calc_pct_experiment(its, max_ap_pos=data_range[-1])
    pct_values_model = calc_pct_model(ts, data_range)

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


def main():

    # Distributions of % of IQ units and #RNA
    naive_initial_transcription_values()

    # Remember, you are overestimating PY even if there is no recycling. Would
    # recycling lead to higher or lower PY? I'm guessing lower.


if __name__ == '__main__':
    main()
