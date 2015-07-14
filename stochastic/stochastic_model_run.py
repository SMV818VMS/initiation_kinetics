import os
import stochpy
stochpy.plt.switch_backend('Agg')
import sys
from math import floor
sys.path.append('/home/jorgsk/Dropbox/phdproject/transcription_initiation/kinetic/model')
import kinetic_transcription_models as ktm
from KineticRateConstants import RateConstants
from ipdb import set_trace as debug  # NOQA


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


def Run(model_name, R, reaction_setup, sim_setup):
    """
    Run stochastic model and return #AbortedRNA and #FL. Is this general
    enough? Should you also return timeseries? Or perhaps you should just
    return the model object and let the receiving end decide what to do with
    it.
    """

    # Make a graph for the system
    model_graph = ktm.CreateModel_3_Graph(R, model_name, reaction_setup,
                                          sim_setup.debug_mode)

    # From the setup, find how many productive and how many unproductive RNAPs
    # should be simulated
    nr_prod_RNAP, nr_unprod_RNAP = calc_nr_RNAPs(reaction_setup, sim_setup)

    # Set initial values and extract values from the graph for simulation
    reactions, initial_values, parameters = \
    ktm.GenerateStochasticInput(model_graph, nr_prod_RNAP, nr_unprod_RNAP)

    # Write a .psc file for the system
    psc_dir = '/home/jorgsk/Dropbox/phdproject/transcription_initiation/kinetic/input/psc_files'
    ktm.write_psc(reactions, initial_values, parameters, model_name, psc_dir)

    model_input = os.path.join(psc_dir, model_name + '.psc')

    model = runStochPy(model_input, trajectories=sim_setup.nr_traj, end=sim_setup.sim_end)

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


def organize_for_plotting():
    """
    You need to associate labels you send in to the plot function with species
    names from the simulation and a label for the plots. This dictionary does
    the trick.
    """

    organizer = {}
    for nt in range(2,21):
        key = '{0}nt'.format(nt)
        if nt < 10:
            species = ['RNAP{0}_b'.format(nt), 'RNAP{0}_f'.format(nt)]
        else:
            species = ['RNAP{0}b'.format(nt), 'RNAP{0}f'.format(nt)]
        legend = '{0}nt abortive RNA'.format(nt)
        organizer[key] = {'species': species, 'legend': legend}

    organizer['productive_open_complex'] = {'species': ['RNAPpoc'],
                                            'legend': 'Productive open complex'}

    organizer['unproductive_open_complex'] = {'species': ['RNAPuoc'],
                                              'legend': 'Unproductive open complex'}

    organizer['FL'] = {'species': ['RNAPflt'], 'legend': 'Full length product'}

    return organizer


def calc_timeseries(organizer, plot_name, sim_species, sp):
    """
    Get the timeseries for the given plot name
    """

    import numpy as np

    # Model species for this name
    species = organizer[plot_name]['species']

    # First deal with the easy ones. FL is additative so we keep the
    # whole timeseries .Open complex should just reflect what is provided:
    # number of species at eacah time point
    if plot_name in ['FL', 'productive_open_complex', 'unproductive_open_complex']:
        sp_name = species[0]
        if sp_name not in sim_species:
            continue
        sp_index = sim_species.index(sp_name)
        ts = sp[:, sp_index]
    # But for #RNA, we need to calculate the nr of species
    # (These are the observables of the system)
    # This must work for both 1 and 2 species producing abortive RNAs, for
    # when we are simulating both productive and unproductive species
    else:
        ts = np.zeros(sp.shape[0])
        for sp_name in species:

            # Silently skip species not modelled
            if sp_name not in sim_species:
                continue

            sp_index = sim_species.index(sp_name)
            # Find -change in #species each timestep (+1=aborted RNA, -1=backtracked)
            change = sp[:-1,sp_index] - sp[1:,sp_index]
            # Set to zero those values that represent entry into the backstepped state
            change[change==-1] = 0
            # Get timeseries of # abortive RNA
            cumul = np.cumsum(change)
            # We lose the first timestep. It's anyway impossive to produce an
            # abortive RNA in the first timestep. So pad a 0 value to the
            # array to make it fit with the time index.
            cumul = np.insert(cumul, 0, 0)

            ts += cumul

    return ts


def plot_timeseries_copy(sim_name, sim, plotme=False):
    """
    Plot timeseries of different molecular species from initial
    transcription modelling.
    """
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    # Just organize it all
    organizer = organize_for_plotting()

    # If no species are specified, plot them all
    if plotme is False:
        plotme = organizer.keys()

    sim_species = sim.data_stochsim.species_labels

    # Convert species array to int32
    sp = np.array(sim.data_stochsim.species, dtype=np.int32)

    data = {}
    for plot_name in plotme:
        ts = calc_timeseries(organizer, plot_name, sim_species, sp)
        data[organizer[plot_name]['legend']] = ts

    df = pd.DataFrame(data, index=sim.data_stochsim.time)

    # Drop duplicate lines?
    #df = df.drop_duplicates()
    # with 3 species dfd it's about 10 times smaller than df
    # Compared to full species array, dfd is 100 times smaller.

    f, ax = plt.subplots()
    df.plot(ax=ax)

    ax.set_ylabel('# of species')
    ax.set_xlabel('Time (seconds)')

    file_name = 'Timeseries_{0}.pdf'.format(sim_name)

    f.suptitle('Stochastic simulation of initial transcription for {0}'.format(sim_name))
    f.tight_layout()
    f.savefig(file_name, format='pdf')


def main():

    # Setup
    backtrack_method = 'backstep'

    # After 10 min:
    # Method 1: 61 abortive 7.35 FL: 8.299 times more abortive than FL
    # Method 2: 5.03 abortive and 0.62 FL; 8.11 times more abortive than FL
    # How does this fly with the DG100 measurements?
    # In DG100 there is 6% FL.
    # But in Vo 2003 there is 11 and 10.8 FL.
    # So this is almost twice as much.
    # Does that mean that with multiple rounds of re-initiation, there is a
    # greater likelihood for entering into an unproductive phase?.
    # But what's the use of this information anyway? It's much more useful to
    # compare the initial growth of abortive to the unproductive-only growth
    # of abortive.

    #reaction_setup = ktm.ReactionSystemSetup(backtrack_method, escape_start=11,
                                            #final_escape_RNA_length=11, abortive_end=11,
                                            #abortive_beg=2, unproductive_only=True)

    reaction_setup = ktm.ReactionSystemSetup(backtrack_method, escape_start=11,
                                            final_escape_RNA_length=11, abortive_end=11,
                                            abortive_beg=2, unproductive=True)

    #reaction_setup = ktm.ReactionSystemSetup(backtrack_method, escape_start=11,
                                            #final_escape_RNA_length=11, abortive_end=11,
                                            #abortive_beg=2)

    # Set fraction of unproductive complexes
    # Should be ignored when using unproductive_only or only productive
    unproductive_pct = 0.10

    # XXX OK!! youdonit. Now you got the effect of rapidly producing abortive
    # suff in the beginning, and then more slowly at the end.

    # SO! Right now, the backtstepping probability for productive RNAPs is too
    # high at +2, and possibly at +3, and 4 too. So you need to adjust the
    # abortive probabilities of productive complexes. This can be done by
    # first running a simulation with unproductive_only, storing the #RNA.

    # XXX New mornig, new thoughts. You need to approach the problem like
    # this: 1) during X seconds, get the #RNA for unproductive ONLY. Then, you
    # actually have to run a parameter estimation process to find the AP for
    # the productive. Then, you can combine the two. But you must keep using
    # the original AP for the simple model. Q: can you do this analytically
    # using the formula for AP-calculation? Test the formula.
    # You should unify your way of calculating APs, dude. You must have
    # repeated that calculation so many times.

    model_name = 'N25'
    #variant = 'DG137a'
    R = RateConstants(variant=model_name, use_AP=True, nac=10, abortive=30, to_fl=1.7)

    sim_name = get_sim_name(reaction_setup, model_name)

    sim_setup = SimSetup(initial_RNAP=200, sim_end=40,
                         unproductive_pct=unproductive_pct, debug_mode=False)

    unproductive = Run(sim_name, R, reaction_setup, sim_setup)

    species2plot = ['2nt', '3nt', '8nt', 'FL', 'productive_open_complex',
                    'unproductive_open_complex']

    plot_timeseries_copy(sim_name, unproductive, species2plot)


def get_sim_name(reaction_setup, model_name):

    if reaction_setup['unproductive']:
        sim_name = '{0}_also_unproductive'.format(model_name)
    elif reaction_setup['unproductive_only']:
        sim_name = '{0}_unproductive_only'.format(model_name)
    else:
        sim_name = '{0}_productive_only'.format(model_name)

    return sim_name


if __name__ == '__main__':
    main()
