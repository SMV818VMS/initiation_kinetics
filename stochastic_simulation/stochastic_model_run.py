import os
import stochpy
stochpy.plt.switch_backend('Agg')
import sys
sys.path.append('/home/jorgsk/Dropbox/phdproject/transcription_initiation/kinetic_model/model')
import kinetic_transcription_models as ktm
from KineticRateConstants import RateConstants
from ipdb import set_trace as debug  # NOQA


class SimSetup(object):
    """
    Setup for stochastic simulation.
    """

    def __init__(self, initial_RNAP=100, nr_trajectories=1, sim_end=60):

        self.init_RNAP = initial_RNAP
        self.nr_traj = nr_trajectories
        self.sim_end = sim_end


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
    #mod.DoStochSim(end=6000, mode='steps')

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

    model_graph = ktm.CreateModel_3_Graph(R, model_name, reaction_setup)

    # Parse the graph to make some nice output
    reactions, initial_values, parameters = \
    ktm.GenerateStochasticInput(model_graph, initial_RNAP=sim_setup.init_RNAP)

    # Write a .psc file for the system
    psc_dir = '/home/jorgsk/Dropbox/phdproject/transcription_initiation/kinetic_model//input_data/stochastic_psc_input'
    ktm.write_psc(reactions, initial_values, parameters, model_name, psc_dir)

    model_input = os.path.join(psc_dir, model_name + '.psc')

    model = runStochPy(model_input, trajectories=sim_setup.nr_traj,
                        end=sim_setup.sim_end)

    return model


def main():

    # Setup
    backtrack_method = 'backstep'

    setup = ktm.ReactionSystemSetup(backtrack_method, escape_start=15,
            final_escape_RNA_length=15, abortive_end=15, abortive_beg=2)

    # Get Rate constants for this promoter variant
    # XXX: Cool: the net result is only weakly dependent on NAC! Perhaps that
    # is why the [NTP] variation of Lilian made little difference?
    # a result! The abortive initiation reaction system the rate of
    # transcription has a minor effect. At least for N25!! Could be
    # interesting to test the different types, but I would guess that NAC/s is
    # most important for the high-PY variants.

    variant = 'N25'
    #variant = 'DG137a'
    R = RateConstants(variant=variant, use_AP=True, nac=10, abortive=30,
            #to_fl=3.5)
            to_fl=1.7)

    # Cool: the abortive rate only matters up to a certain point: after that
    # it has no influence on the process what so ever. I'm sure that can be
    # shown with just equations though, if one can produce an analytical
    # expression for the system. Interesting: NAC=10 but abortive =1, 10, 30
    # gives same t_50 but different end results? t_95 for example?
    # I have a feeling that the stuff you are looking for is kind of shown by
    # the distributions that stochpy is outputting by default.

    model_name = 'N25_simple'
    model_graph = ktm.CreateModel_3_Graph(R, model_name, setup)

    # Parse the graph to make some nice output
    reactions, initial_values, parameters = \
    ktm.GenerateStochasticInput(model_graph, initial_RNAP=100)

    # Write a .psc file for the system
    psc_dir = '/home/jorgsk/Dropbox/phdproject/kinetic_paper/input_data/stochastic_psc_input'
    ktm.write_psc(reactions, initial_values, parameters, model_name, psc_dir)

    model_input = os.path.join(psc_dir, model_name + '.psc')

    nr_traj= 1
    model = runStochPy(model_input, trajectories=nr_traj, end=180)

    #model.PlotSpeciesTimeSeries()
    model.PlotSpeciesTimeSeries(species2plot=['RNAP300', 'RNAP700', 'RNAP000', 'RNAP600'])
    # XXX interesting: the kinetics is waay to fast for the slow PY variants.


if __name__ == '__main__':
    main()
