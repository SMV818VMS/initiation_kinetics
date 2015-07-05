import os
import stochpy
import sys
sys.path.append('/home/jorgsk/Dropbox/phdproject/kinetic_paper/model')
import kinetic_transcription_models as ktm
from KineticRateConstants import RateConstants
from ipdb import set_trace as debug  # NOQA


def runStochPy(model_path):

    mod = stochpy.SSA()
    model_dir, model_filename = os.path.split(model_path)
    mod.Model(model_filename, dir=model_dir)

    mod.DoStochSim(end=10000)

    #mod.DoStochKitStochSim(trajectories=1, IsTrackPropensities=True,
                           #customized_reactions=None, solver=None, keep_stats=False,
                           #keep_histograms=False, endtime=70)

    return mod


def main():

    # Setup
    setup = ktm.ReactionSystemSetup(backstep_mode=True)
    #setup = ktm.ReactionSystemSetup(backstep_mode=True, allow_escape=False,
            #final_escape_RNA_length=8)
    #setup = ktm.ReactionSystemSetup(direct_backtrack=True, allow_escape=False,
            #final_escape_RNA_length=7)  # interesting; if they had used <=7
    #instead of <=8, RNAP would stall a lot more in the forward state.
    # NO, that analysis is not valid. Look at Revyakin SOM. When restricting
    # up to a +7 for example, AP at +7 will be a lot higher than if you can
    # carry on to +8.

    #setup = ktm.ReactionSystemSetup(direct_backtrack=True)

    # Get Rate constants for this promoter variant
    # XXX: Cool: the net result is only weakly dependent on NAC! Perhaps that
    # is why the [NTP] variation of Lilian made little difference :)
    # a result! The abortive initiation reaction system the rate of
    # transcription has a minor effect. At least for N25!! Could be
    # interesting to test the different types, but I would guess that NAC/s is
    # most important for the high-PY variants. Remember: the AP you got are
    # entirely specific to the conditions of Lilian's experiments, [NTP]=100
    # for example. So just increasing NAC with the same AP will not be valid.
    # But you can test the sensitivty of the effect of decreasing [NTP].

    #R = RateConstants(variant='N25', use_AP=True, nac=10)
    R = RateConstants(variant='N25', use_AP=True, nac=10, abortive=30)
    # Cool: the abortive rate only matters up to a certain point: after that
    # it has no influence on the process what so ever. I'm sure that can be
    # shown with just equations though, if one can produce an analytical
    # expression for the system. Interesting: NAC=10 but abortive =1, 10, 30
    # gives same t_50 but different end results?

    model_name = 'N25_simple'
    model_graph = ktm.CreateModel_3_Graph(R, model_name, setup)

    # Parse the graph to make some nice output
    reactions, initial_values, parameters = ktm.GenerateStochasticInput(model_graph)

    # Write a .psc file for the system
    ktm.write_psc(reactions, initial_values, parameters, model_name)

    psc_dir = '/home/jorgsk/Dropbox/phdproject/kinetic_paper/input_data/stochastic_psc_input'
    model_input = os.path.join(psc_dir, model_name + '.psc')

    model = runStochPy(model_input)
    print("ran model")

    model.PlotSpeciesTimeSeries(species2plot=['RNAP_flt', 'RNAP_600', 'RNAP_000'])
    #model.PlotSpeciesTimeSeries(species2plot=['RNAP_300', 'RNAP_700', 'RNAP_000', 'RNAP_600'])


if __name__ == '__main__':
    main()
