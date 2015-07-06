import os
import stochpy
import sys
import numpy as np
sys.path.append('/home/jorgsk/Dropbox/phdproject/transcription_initiation/kinetic_model/model')
import kinetic_transcription_models as ktm
from KineticRateConstants import RateConstants
from ipdb import set_trace as debug  # NOQA


def runStochPy(model_path, method='direct'):
    """
         Only for StochPy:
         - *method* [default='Direct'] stochastic algorithm (Direct, FRM, NRM, TauLeaping)
    """

    mod = stochpy.SSA()
    model_dir, model_filename = os.path.split(model_path)
    mod.Model(model_filename, dir=model_dir)

    # When you run this one, you get in return each single timestep when
    # something happend. From this it is easy to calculate #RNA
    mod.DoStochSim(end=10000, method=method)
    #mod.DoStochSim(end=60, mode='time')
    #mod.DoStochSim(end=6000, mode='steps')

    #mod.DoStochKitStochSim(trajectories=1, IsTrackPropensities=True,
                           #customized_reactions=None, solver=None, keep_stats=True,
                           #keep_histograms=True, endtime=70)

    return mod


def CalculateTranscripts(model, backtrack_method):
    """
    Get #RNA: aborted and FL. Calculate intensities and PY.

    Mmmh. A big challenge. The StochKit solvers are super fast, but do not
    save info about transitions (I need to know the flux of species).
    the timestep. Can you get it to record every timestep? Won't be easy.

    So OK. Estimated run time is about 20 seconds for each simulation (1000
    RNAPs) if you want to get the exact #RNA. But I don't think you need the
    exact #RNA for each of your intended use cases.
    """

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

    fl_change = sp[1:,fl_indx] - sp[:-1,fl_indx]
    indx_before_escape = fl_change == +1
    indx_after_escape = np.roll(indx_before_escape, 1)
    fl_rna = sum(sp[indx_after_escape, fl_indx] - sp[indx_before_escape, fl_indx])

    # Tests. Check that the - number for the open complex corresponds to all
    # the other aborted ones
    assert sum(abs(abrt_rna)) == 2 * abs(abrt_rna[oc_indx])
    assert abrt_rna[fl_indx] == 0  # full length abort not possible

    # Summarize in a lookup table
    transcription_result = {'FL': fl_rna}

    for nr_hits, code in zip(abrt_rna, lb):

        # Skip non-abortive stuff
        if nr_hits == 0:
            continue

        # Skip open complex
        if code == 'RNAP000':
            continue

        if backtrack_method == 'backstep':
            assert code[-1] == 'b'
        else:
            assert code[-1] == '0'

        # Bleh
        infos = code[-3:]
        if infos[1] == '_':
            rna_len = infos[0]
        else:
            rna_len = infos[:2]

        transcription_result[rna_len] = nr_hits


def main():

    # Setup
    backtrack_method = 'backstep'
    setup = ktm.ReactionSystemSetup(backtrack_method)
    #setup = ktm.ReactionSystemSetup(backtrack_method, escape_start=4,
            #final_escape_RNA_length=4, abortive_end=4, abortive_beg=2)
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
    reactions, initial_values, parameters = \
    ktm.GenerateStochasticInput(model_graph, initial_RNAP=500)

    # Write a .psc file for the system
    ktm.write_psc(reactions, initial_values, parameters, model_name)

    psc_dir = '/home/jorgsk/Dropbox/phdproject/kinetic_paper/input_data/stochastic_psc_input'
    model_input = os.path.join(psc_dir, model_name + '.psc')

    model = runStochPy(model_input)
    #print("ran model")

    # Calculate #RNA, FL and abortive
    ts = CalculateTranscripts(model, backtrack_method)
    # XXX TODO: the real test: test with IQ values
    # XXX Wait a minute. Have I gone in a loop here? The AP values were
    # constructed assuming a certain model of transcription. Am I just
    # creating a self-fulfilling prophecy here? You need to think this over
    # really carefully.

    #debug()

    #model.PlotSpeciesTimeSeries()
    #model.PlotSpeciesTimeSeries(species2plot=['RNAPflt', 'RNAP600', 'RNAP000'])
    #model.PlotSpeciesTimeSeries(species2plot=['RNAP300', 'RNAP700', 'RNAP000', 'RNAP600'])


if __name__ == '__main__':
    main()
