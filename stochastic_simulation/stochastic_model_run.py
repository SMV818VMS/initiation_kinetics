import os
import stochpy
import sys
sys.path.append('/home/jorgsk/Dropbox/phdproject/kinetic_paper/model')
import kinetic_transcription_models as ktm
from KineticRateConstants import RateConstants
from ipdb import set_trace as debug  # NOQA
import matplotlib.pyplot as plt


def runStochPy(model_path):

    mod = stochpy.SSA()
    model_dir, model_filename = os.path.split(model_path)
    mod.Model(model_filename, dir=model_dir)

    #mod.DoStochSim(end=20000)

    mod.DoStochKitStochSim(trajectories=1, IsTrackPropensities=True,
                           customized_reactions=None, solver=None, keep_stats=False,
                           keep_histograms=False, endtime=70)

    return mod


def main():
    # Setup
    setup = ktm.ReactionSystemSetup()

    # Get Rate constants for this promoter variant
    R = RateConstants(variant='N25', use_AP=True, pos_dep_abortive=True, nac=10)

    model_graph = ktm.CreateModel_3_Graph(R, 'N25_simple', setup)

    # Parse the graph to make some nice output
    reactions, initial_values, parameters = ktm.GenerateStochasticInput(model_graph)

    # Write a .psc file for the system
    ktm.write_psc(reactions, initial_values, parameters, model_graph.name)

    model_input = '/home/jorgsk/Dropbox/phdproject/kinetic_paper/model/psc_files/N25_simple.psc'

    model = runStochPy(model_input)
    print("ran model")

    model.PlotSpeciesTimeSeries()


if __name__ == '__main__':
    main()
