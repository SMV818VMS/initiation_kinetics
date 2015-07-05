import time
from PyDSTool import args, Generator
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import kinetic_transcription_models as ktm
from KineticRateConstants import RateConstants


def SolveModel(model):

    t1 = time.time()
    traj = model.compute('demo')
    print('Solved model in {0} seconds'.format(time.time()-t1))

    return traj


def BuildModel(reaction_system, parameters, initial_conditions, solver):
    """
    Create and solve model using PyDSTool.
    """

    ### Specify the model
    DSargs = args()
    DSargs.name = 'Initial_transcription'
    DSargs.ics = initial_conditions
    DSargs.pars = parameters
    DSargs.tdata = [0, 100]
    DSargs.varspecs = reaction_system

    # Create the solver object
    t0 = time.time()

    if solver == 'vode':
        DS = Generator.Vode_ODEsystem(DSargs)
    elif solver == 'dopri':
        DS = Generator.Dopri_ODEsystem(DSargs)
    elif solver == 'radau':
        DS = Generator.Radau_ODEsystem(DSargs)
    else:
        print('Wtf no solver 1/0')
        1/0

    print('Generated model in {0} seconds'.format(time.time()-t0))

    return DS


def PlotTrajectoryModel(trajectory, alternative='1A'):

    pts = trajectory.sample()
    fig, ax = plt.subplots()

    if alternative == '2B':
        ax.plot(pts['t'], pts['RNAP_fre'], label='RNAP_000')

    ax.plot(pts['t'], pts['RNAP_000'], label='RNAP_000')
    ax.plot(pts['t'], pts['RNAP_100'], label='RNAP_100')
    ax.plot(pts['t'], pts['RNAP_200'], label='RNAP_200')
    ax.plot(pts['t'], pts['RNAP_1al'], label='RNAP_1al')
    ax.plot(pts['t'], pts['RNAP_2al'], label='RNAP_2al')
    ax.plot(pts['t'], pts['RNAP_flt'], label='RNAP_FL')
    ax.legend(loc='lower right')
    ax.set_xlabel('t')

    plt.show()


def SolveODE(reaction_system, parameters, initial_values, model_graph):

    solver = 'vode'  # 1s setup, 9.5s, 18s, 23s (200, 400, 500 time) solver
    solver = 'radau'  # 15s setup, 0.26s solver # does not work on laptop...
    solver = 'dopri'  # 5s setup, 0.1s solver, only 10% multiproc speedup

    pdt_model = BuildModel(reaction_system, parameters, initial_values, solver)
    trajectory = SolveModel(pdt_model)

    #When changing the model for parameter estimation, do through set method
    pdt_model.set(tdata=[0, 100])

    ktm.CheckConservationOfMass(model_graph, initial_values, trajectory)

    #This NTP calculation is actually unrealistic: in reality, the reactions
    #would have decreased with decreasing [NTP]: here you are assuming Vmax the
    #whole way.
    #NTP = CalcNTP(trajectory, starting_amount=30)

    PlotTrajectoryModel(trajectory)


def main():
    # Setup
    setup = ktm.ReactionSystemSetup()

    # Get Rate constants for this promoter variant
    R = RateConstants(variant='N25', use_AP=True, pos_dep_abortive=True, nac=10)

    model_graph = ktm.CreateModel_3_Graph(R, 'N25_simple', setup)

    # Parse the graph to make some nice output
    reaction_system, reactions, initial_values, parameters = ktm.GenerateODEInput(model_graph)

    SolveODE(reaction_system, parameters, initial_values, model_graph)

if __name__ == '__main__':
    main()
