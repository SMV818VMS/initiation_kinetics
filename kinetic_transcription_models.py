from PyDSTool import args, Generator
from ipdb import set_trace as debug  # NOQA
from matplotlib.pyplot import ion  # NOQA
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import networkx as nx
import time
from datetime import datetime
import os

# Non bliocking graphics
#ion()


class RateConstants(object):
    """
    Class for providing and calculating rate constants

    For now, this class only returns constants for the rate constants. In the
    future, it will use duplex length, rna length, number of backtracked steps,
    etc, to calculate the various rates. A lot of the actual research is going
    to happen on these rates, so the class must be flexible and powerful.
    """

    # default all values to nan to avoid missing obvious mistakes
    def __init__(self, forwardtl=0.9, reversetl=0.1, nac=0.2,
                        abortive=0.2, backtrack=0.3, to_fl=0.7,
                        backstep=0.3):

        self.forwardtl = forwardtl
        self.reversetl = reversetl
        self.nac = nac
        self.to_fl = to_fl
        self.abortive = abortive
        self.backtrack = backtrack
        self.backstep = backstep

    def Forward(self):
        return self.forwardtl

    def Reverse(self):
        return self.reversetl

    def Backtrack(self):
        return self.backtrack

    def Backstep(self):
        return self.backstep

    def Nac(self):
        return self.nac

    def ToFL(self):
        return self.to_fl

    def Abortive(self):
        return self.abortive

    def FirstStep(self, starting_duplex_length):
        """
        The initial step from open complex formation to a conformation with a
        starting duplex length.

        Method: take a dubious average of the net forward/reverse constants plus the
        nucleotide addition constants, and the divide this by the number of
        steps that should be taken. This is a bit wonky, but it does not matter
        since it is sequence independent.
        """
        funky_rate_constant_average = (self.Forward() - self.Reverse() + self.Nac())/2.
        return funky_rate_constant_average / float(starting_duplex_length)


def DuplexLength(rna_length, nr_backtracked_steps):
    """
    Duplex length is set to max 10
    """
    return min(rna_length - nr_backtracked_steps, 10)


def BasicTranscription():
    """
    Basic transcription. Start with len=0 RNA, translocate; add first
    nucleotide; repeat; promoter escape at len=2 RNA. Direct abortive release possible
    from the 10 state. Backtracking possible from the 20 state, to a
    from_2_now_1 state. Abortive release possible from this state.

    Challenge: how to keep track of the amount of aborted stuff? We need to
    create an extra state: 1al, 2al for abortive _l_og, since this state takes
    care of logging the amount of abortive product that is being produced. The
    rate at which abortive RNAs are formed will be the same as the rate at
    which RNAP aborts. Therefore, the accumulated amount of abortive RNA will
    be the same as the accumulated amount of abortive RNAP! That means that if
    X rnap initates transcription, and the final amount of RNAP in state 2a is
    1.7X, it means that 1.7 times more abortive RNA was formed than the
    initiating amount of RNAP.
    """

    # State and parameter logic:
    # 20 means RNA length 2, pre-translocated state
    # 21 means RNA length 2, post-translocated state
    # 201 means RNA length 2, backtracked 1 step from pre-translocated state
    # 2a means RNA length 2, aborted
    # FL means full length transcript

    initial_conditions = {'RNAP_00':1,
                          'RNAP_01':0,
                          'RNAP_10':0,
                          'RNAP_1al':0,
                          'RNAP_11':0,
                          'RNAP_20':0,
                          'RNAP_201':0,
                          'RNAP_2al':0,
                          'RNAP_21':0,
                          'RNAP_FL':0
                          }

    # Rates
    forward = 0.9
    reverse = 0.1
    nac = 0.2
    to_fl = 0.7
    abortive = 0.2
    backtrack = 0.3

    #abortive_from_10 = 0.2
    #abortive_from_20 = 0.2
    #backtrack_from_20_to_201 = 0.3
    #abortive_from_201 = 0.2

    # These give the releation between the states
    # All possible relevant state information should be obtainable from the
    # parameter name: length of RNA, translocation state, and backtracked state.
    parameters = {
            'k_00_to_01':forward,
            'k_01_to_00':reverse,
            'k_01_to_10':nac,

            'k_10_to_11':forward,
            'k_10_to_1a':abortive,
            'k_11_to_10':reverse,
            'k_11_to_20':nac,

            'k_20_to_21':forward,
            'k_20_to_2a':abortive,
            'k_20_to_201':backtrack,
            'k_201_to_2a':abortive,
            'k_21_to_20':reverse,
            'k_21_to_FL':to_fl,
                  }

    # Reaction system
    RNAP_00_rhs = 'k_01_to_00*RNAP_01 - k_00_to_01*RNAP_00 '\
                  '+ k_10_to_1a*RNAP_10 '\
                  '+ k_20_to_2a*RNAP_20 '\
                  '+ k_201_to_2a*RNAP_201 '

    RNAP_01_rhs = '-k_01_to_00*RNAP_01 + k_00_to_01*RNAP_00 - k_01_to_10*RNAP_01'

    RNAP_10_rhs = 'k_11_to_10*RNAP_11 - k_10_to_11*RNAP_10 + k_01_to_10*RNAP_01 '\
                  '-k_10_to_1a*RNAP_10'

    RNAP_11_rhs = '-k_11_to_10*RNAP_11 + k_10_to_11*RNAP_10 - k_11_to_20*RNAP_11'

    RNAP_1al_rhs = 'k_10_to_1a*RNAP_10'

    RNAP_20_rhs = 'k_21_to_20*RNAP_21 - k_20_to_21*RNAP_20 + k_11_to_20*RNAP_11 '\
                  '-k_20_to_201*RNAP_20 - k_20_to_2a*RNAP_20'

    RNAP_21_rhs = '-k_21_to_20*RNAP_21 + k_20_to_21*RNAP_20 - k_21_to_FL*RNAP_21'

    RNAP_201_rhs = 'k_20_to_201*RNAP_20 - k_201_to_2a*RNAP_201'

    RNAP_2al_rhs = 'k_201_to_2a*RNAP_201 + k_20_to_2a*RNAP_20'

    RNAP_FL_rhs = 'k_21_to_FL*RNAP_21'

    #print RNAP_00_rhs
    #debug()

    system = {'RNAP_00':RNAP_00_rhs,
              'RNAP_01':RNAP_01_rhs,
              'RNAP_10':RNAP_10_rhs,
              'RNAP_11':RNAP_11_rhs,
              'RNAP_1al':RNAP_1al_rhs,
              'RNAP_20':RNAP_20_rhs,
              'RNAP_21':RNAP_21_rhs,
              'RNAP_201':RNAP_201_rhs,
              'RNAP_2al':RNAP_2al_rhs,
              'RNAP_FL':RNAP_FL_rhs
              }

    ### Specify the model
    DSargs = args()
    DSargs.name ='Simple_transcription_and_escape'
    DSargs.ics = initial_conditions
    DSargs.pars = parameters
    DSargs.tdata = [0, 150]
    DSargs.varspecs = system

    # Create the solver object
    DS = Generator.Vode_ODEsystem(DSargs)

    traj = DS.compute('demo')
    pts = traj.sample()

    fig, ax = plt.subplots()

    ax.plot(pts['t'], pts['RNAP_00'], label='RNAP_00')
    ax.plot(pts['t'], pts['RNAP_10'], label='RNAP_10')
    ax.plot(pts['t'], pts['RNAP_20'], label='RNAP_20')
    ax.plot(pts['t'], pts['RNAP_1al'], label='RNAP_1a')
    ax.plot(pts['t'], pts['RNAP_2al'], label='RNAP_2a')
    ax.plot(pts['t'], pts['RNAP_FL'], label='RNAP_FL')
    ax.legend(loc='lower right')
    ax.set_xlabel('t')

    fig.suptitle('Hard coded system')

    plt.show()


def QualityCheckModel_1(backtrack_start, initial_rna_length, final_escape_RNA_length,
                        abortive_range, backtrack_until, escape_start):
    """
    Do some sanity checking to verify that the settings for creating the
    reaction network graph is correct.
    """
    warnings = []

    if escape_start > final_escape_RNA_length:
        warning = 'Escape starts after it ends!'
        warnings.append(warning)

    if backtrack_until not in abortive_range:
        warning = 'Backtracking states exist that do not abort!'
        warnings.append(warning)

    if initial_rna_length > final_escape_RNA_length:
        warning = 'Initial RNA length longer than escape length!'
        warnings.append(warning)

    if backtrack_start > final_escape_RNA_length:
        warning = 'No backtracking possible!'
        warnings.append(warning)

    for warning in warnings:
        print 'WARNING: ' + warning


def RateConstantName(from_state, to_state):
    return '_to_'.join([from_state, to_state])


def AddAbortiveLogNode(G, abortive_log):
    # Add the abortive log state
    G.add_node(abortive_log)


def AddReaction(G, rc, from_state, to_state):
    rc_name = RateConstantName(from_state, to_state)
    G.add_edge(from_state, to_state, rate_constant_value=rc, rate_constant_name=rc_name)


def Model_1_Graph(R, setup):
    """

    Most general model of transcription initiation. Contains the following reactions:

    * Nucleotide addition / PPi release
    * Forward translocation
    * Reverse translocation
    * Backtracking from pre-translocated state
    * Backtracking from each backtracked state

    Code for initially transcribing RNAP: RNAP_XYZ

    X -> length of RNA [0,...,inf]
    Y -> transcloation state if [0,1] ; if abortive log == 'a'
    Z -> nr_backtracked steps [0,...,inf]; if abortive log == 'l'

    Examples of RNAP in different states:
    500 : len 5 duplex in the pre-translocated state
    510 : len 5 duplex in the post-translocated state
    503 : len 2 duplex backtracked 3 steps
    5al : len 5 RNA aborted log -- total aborted RNAP / proxy for aborted RNA of length 5
    flt : full length transcript

    Assuming backtracking and direct abortive release happens from the pretranslocated state.

    """

    # read setup
    allow_escape = setup['allow_escape']
    escape_start = setup['escape_start']
    backtrack_start = setup['backtrack_start']
    initial_rna_length = setup['initial_rna_length']
    final_escape_RNA_length = setup['final_escape_RNA_length']  # maximum length
    abortive_range = setup['abortive_range']
    backtrack_until_duplex_len = setup['backtrack_until_duplex_len']

    # QA on setup
    QualityCheckModel_1(backtrack_start, initial_rna_length, final_escape_RNA_length,
            abortive_range, backtrack_until_duplex_len, escape_start)

    # Template string for state variables
    state_template = 'RNAP_{0}{1}{2}'

    # Initialize a graph object
    G = nx.DiGraph()

    # Create the first reaction step -> from open complex to starting complex
    open_complex = state_template.format(0, 0, 0)
    if initial_rna_length > 0:
        starting_complex = state_template.format(initial_rna_length, 0, 0)
        first_step = R.FirstStep(initial_rna_length)
        G.add_edge(open_complex, starting_complex, rate_constant=first_step)
    else:
        starting_complex = open_complex

    # Create the full length (FL) state
    if allow_escape:
        full_length = state_template.format('f', 'l', 't')

    # Tag the starting complex -- and create it, if not already made
    G.add_node(starting_complex, initial_node=True)

    # Following the growing RNA to explore all possible model states
    for rna_len in range(initial_rna_length, final_escape_RNA_length + 1):

        # Define the different possible states associatd with an RNA of this length
        pre_translocated = state_template.format(rna_len, 0, 0)
        post_translocated = state_template.format(rna_len, 1, 0)

        forward_rc = R.Forward()
        reverse_rc = R.Reverse()

        AddReaction(G, forward_rc, pre_translocated, post_translocated)
        AddReaction(G, reverse_rc, post_translocated, pre_translocated)

        # If RNA length longer than 0, add nucleotide addition cycle
        if rna_len > 0:

            previous_post_translocated = state_template.format(rna_len-1, 1, 0)
            nac_rc = R.Nac()
            AddReaction(G, nac_rc, previous_post_translocated, pre_translocated)

        # Add abortive log state if grown into the abortive range
        if rna_len >= abortive_range[0]:
            abortive_log = state_template.format(rna_len, 'a', 'l')
            AddAbortiveLogNode(G, abortive_log)

        # backtrack until backtrack stop
        if rna_len >= backtrack_start:
            for nr_backtracked_steps in range(1, rna_len - backtrack_until_duplex_len + 1):
                this_backtracked = state_template.format(rna_len, 0, nr_backtracked_steps)

                # first step from pre-translocated
                if nr_backtracked_steps == 1:
                    first_backtrack_rc = R.Backstep()
                    AddReaction(G, first_backtrack_rc, pre_translocated, this_backtracked)

                # subsequent from previous backtracked state
                else:
                    last_backtracked = state_template.format(rna_len, 0, nr_backtracked_steps-1)
                    this_backtrack_rc = R.Backtrack()
                    AddReaction(G, this_backtrack_rc, last_backtracked, this_backtracked),

                # check if we have backtracked into abortive-territory!
                if DuplexLength(rna_len, nr_backtracked_steps) in abortive_range:
                    abortive_rc = R.Abortive()
                    AddReaction(G, abortive_rc, this_backtracked, open_complex)
                    # add log
                    AddReaction(G, abortive_rc, this_backtracked, abortive_log)

        # direct abortive release
        if rna_len in abortive_range:
            abortive_rc = R.Abortive()
            AddReaction(G, abortive_rc, pre_translocated, open_complex)
            # add log
            AddReaction(G, abortive_rc, pre_translocated, abortive_log)

        if allow_escape:
            if rna_len >= escape_start:
                escape_rc = R.ToFL()
                AddReaction(G, escape_rc, post_translocated, full_length)

    return G


def ReactionSystemSetupModel_1():
    """
    Setup for the reaction system of Model 1
    """
    setup = {}

    # You can choose not to let the sytem proceed to escape
    setup['allow_escape'] = True
    # Where promoter escape may start from
    setup['escape_start'] = 3
    # RNA length where backtracking may start
    setup['backtrack_start'] = 2
    # starting RNA length
    setup['initial_rna_length'] = 0
    # escape RNA-length: last point of escape
    setup['final_escape_RNA_length'] = 5
    # RNA-DNA duplex abortive range: abortive release may happen here, either by
    # backtracking to this duplex length or directly from this duplex length
    setup['abortive_range'] = range(1, 5+1)
    # minimal duplex length for backtracking -- may be useful for reducing the
    # number of states. For example, backtracking to a length 1 bp hybrid may
    # have such a low probability that it is not worth considering making 19
    # states just to represent this option. An alternative way of not including
    # this state is by setting the rate constant to zero for these states and
    # testing for zero valued rate constants.
    setup['backtrack_until_duplex_len'] = 1

    return setup


def CreateReactionSystem(G):
    """
    Create reaction system from directed graph with 'rate_constant'.

    Requires: directed graph; each directed edge must have a rate constant.
    Optional: if a 'to-node' has a name that ends with 'al', the 'from-node'
              will not be associated with a loss of mass to this 'log' node.
    """

    system = {}
    for from_node, to_node, data in G.edges(data=True):

        # add nodes if not in system
        if from_node not in system:
            system[from_node] = ''
        if to_node not in system:
            system[to_node] = ''

        # extract reaction rate constant
        rc = data['rate_constant_name']
        # add contribution to system from this connection
        factor = '*'.join([str(rc), from_node])

        # Remove spacing to make small printed systems look nice
        if system[from_node] == '':
            loss = '-' + factor
        else:
            loss = ' - ' + factor

        if system[to_node] == '':
            gain = factor
        else:
            gain = ' + ' + factor

        # Do not subtract for mass going to a log state
        if to_node.endswith('al'):
            pass
        else:
            system[from_node] = system[from_node] + loss

        system[to_node] = system[to_node] + gain

    return system


def SetInitialValues(G):
    """
    Set initial values for all nodes except initial node to 0.
    Start with 1 for initial node.

    Requires: graph G with at most one initial node.
    Dislikes: more than one initial node. Will bark.
    """

    initial_found = False
    initial_conditions = {}

    for node, data in G.nodes(data=True):
        if 'initial_node' in data:
            if not initial_found:
                initial_conditions[node] = 1
                initial_found = True
            else:
                print('Multiple initials found!')
                1/0
        else:
            initial_conditions[node] = 0

    if not initial_found:
        print('No initial nodes found')
    else:
        return initial_conditions


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
    DSargs.name ='Initial_transcription'
    DSargs.ics = initial_conditions
    DSargs.pars = parameters
    DSargs.tdata = [0, 400]
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


def GetParameters(G):
    """
    Parse graph to link parameter name with parameter value.

    Requires: directd graph G with attributes rate_constant_name and rate_constant_value.
    """

    parameters = {}
    for from_node, to_node, data in G.edges(data=True):
        parameters[data['rate_constant_name']] = data['rate_constant_value']

    return parameters


def PlotTrajectoryModel_1(trajectory):

    pts = trajectory.sample()

    fig, ax = plt.subplots()

    ax.plot(pts['t'], pts['RNAP_000'], label='RNAP_000')
    ax.plot(pts['t'], pts['RNAP_100'], label='RNAP_100')
    ax.plot(pts['t'], pts['RNAP_200'], label='RNAP_200')
    ax.plot(pts['t'], pts['RNAP_1al'], label='RNAP_1al')
    ax.plot(pts['t'], pts['RNAP_2al'], label='RNAP_2al')
    ax.plot(pts['t'], pts['RNAP_flt'], label='RNAP_FL')
    ax.legend(loc='lower right')
    ax.set_xlabel('t')

    #plt.show()


def WriteGraphAndSetup(G, reaction_setup):
    """
    Write to file an image of the graph and the corresponding setup.
    """

    fig, ax = plt.subplots()

    nx.draw_graphviz(G, with_labels=True, ax=ax)
    # filename is number of translocation steps followed by a time tag
    time_tag = str(datetime.now()).split('.')[0]
    store_dir = 'Log'
    if not os.path.isdir(store_dir):
        os.makedirs(store_dir)

    if 'final_escape_RNA_length' in reaction_setup:
        beg = str(reaction_setup['final_escape_RNA_length'])
    else:
        beg = '0'

    image_path = os.path.join(store_dir, beg + '_' + time_tag + '.png')
    fig.set_size_inches(17, 8)
    fig.savefig(image_path)

    text_path = os.path.join(store_dir, beg + '_' + time_tag + '.txt')
    with open(text_path, 'wb') as text_handle:
        for key, value in reaction_setup.items():
            text_handle.write(key + '\t\t' + str(value) + '\n')


def CheckConservationOfMass(G, initial_values, trajectory):
    """
    Compare total initial mass with total final mass
    """

    pts = trajectory.sample()
    # Do not plot abortive log
    skip_us = [n for n in G.nodes() if n.endswith('al')]

    # Get the final values of the system
    final_values = {node:pts[node][-1] for node in G.nodes()}

    total_initial = sum(initial_values.values())
    # final sum, excluding log states
    total_final = sum([val for n, val in final_values.items() if not n in skip_us])

    print('Total initial: {0}'.format(total_initial))
    print('Total final: {0}'.format(total_final))

    # bar plots with rotated labels?
    #ax.plot(pts['t'], pts['RNAP_000'], label='RNAP_000')


def GenerateInput(G):
    """
    Create a graph based on the raction setup and rate constants. From the
    graph, obtain the system equations, initial values, and parameters (rate
    coefficients).

    Requires a directed graph G where each directed node is associated with a
    'rate_constant_name' and 'rate_constant_value'. Also requires one of the
    nodes to be called 'initial_node'.

    This method is agnostic towards how the graph is crated.
    """

    # get reaction system, initial conditions, and parameter values
    reaction_system = CreateReactionSystem(G)
    initial_values = SetInitialValues(G)
    parameters = GetParameters(G)

    return reaction_system, initial_values, parameters


def wrapper(simtimes, model):

    trajs = []
    for simtime in simtimes:
        model.set(tdata=[0, simtime])

        trajectory = SolveModel(model)
        trajs.append(trajectory)

    return trajs


def CreateModel_1_Graph():
    """
    Setup and create a graph for Model_1

    Model_1 is the most complete model of transcription initiation. The only
    thing it perhaps lacks is the ability to escape the promoter from different
    positions.
    """

    # Stoichiometry setup
    reaction_setup = ReactionSystemSetupModel_1()

    # Rate constants
    R = RateConstants()

    # Create the graph
    G = Model_1_Graph(R, reaction_setup)

    #Plot model graph and write setup log
    WriteGraphAndSetup(G, reaction_setup)

    print("Number of model states: {0}".format(len(G.nodes())))

    return G


def main():
    # TODO next time: start working with an ITS: generate rates and run
    # simulation. Start getting a feel for the kinetic parameters.

    # TODO next time: start specifying several models. You are going to
    # investigate increasingly complex models to evaluate their individual
    # usefullness. You'll need an easy way to specify models and rate
    # coefficients so that it is perfectly clear which model you are using and
    # which rate coefficients

    # XXX two ways of doing this. 1) make a sufficiently complex single class
    # that can generate all these models given basic instructions. 2) make one
    # class per model. If you have different variants of each model, such as
    # different regimes for finding the rate coefficients, these can be named 1a, 1b, 1c.
    # I think alternative 2) is the best.

    # What should influence the different rates
    # Forward translocation: #1 Malinen, #2 RNA-DNA/DNA-DNA, #3 length-dependent, #4 constant
    # Reverse translocation: #1 Malinen/Hein, #2 constant, #3 RNA-DNA/DNA-DNA
    # Nucleotide addition #1 Malinen, #2 constant
    # Entry into backtracked state: #1 constant (ala Patel), #2 length-dependent, #DNA-DNA (scrunch)
    # Rate of backtracking #1 constant, 2# RNA-DNA energy; 3# length-dependent (hybrid)

    model_1_G = CreateModel_1_Graph()
    debug()

    #model_2_G

    # Valid for all directed graphs with associated rates
    reaction_system, initial_values, parameters = GenerateInput(model_1_G)

    #solver = 'vode'  # 1s setup, 9.5s, 18s, 23s (200, 400, 500 time) solver
    #solver = 'radau'  # 15s setup, 0.26s solver
    solver = 'dopri'  # 5s setup, 0.1s solver, only 10% multiproc speedup

    model = BuildModel(reaction_system, parameters, initial_values, solver)
    trajectory1 = SolveModel(model)

    # When changing the model for parameter estimation, do through set method
    model.set(tdata=[0, 100])
    trajectory2 = SolveModel(model)

    CheckConservationOfMass(model_1_G, initial_values, trajectory1)
    CheckConservationOfMass(model_1_G, initial_values, trajectory2)

    #PlotTrajectoryModel_1(trajectory1)

if __name__ == '__main__':
    main()
