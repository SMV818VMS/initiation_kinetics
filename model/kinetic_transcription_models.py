from PyDSTool import args, Generator
from ipdb import set_trace as debug  # NOQA
from matplotlib.pyplot import ion  # NOQA
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import networkx as nx
import time
from datetime import datetime
import os
import numpy as np

# Non blocking graphics
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
    def __init__(self, variant='', use_AP=False, forwardtl=0.9, reversetl=0.1, nac=0.2,
                 abortive=0.2, backtrack=0.3, to_fl=0.7, backstep=0.3, to_open_complex=0.7):

        self.forwardtl = forwardtl
        self.reversetl = reversetl
        self.nac = nac
        self.to_fl = to_fl
        self.abortive = abortive
        self.backtrack = backtrack
        self.backstep = backstep
        self.to_open_complex = to_open_complex

        self.use_AP = use_AP  # use AP to calculate backstepping rate
        self.variant = variant

    def Forward(self):
        return self.forwardtl

    def Reverse(self):
        return self.reversetl

    def Backtrack(self):
        return self.backtrack

    def Backstep(self):
        return self.backstep

    def InstantAbort(self):
        return self.backstep

    def Nac(self):
        return self.nac

    def NTP_and_PPi(self):
        return self.nac

    def ToFL(self):
        return self.to_fl

    def Abortive(self):
        return self.abortive

    def FreeRNAPtoOpenComplex(self):
        return self.to_open_complex

    def FirstStep(self, starting_duplex_length):
        """
        The initial step from open complex formation to a conformation with a
        starting duplex length.

        Method: take a dubious average of the net forward/reverse constants plus the
        nucleotide addition constants, and the divide this by the number of
        steps that should be taken. This is a bit wonky, but it does not matter
        since it is sequence independent.
        """
        funky_rate_constant_average = (self.Forward() - self.Reverse() + self.Nac()) / 2.
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

    initial_conditions = {'RNAP_00': 1,
                          'RNAP_01': 0,
                          'RNAP_10': 0,
                          'RNAP_1al': 0,
                          'RNAP_11': 0,
                          'RNAP_20': 0,
                          'RNAP_201': 0,
                          'RNAP_2al': 0,
                          'RNAP_21': 0,
                          'RNAP_FL': 0
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
            'k_00_to_01': forward,
            'k_01_to_00': reverse,
            'k_01_to_10': nac,

            'k_10_to_11': forward,
            'k_10_to_1a': abortive,
            'k_11_to_10': reverse,
            'k_11_to_20': nac,

            'k_20_to_21': forward,
            'k_20_to_2a': abortive,
            'k_20_to_201': backtrack,
            'k_201_to_2a': abortive,
            'k_21_to_20': reverse,
            'k_21_to_FL': to_fl,
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

    system = {'RNAP_00': RNAP_00_rhs,
              'RNAP_01': RNAP_01_rhs,
              'RNAP_10': RNAP_10_rhs,
              'RNAP_11': RNAP_11_rhs,
              'RNAP_1al': RNAP_1al_rhs,
              'RNAP_20': RNAP_20_rhs,
              'RNAP_21': RNAP_21_rhs,
              'RNAP_201': RNAP_201_rhs,
              'RNAP_2al': RNAP_2al_rhs,
              'RNAP_FL': RNAP_FL_rhs
              }

    ### Specify the model
    DSargs = args()
    DSargs.name = 'Simple_transcription_and_escape'
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


def QualityCheckModel(backtrack_start, initial_rna_length, final_escape_RNA_length,
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


def NonTranslocationModel_Graph(setup, R, tag):
    """
    Even more simple model of transcription initiation.
    Contains:

    * NAC
    * Backstepping OR Abortive collapse
    * Promoter escape

    Code for initially transcribing RNAP: RNAP_XYZ

    X -> length of RNA [0,...,inf]
    Y -> 0 ; if abortive log == 'a' ; if backstep mode == 'b'
    Z -> 0 ; if abortive log == 'l' ; if backstep mode == 's'
    Promoter escape -> full length transcript state: RNAP_flt

    Examples of RNAP in different states:
    500 : len 5 duplex
    5al : proxy for aborted RNA of length 5
    flt : full length transcript
    """

    # Initialize a graph object
    G = nx.DiGraph(name=tag)

    allow_escape = setup['allow_escape']
    escape_start = setup['escape_start']
    backtrack_start = setup['backtrack_start']
    initial_rna_length = setup['initial_rna_length']
    final_escape_RNA_length = setup['final_escape_RNA_length']  # maximum length
    abortive_range = setup['abortive_range']
    backtrack_until_duplex_len = setup['backtrack_until_duplex_len']
    direct_backtrack = setup['direct_backtrack']
    backstep_mode = setup['backstep_mode']
    abortive_log = setup['abortive_log']

    if direct_backtrack == backstep_mode:
        print("Either direct backtrack or backstep model must be enabled for \
                the simple non-translocation model")
        1/0

    ## QA on setup
    QualityCheckModel(backtrack_start, initial_rna_length, final_escape_RNA_length,
                      abortive_range, backtrack_until_duplex_len, escape_start)

    # Template string for state variables
    state_template = 'RNAP_{0}{1}{2}'

    # Create the first reaction step -> from open complex to starting complex
    # Need this because abortive release takes RNAP back to the open complex,
    # but we may wish to model transcription from a 2nt RNA.
    open_complex = state_template.format(0, 0, 0)
    starting_complex = MakeStartingComplex(G, initial_rna_length, open_complex, R, state_template)
    G.add_node(starting_complex, initial_node=True)

    # Create the full length (FL) state
    if allow_escape:
        full_length = state_template.format('f', 'l', 't')

    # Following the growing RNA to explore all possible model states
    for rna_len in range(initial_rna_length, final_escape_RNA_length + 1):

        # This state
        this = state_template.format(rna_len, 0, 0)

        # If RNA length longer than 0, add nucleotide addition cycle
        if rna_len > 0:

            previous = state_template.format(rna_len - 1, 0, 0)
            nac_rc = R.Nac()
            AddReaction(G, nac_rc, previous, this)

        # If into the abortive range, set up instant abort
        if rna_len in abortive_range:

            if direct_backtrack:

                # In future, this should be AP
                direct_backtrack_rc = R.Backstep()
                AddReaction(G, direct_backtrack_rc, this, open_complex)

            elif backstep_mode:

                # Create the backstepped state
                bs = state_template.format(rna_len, 'b', 's')

                backstep_rc = R.Backstep()
                AddReaction(G, backstep_rc, this, bs)

                # Then do the abort step
                abort_rc = R.InstantAbort()
                AddReaction(G, abort_rc, bs, open_complex)

            # optional abortive log states
            if abortive_log:
                al = state_template.format(rna_len, 'a', 'l')
                AddReaction(G, direct_backtrack_rc, this, al)

        # if escape is allowed, can happen after escape start
        if allow_escape:
            if rna_len >= escape_start:
                escape_rc = R.ToFL()
                AddReaction(G, escape_rc, this, full_length)

    return G


def Model_2_Graph(setup, R, alternative='A'):
    """
    Most simple model of transcription initiation.
    For alternative A it contains:

    * Translocation / Nucleotide addition / PPi release
    * Backtracking / RNAP-collapse
    * Promoter escape

    Alternative B additionally models the concentration of free DNA and free
    RNAP. If realistic values can be obtained for RNAP -> promoter -> open
    complex, you may actually have a shot at modelling this perfectly. Feel free
    to use Michaelis Menten on this. However, if you cannot find good rate
    coefficients, just leave this behind and at least you have investigated it.
    After all, it is not rate-limiting on the N25 promoter, compared to others.

    Code for initially transcribing RNAP: RNAP_XYZ

    X -> length of RNA [0,...,inf]
    Y -> 0 ; if abortive log == 'a'
    Z -> 0 ; if abortive log == 'l'
    Promoter escape -> full length transcript state: RNAP_flt

    Examples of RNAP in different states:
    500 : len 5 duplex
    5al : proxy for aborted RNA of length 5
    flt : full length transcript

    Additional codes for alternative B

        Code for free RNAP: RNAP_fre
    """

    # Initialize a graph object
    G = nx.DiGraph()

    allow_escape = setup['allow_escape']
    escape_start = setup['escape_start']
    backtrack_start = setup['backtrack_start']
    initial_rna_length = setup['initial_rna_length']
    final_escape_RNA_length = setup['final_escape_RNA_length']  # maximum length
    abortive_range = setup['abortive_range']
    backtrack_until_duplex_len = setup['backtrack_until_duplex_len']

    ## QA on setup
    QualityCheckModel(backtrack_start, initial_rna_length, final_escape_RNA_length,
                      abortive_range, backtrack_until_duplex_len, escape_start)

    # Template string for state variables
    state_template = 'RNAP_{0}{1}{2}'

    # Create the first reaction step -> from open complex to starting complex
    # Need this because abortive release takes RNAP back to the open complex,
    # but we may wish to model transcription from a 2nt RNA.
    open_complex = state_template.format(0, 0, 0)
    starting_complex = MakeStartingComplex(G, initial_rna_length, open_complex, R, state_template)

    # Tag the starting complex -- and create it, if not already made
    if alternative == 'A':
        G.add_node(starting_complex, initial_node=True)
    else:
        G.add_node(starting_complex)

    if alternative == 'B':
        # Add free RNAP
        free_RNAP = state_template.format('f', 'r', 'e')
        G.add_node(free_RNAP, initial_node=True)

        # Free RNAP to open complex
        AddReaction(G, R.FreeRNAPtoOpenComplex, free_RNAP, starting_complex)

    # Create the full length (FL) state
    if allow_escape:
        full_length = state_template.format('f', 'l', 't')

    # Following the growing RNA to explore all possible model states
    for rna_len in range(initial_rna_length, final_escape_RNA_length + 1):

        # This state
        this = state_template.format(rna_len, 0, 0)

        # If RNA length longer than 0, add nucleotide addition cycle
        if rna_len > 0:

            previous = state_template.format(rna_len - 1, 0, 0)
            nac_rc = R.Nac()
            AddReaction(G, nac_rc, previous, this)

        # If into the abortive range, set up instant abort
        if rna_len in abortive_range:

            direct_backtrack_rc = R.InstantAbort()
            AddReaction(G, direct_backtrack_rc, this, open_complex)

            # remember the abortive log
            al = state_template.format(rna_len, 'a', 'l')
            AddReaction(G, direct_backtrack_rc, this, al)

        # if escape is allowed, can happen after escape start
        if allow_escape:
            if rna_len >= escape_start:
                escape_rc = R.ToFL()
                AddReaction(G, escape_rc, this, full_length)

                if alternative == 'B':
                    AddReaction(G, escape_rc, this, free_RNAP)

    return G


def MakeStartingComplex(G, initial_rna_length, open_complex, R, state_template):
    """
    Starting complex may be open complex, or may for example be from a 2nt RNA
    """

    if initial_rna_length > 0:
        starting_complex = state_template.format(initial_rna_length, 0, 0)
        first_step = R.FirstStep(initial_rna_length)
        G.add_edge(open_complex, starting_complex, rate_constant=first_step)
    else:
        starting_complex = open_complex

    return starting_complex


def Model_1_Graph(R, setup):
    """

    Most general model of transcription initiation. Contains:

    * Nucleotide addition / PPi release
    * Forward translocation
    * Reverse translocation
    * Backtracking from pre-translocated state
    * Backtracking from each backtracked state
    * Promoter escape

    Code for initially transcribing RNAP: RNAP_XYZ

    X -> length of RNA [0,...,inf]
    Y -> transcloation state if [0,1] ; if abortive log == 'a'
    Z -> nr_backtracked steps [0,...,inf]; if abortive log == 'l'
    Promoter escape -> full length transcript state: RNAP_flt

    Examples of RNAP in different states:
    500 : len 5 duplex in the pre-translocated state
    510 : len 5 duplex in the post-translocated state
    503 : len 2 duplex backtracked 3 steps
    5al : len 5 RNA aborted log -- total aborted RNAP / proxy for aborted RNA of length 5
    flt : full length transcript

    Assuming backtracking and direct abortive release happens from the pretranslocated state.
    """
    # Initialize a graph object
    G = nx.DiGraph()

    # Rate constants
    R = RateConstants()

    # read setup
    allow_escape = setup['allow_escape']
    escape_start = setup['escape_start']
    backtrack_start = setup['backtrack_start']
    initial_rna_length = setup['initial_rna_length']
    final_escape_RNA_length = setup['final_escape_RNA_length']  # maximum length
    abortive_range = setup['abortive_range']
    backtrack_until_duplex_len = setup['backtrack_until_duplex_len']

    # QA on setup
    QualityCheckModel(backtrack_start, initial_rna_length, final_escape_RNA_length,
                      abortive_range, backtrack_until_duplex_len, escape_start)

    # Template string for state variables
    state_template = 'RNAP_{0}{1}{2}'

    # Create the first reaction step -> from open complex to starting complex
    open_complex = state_template.format(0, 0, 0)
    starting_complex = MakeStartingComplex(G, initial_rna_length, open_complex, R, state_template)

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

        # If RNA length longer than 0, add NTP incorporation + PPi release steps
        if rna_len > 0:

            previous_post_translocated = state_template.format(rna_len - 1, 1, 0)
            nac_rc = R.NTP_and_PPi()
            AddReaction(G, nac_rc, previous_post_translocated, pre_translocated)

        # Add abortive log state if grown into the abortive range
        if rna_len >= abortive_range[0]:
            al = state_template.format(rna_len, 'a', 'l')
            AddAbortiveLogNode(G, al)

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
                    last_backtracked = state_template.format(rna_len, 0, nr_backtracked_steps - 1)
                    this_backtrack_rc = R.Backtrack()
                    AddReaction(G, this_backtrack_rc, last_backtracked, this_backtracked),

                # check if we have backtracked into abortive-territory!
                if DuplexLength(rna_len, nr_backtracked_steps) in abortive_range:
                    abortive_rc = R.Abortive()
                    AddReaction(G, abortive_rc, this_backtracked, open_complex)
                    # add log
                    AddReaction(G, abortive_rc, this_backtracked, al)

        # direct abortive release
        if rna_len in abortive_range:
            abortive_rc = R.Abortive()
            AddReaction(G, abortive_rc, pre_translocated, open_complex)
            # add log
            AddReaction(G, abortive_rc, pre_translocated, al)

        if allow_escape:
            if rna_len >= escape_start:
                escape_rc = R.ToFL()
                AddReaction(G, escape_rc, post_translocated, full_length)

    return G


def ReactionSystemSetup():
    """
    Setup for the reaction system.

    Sets up the stoichiometry of the system. If modified, must re-create model, thus re-compile binary
    when using C-solvers.
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
    setup['abortive_range'] = range(3, 5+1)

    # minimal duplex length for backtracking until
    # For example, backtracking to a length 1 bp hybrid may be implausible
    setup['backtrack_until_duplex_len'] = 2

    # Direct backtrack: when in "abortive_range", return straight to start
    # NOTE: this disables the "backtrack_start" and "backtrack_until_duplex_len" settings.
    setup['direct_backtrack'] = True

    # Backstep mode: when in "abortive_range", go to a backstepped state. From the
    # backstepped state, return straight to start.
    # NOTE: this disables the "backtrack_start" and "backtrack_until_duplex_len" settings.
    setup['backstep_mode'] = False

    # For some setups, it is useful to keep an "abortive log" state which
    # accounts for aborted RNA
    setup['abortive_log'] = False

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


def GetParameters(G):
    """
    Parse graph to link parameter name with parameter value.

    Requires: directd graph G with attributes rate_constant_name and rate_constant_value.
    """

    parameters = {}
    for from_node, to_node, data in G.edges(data=True):
        parameters[data['rate_constant_name']] = data['rate_constant_value']

    return parameters


def CalcNTP(trajectory, starting_amount=30):
    """
    Calculate NTP consumption. Subtract this from starting_amount.
    """
    pts = trajectory.sample()
    nr_timesteps = len(pts['t'])
    nr_ntp = np.zeros(nr_timesteps)
    for state_name in pts.keys():

        # add nucleotides from the abortive log states
        if state_name.endswith('al'):
            code = state_name.split('_')[1]
            abortive_length = int(code[0])
            nr_ntp = nr_ntp + pts[state_name] * abortive_length

        # add nucleotides from full length
        transcript_length = 74
        if state_name.endswith('flt'):
            nr_ntp = nr_ntp + pts[state_name] * transcript_length

    nr_ntp = starting_amount - nr_ntp

    return nr_ntp


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


def WriteGraphAndSetup(G, reaction_setup, tag, alternative):
    """
    Write to file an image of the graph and the corresponding setup.
    """

    fig, ax = plt.subplots()

    nx.draw_graphviz(G, with_labels=True, ax=ax)
    # filename is number of translocation steps followed by a time tag
    time_tag = str(datetime.now()).split('.')[0]
    time_tag = time_tag.replace(':', '-')
    store_dir = 'Log'
    if not os.path.isdir(store_dir):
        os.makedirs(store_dir)

    if 'final_escape_RNA_length' in reaction_setup:
        beg = str(reaction_setup['final_escape_RNA_length'])
    else:
        beg = '0'

    id_text = tag + alternative + '_' + beg + '_' + time_tag
    image_path = os.path.join(store_dir, id_text + '.png')
    fig.set_size_inches(17, 8)
    fig.savefig(image_path)

    text_path = os.path.join(store_dir, id_text + '.txt')
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
    final_values = {node: pts[node][-1] for node in G.nodes()}

    total_initial = sum(initial_values.values())
    # final sum, excluding log states
    total_final = sum([val for n, val in final_values.items() if n not in skip_us])

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

    I see that programatically this is a weird function. Why not just call
    each function below separtely? in the main program?
    """

    # get reaction system, initial conditions, and parameter values
    reaction_system = CreateReactionSystem(G)
    initial_values = SetInitialValues(G)
    parameters = GetParameters(G)
    reactions = GetReactions(G)

    return reaction_system, reactions, initial_values, parameters


def GetReactions(G):
    """
    For the .psc format you want something like this:

    R1:
        S1 > S2 (description)
        k * S1  (rate)
    """

    reactions = {}

    i = 1
    for from_node, to_node, data in G.edges(data=True):

        name = 'R{0}'.format(i)
        i += 1

        description = from_node + ' > ' + to_node

        # Rate constant
        rc = data['rate_constant_name']
        # Rate
        rate = '*'.join([str(rc), from_node])

        reactions[name] = {'description': description, 'rate': rate}

    return reactions


def CreateModel_1_Graph():
    """
    Setup and create a graph for Model_1
    """

    # Stoichiometry setup
    reaction_setup = ReactionSystemSetup()

    # Create the graph
    G = Model_1_Graph(reaction_setup)

    #Plot model graph and write setup log
    WriteGraphAndSetup(G, reaction_setup, tag='model_1')

    print("Number of model states: {0}".format(len(G.nodes())))

    return G


def CreateModel_3_Graph(R, tag, setup):
    """
    This is a model without translocation and without free RNAP.
    """

    # Create graph
    G = NonTranslocationModel_Graph(setup, R, tag)

    #Plot model graph and write setup log
    WriteGraphAndSetup(G, R, setup, tag, alternative='C')

    return G


def CreateModel_2_Graph(alternative='A'):
    """
    How do I avoid copy-waste code?

    Requirements:
    * provide some model-specific setups
    * provide RateConstants
    * Write the graph

    What are the commonalities? I don't see how to avoid copy waste.

    So far it looks as if you can use the same reaction_setup. However, I think that you will finally
    want to experiment with different setups for different models. Take that when it comes.
    """

    # Setup
    reaction_setup = ReactionSystemSetup()

    # Get Rate constants
    R = RateConstants()

    # Create graph
    G = Model_2_Graph(reaction_setup, R, alternative=alternative)

    WriteGraphAndSetup(G, reaction_setup, tag='model_2', alternative=alternative)

    return G


def write_psc(reactions, initial_values, parameters, graph_name):
    """
    Ecample of .psc format:

        ---------------------
        # Initial transcription with abortive cycling

        # Reactions
        R1:
            OC > nt3
            k1 * OC

        R2:
            nt3 > $null
            k2 * nt3

        # Variable species
        OC = 10  # single molecule experiment yay!
        nt3 = 0

        # Parameters
        k1 = 0.2
        k2 = 0.4
        ---------------------
    """

    lines = []
    lines.append('# ' + graph_name)
    lines.append('')

    lines.append('# Reactions')
    for reacation_name, reaction_info in sorted(reactions.iteritems()):
        description = reaction_info['description']
        rate = reaction_info['rate']
        lines.append(reacation_name + ':')
        lines.append('\t' + description)
        lines.append('\t' + rate)
        lines.append('')

    lines.append('# Variable species')
    for name, value in initial_values.items():
        lines.append(name + ' = ' + str(value))
    lines.append('')

    lines.append('# Parameters')
    for name, value in parameters.items():
        lines.append(name + ' = ' + str(value))

    # Create output directory
    write_dir = 'psc_files'
    if not os.path.isdir(write_dir):
        os.makedirs(write_dir)

    # Write to file
    filepath = os.path.join(write_dir, graph_name + '.psc')
    handle = open(filepath, 'wb')
    for line in lines:
        handle.write(line + os.linesep)
    handle.close()


def SolveODE(reaction_system, parameters, initial_values, model_graph):

    solver = 'vode'  # 1s setup, 9.5s, 18s, 23s (200, 400, 500 time) solver
    solver = 'radau'  # 15s setup, 0.26s solver # does not work on laptop...
    solver = 'dopri'  # 5s setup, 0.1s solver, only 10% multiproc speedup

    pdt_model = BuildModel(reaction_system, parameters, initial_values, solver)
    trajectory = SolveModel(pdt_model)

    #When changing the model for parameter estimation, do through set method
    pdt_model.set(tdata=[0, 100])

    CheckConservationOfMass(model_graph, initial_values, trajectory)

    #This NTP calculation is actually unrealistic: in reality, the reactions
    #would have decreased with decreasing [NTP]: here you are assuming Vmax the
    #whole way.
    #NTP = CalcNTP(trajectory, starting_amount=30)

    PlotTrajectoryModel(trajectory)


def main():

    # What should influence the different rates
    # Forward translocation: #1 Malinen, #2 RNA-DNA/DNA-DNA, #3 length-dependent, #4 constant
    # Reverse translocation: #1 Malinen/Hein, #2 constant, #3 RNA-DNA/DNA-DNA
    # Nucleotide addition #1 Malinen, #2 constant
    # Entry into backtracked state: #1 constant (ala Patel), #2 length-dependent, #DNA-DNA (scrunch)
    # Rate of backtracking #1 constant, 2# RNA-DNA energy; 3# length-dependent (hybrid)

    # OR ... the AP :)

    #model_graph = CreateModel_1_Graph()
    #model_graph = CreateModel_2_Graph(alternative='A')
    #model_graph = CreateModel_2_Graph(alternative='B')

    # Sorry, I made a model 3 :P This is THE most simple model, without
    # translocation, without free RNAP.

    # Setup
    setup = ReactionSystemSetup()

    # Promoter selection
    promoter_variant = 'N25'

    # Get Rate constants for this promoter variant
    R = RateConstants(variant=promoter_variant)

    model_graph = CreateModel_3_Graph(R, 'N25_simple', setup)

    # Parse the graph to make some nice output
    reaction_system, reactions, initial_values, parameters = GenerateInput(model_graph)

    # Write a .psc file for the system
    write_psc(reactions, initial_values, parameters, model_graph.name)

    # Solve ODE sysstem XXX This can be separated out
    #SolveODE(reaction_system, parameters, initial_values, model_graph)


if __name__ == '__main__':
    main()
