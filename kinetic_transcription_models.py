from PyDSTool import args, Generator
from ipdb import set_trace as debug  # NOQA
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import networkx as nx


class RateConstants(object):
    """
    Class for providing and calculating rate constants
    """

    # Rates
    forward = 0.9
    reverse = 0.1
    nac = 0.2
    to_fl = 0.7
    abortive_from_10 = 0.3
    abortive_from_20 = 0.2
    backtrack_from_20_to_201 = 0.3
    abortive_from_201 = 0.3

    # special rate
    abortive_restart = 0.1

    # default all values to nan to avoid missing obvious mistakes
    def __init__(self, forwardtl=float('nan'), reversetl=float('nan'),
                       nac=float('nan'), to_fl=float('nan'),
                       abortive=float('nan'), backtrack=float('nan'),
                       abortive_restart=float('nan'), escape=float('nan')):

        self.forwardtl = forwardtl
        self.reversetl = reversetl
        self.nac = nac
        self.to_fl = to_fl
        self.abortive = abortive
        self.backtrack = backtrack

    def Forward(self):
        return self.forwardtl

    def Reverse(self):
        return self.reversetl

    def Backtrack(self):
        return self.backtrack

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


def basic_transcription():
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
    abortive_from_10 = 0.3
    abortive_from_20 = 0.2
    backtrack_from_20_to_201 = 0.3
    abortive_from_201 = 0.3

    # These give the releation between the states
    # All possible relevant state information should be obtainable from the
    # parameter name: length of RNA, translocation state, and backtracked state.
    parameters = {
            'k_00_to_01':forward,
            'k_01_to_00':reverse,
            'k_01_to_10':nac,

            'k_10_to_11':forward,
            'k_10_to_1a':abortive_from_10,
            'k_11_to_10':reverse,
            'k_11_to_20':nac,

            'k_20_to_21':forward,
            'k_20_to_2a':abortive_from_20,
            'k_20_to_201':backtrack_from_20_to_201,
            'k_201_to_2a':abortive_from_201,
            'k_21_to_20':reverse,
            'k_21_to_FL':to_fl,
                  }

    # Reaction system
    RNAP_00_rhs = 'k_01_to_00*RNAP_01 - k_00_to_01*RNAP_00 '\
                  '+ k_10_to_1a*RNAP_10 '\
                  '+ k_20_to_2a*RNAP_20 '\
                  '+ k_201_to_2a*RNAP_20 '

    RNAP_01_rhs = '-k_01_to_00*RNAP_01 + k_00_to_01*RNAP_00 - k_01_to_10*RNAP_01'

    RNAP_10_rhs = 'k_11_to_10*RNAP_11 - k_10_to_11*RNAP_10 + k_01_to_10*RNAP_01 '\
                  '-k_10_to_1a*RNAP_10'

    RNAP_11_rhs = '-k_11_to_10*RNAP_11 + k_10_to_11*RNAP_10 - k_11_to_10*RNAP_11'

    RNAP_1al_rhs = 'k_10_to_1a*RNAP_10'

    RNAP_20_rhs = 'k_21_to_20*RNAP_21 - k_20_to_21*RNAP_20 + k_11_to_10*RNAP_11 '\
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
    DSargs.tdata = [0, 450]
    DSargs.varspecs = system

    # Create the solver object
    DS = Generator.Vode_ODEsystem(DSargs)

    traj = DS.compute('demo')
    pts = traj.sample()

    plt.plot(pts['t'], pts['RNAP_00'], label='RNAP_00')
    #plt.plot(pts['t'], pts['RNAP_01'], label='RNAP_01')
    #plt.plot(pts['t'], pts['RNAP_10'], label='RNAP_10')
    #plt.plot(pts['t'], pts['RNAP_20'], label='RNAP_20')
    plt.plot(pts['t'], pts['RNAP_1al'], label='RNAP_1a')
    plt.plot(pts['t'], pts['RNAP_2al'], label='RNAP_2a')
    plt.plot(pts['t'], pts['RNAP_FL'], label='RNAP_FL')
    plt.legend()
    plt.xlabel('t')


def basic_transcription_scripted():
    """
    Script up an initial transcription event. Save the resulting reaction
    network as a graph with networkx.

    Code for initially transcribing RNAP: RNAP_XYZ

    X -> length of RNA [0,...,inf]
    Y -> transcloation state if [0,1] ; if abortive log == 'a'
    Z -> nr_backtracked steps [0,...,inf]; if abortive log == 'l'

    Examples of RNAP in different states:
    500 : len 5 duplex in the pre-translocated state
    510 : len 5 duplex in the post-translocated state
    503 : len 2 duplex backtracked 3 steps
    5al : len 5 RNA aborted log -- total aborted RNAP / proxy for aborted RNA of length 5
    flt  : full length transcript

    Assuming backtracking and direct abortive release happens from the
    pretranslocated state
    """

    # Template string for state variables
    state_template = 'RNAP_{0}{1}{2}'

    # RNA length where backtracking may start
    backtrack_start = 2
    # starting RNA length
    initial_rna_length = 0
    # escape RNA-length
    escape_RNA_length = 2
    # RNA-DNA duplex abortive range: abortive release may happen here, either by
    # backtracking to this duplex length or directly from this duplex length
    abort_RNA_range = (1, 5)
    # minimal duplex length for backtracking -- may be useful for reducing the
    # number of states. For example, backtracking to a length 1 bp hybrid may
    # have such a low probability that it is not worth considering making 19
    # states just to represent this option. An alternative way of not including
    # this state is by setting the rate constant to zero for these states and
    # testing for zero valued rate constants.
    backtrack_stop = 1

    # Initialize a rates object to calculate reaction rates for a given
    # translocation state
    R = RateConstants()

    # Initialize a graph object
    G = nx.DiGraph()

    # Create the first reaction step -> from open complex to starting complex
    if initial_rna_length > 0:
        open_complex = state_template.format(0, 0, 0)
        starting_complex = state_template.format(initial_rna_length, 0, 0)
        first_step = R.FirstStep(initial_rna_length)
        G.add_edge(open_complex, starting_complex, rate_constant=first_step)

    for rna_len in range(initial_rna_length, escape_RNA_length + 1):

        # Define the different possible states associatd with an RNA of this length
        pre_translocated = state_template.format(rna_len, 0, 0)
        post_translocated = state_template.format(rna_len, 1, 0)

        forward = R.Forward()
        reverse = R.Reverse()

        # Add translocation states by adding rate coefficients to the graph
        G.add_edge(pre_translocated, post_translocated, rate_constant=forward)
        G.add_edge(post_translocated, pre_translocated, rate_constant=reverse)

        # If RNA length longer than 0, add nucleotide addition cycle
        if rna_len > 0:
            nac = R.Nac()
            previous_post_translocated = state_template.format(rna_len-1, 1, 0)
            G.add_edge(previous_post_translocated, pre_translocated, rate_constant=nac)

        # Add the abortive log state corresponding to this RNA length
        abortive_log = state_template.format(rna_len, 'a', 'l')
        G.add_node(abortive_log)

        # backtrack until backtrack stop
        if rna_len >= backtrack_start:
            for nr_backtrack_steps in range(1, rna_len - backtrack_stop + 1):
                this_backtracked_state = state_template.format(rna_len, 0,
                        nr_backtrack_steps)

                # if first step, go from pre-translocated
                if nr_backtrack_steps == 1:
                    first_backtrack = R.Backtrack()
                    G.add_edge(pre_translocated, this_backtracked_state,
                            rate_constant=first_backtrack)
                # after, go from previous backtracked state
                else:
                    previous_backtracked_state = state_template.format(rna_len,
                            0, nr_backtrack_steps-1)
                    this_backtrack = R.Backtrack()
                    G.add_edge(previous_backtracked_state,
                            this_backtracked_state,
                            rate_constant=this_backtrack)

                # check if we have backtracked into abortive-territory!
                if DuplexLength(rna_len, nr_backtrack_steps) in abort_RNA_range:

                    abortive = R.Abortive()
                    open_complex = state_template.format(0, 0, 0)
                    G.add_edge(this_backtracked_state, open_complex, rate_constant=abortive)
                    # Point to the abortive log state too, to keep tabs
                    G.add_edge(this_backtracked_state, abortive_log, rate_constant=abortive)

        # direct abortive release -- if within direct abortive range!
        if rna_len in abort_RNA_range:
            abortive = R.Abortive()
            open_complex = state_template.format(0, 0, 0)
            G.add_edge(pre_translocated, open_complex, rate_constant=abortive)
            # Point to the abortive log state too, to keep tabs
            G.add_edge(pre_translocated, abortive_log, rate_constant=abortive)

        # last state that should be made
        if rna_len == escape_RNA_length:
            full_length = state_template.format('f', 'l', 't')
            escape = R.ToFL()
            G.add_edge(post_translocated, full_length, rate_constant=escape)
            break

    nx.draw(G, with_labels=True)
    debug()


def main():

    # 0) Hard-code the system
    # Basic, hard-coded transcription initiation.
    #basic_transcription()

    # 1) Script up the system
    # Basic, scripted transcription initiation
    basic_transcription_scripted()


if __name__ == '__main__':
    main()
