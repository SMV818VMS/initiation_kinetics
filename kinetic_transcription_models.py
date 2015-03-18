from PyDSTool import args, Generator
from ipdb import set_trace as debug  # NOQA
import matplotlib.pyplot as plt
plt.style.use('ggplot')


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
                          'RNAP_1a':0,
                          'RNAP_1al':0,
                          'RNAP_11':0,
                          'RNAP_20':0,
                          'RNAP_201':0,
                          'RNAP_2a':0,
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
    backtrack_from_20_to_201 = 0.5
    abortive_from_201 = 0.3

    # special rate
    abortive_restart = 0.1

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
            'k_20_to_201':backtrack_from_20_to_201,
            'k_201_to_2a':abortive_from_201,
            'k_21_to_20':reverse,
            'k_21_to_FL':to_fl,

            'ab_restart': abortive_restart
                  }

    # Reaction system
    RNAP_00_rhs = 'k_01_to_00*RNAP_01 - k_00_to_01*RNAP_00 '\
                  '+ ab_restart*RNAP_1a '\
                  '+ ab_restart*RNAP_2a'

    RNAP_01_rhs = '-k_01_to_00*RNAP_01 + k_00_to_01*RNAP_00 - k_01_to_10*RNAP_01'

    RNAP_10_rhs = 'k_11_to_10*RNAP_11 - k_10_to_11*RNAP_10 + k_01_to_10*RNAP_01 '\
                  '-k_10_to_1a*RNAP_10'

    RNAP_11_rhs = '-k_11_to_10*RNAP_11 + k_10_to_11*RNAP_10 - k_11_to_10*RNAP_11'

    RNAP_1a_rhs = 'k_10_to_1a*RNAP_10 - ab_restart*RNAP_1a'
    RNAP_1al_rhs = 'k_10_to_1a*RNAP_10'

    RNAP_20_rhs = 'k_21_to_20*RNAP_21 - k_20_to_21*RNAP_20 + k_11_to_10*RNAP_11 '\
                  '-k_20_to_201*RNAP_20'

    RNAP_21_rhs = '-k_21_to_20*RNAP_21 + k_20_to_21*RNAP_20 - k_21_to_FL*RNAP_21'

    RNAP_201_rhs = 'k_20_to_201*RNAP_20 - k_201_to_2a*RNAP_201'

    RNAP_2a_rhs = 'k_201_to_2a*RNAP_201 - ab_restart*RNAP_2a'
    RNAP_2al_rhs = 'k_201_to_2a*RNAP_201'

    RNAP_FL_rhs = 'k_21_to_FL*RNAP_21'

    #print RNAP_00_rhs
    #debug()

    system = {'RNAP_00':RNAP_00_rhs,
              'RNAP_01':RNAP_01_rhs,
              'RNAP_10':RNAP_10_rhs,
              'RNAP_11':RNAP_11_rhs,
              'RNAP_1a':RNAP_1a_rhs,
              'RNAP_1al':RNAP_1al_rhs,
              'RNAP_20':RNAP_20_rhs,
              'RNAP_21':RNAP_21_rhs,
              'RNAP_201':RNAP_201_rhs,
              'RNAP_2a':RNAP_2a_rhs,
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


#def basic_transcription_scripted():

    ## number of translocation events
    #nr_transcription_steps = 3

    ## starting RNA length
    #initial_rna_length = 2


def main():

    # Basic, hard-coded transcription initiation.
    basic_transcription()

    # Basict, scripted transcription initiation
    #basic_transcription_scripted()


if __name__ == '__main__':
    main()
