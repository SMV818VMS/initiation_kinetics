from PyDSTool import args, Generator
#from ipdb import set_trace as debug
import matplotlib.pyplot as plt
import matplotlib as mpl

import brewer2mpl
# Use brewer colors instead of default ones in matplotlib
bmap = brewer2mpl.get_map('Set2', 'qualitative', 7)
mpl.rcParams['axes.color_cycle'] = bmap.mpl_colors


def tutorial_linear():

    ### Setup initial conditions, parameters, and dynamical system description ###
    initial_conditions = {'x':1, 'y':0.3}
    parameters = {'k':0.1, 'm':0.5}

    x_rhs = 'y'
    y_rhs = '-k*x/m'

    system = {'x': x_rhs, 'y': y_rhs}

    ### Specify the model
    DSargs = args()
    DSargs.name ='Simple_harmonic_motion'
    DSargs.ics = initial_conditions
    DSargs.pars = parameters
    DSargs.tdata = [0, 20]
    DSargs.varspecs = system

    # Create the solver object
    DS = Generator.Vode_ODEsystem(DSargs)

    # Can change settings afterwards using "set"
    DS.set(pars={'k': 0.1}, ics={'x': 0.8})

    traj = DS.compute('demo')
    pts = traj.sample()

    plt.plot(pts['t'], pts['x'], label='x')
    plt.plot(pts['t'], pts['y'], label='y')
    plt.legend()
    plt.xlabel('t')


def basic_transcription():

    initial_conditions = {'RNAP_00':1,
                          'RNAP_01':0,
                          'RNAP_10':0,
                          'RNAP_11':0,
                          'RNAP_20':0,
                          'RNAP_21':0,
                          'RNAP_FL':0
                          }

    forward = 0.9
    reverse = 0.1
    nac = 0.2
    to_fl = 0.4
    parameters = {
            'k_00_to_01':forward,
            'k_01_to_00':reverse,
            'k_01_to_10':nac,

            'k_10_to_11':forward,
            'k_11_to_10':reverse,
            'k_11_to_20':nac,

            'k_20_to_21':forward,
            'k_21_to_20':reverse,
            'k_21_to_FL':to_fl,
                  }

    RNAP_00_rhs = 'k_01_to_00  * RNAP_01 - k_00_to_01 * RNAP_00'
    RNAP_01_rhs = '-k_01_to_00 * RNAP_01 + k_00_to_01 * RNAP_00 - k_01_to_10 * RNAP_01'

    RNAP_10_rhs = 'k_11_to_10  * RNAP_11 - k_10_to_11 * RNAP_10 + k_01_to_10  * RNAP_01'
    RNAP_11_rhs = '-k_11_to_10 * RNAP_11 + k_10_to_11 * RNAP_10 - k_11_to_10 * RNAP_11'

    RNAP_20_rhs = 'k_21_to_20  * RNAP_21 - k_20_to_21 * RNAP_20 + k_11_to_10  * RNAP_11'
    RNAP_21_rhs = '-k_21_to_20 * RNAP_21 + k_20_to_21 * RNAP_20 - k_21_to_20 * RNAP_21'

    RNAP_FL_rhs = 'k_21_to_FL  * RNAP_21'

    system = {'RNAP_00':RNAP_00_rhs,
              'RNAP_01':RNAP_01_rhs,
              'RNAP_10':RNAP_10_rhs,
              'RNAP_11':RNAP_11_rhs,
              'RNAP_20':RNAP_20_rhs,
              'RNAP_21':RNAP_21_rhs,
              'RNAP_FL':RNAP_FL_rhs
              }

    ### Specify the model
    DSargs = args()
    DSargs.name ='Simple_transcription_and_escape'
    DSargs.ics = initial_conditions
    DSargs.pars = parameters
    DSargs.tdata = [0, 20]
    DSargs.varspecs = system

    # Create the solver object
    DS = Generator.Vode_ODEsystem(DSargs)

    traj = DS.compute('demo')
    pts = traj.sample()

    plt.plot(pts['t'], pts['RNAP_00'], label='RNAP_00')
    plt.plot(pts['t'], pts['RNAP_01'], label='RNAP_01')
    plt.plot(pts['t'], pts['RNAP_10'], label='RNAP_10')
    plt.plot(pts['t'], pts['RNAP_20'], label='RNAP_20')
    plt.plot(pts['t'], pts['RNAP_FL'], label='RNAP_FL')
    plt.legend()
    plt.xlabel('t')


def main():

    #tutorial_linear()
    #basic_transcription()

    #transcription_with_backtracking_and_abortive_states

if __name__ == '__main__':
    main()
