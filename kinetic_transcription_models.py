from PyDSTool import args, Generator
#from ipdb import set_trace as debug
import matplotlib.pyplot as plt
plt.style.use('ggplot')


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

    basic_transcription()


if __name__ == '__main__':
    main()
