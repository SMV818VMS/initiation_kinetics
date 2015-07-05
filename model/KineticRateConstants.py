# At first, keep it simple. Add the path to where the ITS and datahandler
# libraries are.
import sys
sys.path.append('/home/jorgsk/Dropbox/phdproject/hsuVitro')
import data_handler
from ipdb import set_trace as debug  # NOQA


class RateConstants(object):
    """
    Class for providing and calculating rate constants

    For now, this class only returns constants for the rate constants. In the
    future, it will use duplex length, rna length, number of backtracked steps,
    etc, to calculate the various rates. A lot of the actual research is going
    to happen on these rates, so the class must be flexible and powerful.

    We assume 10bp/s to be the standard rate. That means that if nac=10/s and
    AP is 40%, then backstep is 4/s. Will this allow a promoter escape within
    1 second? I think so.
    """

    def __init__(self, variant='', use_AP=False, pos_dep_abortive=False, forwardtl=0.9, reversetl=0.1, nac=0.2,
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
        self.position_dependent_abortive_rc = pos_dep_abortive
        self.variant = variant

        if variant != '':
            # Read data from the DG100 library
            dg100 = data_handler.ReadData('dg100-new')
            self.itso = [i for i in dg100 if i.name == variant][0]
            # Recall: index 0 corresponds to 2nt RNA
            self.abortive_prob = self.itso.abortiveProb
            # interesting: there is a very low amount of AP for positions 12
            # 13, and 14, indicating that most RNAP that reach here manage to
            # escape. This also shows that escape may take place later also
            # for N25!

    def _GetAP(self, rna_length):
        if rna_length < 2:
            print("No AP for RNA < 2nt")
            1/0
        ap = self.abortive_prob[rna_length-2]

        if ap == 0.0:
            print("Warning: AP is 0 at position {0} for {1}".format(rna_length, self.variant))

        return ap

    def _GetNAC(self, rna_length):
        """
        Nac is assumed to be 10bp/s at each position
        """
        return self.nac

    def Forward(self):
        return self.forwardtl

    def Reverse(self):
        return self.reversetl

    def Backtrack(self):
        return self.backtrack

    def Backstep(self, rna_length=-1):

        # Use abortive probability
        if self.use_AP:
            if rna_length == -1:
                print("Specify an RNA length 1/0")
                1/0
            else:

                ap = self._GetAP(rna_length)

                if 0 < ap < 1:
                    # If nac rate coefficient is 10/s and AP is 30% at this point,
                    # then backstep rate coefficient is 3/s
                    return ap * self._GetNAC(rna_length)
                else:
                    print("AP not number between 0 and 1")
        else:
            backstep = self.backstep

        return backstep

    def InstantAbort(self, rna_length):
        """
        If NAC is 10 bp/s, then abortive release is 30 bp/s? x3 must
        qualify as fast.  """
        return self.nac * 3

    def Nac(self):
        if self.use_AP:
            # for now, Nac is not rna_length dependent
            nac = self._GetNAC(rna_length=-1)
        else:
            nac = self.backstep

        return nac

    def NTP_and_PPi(self):
        return self.nac

    def ToFL(self):
        return self.to_fl

    def Abortive(self):
        if self.position_dependent_abortive_rc:
            abortive = 4
        else:
            abortive = self.abortive

        return abortive

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

