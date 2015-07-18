# At first, keep it simple. Add the path to where the ITS and datahandler
# libraries are.
import sys
sys.path.append('/home/jorgsk/Dropbox/phdproject/transcription_initiation/data')
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
    Assume 10bp/s from around +15 to + 67, about 50bt -> 10nt/s / 50nt 1/5 = 0.2
    """

    def __init__(self, variant='', use_AP=False, pos_dep_abortive=False,
                 forwardtl=0.9, reversetl=0.1, nac=0.1, abortive=False,
                 backtrack=0.3, to_fl=0.2, backstep=0.3, to_open_complex=0.7,
                 dataset=False, custom_AP=False, GreB_AP=False, escape=5):

        self.forwardtl = forwardtl
        self.reversetl = reversetl
        self.nac = nac
        self.to_fl = to_fl
        self.abortive = abortive
        self.backtrack = backtrack
        self.backstep = backstep
        self.to_open_complex = to_open_complex
        self.escape = escape

        self.use_AP = use_AP  # use AP to calculate backstepping rate
        self.position_dependent_abortive_rc = pos_dep_abortive
        self.variant = variant

        self.custom_productive_AP = custom_AP
        self.GreB_AP = GreB_AP

        if variant != '':
            # Read data from the DG100 library
            # Optionally supply this directly to avoid reading from disk
            # excessively
            if dataset is False:
                dset = data_handler.ReadData('dg100-new')
            else:
                dset = dataset
            its = [i for i in dset if i.name == variant][0]
            # Recall: index 0 corresponds to 2nt RNA
            self.abortive_prob = its.abortive_prob
            self.unproductive_ap = its.unproductive_ap
            self.greb_abortive_prob = its.abortive_prob_GreB

    def _GetAP(self, rna_length, unproductive=False):

        if rna_length < 2:
            print("No AP for RNA < 2nt")
            1/0

        if unproductive:
            ap = self.unproductive_ap[rna_length-2]
        else:
            ap = self.abortive_prob[rna_length-2]

        if self.GreB_AP:
            ap = self.greb_abortive_prob[rna_length-2]

        if ap == 0.0:
            print("Warning: AP is 0 at position {0} for {1}".format(rna_length, self.variant))

        if 0 <= ap <= 1:
            return ap
        else:
            print("AP not number between 0 and 1")
            #1/0

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

    def Backstep(self, rna_length=-1, unproductive=False):

        # Use abortive probability
        if self.use_AP:
            if rna_length == -1:
                print("Specify an RNA length 1/0")
                1/0
            else:

                # God your setup is becoming more and more spaghetti
                if self.custom_productive_AP is False:
                    ap = self._GetAP(rna_length, unproductive)
                else:
                    ap = self.custom_productive_AP[rna_length-2]

                nac = self._GetNAC(rna_length)

                # Avoid almost dividing by zero for high AP
                if ap > 0.95:
                    r = 50
                else:
                    r = nac * ap / (1-ap)

                return r
        else:
            backstep = self.backstep

        return backstep

    def InstantAbort(self, rna_length, method):
        """
        If NAC is 10 bp/s, then abortive release is 30 bp/s? x3 must
        qualify as fast.  """

        if method == 'simple_length_dependent':
            return (201 - rna_length*10.)/rna_length
        else:
            if self.abortive is False:
                return self.nac * 3
            else:
                return self.abortive

    def Nac(self):
        if self.use_AP:
            # for now, Nac is not rna_length dependent
            nac = self._GetNAC(rna_length=-1)
        else:
            nac = self.backstep

        return nac

    def NTP_and_PPi(self):
        return self.nac

    def Escape(self):
        return self.escape

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

