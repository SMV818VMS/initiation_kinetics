# At first, keep it simple. Add the path to where the ITS and datahandler
# libraries are.
from ipdb import set_trace as debug  # NOQA


class ITRates(object):
    """
    Class for providing and calculating rate constants for ITS variants.
    """

    def __init__(self, its_variant, nac=1, to_fl=5, custom_AP=False,
                 GreB_AP=False, escape=5, unscrunch=2):

        self.name = its_variant.name
        self.nac = nac
        self.to_fl = to_fl
        self.escape = escape
        self.unscrunch = unscrunch

        self.custom_AP = custom_AP
        self.GreB_AP = GreB_AP

        # These have to be present
        self.abortive_prob = its_variant.abortive_prob

        # N25 has some extras
        if self.name == 'N25':
            self.unproductive_ap = its_variant.N25_unproductive_ap
            self.greb_abortive_prob = its_variant.abortive_prob_GreB

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
            print("Warning: AP is 0 at position {0} for {1}".format(rna_length, self.name))

        if 0 <= ap <= 1:
            return ap
        else:
            print("AP not number between 0 and 1")

    def Backtrack(self, rna_length, unproductive=False, use_custom_AP=False):

        # It's implicit that custom AP should only be used for the productive
        # fraction, since we imagine that we have the AP values for the
        # unproductive fraction
        if use_custom_AP:
            ap = self.custom_AP[rna_length-2]
        else:
            ap = self._GetAP(rna_length, unproductive)

        # Avoid almost dividing by zero for high AP
        if ap > 0.95:
            r = 50
        else:
            r = self.nac * ap / (1-ap)

        return r

    def Nac(self):
        # for now, Nac is not rna_length dependent
        return self.nac

    def Escape(self):
        return self.escape

    def ToFL(self):
        return self.to_fl

    def Unscrunch(self):
        return self.unscrunch

    def set_custom_AP(self, ap):
        """
        Can update custom AP from outside
        """
        self.custom_AP = ap

