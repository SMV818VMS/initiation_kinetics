Class iRNAP(object):
    """
    Describes an RNAP undergoing initial transcription.
    """

    def __init__(self,
            rna_length=2,
            pre_translocated=True,
            post_translocated=False,
            paused=False,
            backtracked=False,
            iplus1_site=2,
            RNADNA_duplex_length=2,
            scrunched_DNA_size=0
            max_duplex_length=10):

        # Simplified overall description of RNAP state during initial transcription
        self.RNA_length = rna_length
        self.pre_translocated = pre_translocated
        self.post_translocated = post_translocated
        self.paused = paused
        self.backtracked = backtracked
        self.iplus1_site = iplus1_site
        self.RNADNA_duplex_length = RNADNA_duplex_length
        self.scrunched_DNA_size = scrunched_DNA_size
        self.__max_duplex_length = max_duplex_length

        self.SanityCheck()

    def SanityCheck():
        """
        Internal consistency-checking. Must be called after each update of state space.
        """

        if self.backtracked:
            assert self.pre_translocated is False
            assert self.post_translocated is False
            assert self.paused is False

        if self.paused:
            assert self.pre_translocated is False
            assert self.post_translocated is False
            assert self.backtracked is False

        if self.post_translocated:
            assert self.pre_translocated is False
            assert self.backtracked is False
            assert self.paused is False

        if self.pre_translocated:
            assert self.post_translocated is False
            assert self.backtracked is False
            assert self.paused is False

        assert self.RNA_length >= self.RNADNA_duplex_length

        assert self.scrunched_DNA_size < self.RNA_length

    def GetActiveSiteDinucleotide(its):
        """
        Assumes an its on the form "AT(...)"" where AT are the +1 and +2 positions
        in the transcribed region.
        """
        return its[iplus1_site-1] + its[iplus1_site]

    def GetActiveSiteNucleotide(its):
        return get_active_site_dinucleotide(its)[-1]

    def Translocate():
        if self.post_translocated:
            print('Was already translocated!')
            1/0
        else:
            self.post_translocated = True

        self.SanityCheck()

    def Pause():
        if self.paused:
            print('Was already paused!')
            1/0
        else:
            if self.pre_translocated:
                self.paused = True
            else:
                print('Must be pre-translocated to pause!')
                1/0

        self.SanityCheck()

    def Backtrack():
        """
        Backtrack either from a paused state, or backtrack even further from a
        backtracked state
        """
        if self.paused:
            self.backtracked = True

            self.iplus1_site -= 1
            self.scrunched_DNA_size -= 1
            # increase duplex length only until RNA reaches length 10
            if self.RNADNA_duplex_length < self.__max_duplex_length:
                self.RNADNA_duplex_length = RNA_length

        elif self.backtracked:
            pass
            # TODO backtrack further
        else:
            print('Must be paused or backtracked to backtrack!')
            1/0

    def CanGrow():
        excluded_from_growing = self.pre_translocated and self.backtracked and self.paused
        if excluded_from_growing and self.post_translocated:
            return False
        else:
            return True

    def GrowRNA():
        if self.CanGrow():
            self.iplus1_site += 1
            self.RNA_length += 1
            self.scrunched_DNA_size += 1
            # increase duplex length until RNA reaches length 10
            if self.RNA_length =< self.__max_duplex_length:
                self.RNADNA_duplex_length = RNA_length
        else:
            print('Must be post-translocated to grow RNA!')
            1/0

        self.SanityCheck()

