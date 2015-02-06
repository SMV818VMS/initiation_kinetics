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
            scrunched_DNA_size=0,
            free_5prime_RNA_length=0,
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
        self.free_5prime_RNA_length = free_5prime_RNA_length
        self.__max_duplex_length = max_duplex_length

        self.SanityCheck()

    def SanityCheck():
        """
        Internal consistency-checking of states. Must be called after each update of state.
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

        assert self.free_5prime_RNA_length < self.RNA_length

        if self.free_5prime_RNA_length > 0:
            assert self.free_5prime_RNA_length = self.RNA_length - self.RNADNA_duplex_length

        if self.RNADNA_duplex_length > self.__max_duplex_length:
            assert self.free_5prime_RNA_length > 0

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

    def CanBackTrack():
        self.SanityCheck()
        return self.paused or self.backtracked

    def Backtrack():
        """
        Backtrack either from a paused state, or backtrack even further from an
        already backtracked state
        """
        if self.CanBackTrack():

            if self.paused:
                self.paused = False
                self.backtracked = True

            self.iplus1_site -= 1
            self.scrunched_DNA_size -= 1
            # duplex length decreases by 1 if duplex is not full length
            if self.RNADNA_duplex_length =< self.__max_duplex_length:
                self.RNADNA_duplex_length -= 1

            # free 5' end reduced by one if it exists
            if self.free_5prime_RNA_length > 0:
                self.free_5prime_RNA_length -= 1
        else:
            print('Was not in a state from which to backtrack!')
            1/0

        self.SanityCheck()

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

            # when reaching a full duplex, start growing a free 5' end
            if self.RNADNA_duplex_length == self.__max_duplex_length:
                self.free_5prime_RNA_length += 1
        else:
            print('Must be post-translocated to grow RNA!')
            1/0

        self.SanityCheck()

