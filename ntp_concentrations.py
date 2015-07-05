
DNA = 0.03   # uM (micro molar; 30 nano molar = 0.03 uM)
NTP = 100.   # uM
RNAP = 0.03  # uM (0.05 total with ~ 60% active -> 0.03)

reaction_volume = 10  # uL


# 1M = 1 * An / L
# 1uM = 1e-6 An / L
# 1L = 1e+6 uL
# So 1uM = 1e-6 An * 1e06 / uL = e-12 An / uL
# An = 6.02e23 things
# So 1uM = 6.02e11 / uL

# So in 10 uL, multiply concentration in uM with 6.02e12 to get the number of
# things.

multiplier_10_ul = 6.02e11

nr_DNA = DNA * multiplier_10_ul
nr_NTP = NTP * multiplier_10_ul
nr_RNAP = RNAP * multiplier_10_ul

print('# DNA {0}'.format(nr_DNA))
print('# NTP {0}'.format(nr_NTP))
print('# RNAP {0}'.format(nr_RNAP))

# How quickly is NTP used? Assume that transcription happens at 10bp/s (this
# is about what Bai etc find for [NTP] = 100.

# Extreme case: assume that all NTP has been used by 10 minutes.
# (Hey, here you need to look at Lilian's kinetic paper)

# It's just 3000 NTP for each RNAP
print NTP/RNAP


def NTP_concentration(nac_rate, time, initial_NTP, nr_RNAP, initial_NTP_conc=100):
    """
    Starting from a concentration of [100], what is the final concentration?
    nac_rate in bp/s (eg 10)
    time in s (eg 600 for 10 min)
    """
    nr_NTP_consumed = nr_RNAP * nac_rate * time

    final_NTP = initial_NTP - nr_NTP_consumed

    return initial_NTP_conc * final_NTP / nr_NTP

# After 5 min, 10% left with 10bp/s
# After 5 min, 55% left with 5bp/s
# After 10 min, 10% left with 5bp/s
# After 10 min, 55% left with 5bp/s and 50% transcripbing RNAP
# After 10 min, 10% left with 10bp/s and 50% transcripbing RNAP
new_conc = NTP_concentration(10, 600, nr_NTP, nr_RNAP*0.5)
print(new_conc)
# XXX if you really want to do this, there is only one way to do it, and that
# is to estimate a collision frequency between RNAP and free DNA (ignore
# traveling along DNA). If you spend a week I'm sure you can get some good
# numbers (and learn something along the way) but you don't have time for that
# now.

# If all RNAP are engage, and backtracking is fast, then the intial
# consumption is at 10bp/s
# Can we use a formula for [NTP] to see how fast it's going away?

# What is the rate of RNA synthesis? If abortive release is rapid
# XXX The "rapid" abortive release is only observed for very short
# backtracking events < 6/7 nt!!! Remember that! At +13, backtracking will
# result in a full 9 or 10 BP hybrid.

# So it could be intersting to see the effect of decreasing the rate of
# abortive fallback when RNA is long.
