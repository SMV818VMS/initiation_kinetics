"""
Pilot sandbox model for full ODE model of transcription initiation on the DG100
and DG400 libraries.

There are 5 states with a hypothetical promoter escape at nt = 5

Pe1 <-> Pt1 --> Pe2 <-> Pt2 --> Pe3 <-> Pt3 --> Pe4 <-> Pt4 --> Pe5 <-> Pt5

This hypothetical system has an RNA-DNA hybrid of length 3

Abortive initiation can happen from either pre or post for RNAs of length
1,2,3,4. From state 1 abortive rate is v. high, from state 2 it is 50% of state
1. From state 3 it backtracks to state 2. From backtracked state 4 only state 3
can be reached. From backtracked state 3 only state 2 can be reached, where
abortive initiation can commence.

Example abortive pathway for backtracking that starts with a length 4 RNA.
Eg: Pe4 - > Pe4_bt4 -> Pe4_bt3 -> Pe4_bt2 -> Pe4_bt2 -> Pe4_bt1 -> Abort4
                                          -> Abort4

A required flexibility will be to abort from either the post or the
pre-translocated state, or both. The jacobian should be calculatable from these
states without too much hassle.

Rate constants are going to be a key discision issue. I think we have to
abandom the Keq formulation. In the end, it is the balance that matters. This
allows us to modulate the forward and backward rates. Fortunately you have all
the rates directly from Hein et al.

You still need to think of an architecture for this model! The pilot will also
need the notion of experiment.

What is an experiment?
    1) A collection of ITSs (sequence, AP, PY, etc)
    2) A set of rate constants

------------------------------- Sketch of system -----------------

PeN = Pretranslocated state, N length RNA
PtN = Posttranslocated state, N lenght RNA

kNf = forward translocation coefficient, N length RNA
kNb = backwrard translocation coefficient, N length RNA

AN = abortive coefficient, length N RNA
IN = nucleotide incorporation coefficient, length N RNA

BN = backtracking coefficient, length N RNA
EscN = Escape coefficient, length N RNA


------------ Translocation and full length states ----------
d[Pe1]/dt = -k1f[Pe1] + k1b[Pt1]
d[Pt1]/dt = +k1f[Pe1] - k1b[Pt1] - A1[Pt1] - I1[Pt1]

d[Pe2]/dt = -k2f[Pe2] + k2b[Pt2]
d[Pt2]/dt = +k2f[Pe2] - k2b[Pt2] - B2[Pt2] - A2[Pt2] - I2[Pt2]

d[Pe3]/dt = -k3f[Pe3] + k3b[Pt3]
d[Pt3]/dt = +k3f[Pe3] - k3b[Pt3] - B3[Pt3] - I3[Pt3]

d[Pe4]/dt = -k4f[Pe4] + k4b[Pt4]
d[Pt4]/dt = +k4f[Pe4] - k4b[Pt4] - B4[Pt4] - I4[Pt4]

d[Pe5]/dt = -k5f[Pe5] + k5b[Pt5]
d[Pt5]/dt = +k5f[Pe5] - k5b[Pt5] - B5[Pt5] - Esc5[Pt5]

d[FL]/dt = Esc5[Pt5] (this should go through many steps to get the timing right)

----------- Backtracked and abortive states ------------
B5_4 = Backtracked complex with 5 nt RNA after 1 backtracking step (active
site at position 4). The size of the RNA-DNA hybrid is still 3.
A5 = Aborted transcript of length 5
A5_2 = Aborive coefficient for length 5 RNA with only 2nt hybrid
btN_M = backtrack coefficient for length N RNA, for complex at active site M

--------------------- Backtrack from 5 nt RNA-----------
d[B5_4]/dt = B5[Pt5]     - bt5_4[B5_4]
d[B5_3]/dt = bt5_4[B5_4] - bt5_3[B5_3]
d[B5_2]/dt = bt5_3[B5_3] - bt5_2[B5_2] - A5_2[B5_2] # 2nt hybrid
d[B5_1]/dt = bt5_2[B5_2] - bt5_1[B5_1] - A5_1[B5_1] # 1nt hybrid
---------- The abortive state for this backtracked state.
d[A5] = A5_2[B5_2] + A5_1[B5_1]
---------------------------------------------------------

--------------------- Backtrack from 4 nt RNA-----------
d[B4_3]/dt = B4[Pt4]     - bt4_3[B4_3]
d[B4_2]/dt = bt4_3[B4_3] - bt4_2[B4_2] - A4_2[B4_2] # 2nt hybrid
d[B4_1]/dt = bt4_2[B4_2] - bt4_1[B4_1] - A4_1[B4_1] # 1nt hybrid
---------- The abortive state for this backtracked state.
d[A4] = A4_2[B4_2] + A4_1[B4_1]
---------------------------------------------------------

--------------------- Backtrack from 3 nt RNA-----------
d[B3_2]/dt = B3[Pt3]     - bt3_2[B3_2] - A3_2[B3_2] # 2nt hybrid
d[B3_1]/dt = bt3_2[B4_2] - bt3_1[B3_1] - A3_1[B3_1] # 1nt hybrid
---------- The abortive state for this backtracked state.
d[A3] = A3_2[B4_2] + A3_1[B4_1]
---------------------------------------------------------

--------------------- Backtrack from 2 nt RNA-----------
d[B2_1]/dt = B2[Pt2] - A2_1[B2_1] # 1nt hybrid
---------- The abortive state for this backtracked state.
d[A2] = A2_1[B2_1] + A2[Pt2]
---------------------------------------------------------

--------------------- Abort from 1 nt RNA-----------
d[A1] = A1[Pt1]
---------------------------------------------------------

When solving this system a big challenge will be to keep track of which states
are which when the system is on matrix form.
"""
