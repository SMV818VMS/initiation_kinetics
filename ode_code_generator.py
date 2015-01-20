"""
You need to generate code for your ODE system.

Input parameters will be

    1) last position of abortive release (e.g 5) = LPAR
    2) backtrack from pre/post state
    3) escape position RNA length (N) (can this be predicted somehow?)
    4) first position of direct abortive initiation (+2 for full system, +1 for
    pilot) == FP
    5)

Below you have outlined how many states and parameters are needed.

What's next is to start creating code that looks like this:

[rhs1,
rhs2,
rhs3,
rhs4,
rhs5,
...,
rhs250]

While keeping track of which line in the matrix corresponds to which
state. Then a huge super-sparse Jacobian :S:S I think you have to use sympy,
and then somehow parse sympy output to get it onto the form that scipy wants.

----------------------------------------------------
States:
* N forward translocated states: PeN
* N reverse translocated states: PtN
* 1 full length: FL
* (N-1) backtracked states for the N-nt RNA
* (N-2) backtracked states for the (N-1)-nt RNA
* (N-3) backtracked states for the (N-3)-nt RNA
...
* 1 backtracked state for the (FP+1)-nt RNA [FP=1 -> stop at(N-(N-2))]
* N abortive states, corresponding to abortion from each of the states with
  length N RNA
----------------------------------------------------

= 3N + (N-1) + (N-2) + ... + (FP+1) + 1
if FP = 1, this becomes: 3N + sum(k) - N = 2.5N + (N^2)/2
if FP = 2 this becomes 2.5N + (N^2)/2 - 1

For N=20 this is 250 states
For N=15 this is 150 states

----------------------------------------------------
Parameters (are sequence dependent and vary for each ITS):
    Some of these can be reduced to be the same constant, such as the
    nucleotide incorporation rates. But it's good to keep the option open.

#* N forward translocation:  kNf
#* N backward translocation: kNb
#* LPAR direct-abort (eg. 2nt to 5nt): A2, ..., A5
#* N nucleotide incorporation rates: I2, ..., IN
#* N-FP initial backtracking rates: B3, B4, ..., BN
* 1 escape rate
#* (N-1) + (N-2) + (N-3) + ... + FP+1 backtrack rates
#* N * LPAR from-backtracked-state-abort rates:

    (# indicates those you have implemented)

----------------------------------------------------

There are two different backtrack rates: 1 rate type that is associated with
complexes which have already backtracked, and then there is the initial
backtracked rate.

5-nt RNA has 4 backtrack rates
4-nt RNA has 3 backtrack rates
3-nt RNA has 2 backtrack rates

= (N-1) + (N-2) + (N-3) + ... + FP+1 backtrack rates

2-nt RNA has 0 backtrack rates, but 1 initial backtracked rate
1-nt RNA has 0 backtrack rates and 0 initial backtracked rates, but 1 direct
abort rate

------------------ Nomenclature ------------

"Reaction rate coefficient" is not often used. "Reaction rate constant" is used.
Coefficient may be more appropriate if the constant is not really constant: the
word constant best describes elementary reactions.

dK/dt = k[A], then k is the rate constant [coefficient].

Chemical reaction:

    aA + bB -> cC

    # using mass balance one can write:
    d[C]/dt = k(T)[A]^m[B]^n   # reaction rate of change of concentration

    k(T) is the rate "constant".
    a and b are the stoichiometric coefficients.
    m and n are "partial orders of reaction" and may or may not be equal to a
    and b.

    the "reaction rate" itself is defined as:
    r = 1/c * d[C]/dt
    which makes sense, since then d[C]/dt = r * c, which is reaction rate times
    the number of C's produced.

------------------------------TODO-list--------------------------------


1) Given an ITS-sequence, script up the different parameters
    - Forward translocation
    - Backward translocation

--------------Time constants during Lilians experiments----------------

I'm not sure about the reaction-times for transcription initiation. Can fine
tune this once the model is up and running.

"""
