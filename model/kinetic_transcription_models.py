from ipdb import set_trace as debug  # NOQA
import matplotlib.pyplot as plt
import networkx as nx
from datetime import datetime
import os


class ITStoichiometricSetup(object):
    """
    This is the setup required for the stoichiometry of the system
    """
    def __init__(self, escape_RNA_length, part_unproductive=False,
                 unproductive_only=False, custom_AP=False):

        if unproductive_only and part_unproductive:
            print("Mismatch in unproductive settings")
            1 / 0

        # If this is set, only include the unproductive pathway
        self.unproductive_only = unproductive_only

        # Include a pathway for unproductive complexes
        # These have their own forward-pathway and their own abortive
        # probabilities
        self.part_unproductive = part_unproductive

        # escape RNA-length: last point of escape
        self.escape_RNA_length = escape_RNA_length

        # you can provide custom AP for productive fraction
        self.custom_AP = custom_AP


def RateConstantName(from_state, to_state):
    return '_to_'.join([from_state, to_state])


def AddAbortiveLogNode(G, abortive_log):
    # Add the abortive log state
    G.add_node(abortive_log)


def AddReaction(G, rc, from_state, to_state):
    rc_name = RateConstantName(from_state, to_state)
    G.add_edge(from_state, to_state, rate_constant_value=rc, rate_constant_name=rc_name)


def WriteTemplate(productive_open_complex=False,
                  unproductive_open_complex=False, rna_len=False,
                  backstep_state=False, full_length=False, unproductive=False,
                  unproductive_backstep_state=False, elongating=False):
    """
    The logic of writing the template. It's horrible. Hoping it will settle
    into a "just works" code.
    """

    # Template string for state variables
    state_template = 'RNAP{0}{1}{2}'

    entries = []

    if productive_open_complex:
        entries = ['p', 'o', 'c']
    elif unproductive_open_complex:
        entries = ['u', 'o', 'c']
    elif full_length:
        entries = ['f', 'l', 't']
    elif elongating:
        entries = ['e', 'l', 'c']
    else:
        if rna_len:
            # See if there is 1 or 2 digits
            digits = list(str(rna_len))
            if len(digits) == 1:
                dg1 = digits[0]
                entries.append(dg1)
                entries.append('_')

            elif len(digits) == 2:
                dg1, dg2 = digits
                entries.append(dg1)
                entries.append(dg2)

            if backstep_state:
                entries.append('b')
            elif unproductive:
                entries.append('u')
            elif unproductive_backstep_state:
                entries.append('f')
            else:
                entries.append('_')
        else:
            print('Must provide rna-length or open complex or full length!')

    if len(entries) != 3:
        print("something went wrong here")
        1 / 0

    return state_template.format(*entries)


def BasicITStoichiometry(R, name, s):
    """
    Input: R (reactions), name, s (setup)
    Basic model of transcription initiation.
    Contains:

    * Open complex (productive or unproductive)
    * NAC
    * Backtracking
    * Abortive RNA release
    * Promoter escape
    * Elongating complex

    Code for initially transcribing RNAP: RNAP_XYZ

    X -> length of RNA [0,...,inf]
    Y -> 0 ; if backtracked mode == 'b'
    Promoter escape -> full length transcript state: RNAP_flt
    Promoter escape -> Elongating complex -> Full length transcript

    Examples of RNAP in different states:
    5__ : len 5 complex
    5_u : len 5 unproductive complex
    5_b : len 5 backstepped complex
    5_f : len 5 unproductive backstepped complex

    15_ : len 15 complex
    15u : len 15 unproductive complex
    15b : len 15 backstepped complex
    15f : len 15 unproductive backstepped complex

    5al : proxy for aborted RNA of length 5
    elc : elongating complex
    flt : full length transcript
    poc : productive open complex
    uoc : unproductive open complex
    """

    # Initialize a graph object
    G = nx.DiGraph(name=name)

    # Unproductive complexes do not escape.
    # Conversely, productive complexes do escape.
    if not s.unproductive_only:
        are_productive = True

    # Helper bool
    are_unproductive = s.part_unproductive or s.unproductive_only

    # Create the first reaction step -> from open complex to starting complex
    # Need this because abortive release takes RNAP back to the open complex,
    # but we may wish to model transcription from a 2nt RNA.
    if are_productive:
        open_complex = WriteTemplate(productive_open_complex=True)
        G.add_node(open_complex, initial_node_productive=True)

    if are_unproductive:
        open_complex_unprod = WriteTemplate(unproductive_open_complex=True)
        G.add_node(open_complex_unprod, initial_node_unproductive=True)

    # Create the full length (FL) state if necessary
    if are_productive:
        full_length = WriteTemplate(full_length=True)
        elongating = WriteTemplate(elongating=True)

    # Following the growing RNA to explore all possible model states
    for rna_len in range(1, s.escape_RNA_length + 1):

        # This state
        if are_productive:
            this = WriteTemplate(rna_len=rna_len)

        if are_unproductive:
            this_unprod = WriteTemplate(rna_len=rna_len, unproductive=True)

        # Special case when previous state was open complex
        if rna_len == 1:
            if are_productive:
                previous_prod = open_complex

            if are_unproductive:
                previous_unprod = open_complex_unprod

        else:
            if are_productive:
                previous_prod = WriteTemplate(rna_len=rna_len - 1)

            if are_unproductive:
                previous_unprod = WriteTemplate(rna_len=rna_len - 1, unproductive=True)

        nac_rate = R.Nac()

        if are_productive:
            AddReaction(G, nac_rate, previous_prod, this)

        if are_unproductive:
            AddReaction(G, nac_rate, previous_unprod, this_unprod)

        # If into the abortive range, apply (backtracking) and abort
        if rna_len < 2:
            continue
        else:

            # Common to productive and unproductive
            unscrunch_rate = R.Unscrunch()

            if are_productive:
                bs = WriteTemplate(rna_len=rna_len, backstep_state=True)
                backstep_rate = R.Backtrack(rna_len, use_custom_AP=s.custom_AP)
                AddReaction(G, backstep_rate, this, bs)
                AddReaction(G, unscrunch_rate, bs, open_complex)

            if are_unproductive:
                bsu = WriteTemplate(rna_len=rna_len, unproductive_backstep_state=True)
                bs_unprod_rate = R.Backtrack(rna_len, unproductive=True)
                AddReaction(G, bs_unprod_rate, this_unprod, bsu)
                AddReaction(G, unscrunch_rate, bsu, open_complex_unprod)

        # if escape is allowed, can happen after escape start
        if are_productive:
            if rna_len >= s.escape_RNA_length:
                escape_rate = R.Escape()
                AddReaction(G, escape_rate, this, elongating)

                # Then add escape to full length
                fl_rate = R.ToFL()
                AddReaction(G, fl_rate, elongating, full_length)

    return G


def MakeStartingComplex(G, initial_rna_length, open_complex, R, state_template):
    """
    Starting complex may be open complex, or may for example be from a 2nt RNA
    """

    if initial_rna_length > 0:
        starting_complex = state_template.format(initial_rna_length, 0, 0)
        first_step = R.FirstStep(initial_rna_length)
        G.add_edge(open_complex, starting_complex, rate_constant=first_step)
    else:
        starting_complex = open_complex

    return starting_complex


def CreateReactionSystem(G):
    """
    Create reaction system from directed graph with 'rate_constant'.

    Requires: directed graph; each directed edge must have a rate constant.
    Optional: if a 'to-node' has a name that ends with 'al', the 'from-node'
              will not be associated with a loss of mass to this 'log' node.
    """

    system = {}
    for from_node, to_node, data in G.edges(data=True):

        # add nodes if not in system
        if from_node not in system:
            system[from_node] = ''
        if to_node not in system:
            system[to_node] = ''

        # extract reaction rate constant
        rc = data['rate_constant_name']
        # add contribution to system from this connection
        factor = '*'.join([str(rc), from_node])

        # Remove spacing to make small printed systems look nice
        if system[from_node] == '':
            loss = '-' + factor
        else:
            loss = ' - ' + factor

        if system[to_node] == '':
            gain = factor
        else:
            gain = ' + ' + factor

        # Do not subtract for mass going to a log state
        if to_node.endswith('al'):
            pass
        else:
            system[from_node] = system[from_node] + loss

        system[to_node] = system[to_node] + gain

    return system


def SetInitialValues(G, initial_RNAP_prod, initial_RNAP_unprod):
    """
    Set initial values for all nodes except initial node to 0.
    Start with 1 for initial node.

    Requires: graph G with at most one initial node.
    Dislikes: more than one initial node. Will bark.
    """

    initial_productive_found = False
    initial_unproductive_found = False
    initial_conditions = {}

    for node, data in G.nodes(data=True):
        if 'initial_node_productive' in data:
            if not initial_productive_found:
                initial_conditions[node] = initial_RNAP_prod
                initial_productive_found = True
            else:
                print('Multiple productive initials found!')
                1 / 0

        elif 'initial_node_unproductive' in data:
            if not initial_unproductive_found:
                initial_conditions[node] = initial_RNAP_unprod
                initial_unproductive_found = True
            else:
                print('Multiple unproductive initials found!')
                1 / 0
        else:
            initial_conditions[node] = 0

    if initial_RNAP_prod > 0 and not initial_productive_found:
        print('No initial productive node found!')
        1 / 0

    if initial_RNAP_unprod > 0 and not initial_unproductive_found:
        print('No initial unproductive node found!')
        1 / 0

    return initial_conditions


def GetParameters(G):
    """
    Parse graph to link parameter name with parameter value.

    Requires: directd graph G with attributes rate_constant_name and rate_constant_value.
    """

    parameters = {}
    for from_node, to_node, data in G.edges(data=True):
        parameters[data['rate_constant_name']] = data['rate_constant_value']

    return parameters


def WriteGraphAndSetup(G, reaction_setup, tag):
    """
    Write to file an image of the graph and the corresponding setup.
    """

    fig, ax = plt.subplots()

    nx.draw_graphviz(G, with_labels=True, ax=ax)
    # filename is number of translocation steps followed by a time tag
    time_tag = str(datetime.now()).split('.')[0]
    time_tag = time_tag.replace(':', '-')
    store_dir = 'Log'
    if not os.path.isdir(store_dir):
        os.makedirs(store_dir)

    if 'final_escape_RNA_length' in reaction_setup:
        beg = str(reaction_setup['final_escape_RNA_length'])
    else:
        beg = '0'

    tag = tag.replace('/', '-')
    id_text = tag + '_' + beg + '_' + time_tag

    image_path = os.path.join(store_dir, id_text + '_highres.pdf')
    fig.set_size_inches(34, 16)

    fig.savefig(image_path)
    plt.close(fig)  # Close to avoid piling up

    text_path = os.path.join(store_dir, id_text + '.txt')
    with open(text_path, 'wb') as text_handle:
        for key, value in reaction_setup.items():
            text_handle.write(key + '\t\t' + str(value) + '\n')


def GenerateStochasticInput(G, initial_RNAP_prod=100, initial_RNAP_unprod=0):
    """
    Create a graph based on the raction setup and rate constants. From the
    graph, obtain the system equations, initial values, and parameters (rate
    coefficients).

    Requires a directed graph G where each directed node is associated with a
    'rate_constant_name' and 'rate_constant_value'. Also requires one of the
    nodes to be called 'initial_node'.
    """

    initial_values = SetInitialValues(G, initial_RNAP_prod, initial_RNAP_unprod)
    parameters = GetParameters(G)
    reactions = GetReactions(G)

    return reactions, initial_values, parameters


def GetReactions(G):
    """
    For the .psc format you want something like this:

    R1:
        S1 > S2 (description)
        k * S1  (rate)
    """

    reactions = {}

    i = 1
    for from_node, to_node, data in G.edges(data=True):

        name = 'R{0}'.format(i)
        i += 1

        description = from_node + ' > ' + to_node

        # Rate constant
        rc = data['rate_constant_name']
        # Rate
        rate = '*'.join([str(rc), from_node])

        reactions[name] = {'description': description, 'rate': rate}

    return reactions


def CreateModelGraph(R, name, setup, write_log=False):
    """
    Calculate rate constants and place them in a graph of the model.
    """

    # Calculate rate constants and create graph
    G = BasicITStoichiometry(R, name, setup)

    if write_log:
        #Plot model graph and write setup log
        WriteGraphAndSetup(G, setup, name)

    return G


def write_psc(reactions, initial_values, parameters, graph_name, write_dir):
    """
    Ecample of .psc format:

        ---------------------
        # Initial transcription with abortive cycling

        # Reactions
        R1:
            OC > nt3
            k1 * OC

        R2:
            nt3 > $null
            k2 * nt3

        # Variable species
        OC = 10  # single molecule experiment yay!
        nt3 = 0

        # Parameters
        k1 = 0.2
        k2 = 0.4
        ---------------------
    """

    lines = []
    lines.append('# ' + graph_name)
    lines.append('')

    lines.append('# Reactions')
    for reacation_name, reaction_info in sorted(reactions.iteritems()):
        description = reaction_info['description']
        rate = reaction_info['rate']
        lines.append(reacation_name + ':')
        lines.append('\t' + description)
        lines.append('\t' + rate)
        lines.append('')

    lines.append('# Variable species')
    for name, value in initial_values.items():
        lines.append(name + ' = ' + str(value))
    lines.append('')

    lines.append('# Parameters')
    for name, value in parameters.items():
        lines.append(name + ' = ' + str(value))

    # Create output directory
    if not os.path.isdir(write_dir):
        os.makedirs(write_dir)

    import tempfile
    # Write to file
    #filepath = os.path.join(write_dir, graph_name + '.psc')
    #handle = open(filepath, 'wb')
    handle = tempfile.NamedTemporaryFile(delete=False, suffix='.psc')
    for line in lines:
        #handle.write(line + os.linesep)
        handle.write(line + os.linesep)
    handle.close()

    return handle.name
