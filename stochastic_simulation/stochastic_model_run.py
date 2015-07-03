from libsbml import SBMLDocument, LIBSBML_OPERATION_SUCCESS, OperationReturnValue_toString, UNIT_KIND_SECOND, parseL3Formula, writeSBMLToString
import os
from ipdb import set_trace as debug


# function to check SBML calls
def check(value, message):
    """If 'value' is None, prints an error message constructed using
    'message' and then exits with status code 1.  If 'value' is an integer,
    it assumes it is a libSBML return status code.  If the code value is
    LIBSBML_OPERATION_SUCCESS, returns without further action; if it is not,
    prints an error message constructed using 'message' along with text from
    libSBML explaining the meaning of the code, and exits with status code 1.
    """
    if value is None:
        raise SystemExit('LibSBML returned a null value trying to ' + message + '.')
    elif type(value) is int:
        if value == LIBSBML_OPERATION_SUCCESS:
            return
        else:
            err_msg = 'Error encountered trying to ' + message + '.' \
                 + 'LibSBML returned error code ' + str(value) + ': "' \
                 + OperationReturnValue_toString(value).strip() + '"'
            raise SystemExit(err_msg)
    else:
        return


def SpeciesBoilerplateDimensionless(species, compartment):
    """
    Boilerplate for your dimensionless species. Compartment must be the 'string
    name' of the compartment. Weird how passing the string ID for the object is
    better than passing the object.
    """
    species.setCompartment(compartment)
    species.setConstant(False)
    species.setSubstanceUnits('dimensionless')
    #species.setSubstanceUnits('items')
    species.setBoundaryCondition(False)  # false: is part of reaction
    species.setHasOnlySubstanceUnits(False)  # False: species identifier refers to a concentration when identifier appears in a mathematical formula


def create_test_model():

    # Model
    document = SBMLDocument()
    model = document.createModel('model')

    check(model, 'create model')
    check(model.setTimeUnits('second'), 'set model-wide time units')
    check(model.setExtentUnits('dimensionless'), 'set model units of extent')

    # The unit "second" exists, but not the unit "per second", so we have to
    # make it ourselves.
    create_per_second_unit(model)

    # Must create a compartment: this is required
    c1 = model.createCompartment()
    check(c1, 'create compartment')
    check(c1.setId('c1'), 'set compartment id')
    check(c1.setConstant(True), 'set compartment constant')
    #micro = 10**(-6)
    #check(c1.setSize(micro), 'set compartment size')
    one = 1  # XXX This was required for StochPy. It's OK since you will use
    check(c1.setSize(one), 'set compartment size')
    check(c1.setUnits('litre'), 'set compartment size units')
    check(c1.setSpatialDimensions(3), 'set compartment dimensions')

    # Set species and their initial conditions
    # Can specify amount of substance as "amount" or as concentration.
    # If specified as amount, then the quantity is given from the substanceUnit
    # So 10, and unit is mol, then 10 * 6.022*10^(23)
    # If specified as concentration, then quantity is given as substanceUnit/compartmentSize
    # To input exact species number, you should use the unit 'dimensionless' or 'item'
    #   I have not found a difference between these.

    s1 = model.createSpecies()
    s1.setName('Foxes')
    s1.setId('s1')
    s1.setInitialAmount(10)  # unit of measurement from substanceUnit
    SpeciesBoilerplateDimensionless(s1, 'c1')

    s2 = model.createSpecies()
    s2.setName('Live Rabbits')
    s2.setId('s2')
    s2.setInitialAmount(50)
    SpeciesBoilerplateDimensionless(s2, 'c1')

    # Create rates for your reactions
    # Create reactions
    # If no stoichiometry is given then 1 is assumed

    k = model.createParameter()
    k.setId('k')
    k.setConstant(True)
    k.setValue(0.5)
    k.setUnits('per_second')  # this is the one you made; it's accessible as a string

    # Reaction 1
    r1 = model.createReaction()
    r1.setId('r1')
    r1.setName('Foxes eat rabbits')
    r1.setReversible(False)
    r1.setFast(False)

    # Reactant of reaction 1
    reactant_r1 = r1.createReactant()
    reactant_r1.setSpecies('s2')
    reactant_r1.setConstant(True)  # set fixed stoichiometry
    reactant_r1.setStoichiometry(1)  # THIS WAS ESSENTIAL for Stochpy!

    # Product of reaction 1
    product_r1 = r1.createProduct()
    product_r1.setSpecies('s1')
    product_r1.setConstant(True)
    product_r1.setStoichiometry(1)

    # specify the rate equation
    rate_expression = 'k * s2'
    #rate_expression = 'k * s2 * c1'  # why is compartment included?
    rate_object = parseL3Formula(rate_expression)

    check(rate_object, 'create AST for rate expression')

    kinetic_law = r1.createKineticLaw()
    kinetic_law.setMath(rate_object)

    xml = writeSBMLToString(document)

    fox_sheep = 'foxsheep.xml'
    with open(fox_sheep, 'w') as file_handle:
        file_handle.write(xml)

    fox_sheep_path = os.path.abspath(fox_sheep)

    return fox_sheep_path


def create_per_second_unit(model):
    """
    Weird that this is not done by default.
    """

    per_second = model.createUnitDefinition()
    check(per_second, 'create unit definition')
    check(per_second.setId('per_second'), 'set unit def id')
    unit = per_second.createUnit()
    check(unit, 'create unit')
    check(unit.setKind(UNIT_KIND_SECOND), 'set unit kind')
    # UNIT_KIND_SECOND is the value 28 ...
    check(unit.setExponent(-1), 'set unit exponent')
    check(unit.setScale(0), 'set unit scale')
    check(unit.setMultiplier(1), 'set unit multiplier')


def create_compartment(model, name):
    """
    Take care of boilerplate
    """
    # Must create a compartment: this is required
    # XXX The basic model is just first-order reactions. 5->6 or 5->abort.
    # That will be the same as for deterministic models.
    c = model.createCompartment()
    check(c, 'create compartment')
    check(c.setId(name), 'set compartment id')
    check(c.setConstant(True), 'set compartment constant')
    #micro = 10**(-6)
    #check(c1.setSize(micro), 'set compartment size')
    one = 1  # XXX This was required for StochPy.
    # Volume is arbitrary since we use first order reactions
    check(c.setSize(one), 'set compartment size')
    check(c.setUnits('litre'), 'set compartment size units')
    check(c.setSpatialDimensions(3), 'set compartment dimensions')


def initial_transcription():
    """
    This is a model for initial transcription of N25

    The model setup is quite verbose indeed. After testing the proof of
    principle, you need to generate a model using the tree-method you are
    going to use for deterministic simulations.

    SHIT. I just found out that SMBL uses the NAME of the PYTHON VARIABLE to
    identify species. Why the hell do we set "name" and "id" if finally the
    identifier is the PYTHON VARIABLE NAME. THAT MAKES NO SENSE PLEASE.
    """

    document = SBMLDocument()
    model = document.createModel('Initial transcription on N25')

    check(model, 'create model')
    check(model.setTimeUnits('second'), 'set model-wide time units')
    check(model.setExtentUnits('dimensionless'), 'set model units of extent')

    # The unit "second" exists, but not the unit "per second", so we have to
    # make it ourselves.
    create_per_second_unit(model)

    create_compartment(model, 'compartment')

    # Set species and their initial conditions
    # Can specify amount of substance as "amount" or as concentration.
    # If specified as amount, then the quantity is given from the substanceUnit
    # So 10, and unit is mol, then 10 * 6.022*10^(23)
    # If specified as concentration, then quantity is given as substanceUnit/compartmentSize
    # To input exact species number, you should use the unit 'dimensionless' or 'item'
    # I have not found a difference between these.

    s1 = model.createSpecies()
    s1.setName('Open Complex')
    s1.setId('OC')
    s1.setInitialAmount(100)  # unit of measurement from substanceUnit
    SpeciesBoilerplateDimensionless(s1, 'compartment')

    s2 = model.createSpecies()
    s2.setName('3nt')
    s2.setId('3nt')
    s2.setInitialAmount(0)
    SpeciesBoilerplateDimensionless(s2, 'compartment')

    s3 = model.createSpecies()
    s3.setName('4nt')
    s3.setId('4nt')
    s3.setInitialAmount(0)
    SpeciesBoilerplateDimensionless(s3, 'compartment')

    s4 = model.createSpecies()
    s4.setName('5nt')
    s4.setId('5nt')
    s4.setInitialAmount(0)
    SpeciesBoilerplateDimensionless(s4, 'compartment')

    s5 = model.createSpecies()
    s5.setName('Full length')
    s5.setId('FL')
    s5.setInitialAmount(0)
    SpeciesBoilerplateDimensionless(s5, 'compartment')

    # Create rates for your reactions
    # Abortive
    create_parameter(model, 'ka', True, 0.05, 'per_second')
    create_parameter(model, 'k1', True, 0.1, 'per_second')
    create_parameter(model, 'k2', True, 0.2, 'per_second')

    # Forward reactions
    # OH GOD. THEY REQUIRE THE NAME OF THE PYTHON VARIABLE, NOT THE ID.
    # OH GOD.
    create_reaction(model, 'OC', '3nt', 'r1')
    create_reaction(model, '3nt', '4nt', 'r1')
    create_reaction(model, '4nt', '5nt', 'r1')
    create_reaction(model, '5nt', 'FL', 'r2')

    # Reverse reactions
    create_reaction(model, '3nt', 'OC', 'ra')
    create_reaction(model, '4nt', 'OC', 'ra')
    create_reaction(model, '5nt', 'OC', 'ra')

    xml = writeSBMLToString(document)

    N25 = 'N25.xml'
    with open(N25, 'w') as file_handle:
        file_handle.write(xml)

    N25_path = os.path.abspath(N25)

    return N25_path


def create_parameter(model, name, constant, value, unit):
    """
    Take care of all the boiler plate
    """
    k = model.createParameter()
    k.setId(name)
    k.setConstant(constant)
    k.setValue(value)
    k.setUnits(unit)  # this is the one you made; it's accessible as a string


def create_reaction(model, reactant, product, rate_constant):
    """
    Take care of all the boiler plate
    """

    id = '_'.join([reactant, product, rate_constant])
    print(id)

    r = model.createReaction()
    r.setId(id)
    r.setName(id)
    r.setReversible(False)
    r.setFast(False)

    # Reactant of reaction 1
    reac = r.createReactant()
    reac.setSpecies(reactant)
    reac.setConstant(True)  # set fixed stoichiometry
    reac.setStoichiometry(1)  # THIS WAS ESSENTIAL for Stochpy!

    # Product of reaction 1
    prod = r.createProduct()
    prod.setSpecies(product)
    prod.setConstant(True)
    prod.setStoichiometry(1)

    # specify the rate equation
    rate_expression = rate_constant + ' * ' + reactant
    rate_object = parseL3Formula(rate_expression)
    check(rate_object, 'create AST for rate expression')
    kinetic_law = r.createKineticLaw()
    kinetic_law.setMath(rate_object)

    debug()


def runStochPy(model_path):

    import stochpy

    mod = stochpy.SSA()
    model_dir, model_filename = os.path.split(model_path)
    mod.Model(model_filename, dir=model_dir)  # have to provide name and dir separately ... this took an hour to figure out

    #mod.DoStochSim()

    # !!! YES!!!!! You had to change ownership of the build directoriers, then
    # do just python setup.py install and conda takes care of the rest :)
    mod.DoStochKitStochSim(trajectories=1, IsTrackPropensities=True,
                           customized_reactions=None, solver=None, keep_stats=False,
                           keep_histograms=False, endtime=100)
    mod.PlotSpeciesTimeSeries()


if __name__ == '__main__':
    #model = create_test_model()
    #model = initial_transcription()
    print("made model")

    # XXX The SBML Python API is BAD. It seems to rely on python variable
    # names to identify objects, even when objects have both name and ID. What
    # the hell is up with that. Fortunately, the native .psc format is very
    # easy to use and set up.

    #model = '/home/jorgsk/Stochpy/pscmodels/initial_transcription_test.psc'
    model = '/home/jorgsk/Dropbox/phdproject/kinetic_paper/model/psc_files/N25_simple.psc'
    runStochPy(model)
    print("ran model")

