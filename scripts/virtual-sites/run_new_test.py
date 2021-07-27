# import pprint
import numpy as np
import scipy.optimize
import scipy.spatial.distance
from openff.toolkit.topology import Molecule
from openff.toolkit.typing.engines.smirnoff import ForceField

from openff.recharge.charges.bcc import (
    SMIRNOFFModel,
    SMIRNOFFMoleculeAssignment,
    SMIRNOFFMoleculeAssignmentModel,
    SMIRNOFFTermModel,
)
from openff.recharge.tests.optimize.test_optimize import MockMoleculeESPStoreDinitrogen

#######################################################################################
# Configure the ForceField
#######################################################################################

# ff = ForceField("../openff/recharge/tip4.offxml")
ff = ForceField("../openff/recharge/smirnoff_with_vsite.offxml")
ff._terms_set_active("charge_increment")
terms = []
for handler in ["ChargeIncrementModel", "VirtualSites"]:
    ph = ff.get_parameter_handler(handler)
    for term_name in ph._terms():
        ff_term_name = handler, term_name
        term = SMIRNOFFTermModel(handler=handler, name=ff_term_name)
        terms.append(term)


#######################################################################################
# Generate the recharge models
#######################################################################################
model = SMIRNOFFModel(forcefield=ff, terms=terms)
record = MockMoleculeESPStoreDinitrogen().retrieve()[0]
molecule = Molecule.from_smiles(record.tagged_smiles)

print("MODEL TERMS")
for i, term in enumerate(terms):
    print(f"    {i:3d}", term)


#######################################################################################
# Assignment Matrix
#######################################################################################
assn_model = SMIRNOFFMoleculeAssignmentModel(molecule=molecule, smirnoff_model=model)
assn: SMIRNOFFMoleculeAssignment = assn_model.to_molecule_assignment()

print("TERMS:")
for i, term in enumerate(assn.terms()):
    print(f"    {i:3d}", term)
print("MOLECULEKEYS:")
for i, mol_key in enumerate(assn.molecule_keys()):
    print(f"    {i:3d}", mol_key)

print(assn.to_ndarray())

#######################################################################################
# Optionally build a master Assignment from multiple molecules
#######################################################################################
# assignment_matrix_model = SMIRNOFFMasterAssignmentModel(
#     molecule_assignments=[assn_model]
# )

#######################################################################################
# Configure the grid
#######################################################################################

grid = record.grid_coordinates
print(f"{grid=}")

# Need this because it has the virtual sites added
molecule = assn.molecule()

print(f"{record.conformer=}")
vsite_xyz = molecule.compute_virtual_site_positions_from_atom_positions(
    record.conformer
)
print(f"{vsite_xyz=}")
xyz = np.vstack((record.conformer, vsite_xyz))

R = 1.0 / scipy.spatial.distance.cdist(grid, xyz)
T = assn.to_ndarray()


def distance(A, B):
    return np.linalg.norm(A - B)


def epot(grid, ptls, q):
    v = np.zeros(grid.shape[0])
    for i, g in enumerate(grid):
        for j, ptl in enumerate(ptls):
            v[i] += q[j] / distance(g, ptl)
    return v


def objective(ref_esp, train_esp):
    return np.sum((ref_esp - train_esp) ** 2)


ref_esp = record.esp
q_vsite_init = np.array([0.0, 0.0, 0.0, 0.0])
v_diff = ref_esp - epot(grid, xyz, q_vsite_init)

A = T.T @ R.T @ R @ T
B = T.T @ R.T @ v_diff

n_cc = T.shape[1]
p = np.zeros((n_cc, 1)) + 0.5

print("A")
print(A)

print("B")
print(B)


def fun(x, A, b):
    return ((np.dot(A, x.reshape(-1, 1)) - b.reshape(-1, 1)) ** 2).flat


x0 = np.zeros((n_cc,))
ret = scipy.optimize.least_squares(
    fun, x0, args=(A, B), verbose=2, max_nfev=300, method="lm"
)

print("Final parameters:", ret.x)

q_train = q_vsite_init + np.dot(T, ret.x)
print(f"{q_train=}")

v_train = epot(grid, xyz, q_train)
print(f"{v_train=}")
print(f"{ref_esp=}")

v_calc = epot(grid, record.conformer, np.array([0.0, 0.0]))
print(f"{v_calc=}")

print("INIT:", objective(ref_esp, v_calc))
print("FIT: ", objective(ref_esp, v_train))
