import pprint
import numpy as np
from openff.recharge.charges.bcc import VSiteSMIRNOFFGenerator, VSiteSMIRNOFFCollection
import openff.recharge.smirnoff
from openff.toolkit.typing.engines.smirnoff import ForceField
from openff.toolkit.topology import Molecule

ff = ForceField("../openff/recharge/tip4.offxml")

# all "term"-based API is considered "internal use only" aka "private" for now
ff._terms_set_active("charge_increment")

ff_terms = list(ff._terms())

print("FF TERMS")
for i, term in enumerate(ff_terms):
    print(i, term, ff._term_select(term))

vs = ff.get_parameter_handler("VirtualSites")
mol = Molecule.from_smiles("O")



lbls = ff.label_molecules(mol.to_topology())[0]
# pprint.pprint(lbls)
pprint.pprint(lbls['ChargeIncrementModel'].store)

pprint.pprint(lbls['VirtualSites'].store)

mol_top_orig = mol.to_topology()


# surely we solve this by iterating the vsites
mol_top = vs.create_openff_virtual_sites(mol_top_orig)
molv = next(mol_top.reference_molecules)

assign = np.zeros((molv.n_particles, len(ff_terms)))

print("ASSIGNED TERMS")
terms = ff._term_map_topology(mol_top_orig)

for ((topo, key), pterms) in terms.items():
    print("****", key)

    bcc_terms = filter(lambda x: x[0] == "ChargeIncrementModel", pterms)
    for term in bcc_terms:
        term_i = ff_terms.index(term)

        first_atom = key[0]
        other_atoms = key[1:]

        for other in other_atoms:
            assign[first_atom][term_i] += 1
            assign[other][term_i] -= 1

        print("        {} {}".format(term_i, str(term)))

print("VSITES", molv.n_particles)
for vsite in molv.virtual_sites:
    for vp in vsite.particles:
        ptl = vp.molecule_particle_index
        print("****", ptl, vp.orientation)
        vsite_terms = filter(lambda x: x[0] == "VirtualSites", terms[(mol_top_orig, vp.orientation)])
        for atom, term in zip(vp.orientation, vsite_terms):
            term_i = ff_terms.index(term)
            assign[ptl][term_i] += 1
            assign[atom][term_i] -= 1

            print("        {} {}".format(term_i, str(term)))




print(f"ASSIGNMENT MATRIX ({mol.n_atoms} n_atoms, {len(ff_terms)} terms")

space = ""
print(f"{space:3s} ", end=" ")
for i, _ in enumerate(ff_terms):
    print(f"{i:5d}", end=' ')
print()


for i, row in enumerate(assign):
    print(f"{i:3d} ", end=" ")
    for col in row:
        print(f"{col:5.2f}", end=' ')
    print()


# pprint.pprint(terms.values())

# test when we have term PR active
if 1:
    vsc : VSiteSMIRNOFFCollection = openff.recharge.smirnoff.from_smirnoff_virtual_sites(vs, "VirtualSites")

    from openff.toolkit.topology import Molecule

    oe_mol = Molecule.from_smiles("O").to_openeye()

    VSiteSMIRNOFFGenerator.build_assignment_matrix(oe_mol, vsc)

