import json

import meeko
import prody
import rdkit
import autodockdev as jova
from rdkit import Chem
from rdkit.Chem import AllChem

TIP3P_EPSILON = 0.15210325
TIP3P_RMIN_HALF = 1.7682
TIP3P_O_COULOMB = -0.8340
TIP3P_H_COULOMB = 0.4170

PARAMS = {
    "WEIGHTS": {"lj_12_6": 1.0, "coulomb": 1.0},
    "SCALE_1-4": {"lj_12_6": 0.5, "coulomb": 0.8333},
    "CUTOFF": 999.9,
    "TWO_BODY_TERMS": {
        "lj_12_6": {
            "pair_params": ["vdw_c12", "vdw_c6"]
        },
        "coulomb": {
            "pair_params": ["qij"]
        }
    },
    "PAIRWISE_PARAM_COMPILERS": {
        "vdw_c12": ["rmin_half", "epsilon"],
        "vdw_c6": ["rmin_half", "epsilon"],
        "qij": ["charge"]
    },
    "GRID_PROBE": {"coulomb": {"charge": 1.0}},
    "ADD_NBODY_CONTRIB": {},
    "REMOVE_CONTRIBUTION": {},
    "ATOM_PARAMS": {},
    "VECTORS": {},
    "OFFATOMS": {}
}

def get_data_form_meeko(pdb_file, save=False):
    with open(pdb_file) as fi:
        pdbstring = fi.read()
        
    blunt_ends = [("A:1", 0)]
    mk_prep = meeko.MoleculePreparation(
        merge_these_atom_types=[],
        load_atom_params=["vina_params", "openff"],
        charge_model="espaloma",
    )
    templates = meeko.ResidueChemTemplates.create_from_defaults()
    polymer = meeko.Polymer.from_pdb_string(pdb_string=pdbstring,
                                            chem_templates=templates,
                                            mk_prep=mk_prep,
                                            allow_bad_res=True,
                                            default_altloc="A",
                                            blunt_ends=blunt_ends)
    json_s = polymer.to_json()
    with open("target.json", "w") as fo:
        fo.write(json_s)

    with open("target.json") as fi:
        json_string = fi.read()

    polymer = meeko.Polymer.from_json(json_string)
    return polymer



def load_waters(waters_pdb):
    # mk_prep = meeko.MoleculePreparation()
    # w_rdkit = Chem.MolFromSmiles("[H]O[H]")
    # w_rdkit = Chem.AddHs(w_rdkit)
    # rdkit.Chem.rdDistGeom.EmbedMolecule(w_rdkit)
    # molsetup_template = mk_prep(w_rdkit)[0]
    waters = prody.parsePDB(waters_pdb)
    waters_molsetups = list()
    for residue in waters.iterResidues():
        mk_prep = meeko.MoleculePreparation(load_atom_params="openff",
                                            charge_model="espaloma")
        w_rdkit = Chem.MolFromSmiles("[H]O[H]")
        w_rdkit = Chem.AddHs(w_rdkit)
        rdkit.Chem.rdDistGeom.EmbedMolecule(w_rdkit)
        molsetup = mk_prep(w_rdkit)[0]
        res_coords = residue.getCoords()
        for idx, molsetup_atom in enumerate(molsetup.atoms):
            molsetup_atom.coord = res_coords[idx]
            if molsetup_atom.atom_type == "N-TIP3P-O":
                molsetup_atom.charge = TIP3P_O_COULOMB
                molsetup.atom_params["rmin_half"][molsetup_atom.index] = TIP3P_RMIN_HALF
                molsetup.atom_params["epsilon"][molsetup_atom.index] = TIP3P_EPSILON
            else:
                molsetup_atom.charge = TIP3P_H_COULOMB
                molsetup.atom_params["rmin_half"][molsetup_atom.index] = 0.0
                molsetup.atom_params["epsilon"][molsetup_atom.index] = 0.0

        waters_molsetups.append(molsetup)
    return waters_molsetups

if __name__ == "__main__":
    polymer = get_data_form_meeko("/home/niccolo/phd/waterkit/rust_waterkit/minimal_test/receptor.pdbqt")
    molsetups = load_waters("/home/niccolo/phd/waterkit/rust_waterkit/minimal_test/traj/water_000001.pdb")
    docksys = jova.DockingSystem(
            moving_molsetups=molsetups,
            parameters=PARAMS,
            static_molsetup=None,
            polymer=polymer,
            mapo=None,
            grid_desolv=None
        )


    # print("Everything okay!")
    energies = {}
    g = docksys.get_current_genes()
    e = docksys.eval(g, log=energies)
    print(energies)