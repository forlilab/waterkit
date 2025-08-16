import json
import numpy as np
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
    # with open(pdb_file) as fi:
    #     pdbstring = fi.read()
        
    # blunt_ends = [("A:1", 0)]
    # mk_prep = meeko.MoleculePreparation(
    #     merge_these_atom_types=[],
    #     load_atom_params=["vina_params", "openff"],
    #     charge_model="espaloma",
    # )
    # templates = meeko.ResidueChemTemplates.create_from_defaults()
    # polymer = meeko.Polymer.from_pdb_string(pdb_string=pdbstring,
    #                                         chem_templates=templates,
    #                                         mk_prep=mk_prep,
    #                                         allow_bad_res=True,
    #                                         default_altloc="A",
    #                                         blunt_ends=blunt_ends)
    # json_s = polymer.to_json()
    # with open("target.json", "w") as fo:
    #     fo.write(json_s)

    with open("/data/phd/waterkit/validation/hsp90_target/GCMC/target.json") as fi:
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
                                            charge_model="gasteiger")
        w_rdkit = Chem.MolFromSmiles("[H]O[H]")
        w_rdkit = Chem.AddHs(w_rdkit)
        rdkit.Chem.rdDistGeom.EmbedMolecule(w_rdkit)
        molsetup = mk_prep(w_rdkit)[0]
        res_coords = residue.getCoords()
        for idx, molsetup_atom in enumerate(molsetup.atoms):
            molsetup_atom.coord = res_coords[idx]
            # molsetup.atom_params["rmin_half"][molsetup_atom.index] = TIP3P_RMIN_HALF
            # molsetup.atom_params["epsilon"][molsetup_atom.index] = TIP3P_EPSILON
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

def get_molsetup_coords(molsetup):
    coords = dict()
    for atom in molsetup.atoms:
        if atom.atom_type == "N-TIP3P-O":
            coords["O"] = atom.coord
        elif atom.atom_type == "N-TIP3P-H":
            coords["H1"] = atom.coord
        else:
            coords["H2"] = atom.coord
    return coords

if __name__ == "__main__":
    polymer = get_data_form_meeko("/data/phd/waterkit/example/1uyg_no_ligand.pdb")
    # polymer=None
    
    # molsetups_names = ["/data/phd/waterkit/example/traj/water_000001.pdb",
    molsetups_names = list()
    for i in range(0, 1):
        molsetups_names.append(f"/data/phd/waterkit/validation/hsp90_target/GCMC/frames/water_{i}_optimized.pdb")

    # molsetups_names = ["/data/phd/waterkit/validation/hsp90_target/GCMC/frames/water_0_optimized.pdb"]
    for name in molsetups_names:
        molsetups = load_waters(name)
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
        print(energies.keys())
        # terms_of_interest = ["lj_12_6", "coulomb"]
        # mapping = {0: "Receptor", }
                #    1: f"Water at coords: {get_molsetup_coords(molsetups[0])}", 
                #    2: f"Water at coords: {get_molsetup_coords(molsetups[1])}",
                #    3: f"Water at coords: {get_molsetup_coords(molsetups[2])}",
                #    4: f"Water at coords: {get_molsetup_coords(molsetups[3])}"}
        # for term in terms_of_interest:
        #     data = energies['direct']['terms'][term]
        #     print(f"{term}")
        #     for idx, value in enumerate(data):
        #         print(f"\t{mapping[energies['direct']['pairs'][idx][0]]} - {mapping[energies['direct']['pairs'][idx][1]]}: {value}")
        
        # total_lj = 0
        # total_elec = 0
        # for idx, p in enumerate(energies['direct']['pairs']):
        #     if 1 in p:
        #         print(energies['direct']['terms']['lj_12_6'][idx])
        #         total_lj += energies['direct']['terms']['lj_12_6'][idx]
        #         total_elec += energies['direct']['terms']['coulomb'][idx]
        # print(f"Total LJ for {mapping[1]}:\n")
        # print(f"\t{total_lj}")
        # print(f"Total Coulomb for {mapping[1]}:\n")
        # print(f"\t{total_elec}")
        print(f"Total Energy for {name.split('/')[-1]}: {energies['direct_sum']}")


    #11760.157995647669
    # -797.8549102292691
    # -999.439463372513