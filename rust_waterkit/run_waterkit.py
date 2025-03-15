import numpy as np
import time
import sys
import os
import meeko
import rust_waterkit
from rdkit import Chem

def load_anchor_points(filename, rotatable_hydrogens):
    anchor_points = list()
    disordered_hydrogens = [x.atom_i_xyz() for x in rotatable_hydrogens]
    with open(filename) as fi:
        lines = fi.readlines()

    for (idx, line) in enumerate(lines):
        rotatable_bond = None
        line = line.strip().split(" ")
        anchor_xyz = [float(line[0]), float(line[1]), float(line[2])]
        for h_idx, h in enumerate(disordered_hydrogens):
            if np.allclose(h, anchor_xyz, atol=1e-7):
                rotatable_bond = rotatable_hydrogens[h_idx]
        vector_xyz = [float(line[3]), float(line[4]), float(line[5])]
        hb_type = line[-1]
        ap = rust_waterkit.AnchorPoint(idx, hb_type, anchor_xyz, vector_xyz, rotatable_bond)
        anchor_points.append(ap)
    return anchor_points

def get_data_form_meeko(wanted_residues, pdb_file, save=True):
    rotatable_hydrogens = list()
    rotatable_bonds = parse_rotatable_hydrogens()
    surface_atoms = list()
    box_boundaries = list()
    with open(pdb_file) as fi:
        pdbstring = fi.read()
        
    # # blunt_ends = [("A:1", 0)]
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
                                            default_altloc="A",)
                                            # blunt_ends=blunt_ends)
    json_s = polymer.to_json()
    with open("target.json", "w") as fo:
        fo.write(json_s)

    # if save:
    #     pdb_f = polymer.to_pdb()
    #     with open("meeko.pdb", "w") as fo:
    #         fo.write(pdb_f)
    # with open("/data/phd/waterkit/target.json") as fi:
    # with open("/home/niccolo/phd/waterkit/rust_waterkit/target.json") as fi:
        # json_string = fi.read()

    # polymer = meeko.Polymer.from_json(json_string)

    for res_id, monomer in polymer.get_valid_monomers().items():
        unique_id = f"{res_id.split(':')[0]}:{monomer.input_resname}:{res_id.split(':')[-1]}"
        matching_atoms_rotatable = None
        for rot_bond in rotatable_bonds:
            pattern = rotatable_bonds[rot_bond]
            matching_atoms_rotatable = monomer.rdkit_mol.GetSubstructMatches(pattern)
            if len(matching_atoms_rotatable) > 0: 
                map_idx = {v: k for k, v in monomer.molsetup_mapidx.items()}
                atom_i_xyz = monomer.molsetup.atoms[map_idx[matching_atoms_rotatable[0][0]]].coord
                atom_j_xyz = monomer.molsetup.atoms[map_idx[matching_atoms_rotatable[0][1]]].coord
                atom_k_xyz = monomer.molsetup.atoms[map_idx[matching_atoms_rotatable[0][2]]].coord
                atom_l_xyz = monomer.molsetup.atoms[map_idx[matching_atoms_rotatable[0][3]]].coord
                rotatable_hydrogens.append(rust_waterkit.RotatableBond(atom_i_xyz=atom_i_xyz,
                                                                   atom_j_xyz=atom_j_xyz,
                                                                   atom_k_xyz=atom_k_xyz,
                                                                   atom_l_xyz=atom_l_xyz))
                break
        for atom in monomer.molsetup.atoms:
            if atom.is_ignore:
                continue
            rmin_half = monomer.molsetup.atom_params["rmin_half"][atom.index]
            epsilon = monomer.molsetup.atom_params["epsilon"][atom.index]
            vina_rij = monomer.molsetup.atom_params["vina_ri"][atom.index]
            vina_donor = monomer.molsetup.atom_params["vina_donor"][atom.index]
            vina_acceptor = monomer.molsetup.atom_params["vina_acceptor"][atom.index]
            if vina_rij is None:
                vina_rij = 0.0
                vina_donor = False
                vina_acceptor = False
            charge = atom.charge
            atom_type = atom.pdbinfo.name
            new_atom = rust_waterkit.Atom(atom_type=atom_type,
                        atom_id=f"{unique_id}:{atom_type}",
                        coords_point=atom.coord,
                        rmin_half=rmin_half,
                        epsilon=epsilon,
                        charge=charge,
                        vina_rij=vina_rij,
                        vina_donor=vina_donor,
                        vina_acceptor=vina_acceptor)
            surface_atoms.append(new_atom)
            if wanted_residues is not None and unique_id in wanted_residues:
                box_boundaries.append(atom.coord)
    try:
        box_boundaries = np.array(box_boundaries)
        min_box_boundaries = [np.min(box_boundaries[:, 0]), np.min(box_boundaries[:, 1]), np.min(box_boundaries[:, 2])]
        max_box_boundaries = [np.max(box_boundaries[:, 0]), np.max(box_boundaries[:, 1]), np.max(box_boundaries[:, 2])]
    except:
        min_box_boundaries = list()
        max_box_boundaries = list()
    return surface_atoms, min_box_boundaries, max_box_boundaries, rotatable_hydrogens

def parse_rotatable_hydrogens(path="/data/phd/waterkit/rust_waterkit/disordered_hydrogens.par"):
    rotatable_bonds_smarts = dict()
    with open(path) as f:
        lines = f.readlines()

    for line in lines:
        line = line.strip()
        if line.startswith("#"):
            continue
        # if re.search(r"^[A-Za-z0-9].*?(\[.*?\]){4}( [0-9]){4} -?[0-9]{1,3}", line):
        sline = line.split(" ")
        if len(line) > 1:
            name = sline[0]
            smarts_pattern = Chem.MolFromSmarts(sline[1])
            rotatable_bonds_smarts[name] = smarts_pattern
    return rotatable_bonds_smarts



def load_waters_orientations(orientations="/data/phd/waterkit/waterkit/data/water_orientations.txt"):
# def load_waters_orientations(orientations="/home/niccolo/phd/waterkit/waterkit/data/water_orientations.txt"):
    usecols = [0, 1, 2, 3, 4, 5]
    water_orientations = np.loadtxt(orientations, usecols=usecols)
    return water_orientations

'''
    To compile the code:
        maturin develop -r` --features extension-module
'''
if __name__ == "__main__":
    if len(sys.argv) < 2:
        # select as, i. 111+107+103+162+150+98+97+184+96+93+55+52+51+138+139+136+135
        surface_points = []
        wanted_residues = ["A:ASN:51", "A:SER:52", "A:ALA:55",
                           "A:ASP:93", "A:ILE:96", "A:GLY:97",
                           "A:MET:98", "A:LEU:103", "A:LEU:107",
                           "A:ALA:111", "A:GLY:135", "A:VAL:136",
                           "A:PHE:138", "A:TYR:139", "A:VAL:150",
                           "A:TRP:162", "A:THR:184"]
        # wanted_residues = list()
        parametrized_atoms, min_box_boundaries, max_box_boundaries, rotatable_hydrogens = get_data_form_meeko(wanted_residues, "/data/phd/waterkit/protein_prepared.pdb")
        # parametrized_atoms, min_box_boundaries, max_box_boundaries = get_data_form_meeko(wanted_residues, "/home/niccolo/phd/waterkit/example/1uyg_no_ligand.pdb")

        waters = load_waters_orientations()
        anchor_points = load_anchor_points("/data/phd/waterkit/anchor_points_translated.txt", rotatable_hydrogens)
        # anchor_points = load_anchor_points("/home/niccolo/phd/waterkit/rust_waterkit/anchor_points.txt")
        # with open("anchor_points.xyz", "w") as fo:
        #     fo.write(f"{len(anchor_points)}\n\n")
        #     for anchor_point in anchor_points:
        #         ap_xyz = anchor_point.anchor_point()
        #         fo.write(f"H {ap_xyz[0]} {ap_xyz[1]} {ap_xyz[2]}\n")
        
        spacing = 0.375
        # center = [2.7, 11.45, 24.80]
        center = [32.610, 28.188, 36.505]
        x_size, y_size, z_size = 24.0, 24.0, 24.0

        

        print("Starting waterkit!")
        aps = anchor_points
        n_frames = 1000

        # num_steps = [1, 10, 100, 1000, 10000]
        # optimization_steps = [1, 10, 100, 1000, 10000]
        
        # # Setup grids at the beginning
        start = time.time()
        grid = rust_waterkit.setup_system(parametrized_atoms, x_size, y_size, z_size, spacing, center)
        # optimized_receptor, new_anchor_points = rust_waterkit.optimize_disordered_hydrogens(parametrized_atoms, aps, grid)
        # with open("new_anchor_points.xyz", "w") as fo:
        #     fo.write(f"{len(anchor_points)}\n\n")
        #     for anchor_point in new_anchor_points:
        #         ap_xyz = anchor_point.anchor_point()
        #         fo.write(f"H {ap_xyz[0]} {ap_xyz[1]} {ap_xyz[2]}\n")
        # for point in optimized_receptor:
        #     coords = point.coords()
        #     print(f"{point.atom_type()} {coords[0]} {coords[1]} {coords[2]}")
        # grid = rust_waterkit.setup_system(optimized_receptor, x_size, y_size, z_size, spacing, center)

        # for n_steps in num_steps:
        #     for o_steps in optimization_steps:
        n_steps = 5000
        o_steps = 1000
        save_path = f"test"
        os.makedirs(save_path, exist_ok=True)
        rust_waterkit.run_parallel_waterkit(parametrized_atoms, waters, aps, grid, n_frames, n_steps, o_steps, save_path)
        print(f"Time necessary for the rust part: {time.time() - start}")
