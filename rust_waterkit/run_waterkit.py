import numpy as np
import prody
import multiprocessing as mp
from tqdm import tqdm
import sys
import os
import meeko
import rust_waterkit

def load_anchor_points(filename):
    anchor_points = list()
    with open(filename) as fi:
        lines = fi.readlines()

    for (idx, line) in enumerate(lines):
        line = line.strip().split(" ")
        anchor_xyz = [float(line[0]), float(line[1]), float(line[2])]
        # if idx == 669:
        #     print(f"{anchor_xyz[0]}, {anchor_xyz[1]}, {anchor_xyz[2]}")
        vector_xyz = [float(line[3]), float(line[4]), float(line[5])]
        hb_type = line[-1]
        ap = rust_waterkit.AnchorPoint(hb_type, anchor_xyz, vector_xyz)
        anchor_points.append(ap)
    return anchor_points

def to_xyz(traj, step_size, fname):
    with open(f"{fname}.xyz", 'w') as fo:
        fo.write(f"{len(traj)}\n")
        fo.write("\n")
        for t in traj:
            t_v = t
            fo.write(f"He {t_v[0]} {t_v[1]} {t_v[2]}\n")
    return

def to_xyz_water(traj, fname):
    with open(f"{fname}.xyz", 'w') as fo:
        fo.write(f"{len(traj)}\n")
        fo.write("\n")
        fo.write(f"O {traj[0][0]} {traj[0][1]} {traj[0][2]}\n")
        for t in range(1, len(traj)):
            fo.write(f"H {traj[t][0]} {traj[t][1]} {traj[t][2]}\n")
        # fo.write(f"O {traj[-1][0]} {traj[-1][1]} {traj[-1][2]}\n")
    return

def get_data_form_meeko(wanted_residues, pdb_file, save=False):
    surface_atoms = list()
    # with open(pdb_file) as fi:
    #     pdbstring = fi.read()
        
    # # blunt_ends = [("A:1", 0)]
    # mk_prep = meeko.MoleculePreparation(
    #     merge_these_atom_types=[],
    #     load_atom_params=["vina_params", "openff"],
    #     charge_model="espaloma",
    # )
    box_boundaries = list()
    # templates = meeko.ResidueChemTemplates.create_from_defaults()
    # polymer = meeko.Polymer.from_pdb_string(pdb_string=pdbstring,
    #                                         chem_templates=templates,
    #                                         mk_prep=mk_prep,
    #                                         allow_bad_res=True,
    #                                         default_altloc="A",)
    #                                         # blunt_ends=blunt_ends)
    # json_s = polymer.to_json()
    # with open("target.json", "w") as fo:
    #     fo.write(json_s)

    # if save:
    #     pdb_f = polymer.to_pdb()
    #     with open("meeko.pdb", "w") as fo:
    #         fo.write(pdb_f)
    # with open("/data/phd/waterkit/rust_waterkit/target.json") as fi:
    with open("/home/niccolo/phd/waterkit/rust_waterkit/target.json") as fi:
        json_string = fi.read()

    polymer = meeko.Polymer.from_json(json_string)

    for res_id, monomer in polymer.get_valid_monomers().items():
        unique_id = f"{res_id.split(':')[0]}:{monomer.input_resname}:{res_id.split(':')[-1]}"
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
            # print(new_atom.rmin_half())
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
    return surface_atoms, min_box_boundaries, max_box_boundaries

def load_data_from_pdb(pdb_file):
    return

def to_pdb(pdb_file, w_map):
    ag = prody.AtomGroup('Surface')
    coords = []
    names = []
    resnames = []
    resnums = []
    cnt = 0
    for (idx, atom) in enumerate(w_map):
        coords.append(atom.coords())
        names.append(atom.atom_type())
        resnames.append("HOH")
        if idx % 3 == 0:
            cnt += 1
        resnums.append(cnt)

    ag.setCoords(coords)
    ag.setNames(names)
    ag.setResnames(resnames)
    ag.setResnums(resnums)
    # ag.setBetas(capped_energies)
    prody.writePDB(pdb_file, ag)
    return

def pdb_with_temp(pdb_file, traj, energies, atom_type="He"):
    capped_energies = list()
    for e in energies:
        if e > 100:
            capped_energies.append(100)
        else:
            capped_energies.append(e)
    ag = prody.AtomGroup('Surface')
    ag.setCoords(traj)
    ag.setNames([atom_type for _ in traj])
    ag.setResnames(["MOL" for _ in traj])
    ag.setResnums([1 for _ in traj])
    ag.setBetas(capped_energies)
    prody.writePDB(pdb_file, ag)
    return

def pdb_corners(pdb_file, traj, atom_type="He"):
    x_min, y_min, z_min = np.min(traj, axis=0)
    x_max, y_max, z_max = np.max(traj, axis=0)

    # Generate all combinations of extrema
    corners = np.array([
        [x_min, y_min, z_min],
        [x_min, y_min, z_max],
        [x_min, y_max, z_min],
        [x_min, y_max, z_max],
        [x_max, y_min, z_min],
        [x_max, y_min, z_max],
        [x_max, y_max, z_min],
        [x_max, y_max, z_max],
    ])
    ag = prody.AtomGroup('Surface')
    ag.setCoords(corners)
    ag.setNames([atom_type for _ in corners])
    ag.setResnames(["MOL" for _ in corners])
    ag.setResnums([1 for _ in corners])
    prody.writePDB(pdb_file, ag)

    x_center = sum(corner[0] for corner in corners) / 8

    y_center = sum(corner[1] for corner in corners) / 8

    z_center = sum(corner[2] for corner in corners) / 8

    print(x_center, y_center, z_center)
    return

# def load_waters_orientations(orientations="/data/phd/waterkit/waterkit/data/water_orientations.txt"):
def load_waters_orientations(orientations="/home/niccolo/phd/waterkit/waterkit/data/water_orientations.txt"):
    usecols = [0, 1, 2, 3, 4, 5]
    water_orientations = np.loadtxt(orientations, usecols=usecols)
    return water_orientations

def split_list_in_chunks(size, n):
    if size < n:
        n = size
    return [(l[0], l[-1]) for l in np.array_split(range(size), n)]

def fire_waterkit(parametrized_atoms, waters, aps, grid, use_grids, start=0, stop=1, position=0):
    progress = tqdm(total=stop - start, position=position,
                    desc='job %02d' % (position + 1),
                    bar_format='{l_bar}{bar:50}{r_bar}{bar:-10b}')

    for frame_id in range(start, stop + 1):
        rust_waterkit.run_waterkit(parametrized_atoms, waters, aps, grid, frame_id, use_grids)
        progress.update(1)
    progress.close()
    return

def parse_waters_frame(fname):
    frame = prody.parsePDB(fname)
    atoms = list()
    retvalue = list()
    for idx, atom in enumerate(frame):
        atom_type = "HW"
        rmin_half = 0.0
        epsilon = 0.0
        charge = 0.4170
        vina_rij = 0.0
        vina_donor = False
        vina_acceptor = False
        if atom.getElement() == "O":
            atom_type = "OW"
            rmin_half = 1.7682
            # rmin_half = 3.15061
            epsilon = 0.15210325
            charge = -0.834
            vina_rij = 1.7
            vina_donor = True
            vina_acceptor = True
        
        rust_atom = rust_waterkit.Atom(atom_type=atom_type,
                        atom_id=f"{idx}:{atom_type}",
                        coords_point=atom.getCoords(),
                        rmin_half=rmin_half,
                        epsilon=epsilon,
                        charge=charge,
                        vina_rij=vina_rij,
                        vina_donor=vina_donor,
                        vina_acceptor=vina_acceptor)
        atoms.append(rust_atom)
        if len(atoms) == 3:
            assert(len(atoms) == 3)
            retvalue.append(atoms)
            # print(retvalue)
            atoms = list()
    # print(retvalue[0])
    return retvalue

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
        # parametrized_atoms, min_box_boundaries, max_box_boundaries = get_data_form_meeko(wanted_residues, "/data/phd/waterkit/example/1uyg_no_ligand.pdb")
        parametrized_atoms, min_box_boundaries, max_box_boundaries = get_data_form_meeko(wanted_residues, "/home/niccolo/phd/waterkit/example/1uyg_no_ligand.pdb")

        waters = load_waters_orientations()
        # anchor_points = load_anchor_points("/data/phd/waterkit/rust_waterkit/anchor_points.txt")
        anchor_points = load_anchor_points("/home/niccolo/phd/waterkit/rust_waterkit/anchor_points.txt")
        spacing = 0.375
        center = [2.7, 11.45, 24.80]
        x_size, y_size, z_size = 24.0, 24.0, 24.0


        print("Starting waterkit!")
        aps = anchor_points
        n_frames = 1
        
        num_steps = [1, 10, 100, 1000, 10000]
        optimization_steps = [1, 10, 100, 1000, 10000]
        
        # Setup grids at the beginning
        grid = rust_waterkit.setup_system(parametrized_atoms, x_size, y_size, z_size, spacing, center)
        
        # for n_steps in num_steps:
        #     for o_steps in optimization_steps:
        n_steps = 40000
        o_steps = 10000
        save_path = f"test"
        os.makedirs(save_path, exist_ok=True)
        rust_waterkit.run_parallel_waterkit(parametrized_atoms, waters, anchor_points, grid, n_frames, n_steps, o_steps, save_path)

    # elif sys.argv[1] == "--energies":
    #     parametrized_atoms, min_box_boundaries, max_box_boundaries = get_data_form_meeko(None, "/data/phd/waterkit/rust_waterkit/minimal_test/receptor.pdbqt")
    #     # center = [71.5, 73.1, 243.3]
    #     # x_size, y_size, z_size = 21.0, 24.0, 26.0
        
    #     center = [12.4, 12.3, 14.1]
    #     x_size, y_size, z_size = 24.0, 24.0, 24.0

    #     frame_waters = parse_waters_frame("/data/phd/waterkit/rust_waterkit/minimal_test/traj/water_000001.pdb")
    #     rust_waterkit.get_energies_for_system(parametrized_atoms, frame_waters, center, x_size, y_size, z_size)
