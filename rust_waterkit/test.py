import numpy as np
import prody

import meeko
import rust_waterkit

def load_anchor_points(filename):
    ap_coords = list()
    with open(filename) as fi:
        lines = fi.readlines()

    for line in lines:
        line = line.strip().split(",")
        ap_coords.append(np.array([float(line[0]), float(line[1]), float(line[2])]))
    return ap_coords

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

def get_data_form_meeko(wanted_residues, pdb_file):
    surface_atoms = list()
    with open(pdb_file) as fi:
        pdbstring = fi.read()
    mk_prep = meeko.MoleculePreparation(
        merge_these_atom_types=[],
        load_atom_params=["openff"],
        charge_model="gasteiger",
    )
    box_boundaries = list()
    templates = meeko.ResidueChemTemplates.create_from_defaults()
    polymer = meeko.Polymer.from_pdb_string(pdb_string=pdbstring, 
                                            chem_templates=templates, 
                                            mk_prep=mk_prep, 
                                            allow_bad_res=True, 
                                            default_altloc="A")
    for res_id, monomer in polymer.get_valid_monomers().items():
        unique_id = f"{res_id.split(':')[0]}:{monomer.input_resname}:{res_id.split(':')[-1]}"
        for atom in monomer.molsetup.atoms:
            if atom.is_ignore:
                continue
            rmin_half = monomer.molsetup.atom_params["rmin_half"][atom.index]
            epsilon = monomer.molsetup.atom_params["epsilon"][atom.index]
            charge = atom.charge
            atom_type = atom.pdbinfo.name
            new_atom = rust_waterkit.Atom(atom_type=atom_type, 
                        atom_id=f"{unique_id}:{atom_type}", 
                        coords_point=atom.coord, 
                        rmin_half=rmin_half, 
                        epsilon=epsilon, 
                        charge=charge)
            surface_atoms.append(new_atom)
            if unique_id in wanted_residues:
                box_boundaries.append(atom.coord)
    box_boundaries = np.array(box_boundaries)
    min_box_boundaries = [np.min(box_boundaries[:, 0]), np.min(box_boundaries[:, 1]), np.min(box_boundaries[:, 2])]
    max_box_boundaries = [np.max(box_boundaries[:, 0]), np.max(box_boundaries[:, 1]), np.max(box_boundaries[:, 2])]
    return surface_atoms, min_box_boundaries, max_box_boundaries

def to_pdb(pdb_file, w_map):
    ag = prody.AtomGroup('Surface')
    coords = []
    names = []
    resnames = []
    for atom in w_map:
        coords.append(atom.coords())
        names.append(atom.atom_type())
        resnames.append("HOH")
    ag.setCoords(coords)
    ag.setNames(names)
    ag.setResnames(resnames)
    ag.setResnums([1 for _ in w_map])
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

def load_waters_orientations(orientations="/data/phd/waterkit/waterkit/data/water_orientations.txt"):
    usecols = [0, 1, 2, 3, 4, 5]
    water_orientations = np.loadtxt(orientations, usecols=usecols)
    # shape = (water_orientations.shape[0], 2, 3)
    # water_orientations_reshaped = water_orientations.reshape(shape)
    return water_orientations

if __name__ == "__main__":
    # select as, i. 111+107+103+162+150+98+97+184+96+93+55+52+51+138+139+136+135
    import time
    
    surface_points = []
    wanted_residues = ["A:ASN:51", "A:SER:52", "A:ALA:55",
                       "A:ASP:93", "A:ILE:96", "A:GLY:97",
                       "A:MET:98", "A:LEU:103", "A:LEU:107", 
                       "A:ALA:111", "A:GLY:135", "A:VAL:136", 
                       "A:PHE:138", "A:TYR:139", "A:VAL:150", 
                       "A:TRP:162", "A:THR:184"]
    # wanted_residues = list()
    parametrized_atoms, min_box_boundaries, max_box_boundaries = get_data_form_meeko(wanted_residues, "/data/phd/waterkit/example/1uyg_no_ligand.pdb")
    # parametrized_atoms = get_data_form_meeko(wanted_residues, "/home/niccolo/phd/waterkit/example/1uyg.pdb")
    waters = load_waters_orientations()
    anchor_points = load_anchor_points("/data/phd/waterkit/rust_waterkit/anchor_points_hsp90.txt")
    spacing = 0.375
    center = [2.7, 11.45, 24.80]
    # center = [2.699591, 11.453864, 24.802502]
    x_size, y_size, z_size = 24.0, 24.0, 24.0
    # print("Starting waterkit!")
    # start = time.time()
    # energies, waters_map = rust_waterkit.get_map(parametrized_atoms, waters, x_size, y_size, z_size, spacing, center)
    # # trajectories, energies = rust_waterkit.run_waterkit(parametrized_atoms, waters, step_size)
    # # energies, trajectories = rust_waterkit.roll_sphere_and_compute_energies(parametrized_atoms, step_size)
    # print(f"Time to grid: {time.time() - start}")
    # # to_xyz(trajectories, step_size, "trajectory")
    # pdb_with_temp(f"map.pdb", energies=energies, traj=waters_map)
    # # pdb_corners(f"box.pdb", traj=waters_map)

    # Allowed points
    # start = time.time()
    # energies, waters_map = rust_waterkit.test_allowed_points(parametrized_atoms, x_size, y_size, z_size, spacing, center)
    # # trajectories, energies = rust_waterkit.run_waterkit(parametrized_atoms, waters, step_size)
    # # energies, trajectories = rust_waterkit.roll_sphere_and_compute_energies(parametrized_atoms, step_size)
    # print(f"Time to grid: {time.time() - start}")
    # # to_xyz(trajectories, step_size, "trajectory")
    # pdb_with_temp(f"map_allowed.pdb", energies=energies, traj=waters_map)


    # Shell related stuff
    # start = time.time()
    # (e, map) = rust_waterkit.save_shell_points_with_energies(parametrized_atoms, waters, anchor_points, x_size, y_size, z_size, spacing, center, shells=1)
    # print(f"Time to grid: {time.time() - start}")
    # pdb_with_temp("shell.pdb", map, e)

    print("Starting waterkit!")
    start = time.time()
    # aps = [[-5.15488359,  9.00213151, 34.99793656]]
    aps = anchor_points
    wk_map = rust_waterkit.run_waterkit(parametrized_atoms, waters, aps, x_size, y_size, z_size, spacing, center, epochs=1)
    for (idx, m) in enumerate(wk_map):
        waters_map = [x for x in m if x.atom_type() == "HW" or x.atom_type() == "OW"]
        # print(f"Time to grid: {time.time() - start}")
        to_pdb(f"waterkit_{idx}.pdb", waters_map)
    
    
    # Test ordered
    # start = time.time()
    # (e, map) = rust_waterkit.order_ap(parametrized_atoms, waters, anchor_points, x_size, y_size, z_size, spacing, center, shells=1)
    # print(f"Time to grid: {time.time() - start}")
    # for idx, v in enumerate(e):
    #     pdb_with_temp(f"ap_{idx}.pdb", [map[idx]], [v])
    
    # Test new anchor points
    # start = time.time()
    # ap = rust_waterkit.test_new_aps(parametrized_atoms, waters, anchor_points, x_size, y_size, z_size, spacing, center, shells=1)
    # print(f"Time to grid: {time.time() - start}")
    # to_xyz_water(ap, "new_aps.xyz")


    # print("Starting waterkit!")
    # start = time.time()
    # (j_wat, n_wat) = rust_waterkit.get_energy_for_water(parametrized_atoms)
    # to_xyz_water(j_wat, "jerome_wat")
    # to_xyz_water(n_wat, "nico_wat")
    