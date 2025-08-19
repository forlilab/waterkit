import io
import numpy as np
import time
import sys
import os
import meeko
import rust_waterkit
from rdkit import Chem
from openmm.app import *
from openmm import *
import openmm.unit as openmmunit

def get_data_from_meeko(pdb_file, project_path, save=True):
    rotatable_hydrogens = list()
    surface_atoms = list()
    box_boundaries = list()
    with open(pdb_file) as fi:
        pdbstring = fi.read()
        
    # # blunt_ends = [("A:1", 0)]
    mk_prep = meeko.MoleculePreparation(
        merge_these_atom_types=[],
        load_atom_params=["vina_params", "openff"],
        charge_model="gasteiger",
    )
    
    templates = meeko.ResidueChemTemplates.create_from_defaults()
    polymer = meeko.Polymer.from_pdb_string(pdb_string=pdbstring,
                                            chem_templates=templates,
                                            mk_prep=mk_prep,
                                            allow_bad_res=True,
                                            default_altloc="A",)
                                            # blunt_ends=blunt_ends)
    json_s = polymer.to_json()
    with open(f"{project_path}/target.json", "w") as fo:
        fo.write(json_s)

    if save:
        pdb_f = polymer.to_pdb()
        with open(f"{project_path}/meeko.pdb", "w") as fo:
            fo.write(pdb_f)
    with open(f"{project_path}/target.json") as fi:
    # with open("/Users/niccolobruciaferri/phd/waterkit/rust_waterkit/target.json") as fi:
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
            surface_atoms.append(new_atom)
    return surface_atoms

def _build_box(positions, padding=2*openmmunit.angstrom):
        """Builds the simulation box
        """
        padding = padding.value_in_unit(openmmunit.nanometer)
        positions = positions.value_in_unit(openmmunit.nanometer)
        minRange = Vec3(*(min((pos[i] for pos in positions)) for i in range(3)))
        maxRange = Vec3(*(max((pos[i] for pos in positions)) for i in range(3)))
        center = 0.5*(minRange+maxRange)
        radius = max(unit.norm(center-pos) for pos in positions)
        width = max(2*radius+padding, 2*padding)
        vectors = (Vec3(width, 0, 0), Vec3(0, width, 0), Vec3(0, 0, width))
        box = Vec3(vectors[0][0], vectors[1][1], vectors[2][2])
        origin = center - (np.ceil(np.array((width, width, width))))
        _box_origin = origin
        _box_size = np.ceil(np.array([maxRange[0]-minRange[0],
                                           maxRange[1]-minRange[1],
                                           maxRange[2]-minRange[2]])).astype(int)
        lowerBound = center-box/2
        upperBound = center+box/2
        return vectors

def get_data_from_openmm(pdb_file, project_path, protein_ff="amber14-all.xml", small_molecule=False, small_molecule_ff="espaloma"):
    with open(pdb_file) as f:
        pdb_string = f.read()
    pdb_string = io.StringIO(pdb_string)
    if not pdb_file.endswith(".pdb"):
        pdb = PDBxFile(pdb_string)
    else:
        pdb = PDBFile(pdb_string)
    protein_topology, protein_positions = pdb.topology, pdb.positions
    protein_modeller = Modeller(protein_topology, protein_positions)
    vectors = _build_box(protein_positions)
    protein_modeller.topology.setPeriodicBoxVectors(vectors)
    protein_atoms = [atom for atom in protein_topology.atoms() if atom.residue.name not in ["HOH"]]
    forcefield = ForceField(protein_ff)
    # if small_molecule:
    #     small_molecule_ff = self._parametrize_cosolvents(cosolvents, small_molecule_ff=sm_ff)
    #     forcefield.registerTemplateGenerator(small_molecule_ff.generator)
    system = forcefield.createSystem(protein_topology,
                                    nonbondedMethod=PME,
                                    nonbondedCutoff=10*openmmunit.angstrom,
                                    switchDistance=9*openmmunit.angstrom,
                                    removeCMMotion=True,
                                    constraints=HBonds,
                                    hydrogenMass=1.0*openmmunit.amu)

    # Retrieve all Force objects from the System
    forces = system.getForces()
    mcswell_atoms = list()
    # Find the NonbondedForce object
    nonbonded_force = [f for f in forces if isinstance(f, NonbondedForce)][0]
    for idx, atom in enumerate(protein_atoms):
        charge, sigma, epsilon = nonbonded_force.getParticleParameters(atom.index)
        atom_type = atom.name
        residue = atom.residue
        chain = residue.chain
        chain_id = chr(ord('@')+ chain.index + 1)
        unique_id = f"{chain_id}:{residue.name}:{residue.index}"
        atom_id = f"{unique_id}:{atom_type}"
        coords = protein_positions[idx].in_units_of(openmmunit.angstrom)._value
        rmin_half_value = rmin_half(sigma.in_units_of(openmmunit.angstrom))._value
        epsilon_value = epsilon.in_units_of(openmmunit.kilocalories_per_mole)._value
        charge_value = charge.in_units_of(openmmunit.elementary_charge)._value
        new_atom = rust_waterkit.Atom(atom_type=atom_type,
                                      atom_id=atom_id,
                                      coords_point=[coords.x, coords.y, coords.z],
                                      rmin_half=rmin_half_value,
                                      epsilon=epsilon_value,
                                      charge=charge_value,
                                      vina_rij=0.0,
                                      vina_donor=False,
                                      vina_acceptor=False)
        mcswell_atoms.append(new_atom)
    return mcswell_atoms

def rmin_half(sigma):
        """
        The VdW rmin_half term from sigma
        """
        rmin = 2.0 ** (1.0 / 6) * sigma
        rmin_half = rmin / 2
        return rmin_half

def load_waters_orientations(orientations="/data/phd/waterkit/waterkit/data/water_orientations.txt"):
# def load_waters_orientations(orientations="/Users/niccolobruciaferri/phd/waterkit/waterkit/data/water_orientations.txt"):
    usecols = [0, 1, 2, 3, 4, 5]
    water_orientations = np.loadtxt(orientations, usecols=usecols)
    return water_orientations


def run_mcswell(receptor_path, project_path, center, alg_type="gcmc"):
    parametrized_atoms = get_data_from_openmm(pdb_file=receptor_path, project_path=project_path)
    spacing = 0.375
    x_size, y_size, z_size = 24.0, 24.0, 24.0
    n_frames = 500
    print("Starting MCSwell!")
    start = time.time()
    grid = rust_waterkit.setup_system(parametrized_atoms, x_size, y_size, z_size, spacing, center)
    save_path = f"{project_path}/frames/"
    os.makedirs(save_path, exist_ok=True)
    # sa_steps to be adjusted
    if alg_type == "gcmc":
        # rust_waterkit.run_waterkit_gcmc(parametrized_atoms, [], grid, n_frames, 400000, save_path)
        rust_waterkit.test_gpu(parametrized_atoms, [], grid, n_frames, 400000, save_path)  
    elif alg_type == "gcmcmc":
        rust_waterkit.run_waterkit_gcmcmc(parametrized_atoms, [], grid, n_frames, 400000, 75000, save_path)
    elif alg_type == "gcmcsa":
        rust_waterkit.run_parallel_waterkit(parametrized_atoms, [], [], grid, n_frames, 400000, 75000, save_path)
    else:
        print(f"Error! {alg_type} not available in the allowed algorithms!\nPlease chooes between 1) gcmc 2) gcmcmc 3) gcmcsa")
    exec_time = time.time() - start
    print(f"Time necessary for the rust part: {exec_time/60} minutes - {exec_time} seconds")
    return

'''
    To compile the code:
        maturin develop -r` --features extension-module
'''
if __name__ == "__main__":
    #center_prepared = [2.6995912 11.453865  24.802498]
    # center = [33.714676 30.430138 35.779892]
    print(sys.argv)
    pdb_path = sys.argv[1]
    project_path = sys.argv[2]
    center = [float(arg) for arg in sys.argv[3:6]]
    # sa_steps = int(sys.argv[6])
    # sa_steps = 75000
    parametrized_atoms = get_data_from_meeko(pdb_file=pdb_path, project_path=project_path)
    waters = load_waters_orientations()
    spacing = 0.375
    x_size, y_size, z_size = 24.0, 24.0, 24.0

    sa_intervals = [1000, 5000, 10000, 20000, 40000, 50000, 70000, 80000, 90000, 100000, 200000]

    print("Starting waterkit!")
    n_frames = 1000
    # if gcmc_steps < 50000:
    #     n_frames = 10000
    # else:
    #     n_frames = 1000
    
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
    for sa_steps in sa_intervals:
        save_path = f"{project_path}/{sa_steps}_steps/frames/"
        os.makedirs(save_path, exist_ok=True)
        rust_waterkit.run_parallel_waterkit(parametrized_atoms, waters, [], grid, n_frames, 100, sa_steps, save_path)
        # rust_waterkit.run_waterkit_gcmcre(parametrized_atoms, waters, grid, n_frames, save_path)
        exec_time = time.time() - start
        print(f"Time necessary for the rust part: {exec_time/60} minutes - {exec_time} seconds")
