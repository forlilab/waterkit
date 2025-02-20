import numpy as np
import parmed

from .molecule import Molecule
from .receptor_prep import PrepareReceptor
from .trajectory_utils import make_trajectory
from .trajectory_utils import WaterMinimizer
from .autogrid import calc_spherical_water_map
from .waterkit import WaterKit
from .utils import prepare_water_map
from .utils import temporary_directory
import os
import subprocess
import shutil


def run_waterkit_and_gist(
    receptor_parmed_structure_or_filename,
    output_prefix,
    box_center,
    box_size,
    autogrid_exec_path,
    ignore_gaps=False,
    keep_hydrogen=False,
    n_frames=10000,
    n_layer=3,
    n_jobs=-1,
    platform="CUDA",
):

    if len(box_center) != 3 or len(box_size) != 3:
        raise ValueError("length of box_size and box_center must be 3, but it's %d and %d" % (
            len(box_size), len(box_center)))

    size_types = set([type(c) for c in box_size])
    if len(size_types) != 1 or size_types.pop() not in (int, np.int64, np.int32):
        raise ValueError("Size of box must be specified with integers (Angstroms), got %s" % box_size)

    if type(receptor_parmed_structure_or_filename) is str:
        receptor_parmed_structure = parmed.load_file(receptor_parmed_structure_or_filename)
    else:
        receptor_parmed_structure = receptor_parmed_structure_or_filename

    autogrid_exec_path = os.path.abspath(autogrid_exec_path)

    # The following is a list of reasonable defaults that is probably not worth
    # exposing to the outside. There are further options that individual classes
    # expose, for example `no_disulfide` in PrepareReceptor.
    receptor_spherical_water_map_filename = None
    water_spherical_water_map_filename = None
    water_model="tip3p"
    temperature = 300.0
    renumber_receptor = False
    minimization_restraint = 2.5
    nr_minimization_steps = 100
    clean = True

    # will eventually need to expose these for systems with custom parameters
    lib_files = None
    frcmod_files = None
    
    original_dir = os.getcwd()
    output_dir = os.path.abspath(os.path.dirname(output_prefix))
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    prefix = os.path.basename(output_prefix)
    
    with temporary_directory(clean=clean) as tmp_dir:

        print(tmp_dir)
    
        # renumbering=False unlike in wk_prepare_receptor.py
        print("Preparing receptor")
        pr = PrepareReceptor(
            keep_hydrogen=keep_hydrogen,  # True causes trouble with meeko's blunt ends
            ignore_gaps=ignore_gaps,
            renumbering=renumber_receptor,
        )
        pr.prepare_from_parmed_structure(receptor_parmed_structure, clean=clean)
        pdb_fn = "%sprepared.pdb" % prefix
        pr.write_pdb_file(pdb_fn)
        shutil.copyfile(pdb_fn, os.path.join(output_dir, pdb_fn))
        amber_pdbqt_str = pr.write_pdbqt_string(amber_atom_types=True)

        print("Generating waterkit water configurations")
        receptor = Molecule.from_pdbqt_string(amber_pdbqt_str)
        ad_map = calc_spherical_water_map(
            autogrid_exec_path,
            receptor,
            box_center,
            box_size,
            receptor_spherical_water_map_filename,
        )
        prepare_water_map(ad_map, water_model=water_model)
        k = WaterKit(
            temperature,
            water_model,
            water_spherical_water_map_filename,
            n_layer,
            n_frames,
            n_jobs,
        )
        wk_frames_dir = "traj"
        os.mkdir(wk_frames_dir)
        k.hydrate(receptor, ad_map, wk_frames_dir)

        print("Creating trajectory")
        receptor_dry = parmed.load_file(pdb_fn)
        make_trajectory(receptor_dry, wk_frames_dir, prefix, lib_files, frcmod_files)

        print("Minimizing trajectory")
        prmtop_fn = "%ssystem.prmtop" % prefix
        traj_fn = "%ssystem.nc" % prefix
        minimized_traj_fn = "%ssystem_minimized.nc" % prefix
        m = WaterMinimizer(nr_minimization_steps, minimization_restraint, platform)
        m.minimize_trajectory(prmtop_fn, traj_fn, minimized_traj_fn)

        print("Running GIST")
        gist_input = ""
        gist_input += "parm %s\n" % prmtop_fn
        gist_input += "trajin %s\n" % minimized_traj_fn
        gist_input += "gist gridspacn 0.5 gridcntr %.3f %.3f %.3f griddim %d %d %d\n" % (
            box_center[0], box_center[1], box_center[2],
            box_size[0]*2, box_size[1]*2, box_size[2]*2,
        )
        gist_input += "go\n"
        gist_input += "quit\n"
        gist_input_fn = "gist_input.txt"
        with open(gist_input_fn, "w") as w:
            w.write(gist_input)
        out = subprocess.run(["cpptraj", "-i", gist_input_fn], capture_output=True)
        stdout = out.stdout.decode()
        print(stdout)

        for key in ("gO", "Esw-dens", "Eww-dens", "dTStrans-dens", "dTSorient-dens"):
            shutil.copyfile(
                "gist-%s.dx" % key,
                os.path.join(output_dir, "%swk_gist-%s.dx" % (prefix, key)),
            )
    return
