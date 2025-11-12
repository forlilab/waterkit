#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# make_trajectory
#

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import copy
import os
import re
import sys
from packaging.version import Version
from shutil import copyfile

import numpy as np
import parmed as pmd
from openmm.unit import Quantity, picoseconds, kilocalories_per_mole, angstroms, nanometer, kelvin, kilocalories, mole
from openmm import CustomExternalForce, LangevinIntegrator, Platform
from openmm.app import AmberPrmtopFile, HBonds, CutoffNonPeriodic, Simulation
from parmed.amber import NetCDFTraj
from pdb4amber import pdb4amber
from pdb4amber.utils import easy_call

from waterkit import utils


# region Make trajectory

def max_water(water_filenames):
    """Return the max number of water molecules seen
    
    Args:
        water_filenames (array-like): list of filenames of the water files

    Return:
        int: number of max water molecules
        int: index of the file in the water filenames list
    """
    sizes = [os.path.getsize(f) for f in water_filenames]
    idx = np.argmax(sizes)
    m = pmd.load_file(water_filenames[idx])
    # We select only the water molecules, because we might have ions, etc...
    m = m["@O, H1, H2"]
    max_water = len(m.residues)
    return max_water, idx


def min_water(water_filenames):
    """Return the min number of water molecules seen

    Args:
        water_filenames (array-like): list of filenames of the water files

    Return:
        int: number of min water molecules
        int: index of the file in the water filenames list
    """
    sizes = [os.path.getsize(f) for f in water_filenames]
    idx = np.argmin(sizes)
    m = pmd.load_file(water_filenames[idx])
    # We select only the water molecules, because we might have ions, etc...
    m = m["@O, H1, H2"]
    max_water = len(m.residues)

    return max_water, idx


def add_water_to_receptor(receptor, water):
    receptor_wet = copy.deepcopy(receptor)
    receptor_wet += water["@O, H1, H2"]
    # ParmED really want a symmetry attributes to write the PDB file
    receptor_wet.symmetry = None

    return receptor_wet


def write_pdb_file(output_name, molecule,  overwrite=True, **kwargs):
    '''Write PDB file

    Args:
        output_name (str): pdbqt output filename
        molecule (parmed): parmed molecule object

    '''
    try:
        molecule.save(output_name, format='pdb', overwrite=overwrite, **kwargs)
    except IOError:
        raise IOError("Error: file %s already exists." % fname)


def write_tleap_input_file(fname, pdb_filename, lib_files=None, frcmod_files=None):
    """Create tleap input script

    Args:
        fname (str): tleap input filename
        pdb_filename (str): pdb filename
        lib_files (list): Amber lib parameter files for non-standard residues
        frcmod_files (list): Amber frcmod parameter files for non-standard residues

    """
    prefix = pdb_filename.split(".pdb")[0].split("/")[-1]

    output_str = "source leaprc.protein.ff14SB\n"
    output_str += "source leaprc.DNA.OL15\n"
    output_str += "source leaprc.RNA.OL3\n"
    output_str += "source leaprc.water.tip3p\n"
    output_str += "source leaprc.gaff2\n"
    if frcmod_files is not None:
        output_str += ''.join(['loadamberparams %s\n' % fl for fl in frcmod_files])
    if lib_files is not None:
        output_str += ''.join(['loadoff %s\n' % ll for ll in lib_files])
    output_str += "\n"
    output_str += "x = loadpdb %s\n" % pdb_filename.split("/")[-1]
    output_str += "\n"
    output_str += "set default nocenter on\n"
    output_str += "saveAmberParm x %s.prmtop %s.rst7\n" % (prefix, prefix)
    output_str += "quit\n"

    with open(fname, "w") as w:
        w.write(output_str)


def write_trajectory_file(fname, receptor, water_filenames):
    """Create netcdf trajectory from the water pdb file

    Args:
        fname (str): output name for the trajectory
        receptor_filename (parmed): parmed receptor object
        water_filenames (list): list of filenames of the water files

    """
    max_n_waters, idx = max_water(water_filenames)
    min_n_waters, _ = min_water(water_filenames)
    buffer_n_waters = max_n_waters - min_n_waters
    max_n_water_atoms = max_n_waters * 3

    n_atoms = receptor.coordinates.shape[0]
    max_n_atoms = n_atoms + (max_n_waters * 3)
    coordinates = np.zeros(shape=(max_n_atoms, 3))

    # Boz dimension
    box_center = np.mean(receptor.coordinates, axis=0)

    x_min, x_max = np.min(receptor.coordinates[:, 0]), np.max(receptor.coordinates[:, 0])
    y_min, y_max = np.min(receptor.coordinates[:, 1]), np.max(receptor.coordinates[:, 1])
    z_min, z_max = np.min(receptor.coordinates[:, 2]), np.max(receptor.coordinates[:, 2])
    box_size = np.max([np.abs(x_max - x_min), np.abs(y_max - y_min), np.abs(z_max - z_min)]) + 40

    box = [box_size, box_size, box_size, 90, 90, 90]

    # Dummy water coordinates
    dum_water_x = np.array([0, 0, 0])
    dum_water_y = np.array([0, 0.756, 0.586])
    dum_water_z = np.array([0, -0.756, 0.586])

    radius = (box_size / 2.) - 2.
    z = np.random.uniform(-radius, radius, buffer_n_waters)
    p = np.random.uniform(0, np.pi * 2, buffer_n_waters)
    x = np.sqrt(radius**2 - z**2) * np.cos(p)
    y = np.sqrt(radius**2 - z**2) * np.sin(p)
    oxygen_xyz = np.stack((x, y, z), axis=-1)
    oxygen_xyz += box_center

    dummy_water_xyz = np.zeros(shape=(buffer_n_waters * 3, 3))
    dummy_water_xyz[0::3] = oxygen_xyz
    dummy_water_xyz[1::3] = oxygen_xyz + dum_water_y
    dummy_water_xyz[2::3] = oxygen_xyz + dum_water_z

    # Already add the coordinates from the receptor
    coordinates[:n_atoms] = receptor.coordinates

    trj = NetCDFTraj.open_new(fname, natom=max_n_atoms, box=True, crds=True)

    for i, water_filename in enumerate(water_filenames):
        m = pmd.load_file(water_filename)

        last_atom_id = len(m.residues) * 3
        water_xyz = m["@O, H1, H2"].coordinates

        # Get all the TIP3P water molecules
        coordinates[n_atoms:n_atoms + last_atom_id] = water_xyz
        # Add the dummy water molecules
        coordinates[n_atoms + last_atom_id:] = dummy_water_xyz[:max_n_water_atoms - water_xyz.shape[0]]

        trj.add_coordinates(coordinates)
        trj.add_box(box)
        trj.add_time(i + 1)

    trj.close()


def _make_trajectory(receptor_dry, water_directory, output_prefix, lib_files, frcmod_files):

    tleap_input = 'leap.template.in'
    tleap_output = 'leap.template.out'
    tleap_log = 'leap.log'

    water_filenames = []
    for fname in os.listdir(water_directory):
        if re.match(r"water_[0-9]{6}.pdb", fname):
            water_filenames.append(os.path.join(water_directory, fname))

    """ Add water molecules to the dry receptor and write pdb wet receptor
    We are taking the water coordinates from the frame that have the
    max number of water molecules. Because the number of water molecules
    need to be constant during the trajectory. This is just for creating 
    the amber topology (and coordinate) file(s)."""
    water = pmd.load_file(water_filenames[max_water(water_filenames)[1]])
    receptor_wet = add_water_to_receptor(receptor_dry, water)
    pdb_fn = "%ssystem.pdb" % output_prefix
    write_pdb_file(pdb_fn, receptor_wet)

    # Write tleap input script
    write_tleap_input_file(tleap_input, pdb_fn, lib_files, frcmod_files)

    try:
        # Generate amber prmtop and rst7 files
        easy_call('tleap -s -f %s > %s' % (tleap_input, tleap_output), shell=True)
    except RuntimeError:
        error_msg = 'Could not generate topology/coordinates files with tleap.'
        raise RuntimeError(error_msg)

    # Write trajectory
    write_trajectory_file("%ssystem.nc" % output_prefix, receptor_dry, water_filenames)


def make_trajectory(receptor_dry, water_directory, output_prefix, lib_files, frcmod_files):
    water_directory = os.path.abspath(water_directory)
    if lib_files is not None:
        lib_files = [os.path.abspath(fn) for fn in lib_files]
    if frcmod_files is not None:
        frcmod_files = [os.path.abspath(fn) for fn in frcmod_files]
    abs_output_dir = os.path.abspath(os.path.dirname(output_prefix))
    prefix = output_prefix.split("/")[-1]  # split internally in write_tleap_input_file()
    with utils.temporary_directory(clean=True) as tmp_dir:
        _make_trajectory(receptor_dry, water_directory, prefix,lib_files, frcmod_files)
        copyfile("%ssystem.rst7" % prefix, os.path.join(abs_output_dir, "%ssystem.rst7" % prefix))
        copyfile("%ssystem.prmtop" % prefix, os.path.join(abs_output_dir, "%ssystem.prmtop" % prefix))
        copyfile("%ssystem.nc" % prefix, os.path.join(abs_output_dir, "%ssystem.nc" % prefix))
        copyfile("%ssystem.pdb" % prefix, os.path.join(abs_output_dir, "%ssystem.pdb" % prefix))

# endregion

# region Minimize trajectory

def _box_information(traj_filename):
    box = None

    try:
        traj = NetCDFTraj.open_old(traj_filename)

        if traj.hasbox:
            box = traj.box[0]

        traj.close()
    except FileNotFoundError as err:
        print("Cannot find trajectory file: %s" % traj_filename)
        raise err

    return box


class WaterMinimizer:
    def __init__(self, n_steps=100, restraint=None, platform="OpenCL", verbose=True):
        self._n_steps = n_steps
        self._restraint = restraint
        self._platform = platform
        self._verbose = verbose

    def minimize_trajectory(self, prmtop_filename, traj_filename, output_filename):
        nonbondedMethod = CutoffNonPeriodic
        nonbondedCutoff = 9 * angstroms
        rigidWater = True
        constraints = HBonds
        dt = 0.002 * picoseconds
        temperature = 300 * kelvin
        friction = 1.0 / picoseconds
        K = self._restraint * kilocalories_per_mole / angstroms**2

        # If someone still uses an old version of OpenMM
        if Version(Platform.getOpenMMVersion()) > Version("7.5.0"):
            tolerance = 1.0 * kilocalories / (nanometer * mole)
        else:
            tolerance = 1.0 * kilocalories_per_mole

        box = _box_information(traj_filename)

        platform = Platform.getPlatformByName(self._platform)
        platformProperties = {'Precision': 'single'}

        prmtop = AmberPrmtopFile(prmtop_filename)
        parmedtop = pmd.load_file(prmtop_filename)
        old_traj = NetCDFTraj.open_old(traj_filename)

        n_atom = old_traj.atom
        n_frame = old_traj.frame

        new_trj = NetCDFTraj.open_new(output_filename, natom=n_atom, box=True, crds=True)

        system = prmtop.createSystem(nonbondedMethod=nonbondedMethod, nonbondedCutoff=nonbondedCutoff, constraints=constraints)

        for i, coordinates in enumerate(old_traj.coordinates):
            old_positions = Quantity(coordinates.tolist(), unit=angstroms)

            if self._restraint > 0 or self._restraint is not None:
                # Add harmonic constraints on protein
                force = CustomExternalForce("k * ((x-x0)^2 + (y-y0)^2 + (z-z0)^2)")
                force.addGlobalParameter("k", K)
                force.addPerParticleParameter("x0")
                force.addPerParticleParameter("y0")
                force.addPerParticleParameter("z0")
                for atom in parmedtop.view['!@H= & !:WAT']:
                    force.addParticle(atom.idx, old_positions[atom.idx])
                force_idx = system.addForce(force)

            # Create simulation
            integrator = LangevinIntegrator(temperature, friction, dt)
            simulation = Simulation(prmtop.topology, system, integrator, platform)
            simulation.context.setPositions(old_positions)

            # Minimize the water molecules
            simulation.minimizeEnergy(maxIterations=self._n_steps, tolerance=tolerance)
            # Get new positions and store in the new trajectory
            new_positions = simulation.context.getState(getPositions=True).getPositions(asNumpy=True)

            new_trj.add_coordinates(new_positions)
            new_trj.add_box(box)
            new_trj.add_time(i + 1)

            if self._restraint > 0 or self._restraint is not None:
                system.removeForce(force_idx)

            if (i % 100 == 0) and self._verbose:
                sys.stdout.write("\rConformations minimized:  %5.2f / 100 %%" % (float(i) / n_frame * 100.))
                sys.stdout.flush()

        new_trj.close()
        old_traj.close()

# endregion

