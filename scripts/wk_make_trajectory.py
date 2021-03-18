#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# make_trajectory
#

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import argparse
import copy
import os
import re
import sys

import numpy as np
import parmed as pmd
from pdb4amber import pdb4amber
from pdb4amber.utils import easy_call
from parmed.amber import NetCDFTraj


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


def write_tleap_input_file(fname, pdb_filename):
    """Create tleap input script

    Args:
        fname (str): tleap input filename
        pdb_filename (str): pdb filename

    """
    prefix = pdb_filename.split(".pdb")[0].split("/")[-1]

    output_str = "source leaprc.protein.ff14SB\n"
    output_str += "source leaprc.DNA.OL15\n"
    output_str += "source leaprc.RNA.OL3\n"
    output_str += "source leaprc.water.tip3p\n"
    output_str += "source leaprc.gaff2\n"
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


def cmd_lineparser():
    parser = argparse.ArgumentParser(description="make_trajectory")
    parser.add_argument("-r", "--receptor", dest="receptor_filename", required=True,
                        action="store", help="prepared receptor pdb file")
    parser.add_argument("-w", "--dir", dest="water_directory", required=True,
                        action="store", help="path of the directory containing the water \
                        pdb files")
    parser.add_argument('-o', '--out', default='protein',
                        dest='output_prefix', help='output prefix filename (default: protein)')
    return parser.parse_args()


def main():
    args = cmd_lineparser()
    receptor_filename = args.receptor_filename
    water_directory = args.water_directory
    output_prefix = args.output_prefix

    tleap_input = 'leap.template.in'
    tleap_output = 'leap.template.out'
    tleap_log = 'leap.log'

    receptor_dry = pmd.load_file(receptor_filename)

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
    write_pdb_file("%s_system.pdb" % output_prefix, receptor_wet)

    # Write tleap input script
    write_tleap_input_file(tleap_input, "%s_system.pdb" % output_prefix)

    try:
        # Generate amber prmtop and rst7 files
        easy_call('tleap -s -f %s > %s' % (tleap_input, tleap_output), shell=True)
    except RuntimeError:
        error_msg = 'Could not generate topology/coordinates files with tleap.'
        raise RuntimeError(error_msg)

    # Write trajectory
    write_trajectory_file("%s_system.nc" % output_prefix, receptor_dry, water_filenames)

    os.remove(tleap_input)
    os.remove(tleap_output)
    os.remove(tleap_log)


if __name__ == '__main__':
    main()
