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
from parmed.amber import NetCDFTraj


def cmd_lineparser():
    parser = argparse.ArgumentParser(description="make_trajectory")
    parser.add_argument("-r", "--receptor", dest="receptor_filename", required=True,
                        action="store", help="receptor amber pdb file")
    parser.add_argument("-w", "--dir", dest="water_directory", required=True,
                        action="store", help="path of the directory containing the water \
                        pdb files")
    parser.add_argument("-o", "--output", dest="output_name", default="protein_water",
                        action="store", help="output name (netcdf format)")
    return parser.parse_args()


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


def write_system_pdb_file(fname, receptor, water_filenames, overwrite=True):
    """Create topology file
    
    Args:
        fname (str): output name for the topology file
        receptor_filename (str): filename of the receptor file
        water_filenames (list): list of filenames of the water files
        overwrite (bool): overwrite or not the PDB file (default: True)

    """
    max_n_waters, idx = max_water(water_filenames)
    water = pmd.load_file(water_filenames[idx])
    # We do an in-place addition, so first we have to create a copy
    receptor_copy = copy.deepcopy(receptor)
    receptor_copy += water["@O, H1, H2"]
    # ParmED really want a symmetry attributes to write the PDB file
    receptor_copy.symmetry = None
    try:
        receptor_copy.save(fname, format="pdb", overwrite=overwrite)
    except IOError:
        print("Error: file %s already exists." % fname)
        sys.exit(0)


def write_tleap_input_file(fname, pdb_filename):
    """Create tleap input script

    Args:
        fname (str): tleap input filename
        pdb_filename (str): pdb filename

    """
    prefix = pdb_filename.split(".pdb")[0].split("/")[-1]

    output_str = "source leaprc.protein.ff19SB\n"
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
        receptor_filename (str): filename of the receptor pdb file
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


def main():
    args = cmd_lineparser()
    receptor_filename = args.receptor_filename
    water_directory = args.water_directory
    output_name = args.output_name

    water_filenames = []
    for fname in os.listdir(water_directory):
        if re.match(r"water_[0-9]{6}.pdb", fname):
            water_filenames.append(os.path.join(water_directory, fname))

    receptor = pmd.load_file(receptor_filename)

    write_tleap_input_file("%s.leap.in" % output_name, "%s_system.pdb" % output_name)
    write_system_pdb_file("%s_system.pdb" % output_name, receptor, water_filenames)
    write_trajectory_file("%s.nc" % output_name, receptor, water_filenames)


if __name__ == '__main__':
    main()
