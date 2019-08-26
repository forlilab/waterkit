#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# make_trajectory
#

import argparse
import copy
import glob
import os
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
    parser.add_argument("-b", "--box", dest="box_dimension", nargs=6, 
                        default=[100, 100, 100, 90, 90, 90], action="store", 
                        help="box dimension")
    parser.add_argument("-d", "--dummy", dest="dummy_water_xyz", nargs=3, 
                        default=[0, 0, 0], action="store", 
                        help="dummy water coordinates")
    parser.add_argument("-o", "--output", dest="output_name", default="protein_water",
                        action="store", help="output name (netcdf format)")
    return parser.parse_args()


def max_water(water_filenames):
    """Return the max number of water molecules seen
    
    Args:
        water_filenames (array-like): list of filenames of the water files

    Return:
        int: number of max water molecules
    """
    sizes = [os.path.getsize(f) for f in water_filenames]
    idx = np.argmax(sizes)
    m = pmd.load_file(water_filenames[idx])
    max_water = len(m.residues)
    return max_water, idx


def write_system_pdb_file(fname, receptor, water_filenames):
    """Create topology file
    
    Args:
        fname (str): output name for the topology file
        receptor_filename (str): filename of the receptor file
        water_filenames (list): list of filenames of the water files

    """
    max_n_waters, idx = max_water(water_filenames)
    water = pmd.load_file(water_filenames[idx])
    # We do an in-place addition, so first we have to create a copy
    receptor_copy = copy.deepcopy(receptor)
    receptor_copy += water["@O, H1, H2"]
    # ParmED really want a symmetry attributes to write the PDB file
    receptor_copy.symmetry = None
    try:
        receptor_copy.save(fname, format="pdb")
    except IOError:
        print "Error: file %s already exists." % fname
        sys.exit(0)


def write_tleap_input_file(fname, pdb_filename):
    """Create tleap input script

    Args:
        fname (str): tleap input filename
        pdb_filename (str): pdb filename

    """
<<<<<<< HEAD
    prefix = pdb_filename.split(".pdb")[0]
=======
    prefix = pdb_filename.split(".pdb")[0].split("/")[-1]
>>>>>>> disordered_hydrogens

    output_str = "source leaprc.protein.ff14SB\n"
    output_str += "source leaprc.DNA.OL15\n"
    output_str += "source leaprc.RNA.OL3\n"
    output_str += "source leaprc.water.tip3p\n"
    output_str += "source leaprc.gaff2\n"
    output_str += "\n"
<<<<<<< HEAD
    output_str += "x = loadpdb %s\n" % pdb_filename
=======
    output_str += "x = loadpdb %s\n" % pdb_filename.split("/")[-1]
>>>>>>> disordered_hydrogens
    output_str += "\n"
    output_str += "set default PBradii mbondi3\n"
    output_str += "set default nocenter on\n"
    output_str += "saveAmberParm x %s.prmtop %s.rst7\n" % (prefix, prefix)
    output_str += "quit\n"

    with open(fname, "w") as w:
        w.write(output_str)


def write_trajectory_file(fname, receptor, water_filenames, 
                          dummy_water_xyz=[0, 0, 0], box=[100, 100, 100, 90, 90, 90]):
    """Create netcdf trajectory from the water pdb file

    Args:
        fname (str): output name for the trajectory
        receptor_filename (str): filename of the receptor pdb file
        water_filenames (list): list of filenames of the water files
        dummy_water_xyz (array-like): xyz coordinates of the dummy to add
        box (array-like): box dimension (x, y, z, alpha, beta, gamma)

    """
    max_n_waters, idx = max_water(water_filenames)
    n_atoms = receptor.coordinates.shape[0]
    max_n_atoms = n_atoms + (max_n_waters * 3)
    coordinates = np.zeros(shape=(max_n_atoms, 3))

    # Already add the coordinates from the receptor
    coordinates[:n_atoms] = receptor.coordinates

    trj = NetCDFTraj.open_new(fname, natom=max_n_atoms, box=True, crds=True)

    for water_filename in water_filenames:
        m = pmd.load_file(water_filename)

        last_atom_id = len(m.residues) * 3

        # Get all the TIP3P water molecules
        coordinates[n_atoms:n_atoms + last_atom_id] = m["@O, H1, H2"].coordinates
        # Add the dummy water molecules
        coordinates[n_atoms + last_atom_id:] = dummy_water_xyz

        trj.add_coordinates(coordinates)
        trj.add_box(box)

    trj.close()


def main():
    args = cmd_lineparser()
    receptor_filename = args.receptor_filename
    water_directory = args.water_directory
    box_dimension = args.box_dimension
    dummy_water_xyz = args.dummy_water_xyz
    output_name = args.output_name

    water_filenames = sorted(glob.glob("%s/*" % water_directory))

    receptor = pmd.load_file(receptor_filename)

<<<<<<< HEAD
    write_tleap_input_file("%s_leap.in" % output_name, "%s_system.pdb" % output_name)
=======
    write_tleap_input_file("%s.leap.in" % output_name, "%s_system.pdb" % output_name)
>>>>>>> disordered_hydrogens
    write_system_pdb_file("%s_system.pdb" % output_name, receptor, water_filenames)
    write_trajectory_file("%s.nc" % output_name, receptor, water_filenames,
                          dummy_water_xyz, box_dimension)


if __name__ == '__main__':
    main()
