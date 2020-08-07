#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Scripts to get grid parameters
#

import argparse
import os
import warnings

import numpy as np


def cmd_lineparser():
    """ Function to parse argument command line """
    parser = argparse.ArgumentParser(description='grid parameters')
    parser.add_argument("-r", "--receptor", dest="receptor_file", required=True,
                        action="store", help="receptor file (PDBQT)")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("-l", "--ligand", dest="ligand_file", default=None,
                        action="store", help="ligand file")
    group.add_argument("-c", "--center", dest="box_center", nargs=3, type=float,
                        action="store", help="center of the box")
    parser.add_argument("-s", "--size", dest="box_size", nargs=3, type=int, required=True,
                        action="store", help="size of the box in Angstrom")
    parser.add_argument("-w", "--water", dest="water_model", default="tip3p",
                        choices=["tip3p", "tip5p"], action="store",
                        help="water model used (tip3p or tip5p)")
    parser.add_argument("-o", "--output", dest="output_file", default='protein.gpf',
                        action="store", help="output file")
    return parser.parse_args()


def atom_types_from_pdbqt_file(pdbqt_file):
    """ Get all the unique atom types
    
    Args:
        pdbqt_filename (str): pathname to PDBQT file

    Returns:
        list: atom types

    """
    atom_types = []

    with open(pdbqt_file) as f:
        lines = f.readlines()

        for line in lines:
            atom_types.append(line[77:].rstrip())

    atom_types = list(set(atom_types))

    return atom_types


def molecule_centroid(pdb_file):
    """ Get the centroid of the molecule

    Args:
        pdb_file (str): pathname to PDB/PDBQT file
    
    Returns:
        ndarrays: centroid

    """
    coordinates = []

    with open(pdb_file) as f:
        lines = f.readlines()

        for line in lines:
            if "ATOM" in line:
                x = line[30:38]
                y = line[38:46]
                z = line[46:54]
                coordinates.append((x, y, z))

    coordinates = np.array(coordinates, dtype=np.float)
    centroid = np.mean(coordinates, axis=0)

    return centroid


def number_of_grid_points(box_size, spacing=0.375):
    """ Function to compute the number of grid points.

    Args:
        box_size (array-like): 3D dimensions of the box
        spacing (float): spacing between grid point (default: 0.375)

    Returns:
        ndarray: number of grid points in each dimension

    """
    x = np.floor(box_size[0] / spacing) // 2 * 2 + 1
    y = np.floor(box_size[1] / spacing) // 2 * 2 + 1
    z = np.floor(box_size[2] / spacing) // 2 * 2 + 1

    npts = np.array([np.int(x), np.int(y), np.int(z)])

    return npts


def create_gpf_file(fname, receptor_file, receptor_types, atom_types, 
                    center=(0., 0., 0.), npts=(32, 32, 32), spacing=0.375, smooth=0.5, 
                    dielectric=-0.1465):
    """ Write the Protein Grid file
    
    Args:
        receptor_file (str): pathname of the PDBQT receptor file
        receptor_types (list): list of the receptor atom types
        atom_types (list): list of the ligand atom types
        center (array_like): center of the grid (default: (0, 0, 0))
        npts (array_like): size of the grid box (default: (32, 32, 32))
        spacing (float): space between grid points (default: 0.375)
        smooth (float): AutoDock energy smoothing (default: 0.5)

    """
    _, receptor_filename = os.path.split(receptor_file)
    receptor_name = receptor_filename.split(".")[0]

    fld_file =  "%s_maps.fld" % receptor_name
    xyz_file =  "%s_maps.xyz" % receptor_name
    map_files = ["%s_%s.map" % (receptor_name, t) for t in atom_types]
    e_file = "%s_e.map" % receptor_name
    d_file = "%s_d.map" % receptor_name

    ag_str = "npts %d %d %d\n" % (npts[0], npts[1], npts[2])
    ag_str += "parameter_file %s\n" % "AD4_parameters.dat"
    ag_str += "gridfld %s\n" % fld_file
    ag_str += "spacing %.3f\n" % spacing
    ag_str += "receptor_types " + " ".join(receptor_types) + "\n"
    ag_str += "ligand_types " + " ".join(atom_types) + "\n"
    ag_str += "receptor %s\n" % receptor_filename
    ag_str += "gridcenter %.3f %.3f %.3f\n" % (center[0], center[1], center[2])
    ag_str += "smooth %.3f\n" % smooth
    for map_file in map_files:
        ag_str += "map %s\n" % map_file
    ag_str += "elecmap %s\n" % e_file
    ag_str += "dsolvmap %s\n" % d_file
    ag_str += "dielectric %.3f\n" % dielectric

    with open(fname, "w") as w:
        w.write(ag_str)


def main():
    args = cmd_lineparser()
    receptor_file = args.receptor_file
    ligand_file = args.ligand_file
    center = args.box_center
    box_size = args.box_size
    water_model = args.water_model
    output_file = args.output_file

    if ligand_file is not None:
        center = molecule_centroid(ligand_file)

    if water_model == "tip3p":
        atom_types = ["SW", "OW"]
    else:
        atom_types = ["SW", "OT"]

    npts = number_of_grid_points(box_size)
    receptor_types = atom_types_from_pdbqt_file(receptor_file)

    create_gpf_file(output_file, receptor_file, receptor_types, atom_types, 
                    center, npts, smooth=0, dielectric=1)

if __name__ == '__main__':
    main()
