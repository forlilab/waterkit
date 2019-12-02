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
    parser.add_argument("-o", "--output", dest="output_file", default='protein.gpf',
                        action="store", help="output file")
    return parser.parse_args()


def atom_types_from_pdbqt_file(pdbqt_file):
    atom_types = []

    with open(pdbqt_file) as f:
        lines = f.readlines()

        for line in lines:
            atom_types.append(line[77:].rstrip())

    atom_types = list(set(atom_types))

    return atom_types

def molecule_centroid(pdb_file):
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

def create_gpf_file(fname, npts, center, receptor_types, receptor_file):
    """ Write the Protein Grid file"""
    receptor_name = os.path.basename(receptor_file).split('.')[0]

    grid_str = ("npts {npts}\n"                       
                "parameter_file AD4_parameters.dat\n"
                "gridfld {name}_maps.fld\n"     
                "spacing 0.375\n"
                "receptor_types {receptor_types}\n"
                "ligand_types SW OW\n"
                "receptor {name}.pdbqt\n"       
                "gridcenter {center}\n"
                "smooth 0\n"
                "map {name}_SW.map\n"
                "map {name}_OW.map\n"
                "elecmap {name}_e.map\n"
                "dsolvmap {name}_d.map\n"
                "dielectric 1\n"
               )

    tmp_str = grid_str.format(name=receptor_name,
                              receptor_types=' '.join(receptor_types),
                              npts=' '.join([str(x) for x in npts]),
                              center=' '.join(['%8.3f' % x for x in center]))

    with open(fname, 'w') as w:
        w.write(tmp_str)

def main():
    args = cmd_lineparser()
    receptor_file = args.receptor_file
    ligand_file = args.ligand_file
    box_center = args.box_center
    box_size = args.box_size
    output_file = args.output_file

    if ligand_file is not None:
        box_center = molecule_centroid(ligand_file)

    npts = number_of_grid_points(box_size)
    receptor_types = atom_types_from_pdbqt_file(receptor_file)

    create_gpf_file(output_file, npts, box_center, receptor_types, receptor_file)

if __name__ == '__main__':
    main()
