#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Scripts to get grid parameters
#

import argparse
import os
import warnings

import numpy as np
from MDAnalysis import Universe


# Ignore all the warnings (unrecognized atom types MDAnalysis)
warnings.filterwarnings("ignore")

def cmd_lineparser():
    """ Function to parse argument command line """
    parser = argparse.ArgumentParser(description='grid parameters')
    parser.add_argument("-r", "--receptor", dest="receptor_file", required=True,
                        action="store", help="receptor file (PDBQT)")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("-l", "--ligand", dest="ligand_file",
                        action="store", help="ligand file")
    group.add_argument("-c", "--residues", dest="csv_file",
                        action="store", help="csv file with residues (resid,segid)")
    parser.add_argument("-o", "--output", dest="output_file", default='protein.gpf',
                        action="store", help="output file")
    parser.add_argument("--buffer", dest="buffer_space", default=11, type=int,
                        action="store", help="buffer spacing")
    parser.add_argument("--spacing", dest="spacing", default=0.375, type=float,
                        action="store", help="grid spacing")
    return parser.parse_args()


def read_residues_csv_file(csv_file):
    """ Function to read the csv file, one
    residue per line with format <resid,segid> """
    residues = []

    with open(csv_file) as f:
        lines = f.readlines()

        for line in lines:
            try:
                resid, segid = line.split(',')
                residues.append((int(resid), segid.rstrip()))
            except:
                continue

    return residues


def get_npts(box, center, spacing, buffer_space=0):
    """ Function to compute the number of grid points """
    box = box.T
    
    if np.abs(box[0][0] - center[0]) >= np.abs(box[0][1] - center[0]):
        x = np.abs(box[0][0] - center[0]) + buffer_space
    else:
        x = np.abs(box[0][1] - center[0]) + buffer_space
        
    if np.abs(box[1][0] - center[1]) >= np.abs(box[1][1] - center[1]):
        y = np.abs(box[1][0] - center[1]) + buffer_space
    else:
        y = np.abs(box[1][1] - center[1]) + buffer_space
        
    if np.abs(box[2][0] - center[2]) >= np.abs(box[2][1] - center[2]):
        z = np.abs(box[2][0] - center[2]) + buffer_space
    else:
        z = np.abs(box[2][1] - center[2]) + buffer_space
        
    x = np.floor(x / spacing) * 2
    y = np.floor(y / spacing) * 2
    z = np.floor(z / spacing) * 2

    npts = np.array([np.int(x), np.int(y), np.int(z)])

    return npts

def create_gpf_file(npts, center, receptor_types, receptor_file, output_file='protein.gpf'):
    """ Write the Protein Grid file"""
    path_name = os.path.dirname(receptor_file)
    receptor_name = os.path.basename(receptor_file).split('.')[0]

    if not path_name:
        path_name = '.'

    grid_str = ("npts {npts}\n"                       
                "parameter_file AD4_parameters.dat\n"
                "gridfld {path}/{name}_maps.fld\n"     
                "spacing 0.375\n"
                "receptor_types {receptor_types}\n"
                "ligand_types OD OW OT\n"
                "receptor {path}/{name}.pdbqt\n"       
                "gridcenter {center}\n"
                "smooth 0\n"
                "map {path}/{name}_OD.map\n"
                "map {path}/{name}_OW.map\n"
                "map {path}/{name}_OT.map\n"
                "elecmap {path}/{name}_e.map\n"
                "dsolvmap {path}/{name}_d.map\n"
                "dielectric 1\n"
               )

    tmp_str = grid_str.format(path=path_name, name=receptor_name,
                              receptor_types=' '.join(receptor_types),
                              npts=' '.join([str(x) for x in npts]),
                              center=' '.join([str(x) for x in center]))

    with open(output_file, 'w') as w:
        w.write(tmp_str)

def main():
    args = cmd_lineparser()
    receptor_file = args.receptor_file
    ligand_file = args.ligand_file
    csv_file = args.csv_file
    output_file = args.output_file
    buffer_space = args.buffer_space
    spacing = args.spacing

    receptor = Universe(receptor_file)

    if ligand_file:
        ligand = Universe(ligand_file).select_atoms('all')
    elif csv_file:
        residues = read_residues_csv_file(csv_file)
        selection = ' or '.join(['(resid %d and segid %s)' % residue for residue in residues])
        ligand = receptor.select_atoms(selection)
        
    # Get the center/box of the selection
    center = ligand.centroid(pbc=False)
    box = ligand.bbox(pbc=False)
    # Get number of grid points
    npts = get_npts(box, center, spacing, buffer_space)
    # Get receptor types
    receptor_types = np.unique(receptor.atoms.types)

    create_gpf_file(npts, center, receptor_types, receptor_file, output_file)

if __name__ == '__main__':
    main()
