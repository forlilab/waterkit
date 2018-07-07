#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Scripts to get grid parameters
#

import argparse

import numpy as np
from MDAnalysis import Universe


def cmd_lineparser():
    parser = argparse.ArgumentParser(description='grid parameters')
    parser.add_argument("-i", "--mol", dest="mol_file", required=True,
                        action="store", help="molecule file")
    parser.add_argument("-s", "--selection", dest="selection", required=True,
                        action="store", help="MDAnalysis selection string")
    parser.add_argument("--buffer", dest="buffer_space", default=11, type=int,
                        action="store", help="buffer spacing")
    parser.add_argument("--spacing", dest="spacing", default=0.375, type=float,
                        action="store", help="grid spacing")
    return parser.parse_args()


def get_npts(box, center, spacing, buffer_space=0):

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


def main():
    args = cmd_lineparser()
    mol_file = args.mol_file
    selection = args.selection
    buffer_space = args.buffer_space
    spacing = args.spacing

    u = Universe(mol_file)
    ligand = u.select_atoms(selection)
        
    # Get the center/box of the selection
    center = ligand.centroid(pbc=False)
    box = ligand.bbox(pbc=False)
    # Get number of grid points
    npts = get_npts(box, center, spacing, buffer_space)

    print 'npts %d %s %d' % (npts[0], npts[1], npts[2])
    print 'gridcenter %5.3f %5.3f %5.3f' % (center[0], center[1], center[2])

    # Get residues in the pocket
    pocket_selection = '(protein or nucleic) and around 4 %s' % selection
    pocket = u.select_atoms(pocket_selection)
    residues = set([(x.resid, x.segid) for x in pocket])

    with open('pocket_residues.csv', 'w') as w:
        w.write('resid,segid\n')
        for residue in residues:
            w.write('%s,%s\n' % residue)

if __name__ == '__main__':
    main()
