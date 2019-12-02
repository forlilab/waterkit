#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# amber2pdbqt
#

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import argparse

import parmed as pmd


def cmd_lineparser():
    parser = argparse.ArgumentParser(description="amber2pdbqt")
    parser.add_argument("-t", "--top", dest="top_file", required=True,
                        action="store", help="topology file")
    parser.add_argument("-c", "--coords", dest="crd_file", required=True,
                        action="store", help="coordinates file")
    parser.add_argument("-o", "--output", dest="output_name", default=None,
                        action="store", help="output name")
    parser.add_argument("--pdb", dest="make_pdb", default=False,
                        action="store_true", help="convert to pdb also")
    return parser.parse_args()


def write_pdb_file(output_name, molecule,  overwrite=True):
    """Write PDB file

    Args:
        output_name (str): pdbqt output filename
        molecule (parmed): parmed molecule object

    """
    molecule.save(output_name, format="pdb", overwrite=overwrite)

def write_pdbqt_file(output_name, molecule):
    """Write PDBQT file

    Args:
        output_name (str): pdbqt output filename
        molecule (parmed): parmed molecule object

    """
    pdbqt_str = "ATOM  %5d %-4s %3s  %4d    %8.3f%8.3f%8.3f  1.00  1.00    %6.3f %-2s\n"
    output_str = ""

    for atom in molecule.atoms:
        if len(atom.name) < 4:
            name = " %s" % atom.name
        else:
            name = atom.name
        resname = atom.residue.name
        resid = atom.residue.idx + 1

        if atom.type[0].isdigit():
            atom_type = atom.type[::-1]
        else:
            atom_type = atom.type

        # AutoGrid does not accept atom type name of length > 2
        atom_type = atom_type[:2]

        output_str += pdbqt_str % (atom.idx + 1, name, resname, resid, atom.xx, 
                                   atom.xy, atom.xz, atom.charge, atom_type)

    with open(output_name, "w") as w:
        w.write(output_str)


def main():
    args = cmd_lineparser()
    top_file = args.top_file
    crd_file = args.crd_file
    output_name = args.output_name
    make_pdb = args.make_pdb

    if output_name is None:
        output_name = top_file.split('.')[0]

    molecule = pmd.load_file(top_file, crd_file)

    # The PDB file will be use for the trajectory and
    if make_pdb:
        write_pdb_file("%s_prepared.pdb" % output_name, molecule)
    # the PDBQT file for WaterKit
    write_pdbqt_file("%s_prepared.pdbqt" % output_name, molecule)


if __name__ == '__main__':
    main()
