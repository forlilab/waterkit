#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# amber2pdbqt
#

import argparse

import parmed as pmd


def cmd_lineparser():
    parser = argparse.ArgumentParser(description="amber2pdbqt")
    parser.add_argument("-p", "--top", dest="top_file", required=True,
                        action="store", help="topology file")
    parser.add_argument("-c", "--coords", dest="crd_file", required=True,
                        action="store", help="coordinates file")
    parser.add_argument("-o", "--pdbqt", dest="pdbqt_file", default=None,
                        action="store", help="output pdbqt file")
    return parser.parse_args()


def write_pdbqt(mol, pdbqt_file):
    """Write PDBQT file from parmed amber object

    Args:
        mol (parmed.amber._amberparm.AmberParm): amber object from parmed
        pdbqt_file (str): pdbqt output file

    """
    pdbqt_str = "ATOM  %5d %-4s %3s  %4d    %8.3f%8.3f%8.3f  1.00  1.00    %6.3f %-2s\n"
    output_str = ""

    for atom in mol.atoms:
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

        output_str += pdbqt_str % (atom.idx + 1, name, resname, resid, atom.xx, 
                                   atom.xy, atom.xz, atom.charge, atom_type)

    with open(pdbqt_file, "w") as w:
        w.write(output_str)


def main():
    args = cmd_lineparser()
    top_file = args.top_file
    crd_file = args.crd_file
    pdbqt_file = args.pdbqt_file

    mol = pmd.load_file(top_file, crd_file)

    if pdbqt_file is None:
        pdbqt_file = "%s.pdbqt" % top_file.split('.')[0]

    write_pdbqt(mol, pdbqt_file)


if __name__ == '__main__':
    main()
