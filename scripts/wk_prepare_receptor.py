#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# make_trajectory
#

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import argparse

import parmed as pmd
from pdb4amber import AmberPDBFixer
from pdb4amber.leap_runner import _make_leap_template
from pdb4amber.utils import easy_call
from pdb4amber.residue import (
    RESPROT,
    AMBER_SUPPORTED_RESNAMES,
    HEAVY_ATOM_DICT, )


def cmd_lineparser():
    parser = argparse.ArgumentParser(description="prepare receptor")
    parser.add_argument("-i", "--in", required=True,
        dest="pdbin", help="PDB input file (default: stdin)",
        default='stdin')
    parser.add_argument("-o", "--out", required=True,
        dest="pdbout", help="PDBQT output file (default: stdout)",
        default='stdout')
    parser.add_argument("-y", "--nohyd", action="store_true", default=False,
        dest="nohyd", help="remove all hydrogen atoms (default: no)")
    parser.add_argument("-d", "--dry", action="store_true", default=False,
        dest="dry", help="remove all water molecules (default: no)")
    parser.add_argument("--most-populous", action="store_true",
        dest="mostpop", help="keep most populous alt. conf. (default is to keep 'A')")
    parser.add_argument("--model", type=int, default=1,
        dest="model",
        help="Model to use from a multi-model pdb file (integer).  (default: use 1st model). "
        "Use a negative number to keep all models")
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
    pdbin_filename = args.pdbin
    pdbout_filename = args.pdbout
    nohyd = args.nohyd
    dry = args.dry
    mostpop = args.mostpop
    model = args.model
    pdb = args.pdb

    try:
        receptor = pmd.load_file(pdbin_filename)
    except FileNotFoundError:
        print "Error: Receptor file (%s) cannot be found." % pdbin_filename

    pdbfixer = AmberPDBFixer(receptor)

    ns_names = pdbfixer.find_non_standard_resnames()


    # Remove all the hydrogens
    if nohyd:
        pdbfixer.parm.strip('@/H')
    # Remove water molecules
    if dry
        pdbfixer.remove_water()
    # Keep only standard-Amber residues
    pdbfixer.parm.strip('!:' + ','.join(AMBER_SUPPORTED_RESNAMES))
    # Assign histidine protonations
    pdbfixer.assign_histidine()
    # Find all the disulfide bonds
    sslist, cys_cys_atomidx_set = pdbfixer.find_disulfide()
    pdbfixer.rename_cys_to_cyx(sslist)
    # Find all the gaps
    gaplist = pdbfixer.find_gaps()

    write_pdb_file("%s_prepared.pdb" % output_name, pdbfixer.parm)
    with open('leap.template.in', 'w') as w:
        final_ns_names = []
        
        content = _make_leap_template(parm, final_ns_names, gaplist,
                                      sslist, input_pdb=pdbout_filename, 
                                      prmtop='prmtop', rst7='rst7')
        w.write(content)

    easy_call("tleap -s -f leap.template.in > leap.template.out", shell=True)

    molecule = pmd.load_file('prmtop', 'rst7')
    # The PDB file will be use for the trajectory and
    if make_pdb:
        write_pdb_file("%s_prepared.pdb" % output_name, pdbfixer.parm)
    # the PDBQT file for WaterKit
    write_pdbqt_file("%s_prepared.pdbqt" % output_name, pdbfixer.parm)


if __name__ == '__main__':
    main()