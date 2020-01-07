#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# make_trajectory
#

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import argparse
import os
import sys

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
    parser.add_argument("-o", "--out", default='protein',
        dest="out_filename", help="output filename (default: protein)")
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


def write_pdb_file(output_name, molecule,  overwrite=True, **kwargs):
    """Write PDB file

    Args:
        output_name (str): pdbqt output filename
        molecule (parmed): parmed molecule object

    """
    molecule.save(output_name, format="pdb", overwrite=overwrite, **kwargs)


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


def find_alt_residues(molecule):
    alt_residues = set()
        
    for residue in molecule.residues:
        chains.add(residue.chain)
        for atom in residue.atoms:
            if atom.other_locations:
                alt_residues.add(residue)

    return alt_residues

def remove_alt_residues(molecule):
    # remove altlocs label
    for atom in molecule.atoms:
        atom.altloc = ''
        for oatom in atom.other_locations.values():
            oatom.altloc = ''

def main():
    args = cmd_lineparser()
    pdbin_filename = args.pdbin
    out_filename = args.out_filename
    nohyd = args.nohyd
    dry = args.dry
    mostpop = args.mostpop
    model = args.model - 1

    base_filename, extension = os.path.splitext(out_filename)
    pdb_clean_filename = "%s_clean.pdb" % base_filename
    pdbqt_prepared_filename = "%s_prepared.pdbqt" % base_filename

    try:
        receptor = pmd.load_file(pdbin_filename)
    except FileNotFoundError:
        print("Error: Receptor file (%s) cannot be found." % pdbin_filename)
        sys.exit(0)

    pdbfixer = AmberPDBFixer(receptor)

    ns_names = pdbfixer.find_non_standard_resnames()

    # Remove all the hydrogens
    if nohyd:
        pdbfixer.parm.strip('@/H')
    # Remove water molecules
    if dry:
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

    remove_alt_residues(pdbfixer.parm)

    final_coordinates = pdbfixer.parm.get_coordinates()[model]
    write_kwargs = dict(coordinates=final_coordinates)
    write_kwargs["increase_tercount"] = False # so CONECT record can work properly
    write_kwargs["altlocs"] = "occupancy"

    try:
        write_pdb_file(pdb_clean_filename, pdbfixer.parm, **write_kwargs)
    except:
        print("Error: Could not write pdb file %s"  % pdb_clean_filename)
        sys.exit(0)

    with open('leap.template.in', 'w') as w:
        final_ns_names = []
        
        content = _make_leap_template(pdbfixer.parm, final_ns_names, gaplist,
                                      sslist, input_pdb=pdb_clean_filename, 
                                      prmtop='prmtop', rst7='rst7')
        w.write(content)

    easy_call("tleap -s -f leap.template.in > leap.template.out", shell=True)

    try:
        molecule = pmd.load_file('prmtop', 'rst7')
    except:
        print("Error: Cannot load topology and coordinates Amber files")
        sys.exit(0)

    # the PDBQT file for WaterKit
    write_pdbqt_file(pdbqt_prepared_filename, molecule)


if __name__ == '__main__':
    main()