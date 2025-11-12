#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# prepare receptor
#

import argparse
from waterkit import PrepareReceptor


def cmd_lineparser():
    parser = argparse.ArgumentParser(description='prepare receptor')
    parser.add_argument('-i', '--in', required=True,
        dest='pdb_filename', help='PDB input file (default: stdin)',
        default='stdin')
    parser.add_argument('-o', '--out', default='protein',
        dest='output_prefix', help='output prefix filename (default: protein)')
    parser.add_argument('--keep_hydrogen', action='store_true', default=False,
        dest='keep_hydrogen', help='keep all hydrogen atoms (default: no)')
    parser.add_argument('--no_disulfide', action='store_true', default=False,
        dest='no_disulfide', help='ignore difsulfide bridges (default: no)')
    parser.add_argument('--keep_water', action='store_true', default=False,
        dest='keep_water', help='keep all water molecules (default: no)')
    parser.add_argument('--keep_altloc', action='store_true', default=False,
        dest='keep_altloc', help='keep residue altloc (default is to keep "A")')
    parser.add_argument('--model', type=int, default=1,
        dest='use_model',
        help='Model to use from a multi-model pdb file (integer).  (default: use 1st model). '
        'Use a negative number to keep all models')
    parser.add_argument('--pdb', dest='make_pdb', default=False,
        action='store_true', help='generate pdb file')
    parser.add_argument('--pdbqt', dest='make_pdbqt', default=False,
        action='store_true', help='PDBQT file with AutoDock atom types')
    parser.add_argument('--amber_pdbqt', dest='make_amber_pdbqt', default=False,
        action='store_true', help='PDBQT file with Amber atom types')
    parser.add_argument('--ignore_gaps', dest='ignore_gaps', default=False,
        action='store_true', help='ignore gaps between residues (automatically add TER records)')
    parser.add_argument('--renumber', dest='renumbering', default=False,
        action='store_true', help='Residue index will be renumbered (starting from 1).')
    parser.add_argument('--lib', dest='lib_files', default=None, nargs='+',
        action='store', help='Amber lib parameter files.')
    parser.add_argument('--frcmod', dest='frcmod_files', default=None, nargs='+',
        action='store', help='Amber frcmod parameter files.')
    parser.add_argument('--no_clean', dest='no_clean', default=True,
        action='store_false', help='Does not clean Amber temporay files.')
    parser.add_argument('--default_his_protonation', default='HIE',
        dest='default_his_protonation', help='default hisitidine protonation if hydrogen are not kept. (default: HIE)')
    return parser.parse_args()


def main():
    args = cmd_lineparser()
    pdb_filename = args.pdb_filename
    output_prefix = args.output_prefix
    keep_hydrogen = args.keep_hydrogen
    no_disulfide = args.no_disulfide
    keep_water = args.keep_water
    keep_altloc = args.keep_altloc
    use_model = args.use_model
    make_pdb = args.make_pdb
    make_pdbqt = args.make_pdbqt
    make_amber_pdbqt = args.make_amber_pdbqt
    ignore_gaps = args.ignore_gaps
    renumbering = args.renumbering
    lib_files = args.lib_files
    frcmod_files = args.frcmod_files
    clean = -args.no_clean
    default_his_protonation = args.default_his_protonation

    prmtop_filename = '%s.prmtop' % output_prefix
    rst7_filename = '%s.rst7' % output_prefix
    pdb_clean_filename = '%s_clean.pdb' % output_prefix

    pr = PrepareReceptor(keep_hydrogen, keep_water, no_disulfide, 
                         keep_altloc, ignore_gaps, renumbering, use_model,
                         default_his_protonation)
    pr.prepare(pdb_filename, lib_files, frcmod_files, clean)

    if make_pdb:
        pdb_prepared_filename = '%s.pdb' % output_prefix
        pr.write_pdb_file(pdb_prepared_filename)

    if make_pdbqt:
        pdbqt_prepared_filename = '%s.pdbqt' % output_prefix
        pr.write_pdbqt_file(pdbqt_prepared_filename)

    if make_amber_pdbqt:
        pdbqt_prepared_filename = '%s_amber.pdbqt' % output_prefix
        pr.write_pdbqt_file(pdbqt_prepared_filename, amber_atom_types=True)


if __name__ == '__main__':
    main()
