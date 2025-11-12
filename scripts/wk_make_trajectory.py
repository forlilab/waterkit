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
from waterkit import make_trajectory


def cmd_lineparser():
    parser = argparse.ArgumentParser(description="make_trajectory")
    parser.add_argument("-r", "--receptor", dest="receptor_filename", required=True,
                        action="store", help="prepared receptor pdb file")
    parser.add_argument("-w", "--dir", dest="water_directory", required=True,
                        action="store", help="path of the directory containing the water \
                        pdb files")
    parser.add_argument('-o', '--out', default='protein',
                        dest='output_prefix', help='output prefix filename (default: protein)')
    parser.add_argument('--lib', dest='lib_files', default=None, nargs='+',
        action='store', help='Amber lib parameter files.')
    parser.add_argument('--frcmod', dest='frcmod_files', default=None, nargs='+',
        action='store', help='Amber frcmod parameter files.')
    return parser.parse_args()


def main():
    args = cmd_lineparser()
    receptor_filename = args.receptor_filename
    water_directory = args.water_directory
    output_prefix = args.output_prefix
    lib_files = args.lib_files
    frcmod_files = args.frcmod_files

    receptor_dry = pmd.load_file(receptor_filename)

    make_trajectory(
        receptor_dry,
        water_directory,
        output_prefix,
        lib_files,
        frcmod_files,
    )

if __name__ == '__main__':
    main()
