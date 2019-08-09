#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Script to help to assess WaterKit on dataset
#


import argparse
import os
import shlex
import subprocess

import pandas as pd

import waterkit


def execute_command(cmd_line):
    args = shlex.split(cmd_line)
    p = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    output, errors = p.communicate()

    return output, errors

def cmd_lineparser():
    """ Function to parse argument command line """
    parser = argparse.ArgumentParser(description='waterkit batch')
    parser.add_argument("-p", "--pdb", dest="pdb_directory", required=True,
                        action="store", help="pdb_directory")
    parser.add_argument("-o", "--output", dest="output_directory", default='project',
                        action="store", help="output directory")
    parser.add_argument("-c", "--cutoff", dest="cutoff", nargs='*', default=[1.0, 1.4, 2.0],
                        action="store", help="cutoffs")
    parser.add_argument("--buffer", dest="buffer_space", default=11, type=int,
                        action="store", help="buffer spacing")
    parser.add_argument("--spacing", dest="spacing", default=0.375, type=float,
                        action="store", help="grid spacing")
    parser.add_argument("--no-ligand", dest="no_ligand", default=False,
                        action="store_true", help="ligand not included")
    parser.add_argument("--no-autogrid", dest="no_autogrid", default=False,
                        action="store_true", help="do not run autogrid")
    parser.add_argument("--no-waterkit", dest="no_waterkit", default=False,
                        action="store_true", help="do not run waterkit")
    return parser.parse_args()

def main():
    args = cmd_lineparser()
    pdb_directory = args.pdb_directory
    output_directory = args.output_directory
    cutoff = args.cutoff
    no_ligand = args.no_ligand
    no_autogrid = args.no_autogrid
    no_waterkit = args.no_waterkit
    buffer_space = args.buffer_space
    spacing = args.spacing

    """ Check if already exists and also if
    it is containing results.csv from a previous
    calculation. """
    if not os.path.isdir(output_directory):
        os.makedirs(output_directory)
    else:
        results_file = os.path.join(output_directory, 'results.txt')

        if not os.path.exists(results_file):
            print "results.csv file is missing."
            return False

    if os.path.exists(pdb_directory):
        pdb_files = os.listdir(pdb_directory)
        pdb_files = [p for p in pdb_files if p.split('.')[-1] in ['pdb', 'pdbqt']]

        if not pdb_files:
            print "PDB/PDBQT not found."
            return False
    else:
        print "PDB directory is missing."
        return False
    
    pdb_info = {}

    for pdb_file in pdb_files:
        name = pdb_file.split('_')[0]

        if not name in pdb_info:
            pdb_info[name] = [False, False]

        if 'water' in pdb_file:
            pdb_info[name][1] = True
        elif 'ligand' in pdb_file:
            pdb_info[name][0] = True

    columns = ['name', 'ligand', 'water']
    data = [[key] + value for key, value in pdb_info.iteritems()]
    df = pd.DataFrame(data=data, columns=columns)
    print df


if __name__ == '__main__':
    main()
