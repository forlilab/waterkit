#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# WaterKit
#
# Launch waterkit
#

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import argparse

from waterkit import Map
from waterkit import Molecule
from waterkit import WaterKit
from waterkit import utils


def cmd_lineparser():
    parser = argparse.ArgumentParser(description="waterkit")
    parser.add_argument("-i", "--mol", dest="mol_file", required=True,
                        action="store", help="receptor file")
    parser.add_argument("-m", "--fld", dest="fld_file", required=True,
                        action="store", help="autodock fld file")
    parser.add_argument("-l", "--layer", dest="n_layer", default=0, type=int,
                        action="store", help="number of layer to add")
    parser.add_argument("-t", "--temperature", dest="temperature", default=300., type=float,
                        action="store", help="temperature")
    parser.add_argument("-n", "--n_frames", dest="n_frames", default=1, type=int,
                        action="store", help="number of frames to generate")
    parser.add_argument("-j", "--n_jobs", dest="n_jobs", default=1., type=int,
                        action="store", help="number of jobs to run in parallel")
    parser.add_argument("-w", "--water", dest="water_model", default="tip3p",
                        choices=["tip3p", "tip5p"], action="store",
                        help="water model used (tip3p or tip5p)")
    parser.add_argument("-wr", "--water_ref", dest="water_grid_file", default=None,
                        action="store", help="water reference grid map")
    parser.add_argument("-o", "--output", dest="output_dir", default=".",
                        action="store", help="output directory")
    return parser.parse_args()


def main():
    args = cmd_lineparser()
    mol_file = args.mol_file
    fld_file = args.fld_file
    water_model = args.water_model
    water_grid_file = args.water_grid_file
    n_layer = args.n_layer
    n_frames = args.n_frames
    n_jobs = args.n_jobs
    temperature = args.temperature
    output_dir = args.output_dir

    # Read PDBQT/MOL2 file, Waterfield file and AutoDock grid map
    molecule = Molecule.from_file(mol_file)
    ad_map = Map.from_fld(fld_file)

    # Go waterkit!!
    k = WaterKit(water_model, water_grid_file, temperature, n_layer, n_frames, n_jobs)
    k.hydrate(molecule, ad_map, output_dir)

if __name__ == "__main__":
    main()
