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
    parser.add_argument("-l", "--layer", dest="n_layer", default=3, type=int,
                        action="store", help="number of solvation layer to add")
    parser.add_argument("-t", "--temperature", dest="temperature", default=300., type=float,
                        action="store", help="temperature")
    parser.add_argument("-n", "--n_frames", dest="n_frames", default=1, type=int,
                        action="store", help="number of frames to generate")
    parser.add_argument("-j", "--n_jobs", dest="n_jobs", default=1., type=int,
                        action="store", help="number of jobs to run in parallel")
    parser.add_argument("-wm", "--water_model", dest="water_model", default="tip3p",
                        choices=["tip3p", "tip5p"], action="store",
                        help="water model used (tip3p or tip5p)")
    parser.add_argument("-sw", "--spherical_water_map", dest="spherical_water_map", default=None,
                        action="store", help="external water spherical map file")
    parser.add_argument("-o", "--output", dest="output_dir", default=".",
                        action="store", help="output directory")
    return parser.parse_args()


def main():
    args = cmd_lineparser()
    mol_file = args.mol_file
    fld_file = args.fld_file
    water_model = args.water_model
    spherical_water_map = args.spherical_water_map
    n_layer = args.n_layer
    n_frames = args.n_frames
    n_jobs = args.n_jobs
    temperature = args.temperature
    output_dir = args.output_dir

    # Force to use only one thread per job
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"
    os.environ["OMP_NUM_THREADS"] = "1"

    # Read PDBQT/MOL2 file, Waterfield file and AutoDock grid map
    receptor = Molecule.from_file(mol_file)
    ad_map = Map.from_fld(fld_file)

    # Go waterkit!!
    k = WaterKit(water_model, spherical_water_map, temperature, n_layer, n_frames, n_jobs)
    k.hydrate(receptor, ad_map, output_dir)

if __name__ == "__main__":
    main()
