#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# WaterKit
#
# Launch waterkit
#

import os
import argparse
import shutil

from waterkit import calc_spherical_water_map
from waterkit import Map
from waterkit import Molecule
from waterkit import WaterKit
from waterkit import utils


def cmd_lineparser():
    parser = argparse.ArgumentParser(description='waterkit')
    parser.add_argument('-i', '--mol', dest='receptor_pdbqtfilename', required=True,
                        action='store', help='receptor file')
    parser.add_argument('-c', '--center', dest='box_center', nargs=3, type=float,
                        action='store', help='center of the box')
    parser.add_argument('-s', '--size', dest='box_size', nargs=3, type=int, required=True,
                        action='store', help='size of the box in Angstrom')
    parser.add_argument('-l', '--layer', dest='n_layer', default=3, type=int,
                        action='store', help='number of solvation layer to add')
    parser.add_argument('-t', '--temperature', dest='temperature', default=300., type=float,
                        action='store', help='temperature')
    parser.add_argument('-n', '--n_frames', dest='n_frames', default=1, type=int,
                        action='store', help='number of frames to generate')
    parser.add_argument('-j', '--n_jobs', dest='n_jobs', default=1., type=int,
                        action='store', help='number of jobs to run in parallel')
    parser.add_argument('-sw', '--spherical_water_maps', dest='spherical_water_maps', nargs=2, default=[None, None],
                        action='store', 
                        help='spherical water map files for receptor and single water (used for updating maps)')
    parser.add_argument('-o', '--output', dest='output_dir', default='.',
                        action='store', help='output directory')
    parser.add_argument('--autogrid_exec_path', dest='autogrid_exec_path', default='autogrid4',
                        action='store', help='path to the autogrid4 executable (default: autogrid4')
    return parser.parse_args()


def main():
    args = cmd_lineparser()
    receptor_pdbqtfilename = args.receptor_pdbqtfilename
    box_center = args.box_center
    box_size = args.box_size
    n_layer = args.n_layer
    n_frames = args.n_frames
    n_jobs = args.n_jobs
    temperature = args.temperature
    output_dir = args.output_dir
    spherical_water_maps = args.spherical_water_maps
    autogrid_exec_path = os.path.abspath(args.autogrid_exec_path)
    water_model = 'tip3p'

    # Force to use only one thread per job
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['NUMEXPR_NUM_THREADS'] = '1'
    os.environ['OMP_NUM_THREADS'] = '1'

    # Read PDBQT/MOL2 file, Waterfield file and AutoDock grid map
    receptor = Molecule.from_file(receptor_pdbqtfilename)

    receptor_spherical_water_map_filename = spherical_water_maps[0]
    ad_map = calc_spherical_water_map(
        autogrid_exec_path,
        receptor,
        box_center,
        box_size,
        receptor_spherical_water_map_filename,
    )

    # It is more cleaner if we prepare the maps (OW, HW for tip3p, OT, HT, LP for tip5p) before
    utils.prepare_water_map(ad_map, water_model)

    # Go waterkit!!
    # The second spherical map is for the single water (used for updating maps)
    k = WaterKit(temperature, water_model, spherical_water_maps[1], n_layer, n_frames, n_jobs)
    k.hydrate(receptor, ad_map, output_dir)

if __name__ == '__main__':
    main()
