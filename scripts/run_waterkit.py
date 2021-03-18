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

from waterkit import AutoGrid
from waterkit import Map
from waterkit import Molecule
from waterkit import WaterKit
from waterkit import utils
from vina import Vina


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
    autogrid_exec_path = args.autogrid_exec_path
    water_model = 'tip3p'

    # Force to use only one thread per job
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['NUMEXPR_NUM_THREADS'] = '1'
    os.environ['OMP_NUM_THREADS'] = '1'

    # Read PDBQT/MOL2 file, Waterfield file and AutoDock grid map
    receptor = Molecule.from_file(receptor_pdbqtfilename)

    with utils.temporary_directory(prefix='wk_', dir='.', clean=False) as tmp_dir:
        # Generate AutoDock maps using the Amber ff14SB forcefield
        receptor.to_pdbqt_file('receptor.pdbqt')
        ff14sb_param_file = os.path.join(utils.path_module('waterkit'), 'data/ff14SB_parameters.dat')
        ag = AutoGrid(autogrid_exec_path, ff14sb_param_file)
        ad_map = ag.run('receptor.pdbqt', ['OW'], box_center, box_size, smooth=0, dielectric=1)

        if spherical_water_maps[0] is None:
            # Convert amber atom types to AutoDock atom types
            ad_receptor = utils.convert_amber_to_autodock_types(receptor)
            ad_receptor.to_pdbqt_file('receptor_ad.pdbqt')

            # Generate Vina maps for the spherical maps
            v = Vina(verbosity=0)
            v.set_receptor('receptor_ad.pdbqt')
            v.compute_vina_maps(box_center, box_size, force_even_voxels=True)
            v.write_maps('vina')
            sw_map = Map('vina.O_DA.map', 'SW')
        else:
            # The first spherical map is for the receptor
            sw_map = Map(spherical_water_maps[0], 'SW')

        ad_map.add_map('SW', sw_map._maps['SW'])

    # It is more cleaner if we prepare the maps (OW, HW for tip3p, OT, HT, LP for tip5p) before
    utils.prepare_water_map(ad_map, water_model)

    # Go waterkit!!
    # The second spherical map is for the single water (used for updating maps)
    k = WaterKit(temperature, water_model, spherical_water_maps[1], n_layer, n_frames, n_jobs)
    k.hydrate(receptor, ad_map, output_dir)

if __name__ == '__main__':
    main()
