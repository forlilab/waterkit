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
from waterkit import WaterBox
from waterkit import WaterKit
from waterkit import Water
from waterkit import utils
from vina import Vina


def cmd_lineparser():
    parser = argparse.ArgumentParser(description='waterkit')
    parser.add_argument('-w', '--frame', help='input frame with waters [.pdb]', required=True) 
    parser.add_argument('-i', '--mol', dest='receptor_pdbqtfilename', required=True,
                        action='store', help='receptor file')
    parser.add_argument('-c', '--center', dest='box_center', nargs=3, type=float,
                        action='store', help='center of the box')
    parser.add_argument('-s', '--size', dest='box_size', nargs=3, type=int, required=True,
                        action='store', help='size of the box in Angstrom')
    parser.add_argument('--autogrid_exec_path', dest='autogrid_exec_path', default='autogrid4',
                        action='store', help='path to the autogrid4 executable (default: autogrid4')
    return parser.parse_args()


if __name__ == "__main__":
    args = cmd_lineparser()
    receptor_pdbqtfilename = args.receptor_pdbqtfilename
    water_frame_filename = args.frame
    box_center = args.box_center
    box_size = args.box_size
    autogrid_exec_path = args.autogrid_exec_path
    water_model = 'tip3p'
    spherical_water_maps = (None, None)
    temperature = None  # we don't need it here

    print(f"{box_size=}")

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
            v.compute_vina_maps(box_center, box_size, force_even_voxels=False)
            v.write_maps('vina')
            sw_map = Map('vina.O_DA.map', 'SW')
        else:
            raise NotImplementedError("hard-coded to use vina's O_DA map as spherical water map")

        ad_map.add_map('SW', sw_map._maps['SW'])

    # It is more cleaner if we prepare the maps (OW, HW for tip3p, OT, HT, LP for tip5p) before
    utils.prepare_water_map(ad_map, water_model)

    waters, reskeys = Water.from_file(water_frame_filename)
    nr_waters = len(waters)
    for index in range(nr_waters):

        # add other waters to maps
        this_water = waters[index]
        other_waters = waters[:index] + waters[(index + 1):]
        # WaterBox init makes a copy of ad_map internally in self.map
        wbox = WaterBox(receptor, ad_map, temperature, water_model, spherical_water_maps[1])
        for other_water in other_waters:

            wbox._wopt._update_maps(other_water)

        # compute energy of current water
        oxygen_xyz = this_water.coordinates(1)
        water_info = this_water.atom_informations()
        energy = wbox.map.energy_coordinates(oxygen_xyz, water_info["t"][0])[0]
        print(f"{index:3} {reskeys[index]:6} {energy:12.6f} O (rec+wat)")
        energy_rec = ad_map.energy_coordinates(oxygen_xyz, water_info["t"][0])[0]
        print(f"{index:3} {reskeys[index]:6} {energy_rec:12.6f} O (just rec)")
        for i, atom_type in enumerate(water_info["t"][1:]):
            # i + 2 because 1-indexed and because we skipped first element in enumerate
            e = wbox.map.energy_coordinates(this_water.coordinates(i+2), atom_type)[0]
            energy += e 
            print(f"{index:3} {reskeys[index]:6} {e:12.6f} H (rec+wat)")

            er = ad_map.energy_coordinates(this_water.coordinates(i+2), atom_type)[0]
            energy_rec += er
            print(f"{index:3} {reskeys[index]:6} {er:12.6f} H (just rec)")

        print(f"{index:3} {reskeys[index]:6} {energy:12.6f} HOH (rec+wat)")
        print(f"{index:3} {reskeys[index]:6} {energy_rec:12.6f} HOH (just rec)")
        print()

