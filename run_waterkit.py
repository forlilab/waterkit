#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# WaterKit
#
# Launch waterkit
#


import argparse

from waterkit.waterkit import Waterkit
from waterkit.autodock_map import Map
from waterkit.molecule import Molecule
from waterkit.waterfield import Waterfield


def cmd_lineparser():
    parser = argparse.ArgumentParser(description='waterkit')
    parser.add_argument("-i", "--mol", dest="mol_file", required=True,
                        action="store", help="molecule file")
    parser.add_argument("-m", "--map", dest="map_file", required=True,
                        action="store", help="autodock map file")
    parser.add_argument("-o", "--output", dest="output_file", default='waters.pdbqt',
                        action="store", help="water molecule file (pdbqt)")
    parser.add_argument("-f", "--waterfield", dest="waterfield_file", default=None,
                         action="store", help="waterfield file")
    parser.add_argument("-w", "--watermap", dest="water_map_file", default=None,
                        action="store", help="water autodock map file")
    return parser.parse_args()

def main():
    args = cmd_lineparser()
    mol_file = args.mol_file
    map_file = args.map_file
    waterfield_file = args.waterfield_file
    output_file = args.output_file
    water_map_file = args.water_map_file

    # Read PDBQT/MOL2 file, Waterfield file and AutoDock grid map
    molecule = Molecule(mol_file)
    ad_map = Map(map_file)

    d = os.path.dirname(sys.modules[__name__].__file__)

    if waterfield_file is None:
        waterfield_file = os.path.join(d, 'data/waterfield.par')

    if water_map_file is None:
        water_map_file = os.path.join(d, 'data/water/maps.fld')

    waterfield = Waterfield(waterfield_file)
    water_map = Map(water_map_file)

    # Go waterkit!!
    k = Waterkit(waterfield, water_map)
    waters = k.hydrate(molecule, ad_map, n_layer=-1)

    # Write output files
    utils.write_water(output_file, waters)
    ad_map.to_map('HD.map', 'HD')
    ad_map.to_map('Lp.map', 'Lp')
    ad_map.to_map('OA.map', 'OA')

if __name__ == '__main__':
    main()