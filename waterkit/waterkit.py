#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# WaterKit
#
# The core of the WaterKit program
#

import os
import imp
from string import ascii_uppercase

from autodock_map import Map
from water_box import WaterBox
from waterfield import Waterfield
from optimize import WaterNetwork


class Waterkit():

    def __init__(self, waterfield=None, water_map=None):
        self.water_boxes = []
        self._water_map = water_map
        self._waterfield = waterfield

        if self._waterfield is None:
            d = imp.find_module('waterkit')[1]
            waterfield_file = os.path.join(d, 'data/waterfield.par')
            self._waterfield = Waterfield(waterfield_file)

        if self._water_map is None:
            d = imp.find_module('waterkit')[1]
            water_fld_file = os.path.join(d, 'data/water/maps.fld')
            self._water_map = Map.from_fld(water_fld_file)

        # Combine OA and OD to create OW
        self._water_map.combine('OW', ['OA', 'OD'], how='best')

    def hydrate(self, receptor, ad_map, waters=None, n_layer=0):
        """ Hydrate the molecule by adding successive layers
        of water molecules until the box is complety full
        """
        # Combine OA and OD to create OW
        ad_map.combine('OW', ['OA', 'OD'], how='best')

        w = WaterBox(self._waterfield, self._water_map)
        w.add_receptor(receptor, ad_map)

        if waters is not None:
            w.add_crystallographic_waters(waters)

        i = 1
        while True:
            # build_next_shell returns True if
            # it was able to put water molecules,
            # otherwise it returns False and we break 
            if w.build_next_shell():
                pass
            else:
                break

            """Stop after building n_layer or when the number of
            layers is equal to 26. In the PDB format, the segid
            is encoded with only one caracter, so the maximum number
            of layers is 26, which corresponds to the letter Z.
            """
            if i == n_layer or i == 26:
                break

            i += 1

        self.water_boxes.append(w)

    def write_shells(self, prefix='water', only_active=True):
        """ Write layers of water in a PDBQT file """
        i, j = 1, 1
        ernergy = 1.0
        line = "ATOM  %5d  %-3s HOH%2s%4d    %8.3f%8.3f%8.3f%6.2f 1.00    %6.3f %2s\n"
        active = ''

        if not only_active:
            active = '_all'

        shell_id = self.water_boxes[0].number_of_shells()
        waters = [self.water_boxes[0].molecules_in_shell(i, only_active) for i in range(1, shell_id+1)]

        for shell, chain in zip(waters, ascii_uppercase):
            i, j = 1, 1

            fname = '%s_%s%s.pdbqt' % (prefix, chain, active)

            with open(fname, 'w') as w:
                for water in shell:
                    c = water.get_coordinates()

                    try:
                        e = water.energy
                    except:
                        e = 0.0

                    w.write(line % (j, 'O', chain, i, c[0][0], c[0][1], c[0][2], e, 0, 'OA'))

                    if c.shape[0] == 5:
                        w.write(line % (j + 1, 'H', chain, i, c[1][0], c[1][1], c[1][2], e, 0.2410, 'HD'))
                        w.write(line % (j + 2, 'H', chain, i, c[2][0], c[2][1], c[2][2], e, 0.2410, 'HD'))
                        w.write(line % (j + 3, 'H', chain, i, c[3][0], c[3][1], c[3][2], e, -0.2410, 'Lp'))
                        w.write(line % (j + 4, 'H', chain, i, c[4][0], c[4][1], c[4][2], e, -0.2410, 'Lp'))
                        j += 4

                    i += 1
                    j += 1

    def write_maps(self, prefix, map_types=None):
        """ Write maps for each layer of water molecules """
        for water_map, chain in zip(self.water_boxes[0].maps, ascii_uppercase):
            water_map.to_map(map_types, '%s_%s' % (prefix, chain))
