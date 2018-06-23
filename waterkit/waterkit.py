#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# WaterKit
#
# The core of the WaterKit program
#

from string import ascii_uppercase

from water_box import WaterBox
from optimize import WaterNetwork


class Waterkit():

    def __init__(self, waterfield=None, water_map=None):
        self.water_boxes = []
        self._water_map = water_map
        self._waterfield = waterfield

        # Combine OA and OD to create OW
        self._water_map.combine('OW', ['OA', 'OD'], how='best')

    def hydrate(self, molecule, ad_map, n_layer=0):
        """ Hydrate the molecule by adding successive layers
        of water molecules until the box is complety full
        """
        # Guess all hydrogen bond anchors and rotatble bonds
        molecule.guess_hydrogen_bond_anchors(self._waterfield)
        molecule.guess_rotatable_bonds()
        # Combine OA and OD to create OW
        ad_map.combine('OW', ['OA', 'OD'], how='best')

        w = WaterBox(molecule, ad_map, self._water_map, self._waterfield)

        i = 1
        while True:
            # build_next_shell returns True if
            # it was able to put water molecules,
            # otherwise it returns False and we break 
            if w.build_next_shell():
                pass
            else:
                break

            if i == n_layer:
                break

            i += 1

        self.water_boxes.append(w)

    def write_shells(self, prefix='water'):
        """ Write layers of water in a PDBQT file """
        i, j = 1, 1
        ernergy = 1.0
        line = "ATOM  %5d  %-3s HOH%2s%4d    %8.3f%8.3f%8.3f  1.00%5.2f    %6.3f %2s\n"

        shell_id = self.water_boxes[0].get_number_of_shells()
        waters = [self.water_boxes[0].get_molecules_in_shell(i, True) for i in range(1, shell_id+1)]

        for shell, chain in zip(waters, ascii_uppercase):
            i, j = 1, 1

            fname = '%s_%s.pdbqt' % (prefix, chain)

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
