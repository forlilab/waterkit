#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# WaterKit
#
# The core of the WaterKit program
#

import os
import copy
import imp
from string import ascii_uppercase

from autodock_map import Map
from forcefield import AutoDockForceField
from water_box import WaterBox
from waterfield import Waterfield

class Waterkit():

    def __init__(self, hb_forcefield=None):
        """Initialize WaterKit.

        Args:
            hb_forcefield (Waterfield): Hydrogen Bond forcefield (default: None)

        """
        self.water_boxes = []
        self._hb_forcefield = hb_forcefield

        """If the user does not provide any of these elements,
        we take those available per default in waterkit."""
        if self._hb_forcefield is None:
            d = imp.find_module('waterkit')[1]
            hb_forcefield_file = os.path.join(d, 'data/waterfield.par')
            self._hb_forcefield = Waterfield(hb_forcefield_file)

    def hydrate(self, receptor, ad_map, water_model="tip3p", n_layer=1, 
                how="best", temperature=300.):
        """ Hydrate the molecule with water molecucules.

        The receptor is hydrated by adding successive layers
        of water molecules until the box is complety full.

        Args:
            receptor (Molecule): Receptor of the protein
            ad_map (Map): AutoDock map of the receptor
            water_model (str): Model used for the water molecule, tip3p or tip5p (default: tip3p)
            n_layer (int): Number of hydration layer to add (default: 1)
            how (str): Method for water placement: 'best' or 'boltzmann' (default: best)
            temperature (float): Temperature in Kelvin, only used for Boltzmann sampling (default: 300)

        Returns:
            None

        """
        i = 1
        type_e = 'Electrostatics'

        """In TIP3P and TIP5P models, hydrogen atoms and lone-pairs does not
        have VdW radius, so their interactions with the receptor are purely
        based on electrostatics. So the HD and Lp maps are just the electrostatic 
        map. Each map is multiplied by the partial charge. So it is just a
        look-up table to get the energy for each water molecule.
        """
        if water_model == 'tip3p':
            type_ow = 'OW'
            type_hd = 'HW'
            hw_q = 0.417
            ow_q = -0.834
        elif water_model == 'tip5p':
            type_ow = 'OW'
            type_hd = 'HT'
            type_lp = 'LP'
            hw_q = 0.241
            lp_q = -0.241
            # Need to put a charge for the placement of the spherical water
            ow_q = -0.482
        else:
            print "Error: water model %s unknown." % water_model
            return False

        ad_map.apply_operation_on_maps(type_hd, type_e, 'x * %f' % hw_q)
        if water_model == 'tip5p':
            ad_map.apply_operation_on_maps(type_lp, type_e, 'x * %f' % lp_q)
        ad_map.apply_operation_on_maps(type_e, type_e, '-np.abs(x * %f)' % ow_q)
        ad_map.combine(type_ow, [type_ow, type_e], how='add')

        #w_copy = copy.deepcopy(w_ori)
        w = WaterBox(self._hb_forcefield, water_model)
        w.add_receptor(receptor, ad_map)

        while True:
            # build_next_shell returns True if
            # it was able to put water molecules,
            # otherwise it returns False and we break
            if w.build_next_shell(how, temperature):
                pass
            else:
                break

            """Stop after building n_layer or when the number of
            layers is equal to 26. In the PDB format, the segid
            is encoded with only one caracter, so the maximum number
            of layers is 26, which corresponds to the letter Z."""
            if i == n_layer or i == 26:
                break

            i += 1

        self.water_box = w

        return True

    def write_shells(self, prefix='water', only_active=True):
        """Export hydration shells in a PDBQT format.

        Args:
            prefix (str): prefix name of the files
            only_active (bool): Write only active water molecules

        Returns:
            None

        """
        line = "ATOM  %5d  %-3s HOH%2s%4d    %8.3f%8.3f%8.3f%6.2f 1.00    %6.3f %2s\n"

        if not only_active:
            active = '_all'
        else:
            active = ''

        shell_id = self.water_box.number_of_shells()
        waters = [self.water_box.molecules_in_shell(i, only_active) for i in range(1, shell_id+1)]

        for shell, chain in zip(waters, ascii_uppercase):
            i, j = 1, 1

            fname = '%s_%s%s.pdbqt' % (prefix, chain, active)

            with open(fname, 'w') as w:
                for water in shell:
                    c = water.coordinates()

                    try:
                        e = water.energy
                    except:
                        e = 0.0

                    if c.shape[0] == 3:
                        w.write(line % (j, 'O', chain, i, c[0][0], c[0][1], c[0][2], e, -0.834, 'OW'))
                        w.write(line % (j + 1, 'H', chain, i, c[1][0], c[1][1], c[1][2], e,0.417, 'HW'))
                        w.write(line % (j + 2, 'H', chain, i, c[2][0], c[2][1], c[2][2], e, 0.417, 'HW'))
                        j += 2

                    if c.shape[0] == 5:
                        w.write(line % (j, 'O', chain, i, c[0][0], c[0][1], c[0][2], e, 0, 'OT'))
                        w.write(line % (j + 1, 'H', chain, i, c[1][0], c[1][1], c[1][2], e, 0.241, 'HT'))
                        w.write(line % (j + 2, 'H', chain, i, c[2][0], c[2][1], c[2][2], e, 0.241, 'HT'))
                        w.write(line % (j + 3, 'H', chain, i, c[3][0], c[3][1], c[3][2], e, -0.241, 'LP'))
                        w.write(line % (j + 4, 'H', chain, i, c[4][0], c[4][1], c[4][2], e, -0.241, 'LP'))
                        j += 4

                    i += 1
                    j += 1
