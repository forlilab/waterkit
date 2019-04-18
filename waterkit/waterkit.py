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

    def __init__(self, hb_forcefield=None,  ad4_forcefield=None, water_map=None):
        """Initialize WaterKit.

        Args:
            hb_forcefield (Waterfield): Hydrogen Bond forcefield (default: None)
            ad4_forcefield (AutoDockForceField): AutoDock forcefield (default: None)
            water_map (Map): AutoDock Map of a reference water (default: None)

        """
        self.water_boxes = []
        self._hb_forcefield = hb_forcefield
        self._ad4_forcefield = ad4_forcefield
        self._water_map = water_map

        # AD map names
        self._type_lp = 'Lp'
        self._type_hd = 'HD'
        self._type_dhd = 'Hd'
        self._type_oa = 'Oa'
        self._type_od = 'Od'
        self._type_w = 'Ow'
        self._type_e = 'Electrostatics'

        """If the user does not provide any of these elements,
        we take those available per default in waterkit."""
        if self._hb_forcefield is None:
            d = imp.find_module('waterkit')[1]
            hb_forcefield_file = os.path.join(d, 'data/waterfield.par')
            self._hb_forcefield = Waterfield(hb_forcefield_file)

        if self._ad4_forcefield is None:
            d = imp.find_module('waterkit')[1]
            ad4_forcefield_file = os.path.join(d, 'data/AD4_parameters.dat')
            self._ad4_forcefield = AutoDockForceField(ad4_forcefield_file)

        if self._water_map is None:
            d = imp.find_module('waterkit')[1]
            water_fld_file = os.path.join(d, 'data/water/maps.fld')
            self._water_map = Map.from_fld(water_fld_file)

        # Combine OA, OD and e to create OW
        # Since waters are spherical, both donor/acceptor, so e is always favorable
        self._water_map.apply_operation_on_maps('-np.abs(x)', [self._type_e])
        self._water_map.combine(self._type_w, [self._type_oa, self._type_od, self._type_e], how='add')
        # Add e to HD and Lp
        self._water_map.combine('HD', ['HD', self._type_e], how='add')
        self._water_map.combine(self._type_lp, [self._type_lp, self._type_e], how='add')

        # Deactivate HD-Hd interaction in AutoDock ForceField
        self._ad4_forcefield.deactivate_pairs([[self._type_hd, self._type_dhd]])

    def hydrate(self, receptor, ad_map, waters=None, n_layer=1, how='best', temperature=300.):
        """ Hydrate the molecule with water molecucules.

        The receptor is hydrated by adding successive layers
        of water molecules until the box is complety full.

        Args:
            receptor (Molecule): Receptor of the protein
            ad_map (Map): AutoDock map of the receptor
            waters (list): List of X-Ray water molecules (Water) to incorporate (default: None)
            n_layer (int): Number of hydration layer to add (default: 1)
            how (str): Method for water placement: 'best' or 'boltzmann' (default: best)
            temperature (float): Temperature in Kelvin, only used for Boltzmann sampling (default: 300)

        Returns:
            None

        """
        # Combine OA, OD and e to create OW
        # Warning: this is not the same how as the one passed in input
        ad_map.apply_operation_on_maps('-np.abs(x)', [self._type_e])
        ad_map.combine(self._type_w, [self._type_oa, self._type_od, self._type_e], how='add')
        ad_map.combine(self._type_hd, [self._type_hd, self._type_e], how='add')
        ad_map.combine(self._type_lp, [self._type_lp, self._type_e], how='add')

        #w_copy = copy.deepcopy(w_ori)
        w = WaterBox(self._hb_forcefield, self._ad4_forcefield, self._water_map)
        w.add_receptor(receptor, ad_map)

        if waters is not None:
            w.add_crystallographic_waters(waters, how)

        i = 1
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

                    w.write(line % (j, 'O', chain, i, c[0][0], c[0][1], c[0][2], e, 0, 'OA'))

                    if c.shape[0] == 5:
                        w.write(line % (j + 1, 'H', chain, i, c[1][0], c[1][1], c[1][2], e, 0.2410, 'HD'))
                        w.write(line % (j + 2, 'H', chain, i, c[2][0], c[2][1], c[2][2], e, 0.2410, 'HD'))
                        w.write(line % (j + 3, 'H', chain, i, c[3][0], c[3][1], c[3][2], e, -0.2410, 'Lp'))
                        w.write(line % (j + 4, 'H', chain, i, c[4][0], c[4][1], c[4][2], e, -0.2410, 'Lp'))
                        j += 4

                    i += 1
                    j += 1
