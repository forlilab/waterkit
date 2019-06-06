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

import numpy as np

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
        self.water_box = None
        self._hb_forcefield = hb_forcefield

        """If the user does not provide any of these elements,
        we take those available per default in waterkit."""
        if self._hb_forcefield is None:
            d = imp.find_module("waterkit")[1]
            hb_forcefield_file = os.path.join(d, "data/waterfield.par")
            self._hb_forcefield = Waterfield(hb_forcefield_file)

    def hydrate(self, receptor, ad_map, water_model="tip3p", n_layer=1, 
                how="best", temperature=300., smooth=0.5, dielectric=-0.1465):
        """Hydrate the molecule with water molecules.

        The receptor is hydrated by adding successive layers
        of water molecules until the box is complety full.

        Args:
            receptor (Molecule): Receptor of the protein
            ad_map (Map): AutoDock map of the receptor
            water_model (str): Model used for the water molecule, tip3p or tip5p (default: tip3p)
            n_layer (int): Number of hydration layer to add (default: 1)
            how (str): Method for water placement: "best" or "boltzmann" (default: best)
            temperature (float): Temperature in Kelvin, only used for Boltzmann sampling (default: 300)
            smooth (float): AutoDock smooth parameter (default: 0.5)
            dielectric (float): AutoDock dielectric constant (default: -0.1465)

        Returns:
            bool: True if succeeded or False otherwise

        """
        i = 1
        e_type = "Electrostatics"
        ow_q = -0.834
        ow_type = "OW"

        """In TIP3P and TIP5P models, hydrogen atoms and lone-pairs does not
        have VdW radius, so their interactions with the receptor are purely
        based on electrostatics. So the HD and Lp maps are just the electrostatic 
        map. Each map is multiplied by the partial charge. So it is just a
        look-up table to get the energy for each water molecule.
        """
        if water_model == "tip3p":
            hw_type = "HW"
            hw_q = 0.417
        elif water_model == "tip5p":
            ot_type = "OT"
            hw_type = "HT"
            lpw_type = "LP"
            hw_q = 0.241
            lpw_q = -0.241
        else:
            print "Error: water model %s unknown." % water_model
            return False

        # For the TIP3P and TIP5P models
        ad_map.apply_operation_on_maps(hw_type, e_type, "x * %f" % hw_q)
        if water_model == "tip5p":
            ad_map.apply_operation_on_maps(lpw_type, e_type, "x * %f" % lpw_q)
            # Necessary for the TIP5P oxygen
            ad_map.create_empty_map(ot_type)
        # For the spherical model and TIP3P model
        ad_map.apply_operation_on_maps(e_type, e_type, "-np.abs(x * %f)" % ow_q)
        ad_map.combine(ow_type, [ow_type, e_type], how="add")

        #w_copy = copy.deepcopy(w_ori)
        w = WaterBox(self._hb_forcefield, water_model, smooth, dielectric)
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

    def write_shells(self, prefix="water"):
        """Export hydration shells in a PDBQT format.

        Args:
            prefix (str): prefix name of the files

        Returns:
            None

        """
        output_str = ""
        pdbqt_str = "ATOM  %5d  %-3s HOH%2s%4d    %8.3f%8.3f%8.3f%6.2f 1.00    %6.3f %2s\n"

        shell_id = self.water_box.number_of_shells()
        waters = [self.water_box.molecules_in_shell(i) for i in range(1, shell_id + 1)]

        for shell, chain in zip(waters, ascii_uppercase):
            i, j = 1, 1

            fname = "%s_%s.pdbqt" % (prefix, chain)

            for water in shell:
                c = water.coordinates()

                try:
                    e = water.energy
                    # Truncate energy 
                    e = np.clip(e, -99.99, 99.99)
                except:
                    e = 0.0

                if c.shape[0] == 3:
                    output_str += pdbqt_str % (j, "O", chain, i, c[0][0], c[0][1], c[0][2], e, -0.834, "O")
                    output_str += pdbqt_str % (j + 1, "H", chain, i, c[1][0], c[1][1], c[1][2], e,0.417, "H")
                    output_str += pdbqt_str % (j + 2, "H", chain, i, c[2][0], c[2][1], c[2][2], e, 0.417, "H")
                    j += 2

                if c.shape[0] == 5:
                    output_str += pdbqt_str % (j, "O", chain, i, c[0][0], c[0][1], c[0][2], e, 0, "O")
                    output_str += pdbqt_str % (j + 1, "H", chain, i, c[1][0], c[1][1], c[1][2], e, 0.241, "H")
                    output_str += pdbqt_str % (j + 2, "H", chain, i, c[2][0], c[2][1], c[2][2], e, 0.241, "H")
                    output_str += pdbqt_str % (j + 3, "H", chain, i, c[3][0], c[3][1], c[3][2], e, -0.241, "L")
                    output_str += pdbqt_str % (j + 4, "H", chain, i, c[4][0], c[4][1], c[4][2], e, -0.241, "L")
                    j += 4

                i += 1
                j += 1

            with open(fname, "w") as w:
                w.write(output_str)
