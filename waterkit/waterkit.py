#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# WaterKit
#
# The core of the WaterKit program
#

import copy
from string import ascii_uppercase

import numpy as np

from water_box import WaterBox


class Waterkit():

    def __init__(self):
        """Initialize WaterKit."""
        self.water_box = None

    def hydrate(self, receptor, ad_map, ad_forcefield, water_model="tip3p", 
                how="best", temperature=300., n_layer=1):
        """Hydrate the molecule with water molecules.

        The receptor is hydrated by adding successive layers
        of water molecules until the box is complety full.

        Args:
            receptor (Molecule): Receptor of the protein
            ad_map (Map): AutoDock map of the receptor
            ad_forcefield (AutoDockForceField): AutoDock forcefield for pairwise interactions
            water_model (str): Model used for the water molecule, tip3p or tip5p (default: tip3p)
            how (str): Method for water placement: "best" or "boltzmann" (default: best)
            temperature (float): Temperature in Kelvin, only used for Boltzmann sampling (default: 300)
            n_layer (int): Number of hydration layer to add (default: 1)

        Returns:
            bool: True if succeeded or False otherwise

        """
        i = 1

        w = WaterBox(receptor, ad_map, ad_forcefield, water_model, how, temperature)
        #w_copy = copy.deepcopy(w)

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
            of layers is 26, which corresponds to the letter Z."""
            if i == n_layer or i == 26:
                break

            i += 1

        self.water_box = w

        return True

    def write_shells(self, prefix="water", water_model='tip3p'):
        """Export hydration shells in a PDBQT format.

        Args:
            prefix (str): prefix name of the files
            water_model (str): water model to use to write coordinates (choices: tip3p or tip5p)

        Returns:
            None

        """
        output_str = ""
        pdbqt_str = "ATOM  %5d  %-3s HOH%2s%4d    %8.3f%8.3f%8.3f  1.00  1.00          %2s\n"
<<<<<<< HEAD

=======
        fname = "%s.pdb" % prefix
        
>>>>>>> disordered_hydrogens
        shell_id = self.water_box.number_of_shells()
        water_shells = [self.water_box.molecules_in_shell(i) for i in range(1, shell_id + 1)]

        i = 1

<<<<<<< HEAD
            fname = "%s_%s.pdb" % (prefix, chain)
=======
        for water_shell, chain in zip(water_shells, ascii_uppercase):
            j = 1
>>>>>>> disordered_hydrogens

            for water in water_shell:
                c = water.coordinates()

<<<<<<< HEAD
                output_str += pdbqt_str % (j, "O", chain, i, c[0][0], c[0][1], c[0][2], "O")
                output_str += pdbqt_str % (j + 1, "H1", chain, i, c[1][0], c[1][1], c[1][2], "H")
                output_str += pdbqt_str % (j + 2, "H2", chain, i, c[2][0], c[2][1], c[2][2], "H")
                j += 2

                if water_model == 'tip5p':
                    output_str += pdbqt_str % (j + 3, "L1", chain, i, c[3][0], c[3][1], c[3][2], "L")
                    output_str += pdbqt_str % (j + 4, "L2", chain, i, c[4][0], c[4][1], c[4][2], "L")
                    j += 2
=======
                output_str += pdbqt_str % (i, "O", chain, j, c[0][0], c[0][1], c[0][2], "O")
                output_str += pdbqt_str % (i + 1, "H1", chain, j, c[1][0], c[1][1], c[1][2], "H")
                output_str += pdbqt_str % (i + 2, "H2", chain, j, c[2][0], c[2][1], c[2][2], "H")
                i += 2

                if water_model == 'tip5p':
                    output_str += pdbqt_str % (i + 3, "L1", chain, j, c[3][0], c[3][1], c[3][2], "L")
                    output_str += pdbqt_str % (i + 4, "L2", chain, j, c[4][0], c[4][1], c[4][2], "L")
                    i += 2
>>>>>>> disordered_hydrogens

                i += 1
                j += 1

        with open(fname, "w") as w:
            w.write(output_str)
