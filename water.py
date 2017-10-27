#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Kits
#
# Class for water
#


import numpy as np
import openbabel as ob

import utils
from molecule import Molecule


class Water(Molecule):

    def __init__(self, oxygen, anchor, anchor_type):
        # Create ob molecule and add oxygen atom
        self._OBMol = ob.OBMol()
        self.add_atom(oxygen, atomic=8)

        # Store all the informations about the anchoring
        self._anchor = np.array([anchor, anchor + utils.normalize(utils.vector(oxygen, anchor))])
        self._anchor_type = anchor_type

        # Used to store previous coordinates
        self._previous = None

    def add_atom(self, xyz, atomic, bond=None):
        """
        Add an OBAtom to the molecule
        """
        a = self._OBMol.NewAtom()
        a.SetAtomicNum(atomic)
        a.SetVector(xyz[0], xyz[1], xyz[2])

        if bond is not None and self._OBMol.NumAtoms() >= 1:
            self._OBMol.AddBond(bond[0], bond[1], bond[2])

    def update_coordinates(self, xyz, atom_id):
        """
        Update the coordinates of an OBAtom
        """
        ob_atom = self._OBMol.GetAtomById(atom_id)
        ob_atom.SetVector(xyz[0], xyz[1], xyz[2])

    def get_energy(self, ad_map):
        """
        Return the energy of the water molecule
        """
        return np.sum(ad_map.get_energy(self.get_coordinates()))

    def optimize(self, ad_map, radius=3., angle=110.):
        """
        Optimize the position of the oxygen atom. The movement of the 
        atom is contrained by the distance and the angle with the anchor
        """
        # Get all the point around the anchor (sphere)
        coord_sphere = ad_map.get_neighbor_points(self._anchor[0], radius)
        # Compute angles between all the coordinates and the anchor
        angle_sphere = utils.get_angle(coord_sphere, self._anchor[0], self._anchor[1])

        # Select coordinates with an angle superior to the choosen angle
        coord_sphere = coord_sphere[angle_sphere >= angle]

        # Get energy of all the allowed coordinates (distance + angle)
        energy_sphere = ad_map.get_energy(coord_sphere)
        # ... and get energy of the oxygen
        energy_oxygen = ad_map.get_energy(self.get_coordinates(atom_id=0))

        # And if we find something better, we update the coordinate
        if np.min(energy_sphere) < energy_oxygen:
            t = energy_sphere.argmin()

            # Save the old coordinate
            self._previous = self.get_coordinates()
            # ... update with the one
            self.update_coordinates(coord_sphere[t], atom_id=0)

    def build_hydrogens(self):
        pass

    def rotate_water(self, ref_id=1, angle=0.):
        pass

    def scan(self, ref_id=1):
        pass