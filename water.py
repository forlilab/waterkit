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

    def build_tip5p(self):
        """
        Construct hydrogen atoms (H) and lone-pairs (Lp)
        """
        # Order in which we will build H/Lp
        if self._anchor_type == "acceptor":
            d = [0.9572, 0.9572, 0.7, 0.7]
            a = [104.52, 109.47]
        else:
            d = [0.7, 0.7, 0.9572, 0.9572]
            a = [109.47, 104.52]

        coord_oxygen = self.get_coordinates(atom_id=0)[0]

        # Vector between O and the Acceptor/Donor atom
        v = utils.vector(coord_oxygen, self._anchor[0])
        v = utils.normalize(v)
        # Compute a vector perpendicular to v
        p = coord_oxygen + utils.get_perpendicular_vector(v)

        # H/Lp between O and Acceptor/Donor atom
        a1 = coord_oxygen + (d[0] * v)
        # Build the second H/Lp using the perpendicular vector p
        a2 = utils.rotate_atom(a1, coord_oxygen, p, np.radians(a[0]), d[1])

        # ... and rotate it to build the last H/Lp
        p = utils.atom_to_move(coord_oxygen, [a1, a2])
        r = coord_oxygen + utils.normalize(utils.vector(a1, a2))
        a3 = utils.rotate_atom(p, coord_oxygen, r, np.radians(a[1]/2), d[3])
        a4 = utils.rotate_atom(p, coord_oxygen, r, -np.radians(a[1]/2), d[3])

        # Add them in this order: H, H, Lp, Lp
        if self._anchor_type == "acceptor":
            atoms = [a1, a2, a3, a4]
        else:
            atoms = [a3, a4, a1, a2]

        i = 2
        for atom in atoms:
            self.add_atom(atom, atomic=1, bond=(1, i, 1))
            i += 1

    def rotate_water(self, ref_id=1, angle=0.):
        pass

    def scan(self, ref_id=1):
        pass