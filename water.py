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

    def add_atom(self, xyz, atomic, bond=None):
        a = self._OBMol.NewAtom()
        a.SetAtomicNum(atomic)
        a.SetVector(xyz[0], xyz[1], xyz[2])

        if bond is not None and self._OBMol.NumAtoms() >= 1:
            self._OBMol.AddBond(bond[0], bond[1], bond[2])

    def coordinates(self, atom_id=None):
        if atom_id is not None:
            ob_atom = self._OBMol.GetAtomById(atom_id)
            coordinate = [ob_atom.GetX(), ob_atom.GetY(), ob_atom.GetZ()]
        else:
            coordinate = [[x.GetX(), x.GetY(), x.GetZ()] for x in ob.OBMolAtomIter(self._OBMol)]

        return np.atleast_2d(np.array(coordinate))

    def update_coordinates(xyz, atom_id):
        ob_atom = self._OBMol.GetAtomById(atom_id)
        ob_atom.SetVector(xyz[0], xyz[1], xyz[2])

    def optimize(self, ad_map, radius=3., angle=110., ignore=0):
        # Get oxygen coordinate
        coordinate = self.coordinates(atom_id=0)
        # Get energy of the current position
        energy = ad_map.get_energy(coordinate)

        # Get all the point around a coordinate (sphere)
        ad_map.get_neighbor_points(coordinate, radius)

        """
        # Compute angle
        angles = utils.get_angle(coordinates, self._anchor[0], self._anchor[1])
        # Keep only coordinates within the angle
        idx = np.array((angles <= angle).nonzero()).T

        if new_energy < energy:
            energy = new_energy
            coordinate = new_coordinate

        if energy < ignore:
            self.update_coordinates(coordinate, atom_id=0)
        """

    def build_hydrogens(self):
        pass

    def rotate_water(self, ref_id=1, angle=0.):
        pass

    def scan(self, ref_id=1):
        pass