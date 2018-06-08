#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# WaterKit
#
# Class for water
#


import numpy as np
import openbabel as ob

import utils
from molecule import Molecule


class Water(Molecule):

    def __init__(self, oxygen_xyz, oxygen_type, anchor_xyz, anchor_type):
        # Create ob molecule and add oxygen atom
        self._OBMol = ob.OBMol()
        self.add_atom(oxygen_xyz, atom_type=oxygen_type, atom_num=8)

        # Store all the informations about the anchoring
        anchor_vector = anchor_xyz + utils.normalize(utils.vector(oxygen_xyz, anchor_xyz))
        self._anchor = np.array([anchor_xyz, anchor_vector])
        self._anchor_type = anchor_type

        self._previous = None

    def add_atom(self, atom_xyz, atom_type='OA', atom_num=1, bond=None):
        """
        Add an OBAtom to the molecule
        """
        a = self._OBMol.NewAtom()
        a.SetVector(atom_xyz[0], atom_xyz[1], atom_xyz[2])
        a.SetType(atom_type)
        # Weird thing appends here...
        # If I remove a.GetType(), the oxygen type become O3 instead of OA/HO
        a.GetType()
        a.SetAtomicNum(np.int(atom_num))

        if bond is not None and self._OBMol.NumAtoms() >= 1:
            self._OBMol.AddBond(bond[0], bond[1], bond[2])

    def update_coordinates(self, atom_xyz, atom_id):
        """
        Update the coordinates of an OBAtom
        """
        ob_atom = self._OBMol.GetAtomById(atom_id)
        ob_atom.SetVector(atom_xyz[0], atom_xyz[1], atom_xyz[2])

    def get_energy(self, ad_map, atom_id=None):
        """
        Return the energy of the water molecule
        """
        if atom_id is None:
            n_atoms = self._OBMol.NumAtoms()
            # Spherical water is only one atom, the oxygen
            if n_atoms == 1:
                atom_id = 0
            # TIP5P water is 5 atoms, we ignore the oxygen
            elif n_atoms == 5:
                atom_id = [1, 2, 3, 4]

        coordinates = self.get_coordinates(atom_id)
        atom_types = self.get_atom_types(atom_id)

        energy = 0.

        for coordinate, atom_type in zip(coordinates, atom_types):
            energy += ad_map.get_energy(coordinate, atom_type)

        return energy[0]

    def build_tip5p(self):
        """
        Construct hydrogen atoms (H) and lone-pairs (Lp)
        TIP5P parameters: http://www1.lsbu.ac.uk/water/water_models.html
        """
        # Order in which we will build H/Lp
        if self._anchor_type == "acceptor":
            d = [0.9572, 0.9572, 0.7, 0.7]
            a = [104.52, 109.47]
        else:
            d = [0.7, 0.7, 0.9572, 0.9572]
            a = [109.47, 104.52]

        coord_oxygen = self.get_coordinates(0)[0]

        # Vector between O and the Acceptor/Donor atom
        v = utils.vector(coord_oxygen, self._anchor[0])
        v = utils.normalize(v)
        # Compute a vector perpendicular to v
        p = coord_oxygen + utils.get_perpendicular_vector(v)

        # H/Lp between O and Acceptor/Donor atom
        a1 = coord_oxygen + (d[0] * v)
        # Build the second H/Lp using the perpendicular vector p
        a2 = utils.rotate_point(a1, coord_oxygen, p, np.radians(a[0]))
        a2 = utils.resize_vector(a2, d[1], coord_oxygen)

        # ... and rotate it to build the last H/Lp
        p = utils.atom_to_move(coord_oxygen, [a1, a2])
        r = coord_oxygen + utils.normalize(utils.vector(a1, a2))
        a3 = utils.rotate_point(p, coord_oxygen, r, np.radians(a[1] / 2))
        a3 = utils.resize_vector(a3, d[3], coord_oxygen)
        a4 = utils.rotate_point(p, coord_oxygen, r, -np.radians(a[1] / 2))
        a4 = utils.resize_vector(a4, d[3], coord_oxygen)

        # Add them in this order: H, H, Lp, Lp
        if self._anchor_type == "acceptor":
            atoms = [a1, a2, a3, a4]
        else:
            atoms = [a3, a4, a1, a2]

        atom_types = ['HD', 'HD', 'Lp', 'Lp']

        i = 2
        for atom, atom_type in zip(atoms, atom_types):
            self.add_atom(atom, atom_type=atom_type, atom_num=1, bond=(1, i, 1))
            i += 1

    def get_hb_anchors(self):
        """ Get the ID and atom type of all available atoms """
        if self._anchor_type == 'acceptor':
            atom_ids = [3, 4, 5]
            names = ['H_O_004', 'O_L_000', 'O_L_000']
        else:
            atom_ids = [2, 3, 5]
            names = ['H_O_004', 'H_O_004', 'O_L_000']

        return names, atom_ids

    def rotate_water(self, ref_id=1, angle=0.):
        """
        Rotate water molecule along the axis Oxygen and a choosen atom (H or Lp)
        """
        coord_water = self.get_coordinates()

        # Get the oxygen and the ref atom for the rotation axis
        coord_oxygen = coord_water[0]
        coord_ref = coord_water[ref_id]

        r = coord_oxygen + utils.normalize(utils.vector(coord_ref, coord_oxygen))

        # Ref of all the atom we want to move, minus the ref
        atom_ids = list(range(1, coord_water.shape[0]))
        atom_ids.remove(ref_id)

        for atom_id in atom_ids:
            a = utils.rotate_point(coord_water[atom_id], coord_oxygen, r, np.radians(angle))
            self.update_coordinates(a, atom_id)
