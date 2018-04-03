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

    def __init__(self, oxygen, anchor, anchor_type):
        # Create ob molecule and add oxygen atom
        self._OBMol = ob.OBMol()
        self.add_atom(oxygen, atom_type='OA', atom_num=8)

        # Store all the informations about the anchoring
        self._anchor = np.array([anchor, anchor + utils.normalize(utils.vector(oxygen, anchor))])
        self._anchor_type = anchor_type

        self._previous = None

    def add_atom(self, xyz, atom_type='OA', atom_num=1, bond=None):
        """
        Add an OBAtom to the molecule
        """
        a = self._OBMol.NewAtom()
        a.SetVector(xyz[0], xyz[1], xyz[2])
        a.SetType(atom_type)
        # Weird thing appends here...
        # If I remove a.GetType(), the oxygen type become O3 instead of O
        a.GetType()
        a.SetAtomicNum(np.int(atom_num))

        if bond is not None and self._OBMol.NumAtoms() >= 1:
            self._OBMol.AddBond(bond[0], bond[1], bond[2])

    def update_coordinates(self, xyz, atom_id):
        """
        Update the coordinates of an OBAtom
        """
        ob_atom = self._OBMol.GetAtomById(atom_id)
        ob_atom.SetVector(xyz[0], xyz[1], xyz[2])

    def get_energy(self, ad_map, atom_id=None):
        """
        Return the energy of the water molecule
        """
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

        atom_types = ['HD', 'HD', 'Lp', 'Lp']

        i = 2
        for atom, atom_type in zip(atoms, atom_types):
            self.add_atom(atom, atom_type=atom_type, atom_num=1, bond=(1, i, 1))
            i += 1

    def get_available_anchors(self):
        """ Get the ID and atom type of all available atoms """
        if self._anchor_type == 'acceptor':
            atom_ids = [3, 4, 5]
            names = ['H_O_004', 'O_L_000', 'O_L_000']
        else:
            atom_ids = [2, 3, 5]
            names = ['H_O_004', 'H_O_004', 'O_L_000']

        return names, atom_ids

    @staticmethod
    def complete_map(waters, ad_map, water_map, water_orientation=[[0, 0, 1], [1, 0, 0]]):

        x_len = np.int(np.floor(water_map._grid[0].shape[0]/2.) + 5)
        y_len = np.int(np.floor(water_map._grid[1].shape[0]/2.) + 5)
        z_len = np.int(np.floor(water_map._grid[2].shape[0]/2.) + 5)

        map_types = set(ad_map._maps.keys()) & set(water_map._maps.keys())

        for water in waters:

            o = water.get_coordinates(atom_id=0)[0]
            h1, h2 = water.get_coordinates(atom_id=1)[0], water.get_coordinates(atom_id=2)[0]

            # Create the grid around the protein water molecule
            ix, iy, iz = ad_map._cartesian_to_index(o)

            ix_min = ix - x_len if ix - x_len >= 0 else 0
            ix_max = ix + x_len
            iy_min = iy - y_len if iy - y_len >= 0 else 0
            iy_max = iy + y_len
            iz_min = iz - z_len if iz - z_len >= 0 else 0
            iz_max = iz + z_len

            x = ad_map._grid[0][ix_min:ix_max+1]
            y = ad_map._grid[1][iy_min:iy_max+1]
            z = ad_map._grid[2][iz_min:iz_max+1]

            X, Y, Z = np.meshgrid(x, y, z)
            grid = np.stack((X.ravel(), Y.ravel(), Z.ravel()), axis=-1)

            # Do the translation
            translation = utils.vector(o, water_map._center)
            grid += translation

            # First rotation along z-axis
            u = utils.normalize(utils.vector(o, np.mean([h1, h2], axis=0)))
            rotation_z = utils.get_rotation_matrix(u, water_orientation[0])
            grid = np.dot(grid, rotation_z)

            # Second rotation along x-axis
            h1 = np.dot(h1 + translation, rotation_z)
            h2 = np.dot(h2 + translation, rotation_z)
            v = utils.normalize(np.cross(h1, h2))
            rotation_x = utils.get_rotation_matrix(v, water_orientation[1])
            grid = np.dot(grid, rotation_x)

            for map_type in map_types:
                # Interpolate energy
                energy = water_map.get_energy(grid, map_type)
                # Replace inf by zero, otherwise we cannot add water energy to the grid
                energy[energy == np.inf] = 0.

                # Reshape and swap x and y axis, right? Easy.
                # Thank you Diogo Santos Martins!!
                energy = np.reshape(energy, (y.shape[0], x.shape[0], z.shape[0]))
                energy = np.swapaxes(energy, 0, 1)

                # Add it to the existing grid
                ad_map._maps[map_type][ix_min:ix_max+1, iy_min:iy_max+1, iz_min:iz_max+1] += energy

        # Update interpolator
        for map_type in map_types:
            ad_map._maps_interpn[map_type] = ad_map._generate_affinity_map_interpn(ad_map._maps[map_type])

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
            a = utils.rotate_atom(coord_water[atom_id], coord_oxygen, r, np.radians(angle))
            self.update_coordinates(a, atom_id)
