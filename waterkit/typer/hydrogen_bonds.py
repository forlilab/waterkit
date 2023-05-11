#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# WaterKit
#
# Class to manage hydrogen bond typer
#

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import re
from collections import namedtuple
from collections import OrderedDict

import numpy as np
import openbabel as ob
import pandas as pd

from .. import utils


class HydrogenBonds():

    def __init__(self, fname):
        # Create ordered dict to conserved the order of atom types
        # which is very important in our particular case for the moment
        self._atom_types = OrderedDict()
        field_names = 'hb_type hb_strength hyb n_water hb_length ob_smarts'
        self._Atom_type = namedtuple('Atom_type', field_names)

        self._load_param_file(fname)

    def _load_param_file(self, fname):
        """Load the file to create a hydrogen bond typer object
        """
        with open(fname) as f:
            lines = f.readlines()

            # ATOM NAME(%10s) TYPE(%2s) STRENGTH(%8.3f) HYB(%2s) #WATER(%2s) RADIUS(%8.3f) SMARTS(%s)

            for line in lines:
                if re.search('^ATOM', line):

                    # Split by space and remove them in the list
                    sline = line.split(' ')
                    sline = [e for e in sline if e]

                    ob_smarts = ob.OBSmartsPattern()
                    success = ob_smarts.Init(sline[7])

                    if success:
                        name = sline[1]
                        hb_type = int(sline[2])
                        hb_strength = float(sline[3])
                        hyb = int(sline[4])
                        n_water = int(sline[5])
                        hb_length = float(sline[6])

                        hb_type = self._Atom_type(hb_type, hb_strength, hyb, n_water, hb_length, ob_smarts)
                        self._atom_types[name] = hb_type
                    else:
                        print("Warning: invalid SMARTS pattern %s for atom type %s." % (sline[7], sline[1]))

    def _push_atom_to_end(self, lst, atomic_nums):
        """
        Return a list of OBAtom with all the atom type selected at the end
        """
        if not isinstance(atomic_nums, (list, tuple)):
            atomic_nums = [atomic_nums]

        for atomic_num in atomic_nums:
            pop_count = 0

            idx = [i for i, x in enumerate(lst) if x.GetAtomicNum() == atomic_num]

            for i in idx:
                lst.append(lst.pop(i - pop_count))
                pop_count += 1

        return lst

    def _neighbor_atoms(self, OBMol, start_index=1, depth=1, hydrogen=True):
        """
        Return a nested list of all the neighbor OBAtoms by following the bond connectivity
        https://baoilleach.blogspot.com/2008/02/calculate-circular-fingerprints-with.html
        """
        visited = [False] * (OBMol.NumAtoms() + 1)
        neighbors = []
        queue = list([(start_index, 0)])
        atomic_num_to_keep = 1

        if not hydrogen:
            atomic_num_to_keep = 2

        while queue:
            i, d = queue.pop(0)

            ob_atom = OBMol.GetAtom(int(i))

            # If we construct the data structure before [[n], [n1, n2, ...], ...]
            # and because the depth is too large compared to the molecule
            # we will have some extra [] not filled
            try:
                neighbors[d].append(ob_atom)
            except:
                neighbors.append([])
                neighbors[d].append(ob_atom)

            visited[i] = True

            if d < depth:
                for a in ob.OBAtomAtomIter(ob_atom):
                    if not visited[a.GetIdx()] and a.GetAtomicNum() >= atomic_num_to_keep:
                        queue.append((a.GetIdx(), d + 1))

        # We push all the hydrogen (= 1) atom to the end
        neighbors = [self._push_atom_to_end(x, 1) for x in neighbors]

        return neighbors

    def _neighbor_atom_coordinates(self, OBMol, id_atom, depth=1, hydrogen=True):
        """
        Return a nested list of all the coordinates of all the neighbor
        atoms by following the bond connectivity
        """
        coords = []

        atoms = self._neighbor_atoms(OBMol, id_atom, depth, hydrogen)

        for level in atoms:
            tmp = [[ob_atom.GetX(), ob_atom.GetY(), ob_atom.GetZ()] for ob_atom in level]
            coords.append(np.array(tmp))

        return coords

    def _hb_vectors(self, OBMol, idx, hyb, n_hbond, hb_length):
        """Return all the hydrogen bond vectors the atom idx."""
        vectors = []

        # Get origin atom
        ob_atom = OBMol.GetAtom(int(idx))
        anchor_xyz = np.array([ob_atom.GetX(), ob_atom.GetY(), ob_atom.GetZ()])
        # Get coordinates of all the neihbor atoms
        neighbors_xyz = self._neighbor_atom_coordinates(OBMol, idx, depth=2)
        neighbor1_xyz = neighbors_xyz[1][0]

        if hyb == 1:
            # Position of water is linear
            # And we just need the origin atom and the first neighboring atom
            # Example: H donor
            if n_hbond == 1:
                r = None
                p = anchor_xyz + utils.vector(neighbor1_xyz, anchor_xyz)
                angles = [0]

            if n_hbond == 3:
                hyb = 3

        elif hyb == 2:
            # Position of water is just above the origin atom
            # We need the 2 direct neighboring atoms of the origin atom
            # Example: Nitrogen
            if n_hbond == 1:
                neighbor2_xyz = neighbors_xyz[1][1]

                r = None
                p = utils.atom_to_move(anchor_xyz, [neighbor1_xyz, neighbor2_xyz])
                angles = [0]

            # Position of waters are separated by angle of 120 degrees
            # And they are aligned with the neighboring atoms (deep=2) of the origin atom
            # Exemple: Backbone oxygen
            elif n_hbond == 2:
                neighbor2_xyz = neighbors_xyz[2][0]

                r = utils.rotation_axis(neighbor1_xyz, anchor_xyz, neighbor2_xyz, origin=anchor_xyz)
                p = neighbor1_xyz
                angles = [-np.radians(120), np.radians(120)]

            elif n_hbond == 3:
                hyb = 3

        if hyb == 3:
            # Position of water is just above the origin atom
            # We need the 3 direct neighboring atoms (tetrahedral)
            # Exemple: Ammonia
            if n_hbond == 1:
                neighbor2_xyz = neighbors_xyz[1][1]
                neighbor3_xyz = neighbors_xyz[1][2]

                # We have to normalize bonds, otherwise the water molecule is not well placed
                v1 = anchor_xyz + utils.normalize(utils.vector(anchor_xyz, neighbor1_xyz))
                v2 = anchor_xyz + utils.normalize(utils.vector(anchor_xyz, neighbor2_xyz))
                v3 = anchor_xyz + utils.normalize(utils.vector(anchor_xyz, neighbor3_xyz))

                r = None
                p = utils.atom_to_move(anchor_xyz, [v1, v2, v3])
                angles = [0]

            # Position of waters are separated by angle of 109 degrees
            # Tetrahedral geometry, perpendicular with the neighboring atoms of the origin atom
            # Example: Oxygen of the hydroxyl group
            elif n_hbond == 2:
                neighbor2_xyz = neighbors_xyz[1][1]
                v1 = anchor_xyz + utils.normalize(utils.vector(anchor_xyz, neighbor1_xyz))
                v2 = anchor_xyz + utils.normalize(utils.vector(anchor_xyz, neighbor2_xyz))

                r = anchor_xyz + utils.normalize(utils.vector(v1, v2))
                p = utils.atom_to_move(anchor_xyz, [v1, v2])
                angles = [-np.radians(60), np.radians(60)]

            # Position of waters are separated by angle of 109 degrees
            # Tetrahedral geometry, there is no reference so water molecules are placed randomly
            # Example: DMSO
            elif n_hbond == 3:
                # Vector between anchor_xyz and the only neighbor atom
                v = utils.vector(anchor_xyz, neighbor1_xyz)
                v = utils.normalize(v)

                # Pick a random vector perpendicular to vector v
                # It will be used as the rotation axis
                r = anchor_xyz + utils.get_perpendicular_vector(v)

                # And we place a pseudo atom (will be the first water molecule)
                p = utils.rotate_point(neighbor1_xyz, anchor_xyz, r, np.radians(109.471))
                # The next rotation axis will be the vector along the neighbor atom and the origin atom
                r = anchor_xyz + utils.normalize(utils.vector(neighbor1_xyz, anchor_xyz))
                angles = [0, -np.radians(120), np.radians(120)]

        # We rotate p to get each vectors if necessary
        for angle in angles:
            vector = p
            if angle != 0.:
                vector = utils.rotate_point(vector, anchor_xyz, r, angle)
            vector = utils.resize_vector(vector, hb_length, anchor_xyz)
            vectors.append(vector)

        vectors = np.array(vectors)

        return vectors

    def match(self, OBMol):
        """ Guess all the hydrogen bonds anchors (donor/acceptor)
        in the molecule based on the hydrogen bond forcefield """
        data = []
        columns = ["atom_i", "vector_xyz", "anchor_type", "anchor_name"]
        # Keep track of all the visited atom
        visited = [False] * (OBMol.NumAtoms() + 1)
        atom_types_available = list(self._atom_types.keys())

        for name in atom_types_available[::-1]:
            atom_type = self._atom_types[name]
            atom_type.ob_smarts.Match(OBMol)
            matches = list(atom_type.ob_smarts.GetMapList())

            if atom_type.hb_type == 1:
                hb_type = "donor"
            elif atom_type.hb_type == 2:
                hb_type = "acceptor"
            else:
                hb_type = None

            for match in matches:
                idx = match[0]

                if hb_type is None and not visited[idx]:
                    visited[idx] = True

                if not visited[idx]:
                    visited[idx] = True

                    try:
                        # Calculate the vectors on the anchor
                        vector_xyzs = self._hb_vectors(OBMol, idx, atom_type.hyb, atom_type.n_water, atom_type.hb_length)
                        for vector_xyz in vector_xyzs:
                            data.append([idx, vector_xyz, hb_type, name])
                    except:
                        print("Warning: Could not determine hydrogen bond vectors on atom %s of type %s." % (idx, name))

        df = pd.DataFrame(data=data, columns=columns)
        df.sort_values(by="atom_i", inplace=True)
        df.reset_index(drop=True, inplace=True)

        return df
