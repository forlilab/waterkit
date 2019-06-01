#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# WaterKit
#
# Class for water
#

import copy
import os
from collections import namedtuple

import numpy as np
import openbabel as ob
import pandas as pd

import utils
from molecule import Molecule


class Water(Molecule):

    def __init__(self, xyz, atom_type='Ow', partial_charge=-0.411, anchor_xyz=None, vector_xyz=None, anchor_type=None):
        self._OBMol = ob.OBMol()
        # Add the oxygen atom
        self.add_atom(xyz, atom_type, partial_charge, atom_num=8)
        # Store all the informations about the anchoring
        if all(v is not None for v in [anchor_xyz, vector_xyz, anchor_type]):
            self.set_anchor(anchor_xyz, vector_xyz, anchor_type)

        self._previous = None
        self.hydrogen_bond_anchors = None
        self.rotatable_bonds = None

    def __copy__(self):
        """Create a copy of Water object."""
        cls = self.__class__
        # Initialize a dumb Water
        result = cls([0, 0, 0])
        result.__dict__.update(self.__dict__)
        # Explicitly create a new copy of the OBMol
        result._OBMol = ob.OBMol(self._OBMol)
        return result

    def __deepcopy__(self, memo):
        """Create a deep copy of Water object."""
        cls = self.__class__
        result = cls([0, 0, 0])
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            if not k == '_OBMol':
                setattr(result, k, copy.deepcopy(v, memo))
        # Explicitly create a new copy of the OBMol
        result._OBMol = ob.OBMol(self._OBMol)
        return result

    @classmethod
    def from_file(cls, fname, atom_type='Ow', partial_charge=-0.411):
        """Create list of Water object from a PDB file."""
        waters = []

        # Get name and file extension
        name, file_extension = os.path.splitext(fname)

        if file_extension == '.pdbqt':
            file_extension = 'pdb'

        # Read PDB file
        obconv = ob.OBConversion()
        obconv.SetInFormat(file_extension)
        OBMol = ob.OBMol()
        obconv.ReadFile(OBMol, fname)

        for x in ob.OBMolAtomIter(OBMol):
            if x.IsOxygen():
                xyz = np.array([x.GetX(), x.GetY(), x.GetZ()])
                waters.append(cls(xyz, atom_type, partial_charge))

        return waters

    def add_atom(self, xyz, atom_type, partial_charge, atom_num=1, bond=None):
        """
        Add an OBAtom to the molecule
        """
        a = self._OBMol.NewAtom()
        a.SetVector(xyz[0], xyz[1], xyz[2])
        """Weird stuffs happen here... I have to Set, Get 
        and re-Set the partial charge. Otherwise it does
        not work correctly. Also weird thing appends here... 
        If I remove a.GetType(), the oxygen type become 
        O3 instead of OA/HO."""
        for _ in range(2):
            a.SetPartialCharge(partial_charge)
            a.SetType(atom_type)
            a.GetPartialCharge()
            a.GetType()

        a.SetAtomicNum(np.int(atom_num))

        if bond is not None and self._OBMol.NumAtoms() >= 1:
            self._OBMol.AddBond(bond[0], bond[1], bond[2])

    def delete_atom(self, atom_id):
        """Delete OBAtom from OBMol using atom id."""
        self._OBMol.DeleteAtom(self.atom(atom_id))

    def set_anchor(self, anchor_xyz, vector_xyz, anchor_type):
        """Add information about the anchoring."""
        # IDEA: This info should be accessible with attributes hba, hbv and type
        anchor_vector = anchor_xyz + utils.normalize(utils.vector(vector_xyz, anchor_xyz))
        self._anchor = np.array([anchor_xyz, anchor_vector])
        self._anchor_type = anchor_type

    def update_coordinates(self, atom_xyz, atom_id):
        """
        Update the coordinates of an OBAtom
        """
        ob_atom = self._OBMol.GetAtomById(atom_id)
        ob_atom.SetVector(atom_xyz[0], atom_xyz[1], atom_xyz[2])

    def is_water(self):
        """Tell if it is a water or not."""
        return True

    def is_spherical(self):
        """Tell if water is spherical or not."""
        if self._OBMol.NumAtoms() == 1:
            return True
        return False

    def is_tip5p(self):
        """Tell if water is TIP5P or not."""
        if self._OBMol.NumAtoms() == 5:
            return True
        return False

    def tip3p(self):
        """Return tip3p Water."""
        if self.is_tip5p():
            w = copy.deepcopy(self)
            w.delete_atom(3)
            w.delete_atom(4)

            return w

    def build_explicit_water(self, model="tip3p"):
        """
        Construct hydrogen atoms (H) and lone-pairs (Lp)
        TIP5P parameters: http://www1.lsbu.ac.uk/water/water_models.html
        """
        if model == "tip3p":
            atom_types = ["Hw", "Hw"]
            partial_charges = [0.4170, 0.4170]
        elif model == "tip5p":
            atom_types = ["Hw", "Hw", "Lp", "Lp"]
            partial_charges = [0.241, 0.241, -0.241, -0.241]
        else:
            return False

        # Order in which we will build H/Lp
        if self._anchor_type == "acceptor":
            d = [0.9572, 0.9572, 0.7, 0.7]
            a = [104.52, 109.47]
        else:
            d = [0.7, 0.7, 0.9572, 0.9572]
            a = [109.47, 104.52]

        coord_oxygen = self.coordinates(0)[0]

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

        i = 2
        for atom, atom_type, partial_charge in zip(atoms, atom_types, partial_charges):
            self.add_atom(atom, atom_type, partial_charge, atom_num=1, bond=(1, i, 1))
            i += 1

        return True

    def guess_hydrogen_bond_anchors(self, waterfield):
        """ Guess all the hydrogen bond anchors in the
        TIP5P water molecule. We don't need the waterfield here. """
        columns = ["atom_i", "vector_xyz", "anchor_type", "anchor_name"]
        data = []

        # Get all the available hb types
        atom_types = waterfield.get_atom_types()

        if self._anchor_type == 'acceptor':
            atom_ids = [3, 4, 5]
            names = ['H_O_004', 'O_L_000', 'O_L_000']
        else:
            atom_ids = [2, 3, 5]
            names = ['H_O_004', 'H_O_004', 'O_L_000']

        for name, idx in zip(names, atom_ids):
            atom_type = atom_types[name]

            if atom_type.hb_type == 1:
                hb_type = 'donor'
            elif atom_type.hb_type == 2:
                hb_type = 'acceptor'

            vectors = self._hb_vectors(idx-1, atom_type.hyb, atom_type.n_water, atom_type.hb_length)
            for vector in vectors:
                data.append([idx - 1, vector, hb_type, name])

        self.hydrogen_bond_anchors = pd.DataFrame(data=data, columns=columns)

    def translate(self, vector):
        """Translate the water molecule."""
        water_xyz = self.coordinates() + vector
        for atom_id, coord_xyz in enumerate(water_xyz):
            self.update_coordinates(coord_xyz, atom_id)
        #self._OBMol.Translate(vector)

        # We have also to translate the hydrogen bond vectors if present
        if self.hydrogen_bond_anchors is not None:
            self.hydrogen_bond_anchors['vector_xyz'] += vector

    def rotate(self, angle, ref_id=1):
        """Rotate water molecule.

        The rotation is along the axis Oxygen 
        and a choosen atom (H or Lp)."""
        water_xyz = self.coordinates()

        # Get the rotation between the oxygen and the atom ref
        oxygen_xyz = water_xyz[0]
        ref_xyz = water_xyz[ref_id]
        r = oxygen_xyz + utils.normalize(utils.vector(ref_xyz, oxygen_xyz))

        # Remove the atom ref from the list of atoms we want to rotate
        atom_ids = list(range(1, water_xyz.shape[0]))
        atom_ids.remove(ref_id)

        for atom_id in atom_ids:
            coord_xyz = utils.rotate_point(water_xyz[atom_id], oxygen_xyz, r, np.radians(angle))
            self.update_coordinates(coord_xyz, atom_id)

        # We have also to rotate the hydrogen bond vectors if present
        if self.hydrogen_bond_anchors is not None:
            for index, vector in self.hydrogen_bond_anchors.iterrows():
                vector_xyz = utils.rotate_point(vector['vector_xyz'], oxygen_xyz, r, np.radians(angle))
                self.hydrogen_bond_anchors.at[index, 'vector_xyz'] = vector_xyz
