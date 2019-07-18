#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# WaterKit
#
# Class for water
#

import copy
import os

import numpy as np
import pandas as pd
import openbabel as ob

import utils
from molecule import Molecule


class Water(Molecule):

    def __init__(self, xyz, atom_type="SW", partial_charge=-0.834, anchor_xyz=None, vector_xyz=None, anchor_type=None):
        """Initialize a Water object.
    
        Args:
            xyz (array_like): 3d coordinates
            atom_type (str): atom types of the spherical water molecule (default: OW)
            partial_charge (float): partial charge of the spherical water molecule (default: -0.834)
            anchor_xyz (array_like): 3d coordinates of the anchor point
            vector_xyz (array_like): 3d coordinates of HB vector
            anchor_type (str): type of anchor point (acceptor or donor)

        """
        self.atoms = None
        self.hydrogen_bond_anchors = None
        self.rotatable_bonds = None
        self._anchor = None
        self._anchor_type = anchor_type

        # Add the oxygen atom
        self._add_atom(xyz, atom_type, partial_charge)
        # Store all the informations about the anchoring
        if all(v is not None for v in [anchor_xyz, vector_xyz]):
            self._set_anchor(anchor_xyz, vector_xyz)

    @classmethod
    def from_file(cls, fname, atom_type="SW", partial_charge=-0.834):
        """Create list of Water objects from a PDB file.
        
        The water molecules created are spherical.

        Args:
            fname (str): molecule filename
            atom_type (str): atom types of the spherical water molecules (default: OW)
            partial_charge (float): partial charge of the spherical water molecules (default: -0.834)

        Returns:
            list: list of Water molecule objects

        """
        waters = []

        # Get name and file extension
        name, file_extension = os.path.splitext(fname)

        if file_extension == ".pdbqt":
            file_extension = "pdb"

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

    def _add_atom(self, xyz, atom_type, partial_charge):
        """Add an OBAtom to the molecule."""
        dtype = [("i", "i4"), ("name", "S4"), ("resname", "S3"), ("resnum", "i4"),
                 ("xyz", "f4", (3)), ("q", "f4"), ("t", "S5")]
        resname = "HOH"
        resnum = 1

        if self.atoms is None:
            num_atoms = -1
        else:
            num_atoms = np.max(self.atoms['i'])

        new_atom = (num_atoms + 1, atom_type[0], resname, resnum, xyz, partial_charge, atom_type)
        new_atom = np.array(new_atom, dtype)

        if self.atoms is not None:
            self.atoms = np.hstack((self.atoms, new_atom))
        else:
            self.atoms = new_atom

    def _delete_atoms(self, atom_ids):
        """Delete OBAtom from OBMol using atom id."""
        if self.atoms.size > 1:
            self.atoms = np.delete(self.atoms, atom_ids)
            self.atoms["i"] = np.arange(0, self.atoms.shape[0])
            return True
        else:
            return False

    def _set_anchor(self, anchor_xyz, vector_xyz):
        """Add information about the anchoring."""
        # IDEA: This info should be accessible with attributes hba, hbv and type
        anchor_vector = anchor_xyz + utils.normalize(utils.vector(vector_xyz, anchor_xyz))
        self._anchor = np.array([anchor_xyz, anchor_vector])

    def is_water(self):
        """Tell if it is a water or not."""
        return True

    def is_spherical(self):
        """Tell if water is spherical or not."""
        if self.atoms.size == 1:
            return True
        return False

    def is_tip3p(self):
        """Tell if water is TIP3P or not."""
        if self.atoms.size == 3:
            return True
        return False

    def is_tip5p(self):
        """Tell if water is TIP5P or not."""
        if self.atoms.size == 5:
            return True
        return False

    def tip3p(self):
        """Return a tip3p version of the Water (deepcopy)."""
        if self.is_tip5p():
            w = copy.deepcopy(self)
            w._delete_atoms([3, 4])
            w.hydrogen_bond_anchors.drop([2, 3], inplace=True)

            return w
            
        return self

    def build_explicit_water(self, water_model="tip3p"):
        """ Construct hydrogen atoms (H) and lone-pairs (Lp)
        TIP3P or TIP5P parameters: http://www1.lsbu.ac.uk/water/water_models.html
        
        Args:
            water_model (str): water model (choice: tip3p or tip5)(default: tip3p)

        Returns:
            bool: True if successfull, False otherwise

        """
        i = 2
        distances = [0.9572, 0.9572, 0.7, 0.7]
        angles = [104.52, 109.47]

        models = {
            "tip3p": {"atom_types": ["OW", "HW", "HW"],
                      "partial_charges": [-0.834, 0.417, 0.417]
                     },
            "tip5p": {"atom_types": ["OT", "HT", "HT", "LP", "LP"],
                      "partial_charges": [0.0, 0.241, 0.241, -0.241, -0.241]
                     }
            }

        if water_model in models:
            atom_types = models[water_model]["atom_types"]
            partial_charges = models[water_model]["partial_charges"]
        else:
            print "Error: water model %s unknown." % water_model
            return False

        """ If no anchor information was defined, we define
        it as a donor water molecule with the hydrogen atom
        pointing to the a random direction.
        """
        if self._anchor_type is None:
            self._anchor_type = "acceptor"
        if self._anchor is None:
            self._anchor = [np.random.rand(3), None]

        # If donor, we started by building the lone-pairs first
        if self._anchor_type == "donor":
            distances.reverse()
            angles.reverse()

        coord_oxygen = self.coordinates(0)[0]

        # Vector between O and the Acceptor/Donor atom
        v = utils.vector(coord_oxygen, self._anchor[0])
        v = utils.normalize(v)
        # Compute a vector perpendicular to v
        p = coord_oxygen + utils.get_perpendicular_vector(v)
        # H/Lp between O and Acceptor/Donor atom
        a1 = coord_oxygen + (distances[0] * v)
        # Build the second H/Lp using the perpendicular vector p
        a2 = utils.rotate_point(a1, coord_oxygen, p, np.radians(angles[0]))
        a2 = utils.resize_vector(a2, distances[1], coord_oxygen)
        # ... and rotate it to build the last H/Lp
        p = utils.atom_to_move(coord_oxygen, [a1, a2])
        r = coord_oxygen + utils.normalize(utils.vector(a1, a2))
        a3 = utils.rotate_point(p, coord_oxygen, r, np.radians(angles[1] / 2.))
        a3 = utils.resize_vector(a3, distances[3], coord_oxygen)
        a4 = utils.rotate_point(p, coord_oxygen, r, -np.radians(angles[1] / 2.))
        a4 = utils.resize_vector(a4, distances[3], coord_oxygen)

        """ Only now we do all the modifications to the 
        atoms. We never know, we might have an error 
        before while calculating positions of the new atoms.
        So no need to revert to the previous state if 
        something is happening.
        """

        # Change the type and partial charges of the oxygen atom
        # And remove any existing atoms (except oxygen of course)
        if self.is_spherical():
            self.atoms["q"] = partial_charges[0]
            self.atoms["t"] = atom_types[0]
        if not self.is_spherical():
            self.atoms[0]["q"] = partial_charges[0]
            self.atoms[0]["t"] = atom_types[0]
            self._delete_atoms(range(1, self.atoms.size))

        # Order them: H, H, Lp, Lp, we want hydrogen atoms first
        if self._anchor_type == "acceptor":
            atoms = [a1, a2, a3, a4]
        else:
            atoms = [a3, a4, a1, a2]

        # ... and add the new ones
        for atom, atom_type, partial_charge in zip(atoms, atom_types[1:], partial_charges[1:]):
            self._add_atom(atom, atom_type, partial_charge)

        # Do the HB typing at the end
        self._guess_hydrogen_bond_anchors()

        return True

    def _guess_hydrogen_bond_anchors(self):
        """Guess all the hydrogen bond anchors in the
        TIP5P water molecule. We do not need the waterfield here. """
        hyb = 1
        n_water = 1
        hb_length = 2.8
        columns = ["atom_i", "vector_xyz", "anchor_type", "anchor_name"]
        data = []

        oxygen_xyz = self.coordinates(0)[0]

        for i, atom in enumerate(self.atoms[1:]):
            if atom["name"] == "H":
                hb_type = "donor"
                hb_name = "H_O_004"
            else:
                hb_type = "acceptor"
                hb_name = "O_L_000"

            vector_xyz = utils.resize_vector(atom["xyz"], hb_length, oxygen_xyz)
            data.append((i + 1, vector_xyz, hb_type, hb_name))

        self.hydrogen_bond_anchors = pd.DataFrame(data=data, columns=columns)

    def update_coordinates(self, xyz, atom_id):
        """Update the coordinates of an atom.

        Args:
            xyz (array_like): 3d coordinates of the new atomic position
            atom_id (int): atom id

        Returns:
            bool: True if successfull, False otherwise

        """
        if atom_id < self.atoms.size:
            if self.atoms.size > 1:
                self.atoms[atom_id]["xyz"] = xyz
            else:
                self.atoms["xyz"] = xyz
            return True
        else:
            return False

    def translate(self, vector):
        """Translate the water molecule.
        
        Args:
            vector (array_like): 3d vector

        Returns:
            None

        """
        water_xyz = self.coordinates() + vector
        for atom_id, coord_xyz in enumerate(water_xyz):
            self.update_coordinates(coord_xyz, atom_id)

        # We have also to translate the hydrogen bond vectors if present
        if self.hydrogen_bond_anchors is not None:
            self.hydrogen_bond_anchors["vector_xyz"] += vector

    def rotate(self, angle, ref_id=1):
        """Rotate water molecule.

        The rotation is along the axis Oxygen 
        and a choosen atom (H or Lp).
        
        Args:
            angle (float): rotation angle
            ref_id (int): atom index for the rotation axis

        Returns:
            None

        """
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
                vector_xyz = utils.rotate_point(vector["vector_xyz"], oxygen_xyz, r, np.radians(angle))
                self.hydrogen_bond_anchors.at[index, "vector_xyz"] = vector_xyz
