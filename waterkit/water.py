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

    def __init__(self, xyz, atom_type="W", partial_charge=0., anchor_xyz=None, vector_xyz=None, anchor_type=None):
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
        self.hydrogen_bonds = None
        self.rotatable_bonds = None
        self._anchor = None
        self._anchor_type = anchor_type

        # Add the oxygen atom
        self._add_atom(xyz, atom_type, partial_charge)
        # Store all the informations about the anchoring
        if all(v is not None for v in [anchor_xyz, vector_xyz]):
            self._set_anchor(anchor_xyz, vector_xyz)

    @classmethod
    def from_file(cls, fname, atom_type="W", partial_charge=0.):
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
        """Add an atom to the molecule."""
        dtype = [("i", "i4"), ("name", "S4"), ("resname", "S3"), ("resnum", "i4"),
                 ("xyz", "f4", (3)), ("q", "f4"), ("t", "S5")]
        resname = "HOH"
        resnum = 1

        if self.atoms is None:
            num_atoms = 0
        else:
            num_atoms = np.max(self.atoms['i'])

        new_atom = (num_atoms + 1, atom_type[0], resname, resnum, xyz, partial_charge, atom_type)
        new_atom = np.array(new_atom, dtype)

        if self.atoms is not None:
            self.atoms = np.hstack((self.atoms, new_atom))
        else:
            self.atoms = new_atom

    def _delete_atoms(self, atom_ids):
        """Delete atoms from the water. ids are atom ids and not array indices.
        Only the hydrogens and lone pairs are deleted, never the oxygen atom."""
        if not isinstance(atom_ids, np.ndarray):
            atom_ids = np.array(atom_ids)

        # Make sure, we never delete the oxygen atom (1)
        atom_ids = np.delete(atom_ids, np.where(atom_ids == 1))

        if self.atoms.size > 1 and atom_ids.size > 0:
            # atoms_ids - 1, because the array is 0-based
            self.atoms = np.delete(self.atoms, atom_ids - 1)
            # From 1 to num_atom + 1, because atom ids are 1-based
            self.atoms["i"] = np.arange(1, self.atoms.shape[0] + 1)
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
            # Atom ids are 1-based, and LP corresponds to atom 4 and 5
            w._delete_atoms([4, 5])
            # hbond df is 0-based, and so LP corresponds to index 2 and 3
            w.hydrogen_bonds.drop([2, 3], inplace=True)

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
        it as a donor water molecule with the first hydrogen atom
        pointing to the a random direction.
        """
        if self._anchor_type is None:
            self._anchor_type = "acceptor"
        if self._anchor is None:
            self._anchor = [np.random.rand(3), None]

        if water_model == "tip3p":
            distances = [0.9572, 0.9572, 1.0]
            angles = [104.52, 127.74]
        elif water_model == "tip5p":
            distances = [0.9572, 0.9572, 0.7, 0.7]
            angles = [104.52, 109.47]

        # If donor, we started by building the lone-pairs first
        if self._anchor_type == "donor":
            distances.reverse()
            angles.reverse()

        oxygen_xyz = self.coordinates(1)[0]

        # For both TIP3P and TIP5P
        # Vector between O and the Acceptor/Donor atom
        v = utils.vector(oxygen_xyz, self._anchor[0])
        v = utils.normalize(v)
        # Compute a vector perpendicular to v
        p = oxygen_xyz + utils.get_perpendicular_vector(v)
        # H/Lp between O and Acceptor/Donor atom
        a1_xyz = oxygen_xyz + (distances[0] * v)
        # Build the second H/Lp using the perpendicular vector p
        a2_xyz = utils.rotate_point(a1_xyz, oxygen_xyz, p, np.radians(angles[0]))
        a2_xyz = utils.resize_vector(a2_xyz, distances[1], oxygen_xyz)

        if water_model == "tip3p" and self._anchor_type == "donor":
            # Build the second H/Lp using the perpendicular vector p
            a3_xyz = utils.rotate_point(a2_xyz, oxygen_xyz, p, np.radians(angles[1]))
            a3_xyz = utils.resize_vector(a3_xyz, distances[2], oxygen_xyz)
        elif water_model == "tip5p":
            # ... and rotate it to build the last H/Lp
            v = utils.atom_to_move(oxygen_xyz, [a1_xyz, a2_xyz])
            r = oxygen_xyz + utils.normalize(utils.vector(a1_xyz, a2_xyz))
            a3_xyz = utils.rotate_point(v, oxygen_xyz, r, np.radians(angles[1] / 2.))
            a3_xyz = utils.resize_vector(a3_xyz, distances[3], oxygen_xyz)
            a4_xyz = utils.rotate_point(v, oxygen_xyz, r, -np.radians(angles[1] / 2.))
            a4_xyz = utils.resize_vector(a4_xyz, distances[3], oxygen_xyz)

        """ Only now we do all the modifications to the 
        atoms. We never know, we might have an error 
        before while calculating positions of the new atoms.
        So no need to revert to the previous state if 
        something wrong is happening.
        """

        # Change the type and partial charges of the oxygen atom
        # And remove any existing atoms (except oxygen of course)
        if self.is_spherical():
            self.atoms["q"] = partial_charges[0]
            self.atoms["t"] = atom_types[0]
        if not self.is_spherical():
            self.atoms[0]["q"] = partial_charges[0]
            self.atoms[0]["t"] = atom_types[0]
            self._delete_atoms(range(2, self.atoms.size + 1))

        # Select the right atom to add and their order
        if water_model == "tip3p":
            if self._anchor_type == "acceptor":
                atoms = [a1_xyz, a2_xyz]
            else:
                atoms = [a2_xyz, a3_xyz]
        elif water_model == "tip5p":
            # Order them: H, H, Lp, Lp, we want hydrogen atoms first
            if self._anchor_type == "acceptor":
                atoms = [a1_xyz, a2_xyz, a3_xyz, a4_xyz]
            else:
                atoms = [a3_xyz, a4_xyz, a1_xyz, a2_xyz]

        # ... and add the new ones
        for atom, atom_type, partial_charge in zip(atoms, atom_types[1:], partial_charges[1:]):
            self._add_atom(atom, atom_type, partial_charge)

        # Do the HB typing at the end
        self._guess_hydrogen_bonds()

        return True

    def _guess_hydrogen_bonds(self):
        """Guess all the hydrogen bond anchors in the
        TIP5P water molecule. We do not need the waterfield here. """
        hb_length = 2.8
        angle_lp = 109.47
        columns = ["atom_i", "vector_xyz", "anchor_type", "anchor_name"]
        data = []

        oxygen_xyz = self.coordinates(1)[0]

        if self.is_tip3p():
            h1_xyz = self.atoms[1]["xyz"]
            h2_xyz = self.atoms[2]["xyz"]

            # For the lone-pairs, we use the same recipe as for the TIP5P water
            v = utils.atom_to_move(oxygen_xyz, [h1_xyz, h2_xyz])
            r = oxygen_xyz + utils.normalize(utils.vector(h1_xyz, h2_xyz))
            lp1_xyz = utils.rotate_point(v, oxygen_xyz, r, np.radians(angle_lp / 2.))
            lp1_xyz = utils.resize_vector(lp1_xyz, hb_length, oxygen_xyz)
            lp2_xyz = utils.rotate_point(v, oxygen_xyz, r, -np.radians(angle_lp / 2.))
            lp2_xyz = utils.resize_vector(lp2_xyz, hb_length, oxygen_xyz)

            data.append((1, lp1_xyz, "acceptor", "O_L_000"))
            data.append((1, lp2_xyz, "acceptor", "O_L_000"))

        for i, atom in enumerate(self.atoms[1:]):
            if atom["name"] == "H":
                hb_type = "donor"
                hb_name = "H_O_004"
            else:
                hb_type = "acceptor"
                hb_name = "O_L_000"

            vector_xyz = utils.resize_vector(atom["xyz"], hb_length, oxygen_xyz)
            data.append((i + 1, vector_xyz, hb_type, hb_name))

        self.hydrogen_bonds = pd.DataFrame(data=data, columns=columns)

    def translate(self, vector):
        """Translate the water molecule.
        
        Args:
            vector (array_like): 3d vector

        Returns:
            None

        """
        water_xyz = self.coordinates() + vector
        for atom_id, coord_xyz in enumerate(water_xyz):
            # +1, because atom ids are 1-based
            self.update_coordinates(coord_xyz, atom_id + 1)

        # We have also to translate the hydrogen bond vectors if present
        if self.hydrogen_bonds is not None:
            self.hydrogen_bonds["vector_xyz"] += vector

    def rotate_around_axis(self, axis, angle):
        """Rotate water molecule.

        The rotation is along the axis Oxygen 
        and a choosen atom (H or Lp).
        
        Args:
            axis (str): axis name (choices: o, h1, h2, lp1, lp2)
            angle (float): rotation angle in degrees

        Returns:
            None

        """
        if self.is_spherical():
            print "Error: Cannot rotate a spherical water."
            return False

        available_axes = ["o", "h1", "h2"]
        if self.is_tip5p():
            available_axes += ["lp1", "lp2"]

        axis = axis.lower()
        water_xyz = self.coordinates()
        oxygen_xyz = water_xyz[0]
        # From 2 to num_atom + 1, because atom ids are 1-based
        # And the oxygen (atom id = 1) is not moving of course...
        atom_ids = list(range(2, water_xyz.shape[0] + 1))

        try:
            ref_atom_id = available_axes.index(axis) + 1
        except ValueError:
            error_str = "Error: Axis %s not recognized. " % axis
            error_str += "Availvable axes for this water molecule: %s" % ' '.join(available_axes)
            print error_str
            return False

        # If we want to rotate around the oxygen axis, the rotation axis
        # will be the vector between the oxygen and the position in
        # between the two hydrogen bonds
        if ref_atom_id == 1:
            ref_xyz = utils.atom_to_move(oxygen_xyz, water_xyz[[1,2]])
        else:
            # -1, because array is 0-based
            ref_xyz = water_xyz[ref_atom_id - 1]
            # Of course the reference atom is not moving because
            # it is part of the rotation axis, so we remove it
            # from the list of atom to rotate
            atom_ids.remove(ref_atom_id)
        
        r = oxygen_xyz + utils.normalize(utils.vector(ref_xyz, oxygen_xyz))

        for atom_id in atom_ids:
            # -1, because array is 0-based
            coord_xyz = utils.rotate_point(water_xyz[atom_id - 1], oxygen_xyz, r, np.radians(angle))
            self.update_coordinates(coord_xyz, atom_id)

        # We have also to rotate the hydrogen bond vectors if present
        if self.hydrogen_bonds is not None:
            for index, vector in self.hydrogen_bonds.iterrows():
                vector_xyz = utils.rotate_point(vector["vector_xyz"], oxygen_xyz, r, np.radians(angle))
                self.hydrogen_bonds.at[index, "vector_xyz"] = vector_xyz
