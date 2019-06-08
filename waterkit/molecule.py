#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# WaterKit
#
# Class for molecule
#

import copy
import os
import re

import numpy as np
import pandas as pd
import openbabel as ob
from scipy import spatial

import utils


class Molecule():

    def __init__(self, OBMol, waterfield=None):
        """Initialize a Molecule object.

        Args:
            OBMol (OBMol): OpenBabel molecule object
            waterfield (Waterfield): Waterfield object for HB typing

        """
        i = 0
        j = 0
        dtype = [("i", "i4"), ("name", "S4"), ("resname", "S3"), ("resnum", "i4"),
                 ("xyz", "f4", (3)), ("q", "f4"), ("t", "S5")]
        self.atoms = np.zeros(OBMol.NumAtoms(), dtype)
        self.hydrogen_bond_anchors = None
        self.rotatable_bonds = None

        # Remove all implicit hydrogens because OpenBabel
        # is doing chemical perception, and we want to read the
        # molecule as is.
        for x in ob.OBMolAtomIter(OBMol):
            if not x.IsHydrogen() and x.ImplicitHydrogenCount() != 0:
                x.SetImplicitValence(x.GetValence())
                # Really, there is no implicit hydrogen
                x.ForceImplH()

        for r in ob.OBResidueIter(OBMol):
            for a in ob.OBResidueAtomIter(r):
                atom_type = a.GetType()
                xyz = (a.GetX(), a.GetY(), a.GetZ())
                atom = (i+1, atom_type[0], r.GetName(), j+1, xyz, a.GetPartialCharge(), atom_type)
                self.atoms[i] = atom
                
                i += 1
            j += 1

        # Do the typing for hydrogen bonds and disordered hydrogen atoms
        if waterfield is not None:
            self._guess_hydrogen_bond_anchors(OBMol, waterfield)
        self._guess_rotatable_bonds(OBMol)

    @classmethod
    def from_file(cls, fname, waterfield=None):
        """Create Molecule object from a PDB file.

        Args:
            fname (str): molecule filename
            waterfield (Waterfield): Waterfield object for HB typing

        Returns:
            Molecule

        """
        # Get name and file extension
        name, file_extension = os.path.splitext(fname)
        # Read PDB file
        obconv = ob.OBConversion()
        obconv.SetInFormat(file_extension)
        OBMol = ob.OBMol()
        obconv.ReadFile(OBMol, fname)

        m = cls(OBMol, waterfield)

        # OpenBabel do chemical perception to define the type
        # So we override the types with AutoDock atom types
        # from the PDBQT file
        if file_extension == ".pdbqt":
            qs, ts = m._qt_from_pdbqt_file(fname)
            m.atoms['q'] = qs
            m.atoms['t'] = ts

        return m

    def _qt_from_pdbqt_file(self, fname):
        """Get partial charges and atom types from PDBQT file.

        Args:
            fname (str): molecule filename

        Returns:
            list: partial charges
            list: AutoDock atom types

        """
        atom_types = []
        partial_charges = []

        with open(fname) as f:
            lines = f.readlines()
            for line in lines:
                if re.search("^ATOM", line) or re.search("^HETATM", line):
                    atom_types.append(line[77:79].strip())
                    partial_charges.append(np.float(line[70:77].strip()))

        return partial_charges, atom_types

    def is_water(self):
        """Tell if it is a water or not."""
        return False

    def atom(self, i):
        """
        Return the atom i
        """
        return self.atoms[i]

    def coordinates(self, atom_ids=None):
        """
        Return coordinates of all atoms or a certain atom
        We do it like this because OBMol.GetCoordinates is not working
        ... and it never will (https://github.com/openbabel/openbabel/issues/1367)
        """
        if atom_ids is not None and self.atoms.size > 1:
            atoms = self.atoms[atom_ids]["xyz"]
        else:
            atoms = self.atoms["xyz"]

        return np.atleast_2d(atoms).copy()

    def atom_types(self, atom_ids=None):
        """
        Return atom types of all atoms or a certain atom
        """
        if atom_ids is not None and self.atoms.size > 1:
            t = self.atoms[atom_ids]['t']
        else:
            t = self.atoms['t']

        return t.tolist()

    def partial_charges(self, atom_ids=None):
        """Get partial charges."""
        if atom_ids is not None and self.atoms.size > 1:
            q = self.atoms[atom_ids]['q']
        else:
            q =  self.atoms['q']

        return q.copy()

    def atom_informations(self, atom_ids=None):
        """Get atom informations (xyz, q, type)."""
        columns = ["i", "x", "y", "z", "q", "t"]

        if atom_ids is not None:
            atoms = self.atoms[atom_ids][['i', 'xyz', 'q', 't']]
        else:
            atoms = self.atoms[['i', 'xyz', 'q', 't']]

        return atoms.copy()

    def _guess_rotatable_bonds(self, OBMol):
        """ Guess all the rotatable bonds in the molecule
        based the rotatable forcefield """
        columns = ["atom_i", "atom_j", "atom_i_xyz", "atom_j_xyz",
                   "atom_k_xyz", "atom_l_xyz", "name"]
        data = []
        unique = []

        # Find all the hydroxyl
        ob_smarts = ob.OBSmartsPattern()
        success = ob_smarts.Init("[#1][#8;X2;v2;H1][!#1][!#1]")
        ob_smarts.Match(OBMol)
        matches = list(ob_smarts.GetUMapList())

        for match in matches:
            """ We check if the SMART pattern was not matching twice on
            the same rotatable bonds, like hydroxyl in tyrosine. The
            GetUMapList function does not work on that specific case
            """
            if not match[0] in unique:
                atom_i = match[0] - 1
                atom_j = match[1] - 1
                atom_i_xyz = self.coordinates(match[0] - 1)[0]
                atom_j_xyz = self.coordinates(match[1] - 1)[0]
                atom_k_xyz = self.coordinates(match[2] - 1)[0]
                atom_l_xyz = self.coordinates(match[3] - 1)[0]
                data.append([atom_i, atom_j, atom_i_xyz, atom_j_xyz,
                             atom_k_xyz, atom_l_xyz, "hydroxyl"])
                unique.append(match[0])

        self.rotatable_bonds = pd.DataFrame(data=data, columns=columns)
        self.rotatable_bonds.sort_values(by="atom_i", inplace=True)

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

            ob_atom = OBMol.GetAtomById(np.int(i))

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
                    if not visited[a.GetId()] and a.GetAtomicNum() >= atomic_num_to_keep:
                        queue.append((a.GetId(), d + 1))

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

    def _guess_hydrogen_bond_anchors(self, OBMol, waterfield):
        """ Guess all the hydrogen bonds anchors (donor/acceptor)
        in the molecule based on the hydrogen bond forcefield """
        columns = ["atom_i", "vector_xyz", "anchor_type", "anchor_name"]
        data = []

        # Get all the available hb types
        atom_types = waterfield.get_atom_types()
        # Keep track of all the visited atom
        visited = [False] * (OBMol.NumAtoms() + 1)

        for name in atom_types.keys()[::-1]:
            atom_type = atom_types[name]
            matches = waterfield.get_matches(name, OBMol)

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
                        vectors = self._hb_vectors(OBMol, idx - 1, atom_type.hyb, atom_type.n_water, atom_type.hb_length)
                        for vector in vectors:
                            data.append([idx - 1, vector, hb_type, name])
                    except:
                        print "Warning: Could not determine hydrogen bond vectors on atom %s of type %s." % (idx, name)

        self.hydrogen_bond_anchors = pd.DataFrame(data=data, columns=columns)
        self.hydrogen_bond_anchors.sort_values(by="atom_i", inplace=True)

    def _hb_vectors(self, OBMol, idx, hyb, n_hbond, hb_length):
        """Return all the hydrogen bond vectors the atom idx."""
        vectors = []

        # Get origin atom
        anchor_xyz = self.coordinates(idx)[0]
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
            neighbor2_xyz = neighbors_xyz[1][1]

            # Position of water is just above the origin atom
            # We need the 3 direct neighboring atoms (tetrahedral)
            # Exemple: Ammonia
            if n_hbond == 1:
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

    def to_file(self, fname, fformat, options=None, append=False):
        """Write PDBQT file of the water molecule.
    
        OpenBabel is used to write in format other than PDBQT file because
        it is using the default AutoDock atom types (OA, HD, ...). And so
        it is not working for our special water atom types.

        Args:
            fname (str): name of the PDBQT file
            fformat (str): molecule file format
            options (str): OpenBabel wrting options
            append (bool): append to existing PDBQT file (default: False)

        Returns:
            None

        """
        pdbqt_str = "ATOM  %5d %-4s %-3s  %4d    %8.3f%8.3f%8.3f  0.00 0.00     %6.3f %-2s\n"
        output_str = ""

        for atom in self.atoms:
            x, y, z = atom["xyz"]
            output_str += pdbqt_str % (atom["i"], atom["name"], atom["resname"], atom["resnum"],
                                       x, y, z, atom["q"], atom["t"])

        if fformat != "pdbqt":
            OBMol = ob.OBMol()
            obconv = ob.OBConversion()
            obconv.SetInAndOutFormats("pdbqt", fformat)
            obconv.ReadString(OBMol, output_str)

            if options is not None:
                for option in options:
                    obconv.AddOption(option)

            output_str = obconv.WriteString(OBMol)

        if append and os.path.isfile(fname):
            with open(fname, "a") as a:
                a.write(output_str)
        else:
            with open(fname, "w") as w:
                w.write(output_str)

    def export_hb_vectors(self, fname):
        """ Export all the hb vectors to PDB file. """
        pdbqt_str = "ATOM  %5d  %-3s ANC%2s%4d    %8.3f%8.3f%8.3f%6.2f 1.00    %6.3f %2s\n"

        if self.hydrogen_bond_anchors is not None:
            i = 1
            output_str = ""

            for index, anchor in self.hydrogen_bond_anchors.iterrows():
                x, y, z = anchor.vector_xyz
                atom_type = anchor.anchor_type[0].upper()

                output_str += pdbqt_str % (i, atom_type, "A", index, x, y, z, 1, 1, atom_type)
                i += 1

            with open(fname, "w") as w:
                w.write(output_str)
        else:
            print "Error: There is no hydrogen bond anchors."
