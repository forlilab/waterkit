#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# WaterKit
#
# Class for molecule
#

import os

import numpy as np
import openbabel as ob

import utils


class Molecule():

    def __init__(self, fname):

        # Get name and file extension
        self.name, file_extension = os.path.splitext(fname)

        # Read PDBQT file
        obconv = ob.OBConversion()
        obconv.SetInFormat(file_extension)
        
        self._OBMol = ob.OBMol()
        obconv.ReadFile(self._OBMol, fname)

    def get_coordinates(self, atom_id=None):
        """
        Return coordinates of all atoms or a certain atom
        We do it like this because OBMol.GetCoordinates isn't working
        ... and it never will (https://github.com/openbabel/openbabel/issues/1367)
        """
        if atom_id is not None:
            ob_atom = self._OBMol.GetAtomById(atom_id)
            coordinate = [ob_atom.GetX(), ob_atom.GetY(), ob_atom.GetZ()]
        else:
            coordinate = [[x.GetX(), x.GetY(), x.GetZ()] for x in ob.OBMolAtomIter(self._OBMol)]

        return np.atleast_2d(np.array(coordinate))

    def get_atom_types(self, atom_id=None):
        """
        Return atom types of all atoms or a certain atom
        """
        if atom_id is not None:
            ob_atom = self._OBMol.GetAtomById(atom_id)
            atom_type = [ob_atom.GetType()]
        else:
            atom_type = [x.GetType() for x in ob.OBMolAtomIter(self._OBMol)]

        return atom_type

    def get_atom(self, i):
        """
        Return the OBAtom i
        """
        return self._OBMol.GetAtom(i)

    def get_residue(self, i):
        """
        Return the OBResidue i
        """
        return self._OBMol.GetResidue(i)

    def get_atoms_in_map(self, ad_map=None):
        """
        Returns a list of index of all the atoms in the map
        """
        idx = []

        # If we don't provide an AutoDock Map, we return all the atoms
        if ad_map is None:
            idx = [ob_atom.GetIdx() for ob_atom in ob.OBMolAtomIter(self._OBMol)]
            return idx
        else:
            for ob_atom in ob.OBMolAtomIter(self._OBMol):
                x, y, z = ob_atom.GetX(), ob_atom.GetY(), ob_atom.GetZ()

                if ad_map.is_in_map([x, y, z]):
                    idx.append(ob_atom.GetIdx())

            return idx

    def get_residues_in_map(self, ad_map=None):
        """
        Return a list of index of all the residues in the map
        """
        idx = []

        # If we don't provide an AutoDock Map, we return all the resiudes
        if ad_map is None:
            idx = [ob_residue.GetIdx() for ob_residue in ob.OBResidueIter(self._OBMol)]
            return idx
        else:
            for ob_residue in ob.OBResidueIter(self._OBMol):
                for ob_atom in ob.OBResidueAtomIter(ob_residue):
                    x, y, z = ob_atom.GetX(), ob_atom.GetY(), ob_atom.GetZ()

                    # If at least one atom (whatever the type) is in the grid, add the residue
                    if ad_map.is_in_map([x, y, z]):
                        idx.append(ob_residue.GetIdx())
                        break

            return idx
    
    def is_clash(self, xyz, molecule=None, radius=None):
        """
        Check if there is a clash between a coordinate xyz and itself or another molecule
        """
        if molecule is not None:
            atoms = molecule.get_coordinates()
        else:
            atoms = self.get_coordinates()
        
        # Compute all distances between atom and all other atoms
        d = utils.get_euclidean_distance(xyz, atoms)

        # Remove radius
        if radius is not None:
            d -= radius
        
        # Strictly inferior, otherwise it is always False because of itself
        if (d < 0.).any():
            return True
        
        return False

    def _push_atom_to_end(self, lst, atomic_nums):
        """
        Return a list of OBAtom with all the atom type selected at the end
        """
        if not isinstance(atomic_nums, (list, tuple)):
            atomic_nums = [atomic_nums]

        for atomic_num in atomic_nums:
            pop_count = 0

            idx = [i for i,x in enumerate(lst) if x.GetAtomicNum() == atomic_num]

            for i in idx:
                lst.append(lst.pop(i - pop_count))
                pop_count += 1

        return lst

    def get_neighbor_atoms(self, start_index=1, depth=1, hydrogen=True):
        """
        Return a nested list of all the neighbor OBAtoms by following the bond connectivity
        https://baoilleach.blogspot.com/2008/02/calculate-circular-fingerprints-with.html
        """
        visited = [False] * (self._OBMol.NumAtoms() + 1)
        neighbors = []
        queue = list([(start_index, 0)])
        atomic_num_to_keep = 1

        if not hydrogen:
            atomic_num_to_keep = 2
        
        while queue:
            i, d = queue.pop(0)
            
            ob_atom = self._OBMol.GetAtomById(np.int(i))
            
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
                        queue.append((a.GetId(), d+1))

        # We push all the hydrogen (= 1) atom to the end
        neighbors = [self._push_atom_to_end(x, 1) for x in neighbors]

        return neighbors
    
    def get_neighbor_atom_coordinates(self, id_atom, depth=1, hydrogen=True):
        """
        Return a nested list of all the coordinates of all the neighbor 
        atoms by following the bond connectivity
        """
        coords = []
        
        atoms = self.get_neighbor_atoms(id_atom, depth, hydrogen)
        
        for level in atoms:
            tmp = [[ob_atom.GetX(), ob_atom.GetY(), ob_atom.GetZ()] for ob_atom in level]
            coords.append(np.array(tmp))
            
        return coords

    def get_available_anchors(self, waterfield, ad_map=None):
        """ Return all the anchors available on a molecule
        based on the water forcefield.
        """
        atom_ids = []
        names = []

        # Get all the atoms in the map
        idx_map = self.get_atoms_in_map(ad_map)
        # Get all the water types from the waterfield
        atom_types = waterfield.get_atom_types()
        # In order to keep track which one was alredy typed or not
        visited = [False] * (self._OBMol.NumAtoms() + 1)

        for name in atom_types.keys()[::-1]:
            atom_type = atom_types[name]
            matches = waterfield.get_matches(name, self)

            for match in matches:
                idx = match[0]

                if atom_type.hb_type == 0 and not visited[idx]:
                    visited[idx] = True

                if idx in idx_map and not visited[idx]:
                    visited[idx] = True

                    atom_ids.append(idx)
                    names.append(name)

        return names, atom_ids

    def to_file(self, fname, fformat):
        """
        Write OBMolecule to a file
        """
        obconv = ob.OBConversion()
        obconv.SetOutFormat(fformat)
        obconv.WriteFile(self._OBMol, fname)
