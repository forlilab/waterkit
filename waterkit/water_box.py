#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# WaterKit
#
# Class to manage water box
#

import collections
import copy

import numpy as np
import pandas as pd
from scipy import spatial

import utils
from water import Water
from optimize import WaterOptimizer


class WaterBox():

    def __init__(self, hb_forcefield, ad4_forcefield, water_map):
        self.df = {}
        self._kdtree = None
        self.molecules = {}
        self.map = None

        # Forcefields and WaterMap
        self._ad4_forcefield = ad4_forcefield
        self._hb_forcefield = hb_forcefield
        self._water_map = water_map

        # All the informations are stored into a dict of df
        columns = ['molecule_i', 'atom_i', 'molecule_j', 'atom_j']
        self.df['connections'] = pd.DataFrame(columns=columns)
        columns = ['shell_id', 'active', 'xray']
        self.df['shells'] = pd.DataFrame(columns=columns)
        columns = ['molecule_i', 'atom_i']
        self.df['kdtree_relations'] = pd.DataFrame(columns=columns)
        self.df['profiles'] = pd.DataFrame()

    def copy(self):
        """Return deepcopy of WaterBox."""
        return copy.deepcopy(self)

    def add_receptor(self, receptor, ad_map):
        """Add the receptor and the corresponding ad_map to the waterbox."""
        if not 0 in self.molecules:
            # Find all the HBA and disordered atoms if necessary
            if receptor.hydrogen_bond_anchors is None:
                receptor.guess_hydrogen_bond_anchors(self._hb_forcefield)

            if receptor.rotatable_bonds is None:
                receptor.guess_rotatable_bonds()

            # Add the receptor/map to the waterbox
            self.add_molecules(receptor)
            self.map = ad_map.copy()
            # Add informations about the receptor
            data = pd.DataFrame([[0, True, None]], columns=['shell_id', 'active', 'xray'])
            self.add_informations(data, 'shells')

            return True
        else:
            # The receptor was already added to the waterbox
            return False

    def add_molecules(self, molecules, connections=None, add_KDTree=True):
        """ Add a new molecule to the waterbox """
        if not isinstance(molecules, collections.Iterable):
            molecules = [molecules]

        try:
            last_key = np.max(self.molecules.keys())
        except:
            # We initliaze at -1, make first molecule at index 0
            last_key = -1

        # Add molecules to the dictionary
        new_keys = range(last_key + 1, len(molecules) + last_key + 1)
        d = {key: molecule for key, molecule in zip(new_keys, molecules)}
        self.molecules.update(d)

        if connections is not None:
            self._add_connections(connections)
        if add_KDTree:
            self._add_molecules_to_kdtree(molecules)

    def _add_connections(self, connections):
        """ Add connections between molecules """
        try:
            last_connections = self.df['connections'].tail(1)
            last_molecule_i = last_connections['molecule_i'].values[0]
            last_molecule_j = last_connections['molecule_j'].values[0]
        except:
            last_molecule_i = -1
            last_molecule_j = 0

        connections['molecule_i'] += last_molecule_i + 1
        connections['molecule_j'] += last_molecule_j + 1
        self.add_informations(connections, 'connections')

    def _add_molecules_to_kdtree(self, molecules):
        """ Build or update the cKDTree of all the atoms in
        the water box for quick nearest-neighbor lookup
        """
        if not isinstance(molecules, collections.Iterable):
            molecules = [molecules]

        try:
            last_kdtree_relations = self.df['kdtree_relations'].tail(1)
            last_molecule_i = last_kdtree_relations['molecule_i'].values[0]
        except:
            # We initliaze at -1, make first molecule at index 0
            last_molecule_i = -1

        data = []
        relations = []

        for index, molecule in enumerate(molecules):
            coordinates = molecule.coordinates()
            mol_i = index + last_molecule_i + 1
            relations.append([[mol_i, i] for i in range(coordinates.shape[0])])
            data.append(coordinates)

        # Update the KDTree relation database
        columns = ['molecule_i', 'atom_i']
        relations = np.vstack(relations)
        relations = pd.DataFrame(relations, columns=columns)
        self.add_informations(relations, 'kdtree_relations')

        # Update the KDTree
        data = np.vstack(data)
        try:
            data = np.concatenate((self._kdtree.data, data))
        except:
            pass
        self._kdtree = spatial.cKDTree(data)

    def add_informations(self, data, where):
        """ Append DF to the existing information DF """
        try:
            self.df[where] = self.df[where].append(data, sort=False)
            self.df[where].reset_index(drop=True, inplace=True)
        except:
            print "Error: Cannot add informations to %s dataframe." % where

    def update_informations_in_shell(self, data, shell_id, key):
        """Update shell information."""
        index = self.df['shells']['shell_id'] == shell_id
        self.df['shells'].loc[index, key] = data

    def molecules_in_shell(self, shell_ids=None, active_only=True, xray_only=False):
        """ Get all the molecule in shell """
        if shell_ids is not None:
            if not isinstance(shell_ids, collections.Iterable):
                shell_ids = [shell_ids]
            df = self.df['shells'][self.df['shells']['shell_id'].isin(shell_ids)]
        else:
            df = self.df['shells']

        if active_only:
            df = df[df['active'] == True]

        if xray_only:
            df = df[df['xray'] == True]

        molecules = [self.molecules[i] for i in df.index.tolist()]

        return molecules

    def closest_atoms(self, xyz, radius, exclude=None, active_only=True):
        """ Retrieve indices of the closest atoms around x 
        at a certain radius """
        index = self._kdtree.query_ball_point(xyz, radius)
        df = self.df['kdtree_relations'].loc[index]

        if exclude is not None:
            if not isinstance(exclude, collections.Iterable):
                exclude = [exclude]
            df = df[-df['molecule_i'].isin(exclude)]

        if active_only:
            index = self.df['shells'][self.df['shells']['active'] == True].index
            df = df[df['molecule_i'].isin(index)]

        return df

    def atom_informations(self, df):
        """Get atom informations (xyz, q, type)."""
        data = []
        se = df.groupby('molecule_i')['atom_i'].apply(list)

        for molecule_i, atom_ids in se.iteritems():
            data.append(self.molecules[molecule_i].atom_informations(atom_ids))

        df = pd.concat(data, ignore_index=True)

        return df

    def molecule_informations_in_shell(self, shell_id):
        """Get information of shell."""
        df = self.df['shells']
        # Return a copy to avoid a SettingWithCopyWarning flag
        return df.loc[df['shell_id'] == shell_id].copy()

    def number_of_shells(self, ignore_xray=False):
        """Total number of shells in the WaterBox."""
        shells = self.df['shells']
        
        if ignore_xray:
            shells = shells.loc[shells['xray'] != True]

        # df['column'].max() faster than np.max(df['column'])
        return shells['shell_id'].max()

    def closest_hydrogen_bond_anchor(self, xyz, radius, exclude=None, active_only=True):
        """Find the closest hydrogen bond anchors."""
        best_hba = None
        best_hbv_id = None
        best_hbv_distance = 999.

        df = self.closest_atoms(xyz, radius, exclude, active_only)

        for index, row in df.iterrows():
            try:
                hba = self.molecules[row['molecule_i']].hydrogen_bond_anchors[row['atom_i']]
                hba_xyz = self.molecules[row['molecule_i']].coordinates(row['atom_i'])

                hba_distance = utils.get_euclidean_distance(xyz, hba_xyz)[0]
                hbv_distances = utils.get_euclidean_distance(xyz, hba.vectors)
                hbv_min_distance = np.min(hbv_distances)
                hbv_min_id = np.argmin(hbv_distances)

                # We add 1 A to interpolate the distance to the heavy atom
                if hba.type == 'donor':
                    hba_distance += 1.

                # Select the closest HBV and make sure that the heavy atom is close enough
                if hbv_min_distance < best_hbv_distance and hba_distance <= radius:
                    best_hba = hba
                    best_hbv_id = hbv_min_id
                    best_hbv_distance = hbv_min_distance
            except KeyError:
                continue

        return best_hba, best_hbv_id

    def place_optimal_waters(self, molecules):
        """ Place one or multiple water molecules 
        in the ideal position above an acceptor or donor atom
        """
        waters = []
        data = []

        atom_type = 'OW'
        partial_charge = -0.411

        for i, molecule in enumerate(molecules):
            if molecule.hydrogen_bond_anchors is None:
                try:
                    molecule.guess_hydrogen_bond_anchors(self._hb_forcefield, self.map)
                except:
                    molecule.guess_hydrogen_bond_anchors(self._hb_forcefield)

            for j, hba in molecule.hydrogen_bond_anchors.iteritems():
                anchor_xyz = molecule.coordinates(j)[0]
                for vector_xyz in hba.vectors:
                    # We store the water and the connection
                    waters.append(Water(vector_xyz, atom_type, partial_charge, anchor_xyz, vector_xyz, hba.type))
                    data.append((i, j, len(waters) - 1, None))

        # Convert list of tuples into dataframe
        columns = ['molecule_i', 'atom_i', 'molecule_j', 'atom_j']
        connections = pd.DataFrame(data, columns=columns)

        return (waters, connections)

    def _update_map(self, waters, water_map, water_orientation=[[0, 0, 1], [1, 0, 0]], choices=None):
        """ Update the maps using the water map based
        on the position of the water molecules
        """
        x_len = np.int(np.floor(water_map._grid[0].shape[0] / 2.) + 5)
        y_len = np.int(np.floor(water_map._grid[1].shape[0] / 2.) + 5)
        z_len = np.int(np.floor(water_map._grid[2].shape[0] / 2.) + 5)

        map_types = list(set(self.map._maps.keys()) & set(water_map._maps.keys()))

        if choices is not None:
            map_types = list(set(map_types) & set(choices))

        for water in waters:
            o, h1, h2 = water.coordinates([0, 1, 2])

            # Create the grid around the protein water molecule
            ix, iy, iz = self.map._cartesian_to_index(o)

            ix_min = ix - x_len if ix - x_len >= 0 else 0
            ix_max = ix + x_len
            iy_min = iy - y_len if iy - y_len >= 0 else 0
            iy_max = iy + y_len
            iz_min = iz - z_len if iz - z_len >= 0 else 0
            iz_max = iz + z_len

            x = self.map._grid[0][ix_min:ix_max + 1]
            y = self.map._grid[1][iy_min:iy_max + 1]
            z = self.map._grid[2][iz_min:iz_max + 1]

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
                energy = water_map.energy_coordinates(grid, map_type)
                # Replace inf by zero, otherwise we cannot add water energy to the grid
                energy[energy == np.inf] = 0.

                # Reshape and swap x and y axis, right? Easy.
                # Thank you Diogo Santos Martins!!
                energy = np.reshape(energy, (y.shape[0], x.shape[0], z.shape[0]))
                energy = np.swapaxes(energy, 0, 1)

                # Add it to the existing grid
                self.map._maps[map_type][ix_min:ix_max + 1, iy_min:iy_max + 1, iz_min:iz_max + 1] += energy

        # Update interpolator
        for map_type in map_types:
            self.map._maps_interpn[map_type] = self.map._generate_affinity_map_interpn(self.map._maps[map_type])

    def build_next_shell(self, how='best'):
        """Build the next hydration shell."""
        shell_id = self.number_of_shells(ignore_xray=True)
        molecules = self.molecules_in_shell(shell_id)
        n = WaterOptimizer(self, how)

        # Test if we have all the material to continue
        assert len(molecules) > 0, "There is molecule(s) in the shell %s" % shell_id

        if shell_id == 0:
            opt_disordered = True
        else:
            opt_disordered = False

        waters, connections = self.place_optimal_waters(molecules)
        waters, df = n.optimize(waters, connections, opt_disordered=opt_disordered)

        if len(waters):
            # And add all the waters
            self.add_molecules(waters, df['connections'])

            # Tag as non-Xray waters (for the clustering)
            df['shells']['xray'] = False
            df['shells']['active'] = False

            # Add informations about the new shell
            for key in df.keys():
                self.add_informations(df[key], key)

            # Select water molecules and update shell informations
            n.activate_molecules_in_shell(shell_id + 1)

            if self._water_map is not None:
                # Get only active waters and update the last map OW
                active_waters = self.molecules_in_shell(shell_id + 1)
                self._update_map(active_waters, self._water_map, choices=['OW'])

            return True
        else:
            return False

    def add_crystallographic_waters(self, waters, how='best'):
        """Add crystallographic waters to the waterbox."""
        i = 0
        connections = []
        waters_kept = []
        n = WaterOptimizer(self, how, energy_cutoff=np.inf)

        if 0 in self.molecules:
            for water in waters:
                xyz = water.coordinates()[0]

                # We attach the xray water to the closest HBA acceptor/donor
                if self.map.is_in_map(xyz):
                    hba, hbv_id = self.closest_hydrogen_bond_anchor(xyz, radius=3.5)

                    # Test if there a HBA around, otherwise it is not a first shell water
                    if hba is not None:
                        hba_xyz = self.molecules[0].coordinates(hba.id)[0]
                        water.set_anchor(hba_xyz, hba.vectors[hbv_id], hba.type)

                        connections.append((0, hba.id, i, None))
                        waters_kept.append(water)

                        i += 1
                    else:
                        # For the moment, ignore n-shell water
                        continue

            # We just optimize the orientation
            waters_kept, df = n.optimize(waters_kept, opt_position=False, opt_disordered=False)

            # Guess HBA to be able to put waters on them
            [water.guess_hydrogen_bond_anchors(self._hb_forcefield) for water in waters_kept]

            # Add X-Ray water molecule to the water box
            columns = ['molecule_i', 'atom_i', 'molecule_j', 'atom_j']
            connections = pd.DataFrame(connections, columns=columns)
            self.add_molecules(waters_kept, connections)

            # Tag all water molecules as X-Ray (for the clustering)
            df['shells']['xray'] = True
            df['shells']['active'] = False

            # Add informations about the new shell
            for key in df.keys():
                self.add_informations(df[key], key)

            return True
        else:
            # The receptor wasn't initialized yet.
            return False
