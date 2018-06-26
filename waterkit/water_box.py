#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# WaterKit
#
# Class to manage water box
#

import collections

import numpy as np
import pandas as pd
from scipy import spatial

import utils
from water import Water
from optimize import WaterNetwork


class WaterBox():

    def __init__(self, receptor, ad_map, water_map, waterfield):
        self.molecules = {}
        self.maps = []
        self.df = {}
        self._kdtree = None
        self._water_map = water_map
        self._waterfield = waterfield

        # All the informations are stored into a dict of df
        columns = ['molecule_i', 'atom_i', 'molecule_j', 'atom_j']
        self.df['connections'] = pd.DataFrame(columns=columns)
        columns = ['active', 'shell_id', 'energy', 'cluster_id']
        self.df['shells'] = pd.DataFrame(columns=columns)
        columns = ['molecule_i', 'atom_i']
        self.df['kdtree_relations'] = pd.DataFrame(columns=columns)

        # Add the receptor/map to the waterbox
        self.add_molecules(receptor)
        self.add_map(ad_map)
        # Add informations about the receptor
        data = pd.DataFrame([[True, 0]], columns=['active', 'shell_id'])
        self.add_informations(data, 'shells')

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
            coordinates = molecule.get_coordinates()
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

    def get_molecules_in_shell(self, shell_ids=None, active_only=True):
        """ Get all the molecule in shell """
        if shell_ids is not None:
            if not isinstance(shell_ids, collections.Iterable):
                shell_ids = [shell_ids]
            df = self.df['shells'][self.df['shells']['shell_id'].isin(shell_ids)]
        else:
            df = self.df['shells']

        if active_only:
            df = df[df['active'] == True]

        molecules = [self.molecules[i] for i in df.index.tolist()]

        return molecules

    def add_map(self, map):
        """ Append a map to the existing list of maps """
        return self.maps.append(map)

    def get_map(self, shell_id, copy=False):
        """ Get a map for a particular shell """
        try:
            ad_map = self.maps[shell_id]
            if copy:
                return ad_map.copy()
            else:
                return ad_map
        except:
            print "Error: There is no map %s" % shell_id

    def get_closest_atoms(self, x, radius, exclude=None, active_only=True):
        """ Retrieve indices of the closest atoms around x 
        at a certain radius """
        index = self._kdtree.query_ball_point(x, radius)
        df = self.df['kdtree_relations'].loc[index]

        if exclude is not None:
            if not isinstance(exclude, collections.Iterable):
                exclude = [exclude]
            df = df[-df['molecule_i'].isin(exclude)]

        if active_only:
            index = self.df['shells'][self.df['shells']['active'] == True].index
            df = df[df['molecule_i'].isin(index)]

        return df

    def add_informations(self, data, where):
        """ Append DF to the existing information DF """
        try:
            self.df[where] = self.df[where].append(data)
            self.df[where].reset_index(drop=True, inplace=True)
        except:
            print "Error: Cannot add informations to %s dataframe." % where

    def get_number_of_shells(self):
        """ Get the total number of shells """
        # df['column'].max() faster than np.max(df['column'])
        return self.df['shells']['shell_id'].max()

    def _place_optimal_water(self, molecules, ad_map=None):
        """ Place one or multiple water molecules 
        in the ideal position above an acceptor or donor atom
        """
        waters = []
        data = []

        for i, molecule in enumerate(molecules):
            try:
                molecule.guess_hydrogen_bond_anchors(self._waterfield, ad_map)
            except:
                molecule.guess_hydrogen_bond_anchors(self._waterfield)

            for j, anchor in molecule.hydrogen_bond_anchors.iteritems():
                anchor_xyz = molecule.get_coordinates(j)[0]
                for vector in anchor.vectors:
                    # We store the water and the connection
                    waters.append(Water(vector, 'OW', anchor_xyz, anchor.type))
                    data.append((i, j, len(waters)-1, None))

        # Convert list of tuples into dataframe
        columns = ['molecule_i', 'atom_i', 'molecule_j', 'atom_j']
        connections = pd.DataFrame(data, columns=columns)

        return (waters, connections)

    def _update_map(self, waters, ad_map, water_map, water_orientation=[[0, 0, 1], [1, 0, 0]], choices=None):
        """ Update the maps using the water map based
        on the position of the water molecules
        """
        x_len = np.int(np.floor(water_map._grid[0].shape[0] / 2.) + 5)
        y_len = np.int(np.floor(water_map._grid[1].shape[0] / 2.) + 5)
        z_len = np.int(np.floor(water_map._grid[2].shape[0] / 2.) + 5)

        map_types = list(set(ad_map._maps.keys()) & set(water_map._maps.keys()))

        if choices is not None:
            map_types = list(set(map_types) & set(choices))

        for water in waters:
            o, h1, h2 = water.get_coordinates([0, 1, 2])

            # Create the grid around the protein water molecule
            ix, iy, iz = ad_map._cartesian_to_index(o)

            ix_min = ix - x_len if ix - x_len >= 0 else 0
            ix_max = ix + x_len
            iy_min = iy - y_len if iy - y_len >= 0 else 0
            iy_max = iy + y_len
            iz_min = iz - z_len if iz - z_len >= 0 else 0
            iz_max = iz + z_len

            x = ad_map._grid[0][ix_min:ix_max + 1]
            y = ad_map._grid[1][iy_min:iy_max + 1]
            z = ad_map._grid[2][iz_min:iz_max + 1]

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
                ad_map._maps[map_type][ix_min:ix_max + 1, iy_min:iy_max + 1, iz_min:iz_max + 1] += energy

        # Update interpolator
        for map_type in map_types:
            ad_map._maps_interpn[map_type] = ad_map._generate_affinity_map_interpn(ad_map._maps[map_type])

    def build_next_shell(self):
        """ Build the next hydration shell """
        shell_id = self.get_number_of_shells()
        molecules = self.get_molecules_in_shell(shell_id, active_only=True)
        ad_map = self.get_map(shell_id, copy=True)

        waters, connections = self._place_optimal_water(molecules, ad_map)

        n = WaterNetwork(self)
        waters, connections, info = n.optimize_shell(waters, connections)

        if len(waters):
            # And add all the waters
            self.add_molecules(waters, connections)
            # Add informations about the new shell
            self.add_informations(info, 'shells')
            # Update the last map OW
            self._update_map(waters, ad_map, self._water_map, choices=['OW'])
            self.add_map(ad_map)

            return True
        else:
            return False
