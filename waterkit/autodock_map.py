#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# WaterKit
#
# Class to manage autodock maps
#

import os
import re

import numpy as np
from scipy.interpolate import RegularGridInterpolator

import utils


class Map():

    def __init__(self, fld_file):

        # Read the fld_file
        self._center, self._spacing, self._npts, self._maps = self._read_fld(fld_file)

        # Compute min and max coordinates
        # Half of each side
        l = (self._spacing * self._npts) / 2.
        # Minimum and maximum coordinates
        self._xmin, self._ymin, self._zmin = self._center - l
        self._xmax, self._ymax, self._zmax = self._center + l
        # Generate the cartesian grid
        self._grid = self._generate_cartesian()

        # Get the relative folder path from fld_file
        path = os.path.dirname(fld_file)

        self._maps_interpn = {}
        # Read all the affinity maps
        for map_type, map_file in self._maps.items():
            affinity_map = self._read_affinity_map('%s/%s' % (path, map_file))

            self._maps[map_type] = affinity_map
            self._maps_interpn[map_type] = self._generate_affinity_map_interpn(affinity_map)

    def __str__(self):
        info = 'SPACING %s\n' % self._spacing
        info += 'NELEMENTS %s\n' % ' '.join(self._npts.astype(str))
        info += 'CENTER %s\n' % ' '.join(self._center.astype(str))
        info += 'MAPS %s\n' % ' '.join(self._maps.iterkeys())
        return info

    def _read_fld(self, fld_file):
        """
        Read the fld file and extract spacing, npts, center and the name of all the maps
        """
        labels = []
        map_files = []
        npts = []

        with open(fld_file) as f:
            for line in f:

                if re.search('^#SPACING', line):
                    spacing = np.float(line.split(' ')[1])
                elif re.search('^dim', line):
                    npts.append(line.split('=')[1].split('#')[0].strip())
                elif re.search('^#CENTER', line):
                    center = np.array(line.split(' ')[1:4], dtype=np.float)
                elif re.search('^label=', line):
                    labels.append(line.split('=')[1].split('#')[0].split('-')[0].strip())
                elif re.search('^variable', line):
                    map_files.append(line.split(' ')[2].split('=')[1].split('/')[-1])

        npts = np.array(npts, dtype=np.int)
        maps = {label: map_file for label, map_file in zip(labels, map_files)}

        return center, spacing, npts, maps

    def _read_affinity_map(self, map_file):
        """
        Take a grid file and extract gridcenter, spacing and npts info
        """
        with open(map_file) as f:
            # Read all the lines directly
            lines = f.readlines()

            npts = np.array(lines[4].split(' ')[1:4], dtype=np.int) + 1

            # Get the energy for each grid element
            affinity = [np.float(line) for line in lines[6:]]
            # Some sorceries happen here --> swap x and z axes
            affinity = np.swapaxes(np.reshape(affinity, npts[::-1]), 0, 2)

        return affinity

    def _generate_affinity_map_interpn(self, affinity_map):
        """
        Return a interpolate function from the grid and the affinity map.
        This helps to interpolate the energy of coordinates off the grid.
        """
        return RegularGridInterpolator(self._grid, affinity_map, bounds_error=False, fill_value=np.inf)

    def _generate_cartesian(self):
        """
        Generate all the coordinates xyz for each AutoDock map node
        """
        x = np.linspace(self._xmin, self._xmax, self._npts[0])
        y = np.linspace(self._ymin, self._ymax, self._npts[1])
        z = np.linspace(self._zmin, self._zmax, self._npts[2])

        # We use a tuple of numpy arrays and not a complete numpy array 
        # because otherwise the output will be different if the grid is cubic or not
        arr = tuple([x, y, z])

        return arr

    def _index_to_cartesian(self, idx):
        """
        Return the cartesian coordinates associated to the index
        """
        # Transform 1D to 2D array
        idx = np.atleast_2d(idx)

        # Get coordinates x, y, z
        x = self._grid[0][idx[:,0]]
        y = self._grid[1][idx[:,1]]
        z = self._grid[2][idx[:,2]]

        # Column: x, y, z
        arr = np.stack((x, y, z), axis=-1)

        return arr

    def _cartesian_to_index(self, xyz):
        """
        Return the closest index of the cartesian coordinates
        """
        idx = []

        # Otherwise we can't "broadcast" xyz in the grid
        xyz = np.atleast_2d(xyz)

        for i in range(xyz.shape[0]):
            idx.append([np.abs(self._grid[0] - xyz[i, 0]).argmin(),
                        np.abs(self._grid[1] - xyz[i, 1]).argmin(),
                        np.abs(self._grid[2] - xyz[i, 2]).argmin()])

        # We want just to keep the struict number of dim useful
        idx = np.squeeze(np.array(idx))

        return idx

    def is_in_map(self, xyz):
        """
        Check if coordinates xyz are in the AutoDock map
        and return a boolean numpy array
        """
        xyz = np.atleast_2d(xyz)
        x, y, z = xyz[:,0], xyz[:,1], xyz[:,2]

        x_in = np.logical_and(self._xmin <= x, x <= self._xmax)
        y_in = np.logical_and(self._ymin <= y, y <= self._ymax)
        z_in = np.logical_and(self._zmin <= z, z <= self._zmax)
        all_in = np.all((x_in, y_in, z_in), axis=0)
        #all_in = np.logical_and(np.logical_and(x_in, y_in), z_in)

        return all_in

    def get_energy(self, xyz, atom_type, method='linear'):
        """
        Return the energy of each coordinates xyz
        """
        return self._maps_interpn[atom_type](xyz, method=method)

    def get_neighbor_points(self, xyz, min_radius=0, max_radius=5):
        """
        Return all the coordinates xyz in a certaim radius around a point
        """
        imin, imax = [], []

        idx = self._cartesian_to_index(xyz)

        # Number of grid points based on the grid spacing
        n = np.int(np.rint(max_radius / self._spacing))

        # Be sure, we don't go beyond limits of the grid
        for i in range(idx.shape[0]):
            # This is for the min border
            if idx[i] - n > 0:
                imin.append(idx[i] - n)
            else:
                imin.append(0)

            # This is for the max border
            if idx[i] + n < self._npts[i]:
                imax.append(idx[i] + n)
            else:
                imax.append(self._npts[i]-1)

        x = np.arange(imin[0], imax[0] + 1)
        y = np.arange(imin[1], imax[1] + 1)
        z = np.arange(imin[2], imax[2] + 1)

        X, Y, Z = np.meshgrid(x, y, z)
        data = np.stack((X.ravel(), Y.ravel(), Z.ravel()), axis=-1)

        coordinates = self._index_to_cartesian(data)
        distance = utils.get_euclidean_distance(coordinates, xyz)

        # We want coordinates only in between
        selected_coordinates = coordinates[(distance >= min_radius) & (distance <= max_radius)]

        return selected_coordinates

    def combine(self, name, atom_types, how='best', ad_map=None):
        """
        Funtion to combine autodock map together
        """
        same_grid = True
        indices = np.index_exp[:,:,:]

        if ad_map is not None:
            # Check if the grid are the same between the two ad_maps
            # And we do it like this because grid are tuples of numpy array
            same_grid = all([np.array_equal(x, y) for x, y in zip(self._grid, ad_map._grid)])

        # Check if the grid are the same between the two ad_maps
        if ad_map is not None and same_grid is False:

            ix_min, iy_min, iz_min = self._cartesian_to_index([ad_map._xmin, ad_map._ymin, ad_map._zmin])
            ix_max, iy_max, iz_max = self._cartesian_to_index([ad_map._xmax, ad_map._ymax, ad_map._zmax]) + 1

            x = self._grid[0][ix_min:ix_max]
            y = self._grid[1][iy_min:iy_max]
            z = self._grid[2][iz_min:iz_max]

            X, Y, Z = np.meshgrid(x, y, z)
            grid = np.stack((X.ravel(), Y.ravel(), Z.ravel()), axis=-1)

            indices = np.index_exp[ix_min:ix_max, iy_min:iy_max, iz_min:iz_max]

            # If ad_map smaller than self in every dimension
            # If ad_map is bigger than self in any dimension

        # Get the common maps, the requested ones and the actual ones that we have
        selected_types = set(self._maps.keys()) & set(atom_types)

        if ad_map is not None:
            selected_types = set(selected_types) & set(ad_map._maps.keys())

        # Check maps that we cannot process because there are
        # not present in one of the ad_map
        unselected_types = set(selected_types) - set(atom_types)

        if unselected_types:
            print "Those maps can't be combined: %s" % ', '.join(unselected_types)

        print selected_types, unselected_types

        selected_maps = []

        # Select maps
        for selected_type in selected_types:

            selected_maps.append(self._maps[selected_type][indices])

            if ad_map is not None:
                if same_grid:
                    selected_maps.append(ad_map._maps[selected_type])
                else:
                    energy = ad_map.get_energy(grid, selected_type)
                    print energy.shape
                    energy = np.reshape(energy, (len(x), len(y), len(z)))
                    # I have to check why I have to do this!
                    energy = np.swapaxes(energy, 0, 1)

                    selected_maps.append(energy)

        if not self._maps.has_key(name):
            self._maps[name] = np.zeros(self._npts)

        # Combine all the maps
        if how == 'best':
            self._maps[name][indices] = np.nanmin(selected_maps, axis=0)
        elif how == 'add':
            self._maps[name][indices] = np.nansum(selected_maps, axis=0)

        # Update the interpolate energy function
        self._maps_interpn[name] = self._generate_affinity_map_interpn(self._maps[name])

    def transform_grid(self, rotation, translation):
        """
        Transform the grid by applying rotation and translation
        """
        X, Y, Z = np.meshgrid(self._grid[0], self._grid[1], self._grid[2])
        grid = np.stack((X.ravel(), Y.ravel(), Z.ravel()), axis=-1)

        # Apply rotation and translation
        # http://ajcr.net/Basic-guide-to-einsum/ (we don't use einsum, but it's cool!)
        grid = np.dot(grid, rotation) + translation

        # Keep only unique point
        # We don't want to store the whole grid
        new_grid = tuple((np.unique(grid[:,0]), np.unique(grid[:,1]), np.unique(grid[:,2])))

        self.update_grid(new_grid)

    def update_grid(self, new_grid):
        """
        Update the grid and all the information derived (center, map_interpn..)
        """
        # Make sure that the new grid has the same dimension as the old one
        same_shape = all([x.size == y.size for x,y in zip(self._grid, new_grid)])

        if same_shape is True:

            # Update grid
            self._grid = new_grid
            # Update all other informations
            self._center = np.mean(self._grid, axis=1)
            self._xmin, self._xmax = np.min(self._grid[0]), np.max(self._grid[0])
            self._ymin, self._ymax = np.min(self._grid[1]), np.max(self._grid[1])
            self._zmin, self._zmax = np.min(self._grid[2]), np.max(self._grid[2])

            # Update interpolate energy functions with the new grid
            for map_type in self._maps.iterkeys():
                self._maps_interpn[map_type] = self._generate_affinity_map_interpn(self._maps[map_type])

    def get_volume(self, atom_type, min_energy=0.):
        count = (self._maps[atom_type] >= min_energy).sum()
        volume = count * (self._spacing ** 3)

        return volume

    def to_pdb(self, fname, map_type, max_energy=None):
        """
        Write the AutoDock map in a PDB file
        """
        idx = np.array(np.where(self._maps[map_type] <= max_energy)).T

        i = 0
        line = "ATOM  %5d  D   DUM Z%4d    %8.3f%8.3f%8.3f  1.00%6.2f           D\n"

        with open(fname, 'w') as w:
            for j in range(idx.shape[0]):

                v = self._maps[atom_type][idx[j][0], idx[j][1], idx[j][2]]

                if v > 999.99:
                    v = 999.99

                w.write(line % (i, i, self._grid[0][idx[j][0]], self._grid[1][idx[j][1]], self._grid[2][idx[j][2]], v))
                i += 1

    def to_map(self, fname, map_type, grid_parameter_file='grid.gpf', 
               grid_data_file='maps.fld', macromolecule='molecule.pdbqt'):
        """
        Write the AutoDock map in map format
        """
        with open(fname, 'w') as w:

            npts = np.array([n if not n % 2 else n - 1 for n in self._npts])

            # Write header
            w.write('GRID_PARAMETER_FILE %s\n' % grid_parameter_file)
            w.write('GRID_DATA_FILE %s\n' % grid_data_file)
            w.write('MACROMOLECULE %s\n' % macromolecule)
            w.write('SPACING %s\n' % self._spacing)
            w.write('NELEMENTS %s\n' % ' '.join(npts.astype(str)))
            w.write('CENTER %s\n' % ' '.join(self._center.astype(str)))
            # Write grid (swap x and z axis before)
            m = np.swapaxes(self._maps[map_type], 0, 2).flatten()
            w.write('\n'.join(m.astype(str)))
