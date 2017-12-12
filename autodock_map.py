#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Kits
#
# Class to manage autodock maps
#

import re
import numpy as np

from scipy.interpolate import RegularGridInterpolator

import utils


class Autodock_map():

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
        path = utils.get_folder_path(fld_file)

        self._maps_interpn = {}
        # Read all the affinity maps
        for map_type, map_file in self._maps.items():
            affinity_map = self._read_affinity_map(path + map_file)

            self._maps[map_type] = affinity_map
            self._maps_interpn[map_type] = self._generate_affinity_map_interpn(affinity_map)

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
        return RegularGridInterpolator(self._grid, affinity_map, bounds_error=False, fill_value=np.nan)

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
        arr = np.array([x, y, z]).T

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

    def get_energy(self, xyz, atom_type, method='nearest'):
        """
        Return the energy of each coordinates xyz
        """
        return self._maps_interpn[atom_type](xyz, method=method)

    def get_neighbor_points(self, xyz, radius):
        """
        Return all the coordinates xyz in a certaim radius around a point
        """
        imin, imax = [], []

        idx = self._cartesian_to_index(xyz)

        if radius is not None:
            n = np.int(np.rint(radius / self._spacing))
        else:
            n = 1
            radius = self._spacing

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
        data = np.vstack((X.ravel(), Y.ravel(), Z.ravel())).T

        coordinates = self._index_to_cartesian(data)

        # We can have distance lower than self._spacing
        # because xyz is not necessarily on the grid
        distance = utils.get_euclidean_distance(xyz, coordinates)
        coordinates = coordinates[distance <= radius]

        return coordinates

    def get_volume(self, atom_type, min_energy=0.):
        count = (self._maps[atom_type] >= min_energy).sum()
        volume = count * (self._spacing ** 3)

        return volume

    def to_pdb(self, fname, atom_type, max_energy=None):
        """
        Write the AutoDock map in a PDB file
        """
        idx = np.array(np.where(self._maps[atom_type] <= max_energy)).T

        i = 0
        line = "ATOM  %5d  D   DUM Z%4d    %8.3f%8.3f%8.3f  1.00%6.2f           D\n"

        with open(fname, 'w') as w:
            for j in range(idx.shape[0]):

                v = self._maps[atom_type][idx[j][0], idx[j][1], idx[j][2]]

                if v > 999.99:
                    v = 999.99

                w.write(line % (i, i, self._grid[0][idx[j][0]], self._grid[1][idx[j][1]], self._grid[2][idx[j][2]], v))
                i += 1
