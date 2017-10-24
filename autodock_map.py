#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Kits
#
# Class to manage autodock maps
#


import numpy as np


class Autodock_map():

    def __init__(self, map_file):

        # Read the map
        self._center, self._spacing, self._npts, self._energy = self._read_map(map_file)

        # Compute min and max coordinates
        # Half of each side
        l = (self._spacing * self._npts) / 2.
        # Minimum and maximum coordinates
        self._xmin, self._ymin, self._zmin = self._center - l
        self._xmax, self._ymax, self._zmax = self._center + l

        # Generate the cartesian grid
        self._grid = self._generate_cartesian()

    def _read_map(self, map_file):
        """
        Take a grid file and extract gridcenter, spacing and npts info
        """
        with open(map_file) as f:
            # Read all the lines directly
            lines = f.readlines()

            # Since the format is known, we can retrieve all the information by the position
            spacing = np.float(lines[3].split(' ')[1])
            npts = np.array(lines[4].split(' ')[1:4], dtype=np.int) + 1
            center = np.array(lines[5].split(' ')[1:4], dtype=np.float)

            # Get the energy for each grid element
            energy = [np.float(line) for line in lines[6:]]
            # Some sorceries happen here --> swap axes x and z
            energy = np.swapaxes(np.reshape(energy, npts), 0, 2)

        return center, spacing, npts, energy

    def _generate_cartesian(self):
        x = np.linspace(self._xmin, self._xmax, self._npts[0])
        y = np.linspace(self._ymin, self._ymax, self._npts[1])
        z = np.linspace(self._zmin, self._zmax, self._npts[2])

        # Column: x, y, z (Normally, numpy array are colum-wise. But it is easier to debug.)
        arr = np.array([x, y, z]).T

        return arr

    def _index_to_cartesian(self, idx):
        """
        Return the cartesian coordinates associated to the index
        """
        # Transform 1D to 2D array
        idx = np.atleast_2d(idx)

        # Get coordinates x, y, z
        x = self._grid[idx[:, 0], 0]
        y = self._grid[idx[:, 1], 1]
        z = self._grid[idx[:, 2], 2]

        # Column: x, y, z
        arr = np.array([x, y, z]).T

        return arr

    def _cartesian_to_index(self, xyz):
        """
        Return the closest index of the cartesian coordinates
        """
        idx = []

        if not isinstance(xyz, np.ndarray):
            xyz = np.array(xyz)

        # Otherwise we can't "broadcast"
        xyz = np.atleast_2d(xyz)

        for i in range(xyz.shape[0]):
            idx.append([np.abs(self._grid[:, 0] - xyz[i, 0]).argmin(),
                        np.abs(self._grid[:, 1] - xyz[i, 1]).argmin(),
                        np.abs(self._grid[:, 2] - xyz[i, 2]).argmin()])

        # We want just to keep the struict number of dim useful
        idx = np.squeeze(np.array(idx))

        return idx

    def _is_in_map(self, xyz):

        x, y, z = xyz

        if (self._xmin<=x<=self._xmax) & (self._ymin<=y<=self._ymax) & (self._zmin<=z<=self._zmax):
            return True
        else:
            return False

    def get_energy(self, xyz):
        i, j, k = self._cartesian_to_index(xyz)
        return self._energy[i, j, k]

    def to_pdb(self, fname, max_energy=None):
        idx = np.array(np.where(self._energy <= max_energy)).T

        i = 0
        line = "ATOM  %5d  D   DUM Z%4d    %8.3f%8.3f%8.3f  1.00  1.00     0.000 D\n"

        with open(fname, 'w') as w:
            for j in range(idx.shape[0]):
                #print idx[j], self.get_energy(idx[j])
                w.write(line % (i, i, self._grid[0][idx[j][0]], self._grid[1][idx[j][1]], 
                                self._grid[2][idx[j][2]]))
                i += 1
