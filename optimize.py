#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Kits
#
# Quick, fast and dirty water optimizer
#


import numpy as np

import utils


np.set_printoptions(precision=2)


class Optimize():

    def __init__(self, radius=1, angle=0, ignore=0):
        self._radius = radius
        self._angle = angle
        self._ignore = ignore

    def _get_closest_minima(self, ad_map, xyz):
        
        imin, imax = [], []

        i, j, k = ad_map._cartesian_to_index(xyz)

        if self._radius is not None:
            n = np.int(np.rint(self._radius / ad_map._spacing))
        else:
            n = 1

        # Be sure, we don't go beyond limits of the grid
        for x, y in zip([i, j, k], [0, 1, 2]):

            # This is for the min border
            if x - n > 0:
                imin.append(x - n)
            else:
                imin.append(0)

            # This is for the max border
            if x + n <= ad_map._npts[y]:
                imax.append(x + n + 1)
            else:
                imax.append(ad_map._npts[y])

        # Select only a part of the grid
        t = ad_map._energy[imin[0]:imax[0], imin[1]:imax[1], imin[2]:imax[2]]

        # Get indices of points with a energy lower than the value of ignore
        idx = np.array((t < self._ignore).nonzero()).T
        # Get coordinates of all these indices
        coordinates = ad_map._index_to_cartesian(np.array(imin) + idx)

        # Computes distance
        distance = utils.euclidean_distance(xyz, coordinates)
        # Keep only coordinates within the radius
        coordinates = coordinates[distance <= self._radius]

        # Compute angle
        #angle = utils.get_angle(atom_coord1, atom_coord2, coordinates)
        # Keep only coordinates within the angle
        #coordinates = coordinates[angle <= self._angle]

        # Get the index of the minima
        #l = np.unravel_index(t.argmin(), t.shape)

        # Get the index in the grid, its xyz coordinates and its energy
        #idx_min = np.array(imin) + l
        xyz_min = 0#ad_map._index_to_cartesian(idx_min)
        xyz_min_energy = 0#ad_map.get_energy(xyz_min)

        return xyz_min, xyz_min_energy

    def run(self, ad_map, waters):

        new_waters = []
        
        for water in waters:
            # Get oxygen coordinate
            coordinate = water.coordinates(atom_id=0)
            # Get energy of the current position
            energy = ad_map.get_energy(coordinate)

            # Get the closest minima
            new_coordinate, new_energy = self._get_closest_minima(ad_map, coordinate)

            if new_energy < energy:
                energy = new_energy
                coordinate = new_coordinate

            if energy < self._ignore:
                water.update_coordinate(coordinate, atom_id=0)
                new_waters.append(water)

        # Keep only equivalent old water molecules
        # This should be removed at the end, debug functionality
        #coordinates = np.delete(coordinates, tmp, axis=0)

        return new_waters