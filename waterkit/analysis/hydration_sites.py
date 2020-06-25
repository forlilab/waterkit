#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# WaterKit
#
# Hydration Sites analysis
#

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import numpy as np
from gridData import Grid
from scipy.spatial import distance
from scipy.spatial import cKDTree

from .utils import _coordinates_from_grid, _gaussian_weights


def _hydration_sites(coordinates, values, density=2, min_cutoff=1.4, max_cutoff=2.6):
    # Keep coordinates with a certain density
    tmp_coordinates = coordinates[values >= density]
    tmp_values = values[values >= density]
    mask = np.ones(tmp_values.shape, dtype=np.bool)

    centers = []
    isocontour = []
    labels = []
    i = 1

    while mask.any():
        center_idx = np.argmax(tmp_values)

        d = distance.cdist([tmp_coordinates[center_idx]], tmp_coordinates, 'euclidean')[0]
        close_points = np.where((d <= max_cutoff) & (mask == True))[0]
        extra_close_points = np.where((d <= min_cutoff) & (mask == True))[0]

        # Add center
        centers.extend([tmp_coordinates[center_idx]])
        isocontour.extend([tmp_coordinates[center_idx]])
        labels.extend([i])

        # Remove center from the pool
        mask[center_idx] = False
        tmp_values[center_idx] = -1

        # Add the closest points and remove them from the pool
        if close_points.size > 0:
            isocontour.extend([tmp_coordinates[extra_close_points]])
            labels.extend([i] * extra_close_points.shape[0])

            mask[close_points] = False
            tmp_values[close_points] = -1.

        i += 1

    isocontour = np.vstack(isocontour)
    centers = np.vstack(centers)
    labels = np.array(labels)

    return centers, isocontour, labels


class HydrationSites():

    def __init__(self, gridsize=0.5, water_radius=1.4, min_water_distance=2.6, min_density=2.0):
        self._water_radius = water_radius
        self._min_water_distance = min_water_distance
        self._min_density = min_density
        self._gridsize = gridsize

        self._grid_coordinates = None
        self._hydration_sites = None
        self._isocontour = None
        self._cluster_ids = None

    def find(self, grid):
        """Find all the hydration sites using the density gO map.

        Args:
            gO (str or Grid): filename or Grid object (gridData) of the gO(xygene) density map from GIST

        Returns:
            ndarray: 3D coordinates of all the hydration sites found

        """
        if not isinstance(grid, Grid):
            try:
                grid = Grid(grid)
            except FileNotFoundError:
                print("Error: File %s was not found." % grid)
                return None

        xyz = _coordinates_from_grid(grid)
        values = grid.interpolated(xyz[:, 0], xyz[:, 1], xyz[:, 2])
        self._hydration_sites, self._isocontour, self._cluster_ids = _hydration_sites(xyz, values, self._min_density,
                                                                                      self._water_radius, 
                                                                                      self._min_water_distance)

        self._grid_coordinates = xyz

        return self._hydration_sites

    def hydration_sites_energy(self, grid, gridsize=None, water_radius=None, hydration_sites=None):
        """Get energy of all the hydration sites

        Args:
            grid (Grid): multidimensional grid object (gridData) containing GIST energies (kcal/mol/A**3)
            gridsize (float): size grid (default: 0.5 Angstrom). If 0 does not transform kcal/mol/A**3 to kcal/mol.
            water_radius (float): radius of a water molecule (default: None). If 0 no gaussian filtering.
            hydration_sites (list or ndarray): hydration sites 3D coordinates (default: None)

        Returns:
            ndarray: energy for each hydration sites previously found (in kcal/mol if gridsize > 0)
        """
        assert isinstance(grid, Grid), "Argument passed (%s) is not a Grid object." % type(grid)

        if water_radius is None:
            water_radius = float(self._water_radius)
        if gridsize is None:
            gridsize = self._gridsize

        energy = []
        cutoff = water_radius + 0.5
        # Divide the radius by 3 in order to have 3 sigma (99.7 %) at the radius value
        sigma = water_radius / 3.

        if hydration_sites is None and self._hydration_sites is not None:
            hydration_sites = self._hydration_sites
            grid_coordinates = self._grid_coordinates
        else:
            # If we pass hydration sites coordinates, not sure if it is the same grid
            grid_coordinates = _coordinates_from_grid(grid)

        # If the gridsize is set to 0, we keep the map as is
        # Otherwise we transform kcal/mol/A**3 to kcal/mol
        if gridsize == 0:
            grid_tmp = grid
        else:
            volume = gridsize**3
            grid_tmp = grid * volume

        # Initialize KDTree for faster search
        if water_radius > 0:
            kdtree = cKDTree(grid_coordinates)

        for hydration_site in hydration_sites:
            if water_radius > 0:
                index = kdtree.query_ball_point(hydration_site, cutoff, p=2)
                x, y, z = grid_coordinates[index][:, 0], grid_coordinates[index][:, 1], grid_coordinates[index][:, 2]

                # Gaussian filtering
                weights = _gaussian_weights(hydration_site, grid_coordinates[index], sigma)
                values = grid_tmp.interpolated(x, y, z) * weights
            else:
                x, y, z = hydration_site
                values = grid_tmp.interpolated(x, y, z)

            energy.append(np.sum(values))

        energy = np.array(energy)

        return energy

    def export_to_pdb(self, filename, hydration_sites, values=None):
        """Export hydration sites in a PDB file

        Args:
            filename (str): PDB filename
            hydration_sites (list or ndarray): hydration sites 3D corrdinates
            values (list or array): values associated to each hydration sites (default: None)

        """
        if values is not None:
            assert len(hydration_sites) == len(values), "Number of hydration sites and energy values are not equal."
        else:
            values = np.ones(len(hydration_sites))

        i = 1
        output_str = ''
        pdb_str = 'ATOM  %5d  %-4s%-3s%2s%4d    %8.3f%8.3f%8.3f  1.00%6.2f          %2s\n'

        for hydration_site, value in zip(hydration_sites, values):
            x, y, z = hydration_site
            output_str += pdb_str % (i, 'OW', 'WAT', 'A', i, x, y, z, value, 'O')
            i += 1

        with open(filename, 'w') as w:
            w.write(output_str)
