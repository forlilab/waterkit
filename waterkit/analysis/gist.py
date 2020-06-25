#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# WaterKit
#
# GIST analysis
#

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import numpy as np
from gridData import Grid
from scipy.spatial import distance
from scipy.spatial import cKDTree

from .utils import _coordinates_from_grid, _gaussian_weights


def blur_map(grid, gridsize=0.5, radius=1.4, cutoff=None):
    """Get the smoothed map by summing all the grid points within radius value weighted by Gaussian blurring

    Args:
        grid (Grid): multidimensional grid object (gridData) containing GIST energies (kcal/mol/A**3).
        gridsize (float): size grid (default: 0.5 Angstrom). If 0 does not transform kcal/mol/A**3 to kcal/mol.
        radius (float): Gaussian blurring radius (Default: 1.4, water molecule radius).
        cutoff (float): filetring cutoff distance. (Default: None, radius + 0.5).

    Returns:
        Grid: Gaussian blurred Grid map (kcal/mol, or kcal/mol/A**3 if gridsize equal to 0)
    """
    assert isinstance(grid, Grid), "Argument passed (%s) is not a Grid object." % type(grid)

    if cutoff is None:
        cutoff = radius + 0.5
    else:
        assert radius < cutoff, "Radius (%f) must be smaller than distance cutoff (%f)" % (radius, cutoff)

    energy = []
    # Divide the radius by 3 in order to have 3 sigma (99.7 %) at the radius value
    sigma = float(radius / 3.)

    grid_coordinates = _coordinates_from_grid(grid)

    # If the gridsize is set to 0, we keep the map as is
    # Otherwise we transform kcal/mol/A**3 to kcal/mol
    if gridsize == 0:
        grid_tmp = grid
    else:
        volume = gridsize**3
        grid_tmp = grid * volume

    # Initialize KDTree for fast search
    kdtree = cKDTree(grid_coordinates)

    for xyz in grid_coordinates:
        index = kdtree.query_ball_point(xyz, cutoff, p=2)
        x, y, z = grid_coordinates[index][:, 0], grid_coordinates[index][:, 1], grid_coordinates[index][:, 2]

        weights = _gaussian_weights(xyz, grid_coordinates[index], sigma)
        weighted_values = grid_tmp.interpolated(x, y, z) * weights

        energy.append(np.sum(weighted_values))

    energy = np.array(energy)

    # Swap axis 0 and 1
    new_shape = [grid.grid.shape[1], grid.grid.shape[0], grid.grid.shape[2]]
    new_grid = Grid(np.swapaxes(energy.reshape(new_shape), 0, 1), 
                    origin=grid.origin, delta=grid.delta)

    return new_grid
