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

import sys

import numpy as np
from gridData import Grid
from scipy.spatial import distance
from scipy.spatial import cKDTree
from scipy.interpolate import RegularGridInterpolator

from .utils import _coordinates_from_grid, _gaussian_weights


def blur_map(grid, radius=1.4, gridsize=0.5, center=None, box_size=None, cutoff=None):
    """Get the smoothed map by summing all the grid points within radius value weighted by Gaussian blurring

    Args:
        grid (Grid): multidimensional grid object (gridData) containing GIST energies (kcal/mol/A**3).
        radius (float): Gaussian blurring radius (Default: 1.4, water molecule radius).
        gridsize (float): size grid (default: 0.5 Angstrom). If 0 does not transform kcal/mol/A**3 to kcal/mol.
        center (ndarray): center of the box (Default: None)
        box_size (ndarray): dimension of the box (Default: None)
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

    # Get the grid coordinates
    if center is not None and box_size is not None:
        center = np.array(center)
        box_size = np.array(box_size)

        assert np.ravel(center).size == 3, "Error: center should contain only (x, y, z)."
        assert np.ravel(box_size).size == 3, "Error: grid size should contain only (a, b, c)."
        assert (box_size > 0).all(), "Error: grid size cannot contain negative numbers."

        x, y, z = center
        sd = box_size / 2.
        x = np.arange(x - sd[0], x + sd[0] + gridsize, gridsize) 
        y = np.arange(y - sd[1], y + sd[1] + gridsize, gridsize)
        z = np.arange(z - sd[2], z + sd[2] + gridsize, gridsize)
        X, Y, Z = np.meshgrid(x, y, z)
        grid_coordinates = np.stack((X.ravel(), Y.ravel(), Z.ravel()), axis=-1)
        new_shape = [x.shape[0], y.shape[0], z.shape[0]]
    else:
        grid_coordinates = _coordinates_from_grid(grid)
        new_shape = [grid.grid.shape[1], grid.grid.shape[0], grid.grid.shape[2]]

    # If the gridsize is set to 0, we keep the map as is
    # Otherwise we transform kcal/mol/A**3 to kcal/mol
    if gridsize == 0:
        grid_tmp = grid
        gridsize = grid.delta
    else:
        volume = gridsize**3
        grid_tmp = grid * volume

    # Create grid interpolator
    # The one from Scipy is better than the one in gridData
    grid_interpn = RegularGridInterpolator(grid_tmp.midpoints, grid_tmp.grid)
    # Initialize KDTree for fast search
    kdtree = cKDTree(grid_coordinates)

    for xyz in grid_coordinates:
        index = kdtree.query_ball_point(xyz, cutoff, p=2)

        weights = _gaussian_weights(xyz, grid_coordinates[index], sigma)
        try:
            weighted_values = grid_interpn(grid_coordinates[index]) * weights
        except ValueError:
            print("Error: grid point outside the original box.")
            sys.exit(1)

        energy.append(np.sum(weighted_values))

    energy = np.array(energy)

    # Swap axis 0 and 1
    new_grid = Grid(np.swapaxes(energy.reshape(new_shape), 0, 1), 
                    origin=grid_coordinates[0], delta=gridsize)

    return new_grid
