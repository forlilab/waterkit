#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# WaterKit
#
# Analysis utils
#

import numpy as np
from scipy.spatial import distance


def _gaussian_weights(center, grid_points, sigma):
    d = distance.cdist([center], grid_points, 'euclidean')[0]
    # Gaussian weight filtering based on distance from the center
    weights = np.exp(-(d**2) / (2. * sigma**2))
    return weights


def _coordinates_from_grid(grid):
    x, y, z = grid.midpoints
    X, Y, Z = np.meshgrid(x, y, z)
    grid_coordinates = np.stack((X.ravel(), Y.ravel(), Z.ravel()), axis=-1)
    return grid_coordinates
