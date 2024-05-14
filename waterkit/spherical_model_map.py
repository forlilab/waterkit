#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# WaterKit
#
# Spherical water map
#

import os
import sys
import multiprocessing as mp

import numpy as np

from . import utils


def _water_grid_calculation(xyzs, ad_map, atom_types, temperature, water_orientations, verbose=False):
    energy = np.zeros(shape=(xyzs.shape[0]))
    """ We want to avoid any edges effects, so we exclude 
    the grid points that are too close from the edges.
    """
    not_close_to_egdes = ~ad_map.is_close_to_edge(xyzs, distance=1.)

    energy[not_close_to_egdes] = ad_map.energy_coordinates(xyzs[not_close_to_egdes], atom_types[0])

    n = 0
    n_total = np.sum(not_close_to_egdes)

    for i in np.where(not_close_to_egdes)[0]:
        tmp_energy = energy[i]
        new_orientations = xyzs[i] + water_orientations
        
        # Get the energy of each orientation
        for j, atom_type in enumerate(atom_types[1:]):
            tmp_energy += ad_map.energy_coordinates(new_orientations[:, j], atom_type)

        # Safeguard, inf values mean that we are outside the box
        tmp_energy = tmp_energy[tmp_energy != np.inf]

        """ If there is at least one favorable (< 0 kcal/mol) orientation
        we compute the boltzmann average energy, otherwise it means that
        we are in the receptor (clashes), and so we just compute the classic
        energy average. 
        """
        if any(tmp_energy < 0):
            p = utils.boltzmann_probabilities(tmp_energy, temperature)
            energy[i] = np.sum(tmp_energy * p)
        else:
            energy[i] = np.mean(tmp_energy)

        if (n % 100 == 0) and verbose:
            sys.stdout.write("\rGrid points calculated:  %5.2f / 100 %%" % (float(n) / n_total * 100.))
            sys.stdout.flush()

        n += 1
    
    return energy


def _run_single(job_id, queue, xyzs, ad_map, atom_types, temperature, water_orientations, verbose=False):
    energy = _water_grid_calculation(xyzs, ad_map, atom_types, temperature, water_orientations, verbose)
    queue.put((job_id, energy))
    

class SphericalWaterMap:
    def __init__(self, water_model="tip3p", temperature=300.0, n_jobs=-1, verbose=False):
        self._temperature = temperature
        self._water_model = water_model
        self._verbose = verbose
        
        if n_jobs == -1:
            self._n_jobs = mp.cpu_count()
        else:
            self._n_jobs = n_jobs
            
        if self._water_model == "tip3p":
            self._atom_types = ["OW", "HW", "HW"]
        elif self._water_model == "tip5p":
            self._atom_types = ["OT", "HT", "HT", "LP", "LP"]
        else:
            print("Error: water model %s unknown." % self._water_model)
            return False
            
        """ Load pre-generated water molecules (only hydrogens)
        We will use those to sample the orientation."""
        n_atoms = 2
        usecols = [0, 1, 2, 3, 4, 5]
        if water_model == "tip5p":
            usecols += [6, 7, 8, 9, 10, 11]
            n_atoms += 2
        
        d = utils.path_module("waterkit")
        w_orientation_file = os.path.join(d, "data/water_orientations.txt")
        water_orientations = np.loadtxt(w_orientation_file, usecols=usecols)
        shape = (water_orientations.shape[0], n_atoms, 3)
        self._water_orientations = water_orientations.reshape(shape)
        
    def run(self, ad_map, name="SW"):
        jobs = []
        results = [[] for i in range(self._n_jobs)]
        chunks = utils.split_list_in_chunks(ad_map.size(), self._n_jobs)
        """Why Manager().Queue() and not Queue() directly?
        Well I do not know, it does not work otherwise.
        Source: https://stackoverflow.com/questions/13649625/multiprocessing-in-python-blocked"""
        queue = mp.Manager().Queue()
            
        if name in ad_map._maps:
            print("Error: map %s already exists." % name)
            return False
            
        m_copy = ad_map.copy()
        xyzs = ad_map._kdtree.data

        for i, chunk in enumerate(chunks):
            job = mp.Process(target=_run_single, args=(i, queue, xyzs[chunk[0]:chunk[1]+1], m_copy, 
                                                       self._atom_types, self._temperature, 
                                                       self._water_orientations, self._verbose))
            job.start()
            jobs.append(job)

        for job in jobs:
            job.join()

        while not queue.empty():
            job_id, result = queue.get()
            # The order must be preserved!
            results[job_id] = result
        results = np.concatenate(results)

        # Replace NaN number by max_energy
        max_energy = np.nanmax(results)
        results[np.isnan(results)] = max_energy
        # Make it a 3D grid again
        new_map = np.swapaxes(results.reshape(ad_map._npts), 0, 1)
            
        ad_map.add_map(name, new_map)

        if self._verbose:
            print("\n")

