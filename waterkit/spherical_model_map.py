#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# WaterKit
#
# Spherical water map
#

import os
import imp
import multiprocessing as mp

import numpy as np

import utils


def _water_grid_calculation(xyzs, ad_map, atom_types, temperature, water_orientations):
    energy = ad_map.energy_coordinates(xyzs, atom_types[0])
    
    for i, xyz in enumerate(xyzs):
        tmp_energy = energy[i]
        new_orientations = xyz + water_orientations
        
        for j, atom_type in enumerate(atom_types[1:]):
            tmp_energy += ad_map.energy_coordinates(new_orientations[:, j], atom_type)
            
        p = utils.boltzmann_probabilities(tmp_energy, temperature)
        tmp_energy[tmp_energy == np.inf] = 0.0
        energy[i] = np.sum(tmp_energy * p)
    
    return energy


def _run_single(job_id, queue, xyzs, ad_map, atom_types, temperature, water_orientations):
    energy = _water_grid_calculation(xyzs, ad_map, atom_types, temperature, water_orientations)
    queue.put((job_id, energy))
    

class SphericalWaterMap:
    def __init__(self, water_model="tip3p", temperature=300.0, n_jobs=-1):
        self._temperature = temperature
        self._water_model = water_model
        
        if n_jobs == -1:
            self._n_jobs = mp.cpu_count()
        else:
            self._n_jobs = n_jobs
            
        if self._water_model == "tip3p":
            self._atom_types = ["OW", "HW", "HW"]
        elif self._water_model == "tip5p":
            self._atom_types = ["OT", "HT", "HT", "LP", "LP"]
        else:
            print "Error: water model %s unknown." % self._water_model
            return False
            
        """ Load pre-generated water molecules (only hydrogens)
        We will use those to sample the orientation."""
        n_atoms = 2
        usecols = [0, 1, 2, 3, 4, 5]
        if water_model == "tip5p":
            usecols += [6, 7, 8, 9, 10, 11]
            n_atoms += 2
        
        d = imp.find_module("waterkit")[1]
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
            print "Error: map %s already exists." % name
            return False
            
        m_copy = ad_map.copy()
        xyzs = ad_map._kdtree.data
            
        utils.prepare_water_map(m_copy, self._water_model)

        for i, chunk in enumerate(chunks):
            job = mp.Process(target=_run_single, args=(i, queue, xyzs[chunk[0]:chunk[1]+1], m_copy, 
                                                       self._atom_types, self._temperature, 
                                                       self._water_orientations))
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


