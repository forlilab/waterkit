#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# WaterKit
#
# The core of the WaterKit program
#

import gc
import os
import sys
import time
import multiprocessing as mp

from tqdm import tqdm

from .water_box import WaterBox
from . import utils


def _hydrate_single(water_box, n_layer=0, start=0, stop=1, output_dir=".", position=0):
    """Single job waterkit hydration."""
    progress = tqdm(total=stop - start, position=position,
                    desc='job %02d' % (position + 1),
                    bar_format='{l_bar}{bar:50}{r_bar}{bar:-10b}')

    for frame_id in range(start, stop + 1):
        i = 1

        # Create a copy of the waterbox,
        # because otherwise it won't work...
        w_copy = water_box.copy()

        while True:
            # build_next_shell returns True if
            # it was able to put water molecules,
            # otherwise it returns False and we break
            if w_copy.build_next_shell():
                pass
            else:
                break

            """Stop after building n_layer or when the number of
            layers is equal to 26. In the PDB format, the segid
            is encoded with only one caracter, so the maximum number
            of layers is 26, which corresponds to the letter Z."""
            if i == n_layer or i == 26:
                break

            i += 1

        progress.update(1)

        output_filename = os.path.join(output_dir, "water_%06d.pdb" % (frame_id + 1))
        w_copy.to_pdb(output_filename, include_receptor=False)

        # We have to force python to remove the old box
        # otherwise we are going to have a bad time...
        del w_copy
        gc.collect()

    progress.close()


class WaterKit():

    def __init__(self, temperature=300., water_model="tip3p", spherical_water_map=None, n_layer=1, n_frames=1, n_jobs=1):
        """Initialize WaterKit.

        Args:
            water_model (str): Model used for the water molecule, tip3p or tip5p (default: tip3p)
            spherical_water_map (str): autodock map file for the water spherical model (default: None)
            temperature (float): Temperature in Kelvin, only used for Boltzmann sampling (default: 300)
            n_layer (int): Number of hydration layer to add (default: 1)
            n_frames (int): Number of replicas (default: 1)
            n_jobs (int): Number of parallel processes (default: -1)

        """
        self._water_model = water_model
        self._spherical_water_map = spherical_water_map
        self._temperature = temperature
        self._n_layer = n_layer
        self._n_frames = n_frames

        if n_jobs == -1:
            self._n_jobs = mp.cpu_count()
        else:
            self._n_jobs = n_jobs

    def hydrate(self, receptor, ad_map, output_dir="."):
        """Hydrate the molecule with water molecules.

        The receptor is hydrated by adding successive layers
        of water molecules until the box is complety full.

        Args:
            receptor (Molecule): Receptor of the protein
            ad_map (Map): AutoDock map of the receptor
            output_dir (str): output directory for the trajectory

        """
        utils.is_writable(output_dir)

        jobs = []
        chunks = utils.split_list_in_chunks(self._n_frames, self._n_jobs)

        # Initialize a box, might take some times...
        w = WaterBox(receptor, ad_map, self._temperature, self._water_model, self._spherical_water_map)

        # Fire off!!
        for i, chunk in enumerate(chunks):
            job = mp.Process(target=_hydrate_single, args=(w, self._n_layer, chunk[0], chunk[1], output_dir, i))
            job.start()
            jobs.append(job)

        for job in jobs:
            job.join()
