#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# WaterKit
#
# Class to manage autogrid (wrapper)
#

from __future__ import print_function

import os
import re
from glob import glob

import numpy as np
import openbabel as ob

import utils
from molecule import Molecule
from autodock_map import Map


class AutoGrid():

    def __init__(self, exec_path="autogrid4", param_file="AD4_parameters.dat", gpf_file=None):
        """Initialize AutoGrid.

        Args:
            exec_path (str): pathname of the autogrid executable (default: autogrid4)
            param_file (str): pathname of the AutoDock forcefield (default: AD4_parameters.dat)
            gpf_file (str): gpf file that contains the non-bonded potential (default: None)

        """
        self._exec_path = exec_path
        self._param_file = param_file
        if gpf_file is not None:
            self._nbp_r_eps = self._load_nbp_r_eps_from_gpf(gpf_file)
        else:
            self._nbp_r_eps = None

    def _load_nbp_r_eps_from_gpf(self, gpf_file):
        """Load intnbp_r_eps from dpf file."""
        nbp_r_eps = []

        with open(gpf_file) as f:
            lines = f.readlines()

            for line in lines:
                if re.search("^nbp_r_eps", line):
                    sline = line.split()
                    req = np.float(sline[1])
                    eps = np.float(sline[2])
                    n = np.int(sline[3])
                    m = np.int(sline[4])
                    i = sline[5]
                    j = sline[6]

                    nbp_r_eps.append((req, eps, n, m, i, j))

        return nbp_r_eps

    def run(self, receptor_file, atom_types, center=(0., 0., 0.),
            npts=(32, 32, 32), spacing=0.375, smooth=0.5, dielectric=-0.1465,
            clean=False):
        """Execute AutoGrid on receptor file.

        Args:
            receptor_file (str): pathname of the PDBQT receptor file
            atom_types (list): list of the ligand atom types
            center (array_like): center of the grid (default: (0, 0, 0))
            npts (array_like): size of the grid box (default: (32, 32, 32))
            spacing (float): space between grid points (default: 0.375)
            smooth (float): AutoDock energy smoothing (default: 0.5)
            clean (bool): Remove all the map, fld, gpf and glg files, except the PDBQT receptor file (default: False)

        Returns:
            Map: Return a Map instance

        """
        if not isinstance(atom_types, (list, tuple)):
            atom_types = [atom_types]

        receptor = Molecule.from_file(receptor_file, guess_hydrogen_bonds=False,
                                      guess_disordered_hydrogens=False)
        receptor_types = set(receptor.atom_types())
        receptor_name = receptor_file.split("/")[-1].split(".")[0]

        ag_str = "npts %d %d %d\n" % (npts[0], npts[1], npts[2])
        ag_str += "parameter_file %s\n" % self._param_file
        ag_str += "gridfld %s_maps.fld\n" % receptor_name
        ag_str += "spacing %.3f\n" % spacing
        ag_str += "receptor_types " + " ".join(receptor_types) + "\n"
        ag_str += "ligand_types " + " ".join(atom_types) + "\n"
        ag_str += "receptor %s\n" % receptor_file
        ag_str += "gridcenter %.3f %.3f %.3f\n" % (center[0], center[1], center[2])
        ag_str += "smooth %.3f\n" % smooth
        for atom_type in atom_types:
            ag_str += "map %s.%s.map\n" % (receptor_name, atom_type)
        ag_str += "elecmap %s.e.map\n" % receptor_name
        ag_str += "dsolvmap %s.d.map\n" % receptor_name
        ag_str += "dielectric %.3f\n" % dielectric
        if self._nbp_r_eps is not None:
            for nbp in self._nbp_r_eps:
                ag_str += "nbp_r_eps %.3f %.3f %d %d %s %s\n" % (nbp[0], nbp[1], nbp[2],
                                                                 nbp[3], nbp[4], nbp[5])

        gpf_file = "%s.gpf" % receptor_name
        glg_file = "%s.glg" % receptor_name

        with open(gpf_file, "w") as w:
            w.write(ag_str)

        cmd_line = "%s -p %s -l %s" % (self._exec_path, gpf_file, glg_file)
        utils.execute_command(cmd_line)

        ad_map = Map.from_fld("%s_maps.fld" % receptor_name)

        if clean:
            map_files = glob("%s*.map" % receptor_name)
            for map_file in map_files:
                os.remove(map_file)

            os.remove("%s_maps.fld" % receptor_name)
            os.remove("%s_maps.xyz" % receptor_name)
            os.remove("%s.gpf" % receptor_name)
            os.remove("%s.glg" % receptor_name)

        return ad_map
