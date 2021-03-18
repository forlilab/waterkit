#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# WaterKit
#
# Class to manage autogrid (wrapper)
#

import os
import re
from glob import glob

import numpy as np
import openbabel as ob

from .molecule import Molecule
from .autodock_map import Map
from . import utils


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

    def run(self, receptor_file, atom_types, box_center, box_size, spacing=0.375, 
            smooth=0.5, dielectric=-0.1465, clean=False):
        """Execute AutoGrid on receptor file.

        Args:
            receptor_file (str): pathname of the PDBQT receptor file
            atom_types (list): list of the ligand atom types
            box_center (array_like): center of the grid in Angstrom
            box_size (array_like): size of the grid box in Angstrom
            spacing (float): space between grid points (default: 0.375)
            smooth (float): AutoDock energy smoothing (default: 0.5)
            clean (bool): Remove all the map, fld, gpf and glg files, except the PDBQT receptor file (default: False)

        Returns:
            Map: Return a Map instance

        """
        assert len(box_center) == 3, 'The center of the box must be defined by 3d coordinate (x, y, z)'
        assert len(box_size) == 3, 'The size of the box must be defined by 3d coordinates (x, y, z)'

        if not isinstance(atom_types, (list, tuple)):
            atom_types = [atom_types]

        receptor = Molecule.from_file(receptor_file, False, False)
        receptor_types = set(receptor.atom_types())

        receptor_dir, receptor_filename = os.path.split(receptor_file)
        receptor_name = receptor_filename.split(".")[0]

        npts = np.ceil(np.array(box_size) / spacing)
        # The number of voxels must be even
        npts[npts % 2 == 1] += 1

        gpf_file = os.path.join(receptor_dir, "%s.gpf" % receptor_name)
        glg_file = os.path.join(receptor_dir, "%s.glg" % receptor_name)
        fld_file = os.path.join(receptor_dir, "%s_maps.fld" % receptor_name)
        xyz_file = os.path.join(receptor_dir, "%s_maps.xyz" % receptor_name)
        map_files = [os.path.join(receptor_dir, "%s.%s.map" % (receptor_name, t)) for t in atom_types]
        e_file = os.path.join(receptor_dir, "%s.e.map" % receptor_name)
        d_file = os.path.join(receptor_dir, "%s.d.map" % receptor_name)

        ag_str = "npts %d %d %d\n" % (npts[0], npts[1], npts[2])
        ag_str += "parameter_file %s\n" % self._param_file
        ag_str += "gridfld %s\n" % fld_file
        ag_str += "spacing %.3f\n" % spacing
        ag_str += "receptor_types " + " ".join(receptor_types) + "\n"
        ag_str += "ligand_types " + " ".join(atom_types) + "\n"
        ag_str += "receptor %s\n" % receptor_file
        ag_str += "gridcenter %.3f %.3f %.3f\n" % (box_center[0], box_center[1], box_center[2])
        ag_str += "smooth %.3f\n" % smooth
        for map_file in map_files:
            ag_str += "map %s\n" % map_file
        ag_str += "elecmap %s\n" % e_file
        ag_str += "dsolvmap %s\n" % d_file
        ag_str += "dielectric %.3f\n" % dielectric
        if self._nbp_r_eps is not None:
            for nbp in self._nbp_r_eps:
                ag_str += "nbp_r_eps %.3f %.3f %d %d %s %s\n" % (nbp[0], nbp[1], nbp[2],
                                                                 nbp[3], nbp[4], nbp[5])

        with open(gpf_file, "w") as w:
            w.write(ag_str)

        cmd_line = "%s -p %s -l %s" % (self._exec_path, gpf_file, glg_file)
        utils.execute_command(cmd_line)

        ad_map = Map.from_fld(os.path.join(receptor_dir, "%s_maps.fld" % receptor_name))

        if clean:
            for map_file in map_files:
                os.remove(map_file)

            os.remove(e_file)
            os.remove(d_file)
            os.remove(fld_file)
            os.remove(xyz_file)
            os.remove(gpf_file)
            os.remove(glg_file)

        return ad_map
