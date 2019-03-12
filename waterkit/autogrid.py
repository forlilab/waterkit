#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# WaterKit
#
# Class to manage autogrid (wrapper)
#

from __future__ import print_function

import os
import subprocess
from glob import glob

from molecule import Molecule
from autodock_map import Map


def execute_command(cmd_line):
    args = cmd_line.split()
    p = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    output, errors = p.communicate()
    return output, errors


class AutoGrid():

    def __init__(self, exec_path='autogrid4', param_file='AD4_parameters.dat'):
        self._exec_path = exec_path
        self._param_file = param_file

    def run(self, receptor_file, atom_types, center=(0., 0., 0.),
            npts=(32, 32, 32), spacing=0.375, smooth=0.5, clean=False):

        if not isinstance(atom_types, (list, tuple)):
            atom_types = [atom_types]

        receptor = Molecule.from_file(receptor_file)
        receptor_types = set(receptor.atom_types())
        receptor_name = receptor_file.split('/')[-1].split('.')[0]

        ag_str = 'npts %d %d %d\n' % (npts[0], npts[1], npts[2])
        ag_str += 'parameter_file %s\n' % self._param_file
        ag_str += 'gridfld %s_maps.fld\n' % receptor_name
        ag_str += 'spacing %.3f\n' % spacing
        ag_str += 'receptor_types ' + ' '.join(receptor_types) + '\n'
        ag_str += 'ligand_types ' + ' '.join(atom_types) + '\n'
        ag_str += 'gridcenter %.3f %.3f %.3f\n' % (center[0], center[1], center[2])
        ag_str += 'smooth %.3f\n' % smooth
        for atom_type in atom_types:
            ag_str += 'map %s.%s.map\n' % (receptor_name, atom_type)
        ag_str += 'elecmap %s.e.map\n' % receptor_name
        ag_str += 'dsolvmap %s.d.map\n' % receptor_name
        ag_str += 'dielectric -0.1465\n'

        gpf_file = '%s.gpf' % receptor_name
        glg_file = '%s.glg' % receptor_name
        
        with open(gpf_file, 'w') as w:
            w.write(ag_str)

        cmd_line = '%s -p %s -l %s' % (self._exec_path, gpf_file, glg_file)
        execute_command(cmd_line)

        ad_map = Map.from_fld('%s_maps.fld' % receptor_name)

        if clean:
            map_files = glob('%s*.map' % receptor_name)
            for map_file in map_files:
                os.remove(map_file)

            os.remove('%s_maps.fld' % receptor_name)
            os.remove('%s_maps.xyz' % receptor_name)
            os.remove('%s.gpf' % receptor_name)
            os.remove('%s.glg' % receptor_name)

        return ad_map
