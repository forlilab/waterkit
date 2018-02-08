#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Kits
#
# Class to manage water forcefield
#

import re

import numpy as np
import openbabel as ob


class Waterfield():

    def __init__(self, fname):

        self._atom_types = {}
        self.load_files(fname)

    def load_files(self, fname):
        """
        Load one or more files to create a Forceflied object based on them
        """
        i = 0

        with open(fname) as f:
            lines = f.readlines()

            # ATOM NAME(%10s) TYPE(%2s) STRENGTH(%8.3f) HYB(%2s) #WATER(%2s) RADIUS(%8.3f) SMARTS(%s)

            for line in lines:
                if re.search('^ATOM', line):

                    # Split by space and remove them in the list
                    sline = line.split(' ')
                    sline = filter(None, sline)

                    name = sline[1]
                    hb_type = np.int(sline[2])
                    strength = np.float(sline[3])
                    hyb = np.int(sline[4])
                    n_water = np.int(sline[5])
                    hb_length = np.float(sline[6])
                    smarts =  sline[7]

                    # Initialize the SMARTS pattern and check if is valid
                    ob_smarts = ob.OBSmartsPattern()
                    success = ob_smarts.Init(smarts)

                    # Add atom type if is valid
                    if success:
                        self._atom_types[i] = [name, hb_type, strength, hyb, n_water, hb_length, ob_smarts]
                        i += 1
                    else:
                        print "Warning: SMARTS %s is invalid" % smarts

    def get_atom_types(self, idx=None):
        """
        Return all the atom types or just one based on the idx
        """
        if idx is None:
            return self._atom_types
        else:
            return self._atom_types[idx]