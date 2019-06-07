#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# WaterKit
#
# Class to manage water forcefield
#

import re
from collections import namedtuple
from collections import OrderedDict

import numpy as np
import openbabel as ob


class Waterfield():

    def __init__(self, fname):
        # Create ordered dict to conserved the order of atom types
        # which is very important in our particular case for the moment
        self._atom_types = OrderedDict()
        field_names = 'hb_type hb_strength hyb n_water hb_length ob_smarts'
        self._Atom_type = namedtuple('Atom_type', field_names)

        self._load_param_file(fname)

    def _load_param_file(self, fname):
        """Load the file to create a water Forcefield object
        """
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
                    hb_strength = np.float(sline[3])
                    hyb = np.int(sline[4])
                    n_water = np.int(sline[5])
                    hb_length = np.float(sline[6])
                    smarts = sline[7]

                    if smarts == '[]':
                        smarts = None

                    # And add it
                    self.add_atom_type(name, hb_type, hb_strength, hyb, n_water, hb_length, smarts)

    def create_atom_type(self, hb_type, hb_strength, hyb, n_water, hb_length, smarts=None):
        """Create atom type using a namedtuple
        """
        if smarts is not None:
            # Initialize the SMARTS pattern and check if is valid
            ob_smarts = ob.OBSmartsPattern()
            success = ob_smarts.Init(smarts)

            # Add atom type if is valid
            if success:
                atom_type = self._Atom_type(hb_type, hb_strength, hyb, n_water, hb_length, ob_smarts)
            else:
                print "Warning: SMARTS %s is invalid" % smarts
                return None
        else:
            atom_type = self._Atom_type(hb_type, hb_strength, hyb, n_water, hb_length, None)

        return atom_type

    def add_atom_type(self, name, hb_type, hb_strength, hyb, n_water, hb_length, smarts=None):
        """Add an atom type to the atom types library
        """
        atom_type = self.create_atom_type(hb_type, hb_strength, hyb, n_water, hb_length, smarts)

        if atom_type is not None:
            self._atom_types[name] = atom_type

    def get_atom_types(self, name=None):
        """Return all the atom types or just one based on the idx
        """
        if name is None:
            return self._atom_types
        elif name in self._atom_types:
            return self._atom_types[name]
        else:
            return None

    def get_matches(self, name, molecule):
        """Return matches using smarts on the molecule
        """
        if name in self._atom_types:
            atom_type = self._atom_types[name]

            if atom_type.ob_smarts is not None:
                atom_type.ob_smarts.Match(molecule._OBMol)
                matches = list(atom_type.ob_smarts.GetMapList())
            else:
                matches = []
        else:
            matches = []

        return matches
