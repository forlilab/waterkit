#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# WaterKit
#
# Class to manage the rotatable bond typer
#

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import


import re
from collections import namedtuple
from collections import OrderedDict

import numpy as np
import openbabel as ob
import pandas as pd


class RotatableBonds():

    def __init__(self, fname):
        self._rotatable_bonds = OrderedDict()
        field_names = "name reference_atoms rotamers ob_smarts"
        self._Rotatable_bond = namedtuple("rotatable_bond", field_names)

        self._load_param_file(fname)

    def _load_param_file(self, fname):
        """Load the file to create a rotatable bond typer object
        """
        with open(fname) as f:
            lines = f.readlines()

            for line in lines:
                if re.search("^[A-Za-z0-9].*?(\[.*?\]){4}( [0-9]){4} -?[0-9]{1,3}", line):
                    sline = line.split(" ")

                    ob_smarts = ob.OBSmartsPattern()
                    success = ob_smarts.Init(sline[1])

                    if success:
                        name = sline[0]
                        reference_atoms = np.array(sline[2:6]).astype(int)

                        if "Delta" == sline[7]:
                            rotamers = np.arange(0, 360, int(sline[-1])) + int(sline[6])
                        else:
                            rotamers = np.array(sline[6:]).astype(int)

                        rotatable_bond = self._Rotatable_bond(name, reference_atoms, rotamers, ob_smarts)
                        self._rotatable_bonds[name] = rotatable_bond 

    def match(self, OBMol):
        """Atom typing of the rotatable bonds in the molecule

        Args:
            OBMol (OBMol): OBMol object

        Returns:
            DataFrame: contains all the informations about the rotatable bonds.

            The DataFrame contains 8 different columns: atom_i, atom_j,
            atom_i_xyz, atom_j_xyz, atom_k_xyz, atom_l_xyz, rotamers and
            the name of rotatable bond

        """
        data = []
        columns = ["atom_i", "atom_j", "atom_i_xyz", "atom_j_xyz", 
                   "atom_k_xyz", "atom_l_xyz", "rotamers", "name"]
        
        for name in self._rotatable_bonds:
            unique = []
            matches = []
            
            rotatable_bond = self._rotatable_bonds[name]
            rotatable_bond.ob_smarts.Match(OBMol)
            tmp_matches = list(rotatable_bond.ob_smarts.GetMapList())

            """We check if the SMART pattern was not matching twice on
            the same rotatable bonds, like hydroxyl in tyrosine. The
            GetUMapList function does not work on that specific case
            """
            for match in tmp_matches:
                if not match[0] in unique:
                    unique.append(match[0])
                    
                    ob_atom = OBMol.GetAtom(int(match[0]))
                    atom_i_xyz = np.array([ob_atom.GetX(), ob_atom.GetY(), ob_atom.GetZ()])
                    ob_atom = OBMol.GetAtom(int(match[1]))
                    atom_j_xyz = np.array([ob_atom.GetX(), ob_atom.GetY(), ob_atom.GetZ()])
                    ob_atom = OBMol.GetAtom(int(match[2]))
                    atom_k_xyz = np.array([ob_atom.GetX(), ob_atom.GetY(), ob_atom.GetZ()])
                    ob_atom = OBMol.GetAtom(int(match[3]))
                    atom_l_xyz = np.array([ob_atom.GetX(), ob_atom.GetY(), ob_atom.GetZ()])

                    data.append((match[0], match[1], atom_i_xyz, atom_j_xyz,
                                atom_k_xyz, atom_l_xyz, rotatable_bond.rotamers, name))

        df = pd.DataFrame(data=data, columns=columns)
        df.sort_values(by="atom_i", inplace=True)
        df.reset_index(drop=True, inplace=True)

        return df
