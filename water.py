#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Kits
#
# Class for water
#


import numpy as np
import openbabel as ob

import utils
from molecule import Molecule


class Water(Molecule):

    def __init__(self, oxygen, anchor, hybridization=1):
        self._OBMol = ob.OBMol()
        self.add_atom(oxygen, atomic=8)
        self._anchor = anchor
        self._hybridization = hybridization

    def add_atom(self, xyz, atomic, bond=None):
        a = self._OBMol.NewAtom()
        a.SetAtomicNum(atomic)
        a.SetVector(xyz[0], xyz[1], xyz[2])

        if bond is not None and self._OBMol.NumAtoms() >= 1:
            self._OBMol.AddBond(bond[0], bond[1], bond[2])

    def get_coordinate(self, atom_id=None):
        if atom_id is not None:
            ob_atom = self._.OBMol.GetAtomById(atom_id)
            coordinate = [ob_atom.GetX(), ob_atom.GetY(), ob_atom.GetZ()]
        else:
            coordinate = [[x.GetX(), x.GetY(), x.GetZ()] for x in ob.OBMolAtomIter(self._OBMol)]

        return np.array(coordinate)

    def update_coordinate(xyz, atom_id):
        ob_atom = self._OBMol.GetAtomById(atom_id)
        ob_atom.SetVector(xyz[0], xyz[1], xyz[2])

    def build_hydrogens(self):
        pass

    def rotate_water(self, ref_id=1, angle=0.):
        pass

    def scan(self, ref_id=1):
        pass

def main():
    w = Water(oxygen=[0, 0, 0], anchor=[[0, 0, 1], [0, 0, 2]])
    w.build_hydrogens()
    w.to_file("outputfile.pdb", "pdb")

if __name__ == '__main__':
    main()