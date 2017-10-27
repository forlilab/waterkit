#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Kits
#
# The core of the kits program
#


import argparse

import numpy as np
import openbabel as ob

import utils
from autodock_map import Autodock_map
from optimize import Optimize
from molecule import Molecule
from water import Water


class Kits():

    def __init__(self, vdw_water=2.8):
        
        self._vdw_water = vdw_water
    
    def _place_optimal_water(self, molecule, ad_map):
        
        angles = []
        waters = []
        
        # Lentgh of hbonds between water oxygen and atoms
        hb_donor_length = 2.8 # if water is donor (distance Woxygen(-Whydrogen) -- Acceptor)
        hb_acceptor_length = 1.8 # if water is acceptor (distance Woxygen -- hydrogen(-Donor))

        # Get all the residues in the map
        idx = molecule.get_residues_in_map(ad_map)

        # Place first class water molecules
        for i in idx:
            
            ob_residue = molecule.get_residue(i)
            
            for ob_atom in ob.OBResidueAtomIter(ob_residue):
                
                id_atom = ob_atom.GetId()
                coord_atom = np.array([ob_atom.GetX(), ob_atom.GetY(), ob_atom.GetZ()])
                hyb_atom = ob_atom.GetHyb()
                type_atom = ob_atom.GetType()
                
                if ob_atom.IsHbondDonorH():
                    # Get the coordinates of the neighbor atom
                    coord_neighbor_atoms = molecule.get_neighbor_atom_coordinates(id_atom)
                    # Set HBond length
                    hb_length = hb_acceptor_length
                    anchor_type = 'donor'

                    # The angle will be zero, so no need for a rotation axis
                    r = None
                    p = utils.atom_to_move(coord_atom, coord_neighbor_atoms[1])
                    angles = [0]
                
                if ob_atom.IsHbondAcceptor():
                    # Get the coordinates of the neighbor atom
                    coord_neighbor_atoms = molecule.get_neighbor_atom_coordinates(id_atom, 2)
                    # Set HBond length
                    hb_length = hb_donor_length
                    anchor_type = 'acceptor'
                    
                    #The hybridization of the O3 atom in the tyrosine residue is not
                    #correctly recognized by Open-babel. So I added a quick-fix in the
                    #if condition.
                    
                    if hyb_atom == 2 and not type_atom == 'O3':
                        
                        # It means probably that we have a backbone oxygen
                        if coord_neighbor_atoms[1].ndim == 1:
                            coord_atom1 = coord_neighbor_atoms[2][0]
                            coord_atom2 = coord_neighbor_atoms[2][1]
                            angles = [-np.radians(60), np.radians(60)]

                        # It means probably that we have a planar nitrogen
                        elif coord_neighbor_atoms[1].ndim == 2:
                            coord_atom1 = coord_neighbor_atoms[1][0]
                            coord_atom2 = coord_neighbor_atoms[1][1]
                            angles = [0]

                        r = utils.rotation_axis(coord_atom, coord_atom1, coord_atom2)
                        p = utils.atom_to_move(coord_atom, [coord_atom1, coord_atom2])

                    if hyb_atom == 3 or type_atom == 'O3':
                        
                        # It means that we have probably a tetrahedral nitrogen
                        if coord_neighbor_atoms[1].ndim == 1:
                            # The angle will be zero, so no need for a rotation axis
                            r = None
                            p = utils.atom_to_move(coord_atom, coord_neighbor_atoms[2])
                            angles = [0]
                        
                        # It means that we have probably an hydroxyl group
                        elif coord_neighbor_atoms[1].ndim == 2:
                            # Carbon atom (in -OH context) correspond to the rotation axis
                            r = coord_neighbor_atoms[1][0]
                            # The atom to move correspond to the hydrogen already in position (-OH)
                            p = coord_neighbor_atoms[1][1]
                            angles = [-np.radians(120), np.radians(120)]

                if angles:
                    # Now we place the water molecule(s)!
                    for angle in angles:
                        w = utils.rotate_atom(p, coord_atom, r, angle, hb_length)

                        # ... and check if it's in the map
                        if ad_map.is_in_map(w):
                            waters.append(Water(w, anchor=coord_atom, anchor_type=anchor_type))

                    angles = []
        
        return waters
    
    def hydrate(self, molecule, ad_map):

        # First hydration shell!!
        # Place optimal water molecules everywhere
        waters = self._place_optimal_water(molecule, ad_map)
        # Optimize them
        new_waters = []

        for water in waters:
            water.optimize(ad_map, radius=3.2, angle=145.)

            if water.get_energy(ad_map) < 0.:
                new_waters.append(water)

        # Second hydration shell!!
        # Last hydration shell!!
        # ???
        # PROFIT!
        
        return new_waters
      

def cmd_lineparser():
    parser = argparse.ArgumentParser(description='kits')

    parser.add_argument("-p", "--pdbqt", dest="pdbqt_file", required=True,
                        action="store", help="molecule file")
    parser.add_argument("-m", "--map", dest="map_file", default=None,
                        action="store", help="autodock map file")

    return parser.parse_args()

def main():

    args = cmd_lineparser()

    pdbqt_file = args.pdbqt_file
    map_file = args.map_file

    # Read PDBQT file
    molecule = Molecule(pdbqt_file)
    # Read AutoDock grid map
    ad_map = Autodock_map(map_file)

    # Go kits!!
    k = Kits()
    waters = k.hydrate(molecule, ad_map)

    utils.write_water_pdb('waters.pdb', waters, previous=True)

if __name__ == '__main__':
    main()