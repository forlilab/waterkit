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
from molecule import Molecule
from water import Water
from waterfield import Waterfield


class Kits():

    def __init__(self, waterfield, vdw_water=2.8):
        
        self._vdw_water = vdw_water
        self._waterfield = waterfield
    
    def _place_optimal_water(self, molecule, ad_map=None):
        
        angles = []
        waters = []
        
        # Lentgh of hbonds between water oxygen and atoms
        hb_donor_length = 2.8 # if water is donor (distance Woxygen(-Whydrogen) -- Acceptor)
        hb_acceptor_length = 1.8 # if water is acceptor (distance Woxygen -- hydrogen(-Donor))

        # Get all the atoms in the map
        idx_map = molecule.get_atoms_in_map(ad_map)
        # Get all the water types from the waterfield
        atom_types = self._waterfield.get_atom_types()

        # In order to keep track which one was alredy typed or not
        visited = [False] * (molecule._OBMol.NumAtoms() + 1)

        # We will test every patterns. Not very efficient 
        # but otherwise we have to do the contrary, test if the atom 
        # is in the pattern list or not... more complicated
        for i in atom_types.iterkeys():

            n_water = atom_types[i][2]
            ob_smarts = atom_types[i][3]

            ob_smarts.Match(molecule._OBMol)
            matches = list(ob_smarts.GetMapList())

            for match in matches:
                idx = match[0]

                print match

                if idx in idx_map and not visited[idx]:
                    visited[idx] = True



        """
        # Place first class water molecules
        for i in idx:
            
            ob_residue = molecule.get_residue(i)
            
            for ob_atom in ob.OBResidueAtomIter(ob_residue):
                
                type_atom = ob_atom.GetType()

                # Quick-fix 1: The hybridization of the O3 atom in the tyrosine residue is not
                # correctly recognized by Open-babel.
                if type_atom == 'O3':
                    ob_atom.SetHyb(3)

                id_atom = ob_atom.GetId()
                coord_atom = np.array([ob_atom.GetX(), ob_atom.GetY(), ob_atom.GetZ()])
                hyb_atom = ob_atom.GetHyb()
                val_atom = ob_atom.GetValence()
                
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
                    
                    if hyb_atom == 2:
                        
                        # It means probably that we have a backbone oxygen
                        if coord_neighbor_atoms[1].ndim == 1:
                            coord_atom1 = coord_neighbor_atoms[1]
                            coord_atom2 = np.atleast_2d(coord_neighbor_atoms[2])[0]

                            p = coord_atom + utils.normalize(utils.vector(coord_atom1, coord_atom))
                            r = utils.rotation_axis(coord_atom1, coord_atom, coord_atom2, origin=coord_atom)
                            angles = [-np.radians(60), np.radians(60)]

                        # It means probably that we have a planar nitrogen
                        elif coord_neighbor_atoms[1].ndim == 2:
                            coord_atom1 = coord_neighbor_atoms[1][0]
                            coord_atom2 = coord_neighbor_atoms[1][1]

                            p = utils.atom_to_move(coord_atom, [coord_atom1, coord_atom2])
                            r = utils.rotation_axis(coord_atom, coord_atom1, coord_atom2)
                            angles = [0]

                    if hyb_atom == 3:
                        
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

                        # Create water molecule
                        w = Water(w, anchor=coord_atom, anchor_type=anchor_type)

                        # Check if it's in the map if we provided an AD map
                        if ad_map is None:
                            waters.append(w)
                        else:
                            if ad_map.is_in_map(w):
                                waters.append(w)

                    angles = []
        """
        return waters
    
    def hydrate(self, molecule, ad_map=None):

        # First hydration shell!!
        # Place optimal water molecules everywhere
        waters = self._place_optimal_water(molecule, ad_map)
        # If a AD_map is provide, we optimize water placement
        # and keep only the favorable ones (energy < 0)
        if ad_map is not None:
            # Optimization
            for water in waters:
                water.optimize(ad_map, radius=3.2, angle=140)

            # Energy filter
            waters = [water for water in waters if water.get_energy(ad_map) < 0]

        # Build TIP5P model for each water molecule
        for water in waters:
            water.build_tip5p()

        # Second hydration shell!!
        # Last hydration shell!!
        # ???
        # PROFIT!
        
        return waters
      

def cmd_lineparser():
    parser = argparse.ArgumentParser(description='kits')

    parser.add_argument("-p", "--pdbqt", dest="pdbqt_file", required=True,
                        action="store", help="molecule file")
    parser.add_argument("-m", "--map", dest="map_file", default=None,
                        action="store", help="autodock map file")
    parser.add_argument("-w", "--water", dest="waterfield_file", required=True,
                        action="store", help="waterfield file")

    return parser.parse_args()

def main():

    args = cmd_lineparser()

    pdbqt_file = args.pdbqt_file
    map_file = args.map_file
    waterfield_file = args.waterfield_file

    # Read PDBQT file
    molecule = Molecule(pdbqt_file)
    # Read AutoDock grid map
    ad_map = Autodock_map(map_file)
    # Read the Waterfield file
    waterfield = Waterfield(waterfield_file)

    # Go kits!!
    k = Kits(waterfield)
    waters = k.hydrate(molecule, ad_map)

    utils.write_water('waters.pdb', waters)

if __name__ == '__main__':
    main()