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
        
        waters = []
        coord_waters = []
        
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

                    print "Donor   ", coord_neighbor_atoms
                    print ""
                    
                    # Get the coordinates of the water oxygen atom
                    v = utils.vector(coord_neighbor_atoms[1], coord_atom)
                    n = utils.normalized_vector(v)
                    w = coord_atom + (n * hb_acceptor_length)
                    
                    coord_waters.append(w)
                        
                elif ob_atom.IsHbondAcceptor():
                    
                    coord_neighbor_atoms = molecule.get_neighbor_atom_coordinates(id_atom, 2)
                    
                    """
                    The hybridization of the O3 atom in the tyrosine residue is not
                    correctly recognized by Open-babel. So I added a quick-fix in the
                    if condition.
                    """

                    print "Acceptor", coord_neighbor_atoms
                    print ""
                    
                    if hyb_atom == 2 and not type_atom == 'O3':
                        
                        if coord_neighbor_atoms[1].ndim == 1:
                            # It means probably that we have a backbone oxygen
                            coord_atom1 = coord_neighbor_atoms[2][0]
                            coord_atom2 = coord_neighbor_atoms[2][1]
                            angles = [np.radians(60), -np.radians(60)]

                        elif coord_neighbor_atoms[1].ndim == 2:
                            # It means probably that we have a planar nitrogen
                            atom1 = coord_neighbor_atoms[1][0]
                            atom2 = coord_neighbor_atoms[1][1]
                            angles = [0]

                        v1 = utils.vector(coord_atom1, coord_atom)
                        v2 = utils.vector(coord_atom2, coord_atom)
                        v3 = utils.normalized_vector(v2 + v1)
                        n = utils.normalized_vector(np.cross(v1, v2))

                        tmp_waters = []
                        
                        for angle in angles:
                            w = utils.rotate_atom(coord_atom+v3, coord_atom, coord_atom+n, angle, hb_donor_length)
                            tmp_waters.append(w)
                        
                        coord_waters.extend(tmp_waters)
                    
                    elif hyb_atom == 3 or type_atom == 'O3':
                        
                        if coord_neighbor_atoms[1].ndim == 1:
                            # It means that we have probably a tetrahedral nitrogen
                            p = np.mean(coord_neighbor_atoms[2], axis=0)
                            v = utils.vector(coord_atom, p)
                            n = -1. * utils.normalized_vector(v)
                            w = coord_atom + (n * hb_donor_length)
                            
                            coord_waters.append(w)
                        
                        elif coord_neighbor_atoms[1].ndim == 2:
                            # It means that we have probably an hydroxyl group
                            coord_atom1 = coord_neighbor_atoms[1][0]
                            coord_atom2 = coord_neighbor_atoms[1][1]
                            angle = np.radians(120)
                            
                            w1 = utils.rotate_atom(coord_atom2, coord_atom, coord_atom1, angle, hb_donor_length)
                            w2 = utils.rotate_atom(coord_atom2, coord_atom, coord_atom1, -angle, hb_donor_length)
                            
                            coord_waters.extend([w1, w2])

                elif ob_atom.IsAromatic():
                    pass
        
        # Check if water molecules are in the map, otherwise don't keep them
        coord_waters = [x for x in coord_waters if ad_map._is_in_map(x)]

        # Create all the water molecules
        #for coord_water in coord_waters:
        #    water = Water(coord_water)
        
        return np.array(coord_waters)
    
    def hydrate(self, molecule, ad_map):

        # First hydration shell!!
        # Place optimal water molecules everywhere
        waters = self._place_optimal_water(molecule, ad_map)
        # Optimize them
        #o = Optimize(radius=3., angle=110)
        #waters = o.run(ad_map, waters)

        #ad_map.to_pdb('test.pdb', max_energy=-0.8)

        # Check if there is a clash with other atoms, except donor atom
        # For the moment, the radius is zero. So we keep everything.
        """
        for coord_water, i in enumerate(coord_waters):
            if not self._is_clash(coord_water, all_atoms, exclude=id_atom, radius=0):
                coord_waters.append(coord_water)
                # And maybe optimize
        """

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
    old = k.hydrate(molecule, ad_map)

    utils.write_pdb('old_waters.pdb', old)
    #utils.write_pdb('opt_waters.pdb', new)
    #utils.write_pdb_opt_water('waters.pdb', old, new)

if __name__ == '__main__':
    main()