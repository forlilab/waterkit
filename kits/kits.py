#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# WaterKits
#
# The core of the WaterKits program
#


import argparse

import numpy as np
import openbabel as ob

import utils
from autodock_map import Autodock_map
from molecule import Molecule
from water import Water
from waterfield import Waterfield
from optimize import Water_network


class Kits():

    def __init__(self, waterfield, water_map):
        self._water_map = water_map
        self._waterfield = waterfield
    
    def _place_optimal_water(self, molecule, atom_type, idx):
        """Place one or multiple water molecules in the ideal position 
        above an acceptor or donor atom
        """
        waters = []
        angles = []
        hyb = atom_type.hyb
        
        if atom_type.hb_type == 1:
            anchor_type = 'donor'
        else:
            anchor_type = 'acceptor'

        # It is not the same index
        idx -= 1
        # Get origin atom
        coord_atom = molecule.get_coordinates(idx)[0]
        # Get coordinates of all the neihbor atoms
        coord_neighbor_atoms = molecule.get_neighbor_atom_coordinates(idx, depth=2)

        if hyb == 1:
            coord_atom1 = coord_neighbor_atoms[1][0]

            # Position of water is linear
            # And we just need the origin atom and the first neighboring atom
            # Example: H donor
            if atom_type.n_water == 1:
                r = None
                p = coord_atom + utils.vector(coord_atom1, coord_atom)
                angles = [0]

            if atom_type.n_water == 3:
                hyb = 3

        if hyb == 2:
            # Position of water is just above the origin atom
            # We need the 2 direct neighboring atoms of the origin atom
            # Example: Nitrogen
            if atom_type.n_water == 1:
                coord_atom1 = coord_neighbor_atoms[1][0]
                coord_atom2 = coord_neighbor_atoms[1][1]

                r = None
                p = utils.atom_to_move(coord_atom, [coord_atom1, coord_atom2])
                angles = [0]

            # Position of waters are separated by angle of 120 degrees
            # And they are aligned with the neighboring atoms (deep=2) of the origin atom
            # Exemple: Backbone oxygen
            elif n_water == 2:
                coord_atom1 = coord_neighbor_atoms[1][0]
                coord_atom2 = coord_neighbor_atoms[2][0]

                r = utils.rotation_axis(coord_atom1, coord_atom, coord_atom2, origin=coord_atom)
                p = coord_atom1
                angles = [-np.radians(120), np.radians(120)]

            elif atom_type.n_water == 3:
                hyb = 3

        if hyb == 3:
            #print idx, hb_type
            coord_atom1 = coord_neighbor_atoms[1][0]
            coord_atom2 = coord_neighbor_atoms[1][1]

            # Position of water is just above the origin atom
            # We need the 3 direct neighboring atoms (tetrahedral)
            # Exemple: Ammonia
            if atom_type.n_water == 1:
                coord_atom3 = coord_neighbor_atoms[1][2]

                # We have to normalize bonds, otherwise the water molecule is not well placed
                v1 = coord_atom + utils.normalize(utils.vector(coord_atom, coord_atom1))
                v2 = coord_atom + utils.normalize(utils.vector(coord_atom, coord_atom2))
                v3 = coord_atom + utils.normalize(utils.vector(coord_atom, coord_atom3))

                r = None
                p = utils.atom_to_move(coord_atom, [v1, v2, v3])
                angles = [0]

            # Position of waters are separated by angle of 109 degrees
            # Tetrahedral geometry, perpendicular with the neighboring atoms of the origin atom
            # Example: Oxygen of the hydroxyl group
            elif atom_type.n_water == 2:
                v1 = coord_atom + utils.normalize(utils.vector(coord_atom, coord_atom1))
                v2 = coord_atom + utils.normalize(utils.vector(coord_atom, coord_atom2))

                r = coord_atom + utils.normalize(utils.vector(v1, v2))
                p = utils.atom_to_move(coord_atom, [v1, v2])
                angles = [-np.radians(60), np.radians(60)]

            # Position of waters are separated by angle of 109 degrees
            # Tetrahedral geometry, there is no reference so water molecules are placed randomly
            # Example: DMSO
            elif atom_type.n_water == 3:
                coord_atom1 = coord_neighbor_atoms[1][0]

                # Vector between coord_atom and the only neighbor atom
                v = utils.vector(coord_atom, coord_atom1)
                v = utils.normalize(v)

                # Pick a random vector perpendicular to vector v
                # It will be used as the rotation axis
                r = coord_atom + utils.get_perpendicular_vector(v)

                # And we place a pseudo atom (will be the first water molecule)
                p = utils.rotate_atom(coord_atom1, coord_atom, r, np.radians(109.47), atom_type.hb_length)
                # The next rotation axis will be the vector along the neighbor atom and the origin atom 
                r = coord_atom + utils.normalize(utils.vector(coord_atom1, coord_atom))
                angles = [0, -np.radians(120), np.radians(120)]

        # Now we place the water molecule(s)!
        for angle in angles:
            w = utils.rotate_atom(p, coord_atom, r, angle, atom_type.hb_length)
            # Create water molecule
            waters.append(Water(w, anchor=coord_atom, anchor_type=anchor_type))

        return waters
    
    def hydrate(self, molecule, ad_map):
        """Hydrate the molecule by adding successive layers 
        of water molecules until the box is complety full
        """
        waters = []

        # First hydration shell!!

        # Get all the atoms in the map
        idx_map = molecule.get_atoms_in_map(ad_map)


        # Get all the water types from the waterfield
        atom_types = self._waterfield.get_atom_types()
        # In order to keep track which one was alredy typed or not
        visited = [False] * (molecule._OBMol.NumAtoms() + 1)

        # We will test every patterns. Not very efficient 
        # but otherwise we have to do the contrary, test if the atom 
        # is in the pattern list or not... more complicated
        for key in atom_types.keys()[::-1]:
            atom_type = atom_types[key]
            matches = self._waterfield.get_matches(key, molecule)

            for match in matches:
                idx = match[0]

                if atom_type.hb_type == 0 and not visited[idx]:
                    visited[idx] = True

                if idx in idx_map and not visited[idx]:
                    visited[idx] = True

                    try:
                        waters.extend(self._place_optimal_water(molecule, atom_type, idx))
                    except:
                        print 'Error: Couldn\'t put water(s) on %s using %s atom type' % (idx, key)
                        continue

        # We optimize the water placement
        # and keep only the favorable ones (energy < 0)
        n = Water_network(distance=2.8, angle=145, cutoff=0)
        waters = n.optimize(waters, ad_map)

        # Complete map
        Water.complete_map(waters, ad_map, self._water_map)

        # Second to N hydration shell!!

        add_waters = True
        current_waters = waters

        while add_waters:
            tmp_waters = []

            # Second hydration shell!!
            for water in current_waters:
                atom_ids, names = water.get_available_anchors()

                for idx, name in zip(atom_ids, names):
                    tmp_waters.extend(self._place_optimal_water(water, atom_types[name], idx))
 
            # We optimize the water placement
            # and keep only the favorable ones (energy < 0)
            tmp_waters = n.optimize(tmp_waters, ad_map)

            if tmp_waters:
                # Complete map
                Water.complete_map(tmp_waters, ad_map, self._water_map)

                current_waters = tmp_waters
                waters.extend(tmp_waters)
            else:
                add_waters = False
 
         # ???
         # PROFIT!
         
        return waters
      

def cmd_lineparser():
    parser = argparse.ArgumentParser(description='kits')

    parser.add_argument("-p", "--pdbqt", dest="pdbqt_file", required=True,
                        action="store", help="molecule file")
    parser.add_argument("-f", "--waterfield", dest="waterfield_file", required=True,
                         action="store", help="waterfield file")
    parser.add_argument("-w", "--watermap", dest="water_map_file", required=True,
                        action="store", help="water autodock map file")
    parser.add_argument("-m", "--map", dest="map_file", required=True,
                        action="store", help="autodock map file")
    parser.add_argument("-o", "--output", dest="output_file", default='waters.pdbqt',
                        action="store", help="water molecule file (pdbqt)")
    return parser.parse_args()

def main():
    args = cmd_lineparser()
    pdbqt_file = args.pdbqt_file
    map_file = args.map_file
    waterfield_file = args.waterfield_file
    output_file = args.output_file
    water_map_file = args.water_map_file

    # Read PDBQT file, Waterfield file and AutoDock grid map
    molecule = Molecule(pdbqt_file)
    waterfield = Waterfield(waterfield_file)
    ad_map = Autodock_map(map_file)
    water_map = Autodock_map(water_map_file)

    # Go kits!!
    k = Kits(waterfield, water_map)
    waters = k.hydrate(molecule, ad_map)

    # Write output files
    utils.write_water(output_file, waters)
    ad_map.to_map('HD.map', 'HD')
    ad_map.to_map('Lp.map', 'Lp')
    ad_map.to_map('O.map', 'O')

if __name__ == '__main__':
    main()