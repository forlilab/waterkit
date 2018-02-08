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
    
    def _place_optimal_water(self, molecule, weight, idx, hyb, n_water, hb_type, hb_length):

        waters = []
        angles = []
        
        if hb_type == 1:
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
            if n_water == 1:
                r = None
                p = coord_atom + utils.vector(coord_atom1, coord_atom)
                angles = [0]

            if n_water == 3:
                hyb = 3

        if hyb == 2:

            # Position of water is just above the origin atom
            # We need the 2 direct neighboring atoms of the origin atom
            # Example: Nitrogen
            if n_water == 1:
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

            elif n_water == 3:
                hyb = 3

        if hyb == 3:

            coord_atom1 = coord_neighbor_atoms[1][0]
            coord_atom2 = coord_neighbor_atoms[1][1]

            # Position of water is just above the origin atom
            # We need the 3 direct neighboring atoms (tetrahedral)
            # Exemple: Ammonia
            if n_water == 1:
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
            elif n_water == 2:

                v1 = coord_atom + utils.normalize(utils.vector(coord_atom, coord_atom1))
                v2 = coord_atom + utils.normalize(utils.vector(coord_atom, coord_atom2))

                r = coord_atom + utils.normalize(utils.vector(v1, v2))
                p = utils.atom_to_move(coord_atom, [v1, v2])
                angles = [-np.radians(60), np.radians(60)]

            # Position of waters are separated by angle of 109 degrees
            # Tetrahedral geometry, there is no reference so water molecules are placed randomly
            # Example: DMSO
            elif n_water == 3:

                coord_atom1 = coord_neighbor_atoms[1][0]

                # Vector between coord_atom and the only neighbor atom
                v = utils.vector(coord_atom, coord_atom1)
                v = utils.normalize(v)

                # Pick a random vector perpendicular to vector v
                # It will be used as the rotation axis
                r = coord_atom + utils.get_perpendicular_vector(v)

                # And we place a pseudo atom (will be the first water molecule)
                p = utils.rotate_atom(coord_atom1, coord_atom, r, np.radians(109.47), hb_length)
                # The next rotation axis will be the vector along the neighbor atom and the origin atom 
                r = coord_atom + utils.normalize(utils.vector(coord_atom1, coord_atom))

                angles = [0, -np.radians(120), np.radians(120)]

        # Now we place the water molecule(s)!
        for angle in angles:
            w = utils.rotate_atom(p, coord_atom, r, angle, hb_length)
            # Create water molecule
            waters.append(Water(w, weight, anchor_id=idx+1, anchor=coord_atom, anchor_type=anchor_type))

        return waters
    
    def hydrate(self, molecule, ad_map=None):

        waters = []

        # First hydration shell!!
        # Place optimal water molecules everywhere

        # Get all the atoms in the map
        idx_map = molecule.get_atoms_in_map(ad_map)
        # Get all the water types from the waterfield
        atom_types = self._waterfield.get_atom_types()

        # In order to keep track which one was alredy typed or not
        visited = [False] * (molecule._OBMol.NumAtoms() + 1)

        # We will test every patterns. Not very efficient 
        # but otherwise we have to do the contrary, test if the atom 
        # is in the pattern list or not... more complicated
        for i in atom_types.keys()[::-1]:

            hb_type = atom_types[i][1]
            weight = atom_types[i][2]
            hyb = atom_types[i][3]
            n_water = atom_types[i][4]
            hb_length = atom_types[i][5]
            ob_smarts = atom_types[i][6]

            ob_smarts.Match(molecule._OBMol)
            matches = list(ob_smarts.GetMapList())

            for match in matches:
                idx = match[0]

                if hb_type == 0 and not visited[idx]:
                    visited[idx] = True
                    #print '%10s %2d %s' % (atom_types[i][0], idx, 'NO')

                if idx in idx_map and not visited[idx]:
                    #print '%10s %2d %s' % (atom_types[i][0], idx, 'YES WATER!')
                    visited[idx] = True
                    waters.extend(self._place_optimal_water(molecule, weight, idx, hyb, n_water, hb_type, hb_length))

        # If a AD_map is provide, we optimize the water placement
        # and keep only the favorable ones (energy < 0)
        if ad_map is not None:
            # Optimization
            for i, water in enumerate(waters):
                if ad_map.is_in_map(water.get_coordinates(0)[0]):
                    water.optimize(ad_map, radius=3.2, angle=140)
                else:
                    waters.pop(i)

            # Energy filter
            waters = [w for w in waters if w.get_energy(ad_map) < 0.]

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

    ad_map = None

    # Read PDBQT file
    molecule = Molecule(pdbqt_file)
    # Read the Waterfield file
    waterfield = Waterfield(waterfield_file)

    # Read AutoDock grid map
    if map_file is not None:
        ad_map = Autodock_map(map_file)

    # Go kits!!
    k = Kits(waterfield)
    waters = k.hydrate(molecule, ad_map)

    utils.write_water('waters.pdb', waters)

if __name__ == '__main__':
    main()