#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# WaterKit
#
# The core of the WaterKit program
#


import imp

import numpy as np

import utils
from water import Water
from optimize import Water_network


class Waterkit():

    def __init__(self, waterfield=None, water_map=None):
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

        coord_atom1 = coord_neighbor_atoms[1][0]

        if hyb == 1:
            # Position of water is linear
            # And we just need the origin atom and the first neighboring atom
            # Example: H donor
            if atom_type.n_water == 1:
                r = None
                p = coord_atom + utils.vector(coord_atom1, coord_atom)
                angles = [0]

            if atom_type.n_water == 3:
                hyb = 3

        elif hyb == 2:
            # Position of water is just above the origin atom
            # We need the 2 direct neighboring atoms of the origin atom
            # Example: Nitrogen
            if atom_type.n_water == 1:
                coord_atom2 = coord_neighbor_atoms[1][1]

                r = None
                p = utils.atom_to_move(coord_atom, [coord_atom1, coord_atom2])
                angles = [0]

            # Position of waters are separated by angle of 120 degrees
            # And they are aligned with the neighboring atoms (deep=2) of the origin atom
            # Exemple: Backbone oxygen
            elif atom_type.n_water == 2:
                coord_atom2 = coord_neighbor_atoms[2][0]

                r = utils.rotation_axis(coord_atom1, coord_atom, coord_atom2, origin=coord_atom)
                p = coord_atom1
                angles = [-np.radians(120), np.radians(120)]

            elif atom_type.n_water == 3:
                hyb = 3

        if hyb == 3:
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

    def _complete_map(self, waters, ad_map, water_map, water_orientation=[[0, 0, 1], [1, 0, 0]]):

        x_len = np.int(np.floor(water_map._grid[0].shape[0]/2.) + 5)
        y_len = np.int(np.floor(water_map._grid[1].shape[0]/2.) + 5)
        z_len = np.int(np.floor(water_map._grid[2].shape[0]/2.) + 5)

        map_types = set(ad_map._maps.keys()) & set(water_map._maps.keys())

        for water in waters:
            o = water.get_coordinates(atom_id=0)[0]
            h1, h2 = water.get_coordinates(atom_id=1)[0], water.get_coordinates(atom_id=2)[0]

            # Create the grid around the protein water molecule
            ix, iy, iz = ad_map._cartesian_to_index(o)

            ix_min = ix - x_len if ix - x_len >= 0 else 0
            ix_max = ix + x_len
            iy_min = iy - y_len if iy - y_len >= 0 else 0
            iy_max = iy + y_len
            iz_min = iz - z_len if iz - z_len >= 0 else 0
            iz_max = iz + z_len

            x = ad_map._grid[0][ix_min:ix_max+1]
            y = ad_map._grid[1][iy_min:iy_max+1]
            z = ad_map._grid[2][iz_min:iz_max+1]

            X, Y, Z = np.meshgrid(x, y, z)
            grid = np.stack((X.ravel(), Y.ravel(), Z.ravel()), axis=-1)

            # Do the translation
            translation = utils.vector(o, water_map._center)
            grid += translation

            # First rotation along z-axis
            u = utils.normalize(utils.vector(o, np.mean([h1, h2], axis=0)))
            rotation_z = utils.get_rotation_matrix(u, water_orientation[0])
            grid = np.dot(grid, rotation_z)

            # Second rotation along x-axis
            h1 = np.dot(h1 + translation, rotation_z)
            h2 = np.dot(h2 + translation, rotation_z)
            v = utils.normalize(np.cross(h1, h2))
            rotation_x = utils.get_rotation_matrix(v, water_orientation[1])
            grid = np.dot(grid, rotation_x)

            for map_type in map_types:
                # Interpolate energy
                energy = water_map.get_energy(grid, map_type)
                # Replace inf by zero, otherwise we cannot add water energy to the grid
                energy[energy == np.inf] = 0.

                # Reshape and swap x and y axis, right? Easy.
                # Thank you Diogo Santos Martins!!
                energy = np.reshape(energy, (y.shape[0], x.shape[0], z.shape[0]))
                energy = np.swapaxes(energy, 0, 1)

                # Add it to the existing grid
                ad_map._maps[map_type][ix_min:ix_max+1, iy_min:iy_max+1, iz_min:iz_max+1] += energy

        # Update interpolator
        for map_type in map_types:
            ad_map._maps_interpn[map_type] = ad_map._generate_affinity_map_interpn(ad_map._maps[map_type])
    
    def hydrate(self, molecule, ad_map, n_layer=0):
        """Hydrate the molecule by adding successive layers 
        of water molecules until the box is complety full
        """
        waters = []
        water_layers = []

        # Initialize the water netwrok optimizer
        n = Water_network(distance=2.9, angle=145, cutoff=0)

        # First hydration shell!!
        names, atom_ids = molecule.get_available_anchors(self._waterfield, ad_map)

        for name, idx in zip(names, atom_ids):
            atom_type = self._waterfield.get_atom_types(name)

            try:
                waters.extend(self._place_optimal_water(molecule, atom_type, idx))
            except:
                print 'Error: Couldn\'t put water(s) on %s using %s atom type' % (idx, name)
                continue

        # Optimize waters and complete the map
        waters = n.optimize(waters, ad_map)

        for water in waters:
            water.energy = water.get_energy(ad_map)

        self._complete_map(waters, ad_map, self._water_map)

        water_layers.append(waters)
        
        # Second to N hydration shell!!
        i = 1
        add_waters = True
        previous_waters = waters

        while add_waters:
            # Stop if we reach the layer i
            # If we choose n_shell equal 0, we will never reach that condition
            # and he will continue forever and ever to add water molecules 
            # until the box is full of water molecules
            if i == n_layer:
                break

            waters = []

            # Second hydration shell!!
            for water in previous_waters:
                names, atom_ids = water.get_available_anchors()

                for name, idx in zip(names, atom_ids):
                    atom_type = self._waterfield.get_atom_types(name)
                    waters.extend(self._place_optimal_water(water, atom_type, idx))
 
            # Optimize water placement
            waters = n.optimize(waters, ad_map)

            for water in waters:
                water.energy = water.get_energy(ad_map)

            if waters:
                # Complete map
                self._complete_map(waters, ad_map, self._water_map)

                previous_waters = waters
                water_layers.append(waters)
            else:
                add_waters = False

            i += 1
 
        # ???
        # PROFIT!
         
        return water_layers
      