#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# WaterKit
#
# The core of the WaterKit program
#

import imp
import sys
from string import ascii_uppercase

import openbabel as ob
import numpy as np

import utils
from water import Water
from optimize import Water_network


class Waterkit():

    def __init__(self, waterfield=None, water_map=None):
        self._water_map = water_map
        self._waterfield = waterfield

        self.water_layers = []
        self.map_layers = []

    def _place_optimal_water(self, molecule, atom_type, idx):
        """Place one or multiple water molecules in the ideal position
        above an acceptor or donor atom
        """
        waters = []
        angles = []
        hyb = atom_type.hyb
        oxygen_type = 'OW'

        if atom_type.hb_type == 1:
            anchor_type = 'donor'
        else:
            anchor_type = 'acceptor'

        # It is not the same index
        idx -= 1
        # Get origin atom
        anchor_xyz = molecule.get_coordinates(idx)[0]
        # Get coordinates of all the neihbor atoms
        neighbors_xyz = molecule.get_neighbor_atom_coordinates(idx, depth=2)

        neighbor1_xyz = neighbors_xyz[1][0]

        if hyb == 1:
            # Position of water is linear
            # And we just need the origin atom and the first neighboring atom
            # Example: H donor
            if atom_type.n_water == 1:
                r = None
                p = anchor_xyz + utils.vector(neighbor1_xyz, anchor_xyz)
                angles = [0]

            if atom_type.n_water == 3:
                hyb = 3

        elif hyb == 2:
            # Position of water is just above the origin atom
            # We need the 2 direct neighboring atoms of the origin atom
            # Example: Nitrogen
            if atom_type.n_water == 1:
                neighbor2_xyz = neighbors_xyz[1][1]

                r = None
                p = utils.atom_to_move(anchor_xyz, [neighbor1_xyz, neighbor2_xyz])
                angles = [0]

            # Position of waters are separated by angle of 120 degrees
            # And they are aligned with the neighboring atoms (deep=2) of the origin atom
            # Exemple: Backbone oxygen
            elif atom_type.n_water == 2:
                neighbor2_xyz = neighbors_xyz[2][0]

                r = utils.rotation_axis(neighbor1_xyz, anchor_xyz, neighbor2_xyz, origin=anchor_xyz)
                p = neighbor1_xyz
                angles = [-np.radians(120), np.radians(120)]

            elif atom_type.n_water == 3:
                hyb = 3

        if hyb == 3:
            neighbor2_xyz = neighbors_xyz[1][1]

            # Position of water is just above the origin atom
            # We need the 3 direct neighboring atoms (tetrahedral)
            # Exemple: Ammonia
            if atom_type.n_water == 1:
                neighbor3_xyz = neighbors_xyz[1][2]

                # We have to normalize bonds, otherwise the water molecule is not well placed
                v1 = anchor_xyz + utils.normalize(utils.vector(anchor_xyz, neighbor1_xyz))
                v2 = anchor_xyz + utils.normalize(utils.vector(anchor_xyz, neighbor2_xyz))
                v3 = anchor_xyz + utils.normalize(utils.vector(anchor_xyz, neighbor3_xyz))

                r = None
                p = utils.atom_to_move(anchor_xyz, [v1, v2, v3])
                angles = [0]

            # Position of waters are separated by angle of 109 degrees
            # Tetrahedral geometry, perpendicular with the neighboring atoms of the origin atom
            # Example: Oxygen of the hydroxyl group
            elif atom_type.n_water == 2:
                v1 = anchor_xyz + utils.normalize(utils.vector(anchor_xyz, neighbor1_xyz))
                v2 = anchor_xyz + utils.normalize(utils.vector(anchor_xyz, neighbor2_xyz))

                r = anchor_xyz + utils.normalize(utils.vector(v1, v2))
                p = utils.atom_to_move(anchor_xyz, [v1, v2])
                angles = [-np.radians(60), np.radians(60)]

            # Position of waters are separated by angle of 109 degrees
            # Tetrahedral geometry, there is no reference so water molecules are placed randomly
            # Example: DMSO
            elif atom_type.n_water == 3:
                # Vector between anchor_xyz and the only neighbor atom
                v = utils.vector(anchor_xyz, neighbor1_xyz)
                v = utils.normalize(v)

                # Pick a random vector perpendicular to vector v
                # It will be used as the rotation axis
                r = anchor_xyz + utils.get_perpendicular_vector(v)

                # And we place a pseudo atom (will be the first water molecule)
                p = utils.rotate_atom(neighbor1_xyz, anchor_xyz, r, np.radians(109.47), atom_type.hb_length)
                # The next rotation axis will be the vector along the neighbor atom and the origin atom
                r = anchor_xyz + utils.normalize(utils.vector(neighbor1_xyz, anchor_xyz))
                angles = [0, -np.radians(120), np.radians(120)]

        # Now we place the water molecule(s)!
        for angle in angles:
            oxygen_xyz = utils.rotate_atom(p, anchor_xyz, r, angle, atom_type.hb_length)
            # Create water molecule
            waters.append(Water(oxygen_xyz, oxygen_type, anchor_xyz, anchor_type))

        return waters

    def _complete_map(self, waters, ad_map, water_map, water_orientation=[[0, 0, 1], [1, 0, 0]], choices=None):

        x_len = np.int(np.floor(water_map._grid[0].shape[0] / 2.) + 5)
        y_len = np.int(np.floor(water_map._grid[1].shape[0] / 2.) + 5)
        z_len = np.int(np.floor(water_map._grid[2].shape[0] / 2.) + 5)

        map_types = list(set(ad_map._maps.keys()) & set(water_map._maps.keys()))

        if choices is not None:
            map_types = list(set(map_types) & set(choices))

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

            x = ad_map._grid[0][ix_min:ix_max + 1]
            y = ad_map._grid[1][iy_min:iy_max + 1]
            z = ad_map._grid[2][iz_min:iz_max + 1]

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
                ad_map._maps[map_type][ix_min:ix_max + 1, iy_min:iy_max + 1, iz_min:iz_max + 1] += energy

        # Update interpolator
        for map_type in map_types:
            ad_map._maps_interpn[map_type] = ad_map._generate_affinity_map_interpn(ad_map._maps[map_type])

    def hydrate(self, molecule, ad_map, n_layer=0):
        """Hydrate the molecule by adding successive layers
        of water molecules until the box is complety full
        """
        waters = []
        dict_prot_water = {}
        n_waters = 0

        # Combine OA and HO maps to create the water map
        ad_map.combine('OW', ['OA', 'OD'], how='best')
        self._water_map.combine('OW', ['OA', 'OD'], how='best')

        # Initialize the water netwrok optimizer
        n = Water_network(distance=2.9, angle=145, cutoff=0)

        # First hydration shell!!
        names, atom_ids = molecule.get_available_anchors(self._waterfield, ad_map)

        for name, idx in zip(names, atom_ids):
            atom_type = self._waterfield.get_atom_types(name)

            try:
                tmp_waters = self._place_optimal_water(molecule, atom_type, idx)
                waters.extend(tmp_waters)

                # #### FOR HYDROXYL OPTIMIZATION #####
                # Save the relation between prot and water
                for tmp_water in tmp_waters:
                    if idx in dict_prot_water:
                        dict_prot_water[idx].extend([n_waters])
                    else:
                        dict_prot_water[idx] = [n_waters]

                    n_waters += 1
            except:
                print 'Error: Couldn\'t put water(s) on %s using %s atom type' % (idx, name)
                continue

        # #### FOR HYDROXYL OPTIMIZATION #####
        # Find all the hydroxyl
        ob_smarts = ob.OBSmartsPattern()
        success = ob_smarts.Init('[!#1][!#1][#8;X2;v2;H1][#1]')
        ob_smarts.Match(molecule._OBMol)
        matches = list(ob_smarts.GetMapList())

        idx_map = molecule.get_atoms_in_map(ad_map)
        # atom_children = ob.vectorInt()

        rotation = 10
        angles = [rotation] * (np.int(np.floor((360 / rotation))) - 1)
        angles = np.radians(angles)

        for match in matches:
            if [x for x in match if x in idx_map]:
                best_energy = 0
                hydroxyl_waters = []

                for atom_id in match:
                    if atom_id in dict_prot_water:
                        hydroxyl_waters.extend([waters[i] for i in dict_prot_water[atom_id]])

                hydroxyl_energy = np.array([w.get_energy(ad_map) for w in tmp_waters])
                # Keep only negative energies
                hydroxyl_energy[hydroxyl_energy > 0] = 0
                best_energy += np.sum(hydroxyl_energy)
                current_angle = np.radians(molecule._OBMol.GetTorsion(match[0], match[1], match[2], match[3]))
                best_angle = current_angle

                # print best_energy, best_angle
                # print match, molecule._OBMol.GetTorsion(match[0], match[1], match[2], match[3])
                # molecule._OBMol.FindChildren(atom_children, match[2], match[3])
                # print np.array(atom_children)

                for angle in angles:
                    # print np.degrees(current_angle)
                    molecule._OBMol.SetTorsion(match[0], match[1], match[2], match[3], current_angle + angle)

                    # Move water molecules HERE
                    p1 = molecule.get_coordinates(match[1] - 1)
                    p2 = molecule.get_coordinates(match[2] - 1)

                    for hydroxyl_water in hydroxyl_waters:
                        p = hydroxyl_water.get_coordinates()
                        p_new = utils.rotate_atom(p[0], p1[0], p2[0], -angle)
                        hydroxyl_water.update_coordinates(p_new, atom_id=0)

                    current_energy = np.array([w.get_energy(ad_map) for w in hydroxyl_waters])
                    current_energy[current_energy > 0] = 0
                    current_energy = np.sum(current_energy)
                    current_angle += angle

                    if current_energy < best_energy:
                        best_angle = current_angle
                        best_energy = current_energy
                        # print best_energy

                # Set the hydroxyl to the best angle
                molecule._OBMol.SetTorsion(match[0], match[1], match[2], match[3], best_angle)
                # And also for each water molecule
                best_angle = np.radians((360 - np.degrees(current_angle)) + np.degrees(best_angle))
                for hydroxyl_water in hydroxyl_waters:
                    p = hydroxyl_water.get_coordinates()
                    p_new = utils.rotate_atom(p[0], p1[0], p2[0], -best_angle)
                    hydroxyl_water.update_coordinates(p_new, atom_id=0)
                    # Update also the anchor
                    # print hydroxyl_water._anchor
                    anchor = hydroxyl_water._anchor
                    anchor[0] = utils.rotate_atom(anchor[0], p1[0], p2[0], -best_angle)
                    anchor[1] = utils.rotate_atom(anchor[1], p1[0], p2[0], -best_angle)
        ####################################

        # Optimize waters and complete the map
        waters = n.optimize(waters, ad_map)

        for water in waters:
            water.energy = water.get_energy(ad_map)

        # Save the map and the water layer before updating the map
        self.map_layers.append(ad_map.copy())
        self.water_layers.append(waters)

        self._complete_map(waters, ad_map, self._water_map, choices=['OW', 'HD', 'Lp'])

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
                # Save the map and the water layer before updating the map
                self.map_layers.append(ad_map.copy())
                self.water_layers.append(waters)

                self._complete_map(waters, ad_map, self._water_map, choices=['OW', 'HD', 'Lp'])

                previous_waters = waters
            else:
                add_waters = False

            i += 1

        # ???
        # PROFIT!

    def write_waters(self, filename):
        """ Write layers of water in a PDBQT file """
        i, j = 1, 1
        ernergy = 1.0
        line = "ATOM  %5d  %-3s HOH%2s%4d    %8.3f%8.3f%8.3f  1.00%5.2f    %6.3f %2s\n"

        for waters, chain in zip(self.water_layers, ascii_uppercase):
            i, j = 1, 1

            with open('%s_%s.pdbqt' % (filename, chain), 'w') as w:
                for water in waters:
                    c = water.get_coordinates()

                    try:
                        e = water.energy
                    except:
                        e = 0.0

                    w.write(line % (j, 'O', chain, i, c[0][0], c[0][1], c[0][2], e, 0, 'OA'))

                    if c.shape[0] == 5:
                        w.write(line % (j + 1, 'H', chain, i, c[1][0], c[1][1], c[1][2], e, 0.2410, 'HD'))
                        w.write(line % (j + 2, 'H', chain, i, c[2][0], c[2][1], c[2][2], e, 0.2410, 'HD'))
                        w.write(line % (j + 3, 'H', chain, i, c[3][0], c[3][1], c[3][2], e, -0.2410, 'Lp'))
                        w.write(line % (j + 4, 'H', chain, i, c[4][0], c[4][1], c[4][2], e, -0.2410, 'Lp'))
                        j += 4

                    i += 1
                    j += 1

    def write_maps(self, prefix, map_types=None):
        """ Write maps for each layer of water molecules """
        for map_layer in self._map_layers:
            map_layer.to_map(map_types, prefix)