#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# WaterKit
#
# Class for water network optimizer
#

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, fcluster

import utils

class WaterNetwork():

    def __init__(self, water_box):
        self._water_box = water_box

    def _optimize_rotatable_waters(self, receptor, waters, connections, ad_map, rotation=10):
        """ Optimize water molecules on rotatable bonds """
        rotatable_bonds = receptor.rotatable_bonds

        angles = [rotation] * (np.int(np.floor((360 / rotation))) - 1)
        angles = np.radians(angles)

        for match, value in rotatable_bonds.iteritems():
            best_energy = 0
            rot_waters = []

            for atom_id in match:
                if atom_id in connections['atom_i'].values:
                    index = connections.loc[connections['atom_i'] == atom_id]['molecule_j'].values
                    rot_waters.extend([waters[i] for i in index])

            current_energy = np.array([w.get_energy(ad_map) for w in rot_waters])
            # Keep only favorable energies
            current_energy[current_energy > 0] = 0
            best_energy += np.sum(current_energy)
            current_angle = np.radians(receptor._OBMol.GetTorsion(match[3]+1, match[2]+1, match[1]+1, match[0]+1))
            best_angle = current_angle

            # molecule._OBMol.FindChildren(atom_children, match[2], match[3])
            # print np.array(atom_children)

            # Atoms 2 and 1 define the rotation axis
            p1 = receptor.get_coordinates(match[2])[0]
            p2 = receptor.get_coordinates(match[1])[0]

            # Scan all the angles
            for angle in angles:
                for rot_water in rot_waters:
                    p0 = rot_water.get_coordinates([0])[0]
                    p_new = utils.rotate_point(p0, p1, p2, angle)
                    rot_water.update_coordinates(p_new, atom_id=0)

                current_energy = np.array([w.get_energy(ad_map) for w in rot_waters])
                current_energy[current_energy > 0] = 0
                current_energy = np.sum(current_energy)
                current_angle += angle

                if current_energy < best_energy:
                    best_angle = current_angle
                    best_energy = current_energy

            # Set the best angle
            best_angle = np.radians((360 - np.degrees(current_angle)) + np.degrees(best_angle))

            for rot_water in rot_waters:
                p0 = rot_water.get_coordinates([0])[0]
                p_new = utils.rotate_point(p0, p1, p2, best_angle)
                rot_water.update_coordinates(p_new, atom_id=0)
                # Update also the anchor
                anchor = rot_water._anchor
                anchor[0] = utils.rotate_point(anchor[0], p1, p2, best_angle)
                anchor[1] = utils.rotate_point(anchor[1], p1, p2, best_angle)

    def _cluster_waters(self, waters, distance=2., method='single'):
        """ Cluster water molecule based on their position using hierarchical clustering """
        coordinates = np.array([w.get_coordinates([0])[0] for w in waters])

        # Clustering
        Z = linkage(coordinates, method=method, metric='euclidean')
        clusters = fcluster(Z, distance, criterion='distance')
        return clusters

    def _optimize_position(self, water, ad_map, distance=2.9, angle=145):
        """ Optimize the position of the oxygen atom. The movement of the
        atom is contrained by the distance and the angle with the anchor
        """
        oxygen_type = water.get_atom_types([0])[0]

        # If the anchor type is donor, we have to reduce the
        # radius by 1 angstrom. Because hydrogen!
        if water._anchor_type == 'donor':
            distance -= 1.

        # Get all the point around the anchor (sphere)
        coord_sphere = ad_map.get_neighbor_points(water._anchor[0], 0., distance)
        # Compute angles between all the coordinates and the anchor
        angle_sphere = utils.get_angle(coord_sphere, water._anchor[0], water._anchor[1])
        # Select coordinates with an angle superior to the choosen angle
        coord_sphere = coord_sphere[angle_sphere >= angle]
        # Get energy of all the allowed coordinates (distance + angle)
        energy_sphere = ad_map.get_energy(coord_sphere, atom_type=oxygen_type)
        # ... and get energy of the oxygen
        energy_oxygen = ad_map.get_energy(water.get_coordinates(0), atom_type=oxygen_type)

        if energy_sphere.size:
            # And if we find something better, we update the coordinate
            if np.min(energy_sphere) < energy_oxygen:
                t = energy_sphere.argmin()

                # Save the old coordinate
                water._previous = water.get_coordinates()
                # ... update with the new one
                water.translate(utils.vector(water.get_coordinates(0), coord_sphere[t]))

    def _optimize_rotation(self, water, ad_map, rotation=10):
        """ Optimize the rotation of a TIP5P water molecule """
        if water._anchor_type == 'donor':
            ref_id = 3
        else:
            ref_id = 1

        best_angle = 0
        current_rotation = 0.
        best_energy = water.get_energy(ad_map)

        # Save the old coordinate
        water._previous = water.get_coordinates()

        angles = [rotation] * (np.int(np.floor((360 / rotation))) - 1)

        for angle in angles:
            water.rotate(angle, ref_id=ref_id)

            current_energy = water.get_energy(ad_map)
            current_rotation += angle

            if current_energy < best_energy:
                best_angle = current_rotation
                best_energy = current_energy

        # Once we checked all the angles, we rotate the water molecule to the best angle
        # But also we have to consider how much we rotated the water molecule before
        best_angle = (360. - current_rotation) + best_angle
        water.rotate(best_angle, ref_id=ref_id)

    def optimize_shell(self, waters, connections, distance=2.9, angle=145, cutoff=0):
        """ Optimize position of water molecules """
        rotation = 10
        cluster_distance = 2.

        shell_id = self._water_box.get_number_of_shells()
        ad_map = self._water_box.get_map(shell_id)

        if shell_id == 0:
            receptor = self._water_box.get_molecules_in_shell(0)[0]
            # Start first by optimizing the water on rotatable bonds
            self._optimize_rotatable_waters(receptor, waters, connections, ad_map, rotation)

        energies = []
        to_be_removed = []

        # And now we optimize all water individually
        for i, water in enumerate(waters):
            if ad_map.is_in_map(water.get_coordinates(0)[0]):
                # Optimize the position of the spherical water
                self._optimize_position(water, ad_map, distance, angle)

                # Before going further we check the energy
                if water.get_energy(ad_map) <= cutoff:
                    # ... and we build the TIP5
                    water.build_tip5p()
                    # ... and optimize the rotation
                    self._optimize_rotation(water, ad_map, rotation)
                    # Again, we check the energy and if it is good we keep it
                    energy = water.get_energy(ad_map)

                    if energy <= cutoff:
                        energies.append(energy)
                        water.energy = energy

                    else:
                        to_be_removed.append(i)
                else:
                    to_be_removed.append(i)
            else:
                to_be_removed.append(i)

        # Keep only the good waters
        waters = [water for i, water in enumerate(waters) if not i in to_be_removed]
        # ...and also the connections
        index = connections.loc[connections['molecule_j'].isin(to_be_removed)].index
        connections.drop(index, inplace=True)
        connections['molecule_j'] = range(0, len(waters)) # Renumber the water molecules

        if len(waters) > 1:
            # Identify clusters of waters
            clusters = self._cluster_waters(waters, distance=cluster_distance)
        elif len(waters) == 1:
            clusters = [1]
        else:
            # We return an empty water, connections and info
            return ([], [], [])

        columns = ['active', 'shell_id', 'energy', 'cluster_id']
        data = [(False, shell_id + 1, energy, cluster) for energy, cluster in zip(energies, clusters)]
        info = pd.DataFrame(data, columns=columns)
        
        # Activate only the best one in each cluster
        index = info.groupby('cluster_id', sort=False)['energy'].idxmin()
        info.loc[index, 'active'] = True

        return (waters, connections, info)
