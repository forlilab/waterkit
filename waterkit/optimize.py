#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# WaterKit
#
# Class for water network optimizer
#


import numpy as np
import openbabel as ob
from scipy.cluster.hierarchy import linkage, fcluster

import utils


class Water_network():

    def __init__(self, distance=2.8, angle=90, cutoff=0):
        self.distance = distance
        self.angle = angle
        self.cutoff = cutoff
        #self.alpha = alpha
        #self.beta = beta

    def _cluster_waters(self, waters, distance=2., method='single'):
        """ Cluster water molecule based on their position using hierarchical clustering
        """
        coordinates = np.array([w.get_coordinates(atom_id=0)[0] for w in waters])

        # Clustering
        Z = linkage(coordinates, method=method, metric='euclidean')
        clusters = fcluster(Z, distance, criterion='distance')

        # Group them into list
        water_clusters = [[] for i in range(np.max(clusters))]

        for water, cluster in zip(waters, clusters):
            water_clusters[cluster - 1].append(water)

        return water_clusters

    def _optimize_position(self, water, ad_map):
        """ Optimize the position of the oxygen atom. The movement of the 
        atom is contrained by the distance and the angle with the anchor
        """
        distance = self.distance

        # If the anchor type is donor, we have to reduce the 
        # radius by 1 angstrom. Because hydrogen!
        if water._anchor_type == 'donor':
            distance -= 1.

        # Get all the point around the anchor (sphere)
        coord_sphere = ad_map.get_neighbor_points(water._anchor[0], 0., distance)
        # Compute angles between all the coordinates and the anchor
        angle_sphere = utils.get_angle(coord_sphere, water._anchor[0], water._anchor[1])
        # Select coordinates with an angle superior to the choosen angle
        coord_sphere = coord_sphere[angle_sphere >= self.angle]
        # Get energy of all the allowed coordinates (distance + angle)
        energy_sphere = ad_map.get_energy(coord_sphere, atom_type='O')
        # ... and get energy of the oxygen
        energy_oxygen = ad_map.get_energy(water.get_coordinates(0), atom_type='O')

        if energy_sphere.size:
            # And if we find something better, we update the coordinate
            if np.min(energy_sphere) < energy_oxygen:
                t = energy_sphere.argmin()

                # Save the old coordinate
                water._previous = water.get_coordinates()
                # ... update with the new one
                water.update_coordinates(coord_sphere[t], 0)

    def _optimize_rotation(self, water, ad_map, rotation=10):
        """ Optimize the rotation of a TIP5P water molecule """
        if water._anchor_type == 'donor':
            ref_id = 3
        else:
            ref_id = 1

        best_angle = 0
        best_energy = water.get_energy(ad_map)
        current_rotation = 0.

        # Save the old coordinate
        water._previous = water.get_coordinates()

        angles = [rotation] * (np.int(np.floor((360 / rotation))) - 1)

        for angle in angles:
            water.rotate_water(ref_id=ref_id, angle=angle)

            current_energy = water.get_energy(ad_map)
            current_rotation += angle

            if current_energy < best_energy:
                best_angle = current_rotation
                best_energy = current_energy

        # Once we checked all the angles, we rotate the water molecule to the best angle
        # But also we have to consider how much we rotated the water molecule before
        best_angle = (360. - current_rotation) + best_angle
        water.rotate_water(ref_id, angle=best_angle)

    def optimize(self, waters, ad_map):

        opti_waters = []
        uniq_waters = []

        opt_rotation = 10
        cluster_distance = 2.

        for water in waters:
            if ad_map.is_in_map(water.get_coordinates(0)[0]):

                # Optimize position, build TIP5P, optimize rotation
                self._optimize_position(water, ad_map)
                water.build_tip5p()
                self._optimize_rotation(water, ad_map, rotation=opt_rotation)

                # Check energy
                if water.get_energy(ad_map) <= self.cutoff:
                    opti_waters.append(water)

        if len(opti_waters) > 1:
            # Identify clusters of waters
            clusters = self._cluster_waters(opti_waters, distance=cluster_distance)
        elif len(opti_waters) == 1:
            return opti_waters
        else:
            return []

        # Optimize each cluster
        for cluster in clusters:
            if len(cluster) > 1:

                best_energy = 999.
                best_water = None

                for index, water in enumerate(cluster):
                    water_energy = water.get_energy(ad_map)

                    if water_energy <= best_energy:
                        best_water = water
                        best_energy = water_energy

                uniq_waters.append(best_water)

            else:
                uniq_waters.append(cluster[0])

        return uniq_waters

    def _minima_hopping(self, waters):
        return None

    def identify_rings(self, waters):
        """ Identify rings in the water network """
        return None
