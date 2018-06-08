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

import matplotlib.pyplot as plt

class Water_network():

    def __init__(self, distance=2.8, angle=90, cutoff=0):
        self.distance = distance
        self.angle = angle
        self.cutoff = cutoff

    def _cluster_waters(self, waters, distance=2., method='single'):
        """ Cluster water molecule based on their position using hierarchical clustering """
        coordinates = np.array([w.get_coordinates(atom_ids=[0])[0] for w in waters])

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
        oxygen_type = water.get_atom_types(atom_ids=[0])[0]

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
        energy_profile = [[], [], [], [], []]

        # Save the old coordinate
        water._previous = water.get_coordinates()

        angles = [rotation] * (np.int(np.floor((360 / rotation))) - 1)

        for angle in angles:
            water.rotate_water(ref_id=ref_id, angle=angle)

            current_energy = water.get_energy(ad_map)
            current_rotation += angle

            energy_profile[0].append(water.get_energy(ad_map, 1))
            energy_profile[1].append(water.get_energy(ad_map, 2))
            energy_profile[2].append(water.get_energy(ad_map, 3))
            energy_profile[3].append(water.get_energy(ad_map, 4))
            energy_profile[4].append(current_energy)

            if current_energy < best_energy:
                best_angle = current_rotation
                best_energy = current_energy

        # Once we checked all the angles, we rotate the water molecule to the best angle
        # But also we have to consider how much we rotated the water molecule before
        best_angle = (360. - current_rotation) + best_angle
        water.rotate_water(ref_id, angle=best_angle)

        # Save the energy profile
        water._energy_profile = energy_profile

    def optimize(self, waters, ad_map):
        """ Optimize position of water molecules """
        opti_waters = []
        uniq_waters = []

        opt_rotation = 10
        cluster_distance = 2.
        i = 0

        for water in waters:
            if ad_map.is_in_map(water.get_coordinates(0)[0]):
                # Optimize the position of the spherical water
                self._optimize_position(water, ad_map)

                # Before going further we check the energy
                if water.get_energy(ad_map) <= self.cutoff:
                    # ... and we build the TIP5
                    water.build_tip5p()
                    # ... and optimize the rotation
                    self._optimize_rotation(water, ad_map, rotation=opt_rotation)

                    # Again, we check the energy and if it is good we keep it
                    if water.get_energy(ad_map) <= self.cutoff:
                        opti_waters.append(water)

                    tmp = water._energy_profile
                    angles = np.linspace(0, 360, len(tmp[0]))
                    fig, axarr = plt.subplots(5, figsize=(18, 12), sharex=True)
                    axarr[0].plot(angles, tmp[0], color='turquoise', label='hydrogen 1')
                    axarr[0].legend(loc='upper right')
                    axarr[0].set_ylabel('Energy (kcal/mol)')
                    axarr[1].plot(angles, tmp[1], color='turquoise', label='hydrogen 2')
                    axarr[1].legend(loc='upper right')
                    axarr[1].set_ylabel('Energy (kcal/mol)')
                    axarr[2].plot(angles, tmp[2], color='silver', label='lone pair 1')
                    axarr[2].legend(loc='upper right')
                    axarr[2].set_ylabel('Energy (kcal/mol)')
                    axarr[3].plot(angles, tmp[3], color='silver', label='lone pair 2')
                    axarr[3].legend(loc='upper right')
                    axarr[3].set_ylabel('Energy (kcal/mol)')
                    axarr[4].plot(angles, tmp[4], color='tomato', label='total')
                    axarr[4].legend(loc='upper right')
                    axarr[4].set_xlabel('Orientation (degrees)')
                    axarr[4].set_ylabel('Energy (kcal/mol)')
                    plt.savefig('waters_rotation_profile_%03d.png' % i, bbox_inches='tight')
                    plt.close('all')

                    i += 1

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
