#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# WaterKit
#
# Class for water network optimizer
#


import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster

import utils

class WaterNetwork():

    def __init__(self, water_box):
        self._water_box = water_box

    def score_angle(self, angles):
        score = 1.
        angle_90 = np.pi / 2.

        for angle in angles:
            if angle < angle_90:
                score *= np.cos(angle)**2
            else:
                score *= 0.

        return score

    def _calc_smooth(self, r, rij):
        smooth = 0.5

        if rij - 0.5 * smooth < r < rij + .5 * smooth :
            return rij
        elif r >= rij + .5 * smooth:
            return r - .5 * smooth
        elif r <= rij - 0.5 * smooth:
            return r + .5 * smooth

    def score_distance(self, r, a, c):
        rs = self._calc_smooth(r, 1.9)
        return ((a/(rs**12)) - (c/(rs**10)))

    def score(self, molecule, waters):

        scores = []

        for index, water in enumerate(waters):
            score_total = 0

            water_xyz = water.get_coordinates()
            closest_atom_ids = molecule.get_closest_atoms(water_xyz[0], 3.4)
            closest_anchor_ids = list(set(closest_atom_ids).intersection(molecule.hydrogen_bond_anchors.keys()))

            anchors_xyz = []
            vectors_xyz = []
            anchors_ids = []

            for idx in closest_anchor_ids:
                v = molecule.hydrogen_bond_anchors[idx].vectors
                a = np.tile(np.array(molecule.get_coordinates(idx)), (v.shape[0], 1))

                anchors_xyz.append(a)
                vectors_xyz.append(v)
                anchors_ids.extend([idx]*v.shape[0])

            anchors_xyz = np.vstack(anchors_xyz)
            vectors_xyz = np.vstack(vectors_xyz)

            for i, xyz in enumerate(water_xyz[1:]):

                if i <= 1:
                    ref_xyz = xyz
                else:
                    ref_xyz = water_xyz[0]

                j = 0
                for anchor_xyz, vector_xyz in zip(anchors_xyz, vectors_xyz):
                    beta_1 = utils.get_angle(xyz, anchor_xyz, vector_xyz, False)[0]
                    beta_2 = utils.get_angle(xyz + utils.vector(water_xyz[0], xyz), xyz, anchor_xyz, False)[0]
                    #print i+1, anchors_ids[j], np.degrees(beta_1), np.degrees(beta_2)
                    score_a = self.score_angle([beta_1, beta_2])
                    r = utils.get_euclidean_distance(ref_xyz, np.array([anchor_xyz]))[0]
                    score_d = self.score_distance(r, 55332.873, 18393.199)
                    score_total += score_a * score_d

                    """
                    if 688 in anchors_ids:
                        print i+1, anchors_ids[j]
                        print xyz, anchor_xyz
                        print r, score_d
                        print np.degrees(beta_1), np.degrees(beta_2), score_a
                        print "Score:", score_a * score_d
                        print "Total score: ", score_total
                        print ""
                    """

                    j += 1

            #print "Score total: ", score_total, index+1

            scores.append(score_total)

        return scores

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
                    rot_waters.extend(list(waters[index]))

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
        #energy_profile = [[], [], [], [], []]

        #energy_profile[0].append(water.get_energy(ad_map, 1))
        #energy_profile[1].append(water.get_energy(ad_map, 2))
        #energy_profile[2].append(water.get_energy(ad_map, 3))
        #energy_profile[3].append(water.get_energy(ad_map, 4))
        #energy_profile[4].append(best_energy)

        # Save the old coordinate
        water._previous = water.get_coordinates()

        angles = [rotation] * (np.int(np.floor((360 / rotation))) - 1)

        for angle in angles:
            water.rotate(angle, ref_id=ref_id)

            current_energy = water.get_energy(ad_map)
            current_rotation += angle

            #energy_profile[0].append(water.get_energy(ad_map, 1))
            #energy_profile[1].append(water.get_energy(ad_map, 2))
            #energy_profile[2].append(water.get_energy(ad_map, 3))
            #energy_profile[3].append(water.get_energy(ad_map, 4))
            #energy_profile[4].append(current_energy)

            if current_energy < best_energy:
                best_angle = current_rotation
                best_energy = current_energy

        # Once we checked all the angles, we rotate the water molecule to the best angle
        # But also we have to consider how much we rotated the water molecule before
        best_angle = (360. - current_rotation) + best_angle
        water.rotate(best_angle, ref_id=ref_id)

        # Save the energy profile
        #water._energy_profile = energy_profile

    def optimize_last_shell(self, distance=2.9, angle=145, cutoff=0):
        """ Optimize position of water molecules """
        rotation = 10
        cluster_distance = 2.

        ad_map = self._water_box.get_last_map()
        last_shell = self._water_box.get_last_shell()
        shell_id = self._water_box.get_number_of_shells()

        waters = last_shell['molecule']

        if shell_id == 1:
            receptor = self._water_box.get_molecule(0)
            connections = self._water_box.get_connections_from_molecule(0)
            # Start first by optimizing the water on rotatable bonds
            self._optimize_rotatable_waters(receptor, waters, connections, ad_map, rotation)

        energies = []
        to_be_removed = []

        # And now we optimize all water individually
        for index, water in waters.iteritems():
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

                    else:
                        to_be_removed.append(index)
                else:
                    to_be_removed.append(index)
            else:
                to_be_removed.append(index)

        # Remove the bad water molecules
        self._water_box.remove_molecules(to_be_removed, remove_connections=True)
        # Update energy of the good ones
        self._water_box.update_shell_data(energies, 'energy', shell_id)

        # Get the new last shell
        last_shell = self._water_box.get_shell(shell_id)
        if len(last_shell.index) > 1:
            # Identify clusters of waters
            clusters = self._cluster_waters(last_shell['molecule'], distance=cluster_distance)
        elif len(last_shell.index) == 1:
            clusters = [1]
        else:
            # It means there is no more valid
            # water molecules that we can add
            return False
        # Update the cluster of water molecules
        self._water_box.update_shell_data(clusters, 'cluster', shell_id)
        
        # Get the new last shell
        last_shell = self._water_box.get_shell(shell_id)
        # Activate only the best one in each cluster
        index = last_shell.groupby('cluster', sort=False)['energy'].idxmin()
        self._water_box.activate_molecules(index)

        return True
