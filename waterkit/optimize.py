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

    def _calc_smooth(self, r, rij):
        smooth = 0.5

        if rij - 0.5 * smooth < r < rij + .5 * smooth :
            return rij
        elif r >= rij + .5 * smooth:
            return r - .5 * smooth
        elif r <= rij - 0.5 * smooth:
            return r + .5 * smooth

    def _calc_distance(self, r, a, c):
        rs = self._calc_smooth(r, 1.9)
        return ((a/(rs**12)) - (c/(rs**10)))

    def _calc_angle(self, angles):
        score = 1.
        angle_90 = np.pi / 2.

        for angle in angles:
            if angle < angle_90:
                score *= np.cos(angle)**2
            else:
                score *= 0.

        return score

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

                return np.min(energy_sphere)

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

    def _calc_water_energy_pairwise(self, water_xyz, anchors_xyz, vectors_xyz, anchors_types):
        energy = 0.

        for i, xyz in enumerate(water_xyz[1:]):
            if i <= 1:
                ref_xyz = xyz # use hydrogen for distance
                water_type = 'donor'
            else:
                ref_xyz = water_xyz[0] # use oxygen for distance
                water_type = 'acceptor'

            for anchor_xyz, vector_xyz, anchor_type in zip(anchors_xyz, vectors_xyz, anchors_types):
                """ water and anchor types have to be opposite types
                in order to have an hydrogen bond between them """
                if water_type != anchor_type:
                    beta_1 = utils.get_angle(xyz, anchor_xyz, vector_xyz, False)[0]
                    beta_2 = utils.get_angle(xyz + utils.vector(water_xyz[0], xyz), xyz, anchor_xyz, False)[0]
                    score_a = self._calc_angle([beta_1, beta_2])
                    r = utils.get_euclidean_distance(ref_xyz, np.array([anchor_xyz]))[0]
                    score_d = self._calc_distance(r, 55332.873, 18393.199)
                    energy += score_a * score_d

        return energy

    def _optimize_rotation_pairwise(self, water, rotation=10):
        """ Rescore all the water molecules with pairwise interactions """
        if water._anchor_type == 'donor':
            ref_id = 3
        else:
            ref_id = 1

        # Get all the neighborhood atoms
        water_xyz = water.get_coordinates()
        closest_atom_ids = self._water_box.get_closest_atoms(water_xyz[0], 3.4)

        anchors_xyz = []
        vectors_xyz = []
        anchors_ids = []
        anchors_types = []

        # Retrieve the coordinates of all the anchors
        for _, row in closest_atom_ids.iterrows():
            molecule = self._water_box.molecules[row['molecule_i']]

            # Get anchors ids
            anchor_ids = molecule.hydrogen_bond_anchors.keys()
            closest_anchor_ids = list(set([row['atom_i']]).intersection(anchor_ids))

            # Get rotatable bonds ids
            try:
                rotatable_bond_ids = molecule.rotatable_bonds.keys()
            except:
                rotatable_bond_ids = []

            for idx in closest_anchor_ids:
                xyz = molecule.get_coordinates(idx)
                anchor_type = molecule.hydrogen_bond_anchors[idx].type

                if [idx for i in rotatable_bond_ids if idx in i]:
                    """ If the vectors are on a rotatable bond 
                    we change them in order to be always pointing to
                    the water molecule (perfect HB). In fact the vector
                    is now the coordinate of the oxygen atom. And we
                    keep only one vector, we don't want to count it twice"""
                    v = water_xyz[0]
                    a = xyz
                else:
                    # Otherwise, get all the vectors on this anchor
                    v = molecule.hydrogen_bond_anchors[idx].vectors
                    a = np.tile(xyz, (v.shape[0], 1))

                anchors_xyz.append(a)
                vectors_xyz.append(v)
                anchors_ids.extend([idx] * v.shape[0])
                anchors_types.extend([anchor_type] * v.shape[0])

        anchors_xyz = np.vstack(anchors_xyz)
        vectors_xyz = np.vstack(vectors_xyz)

        angles = [rotation] * (np.int(np.floor((360 / rotation))) - 1)
        best_angle = 0
        current_rotation = 0.
        best_energy = self._calc_water_energy_pairwise(water_xyz, anchors_xyz, vectors_xyz, anchors_types)
        energy_profile = [best_energy]

        # Rotate the water molecule and get its energy
        for angle in angles:
            water.rotate(angle, ref_id=ref_id)
            water_xyz = water.get_coordinates()

            current_energy = self._calc_water_energy_pairwise(water_xyz, anchors_xyz, vectors_xyz, anchors_types)
            current_rotation += angle
            energy_profile.append(current_energy)

            if current_energy < best_energy:
                best_angle = current_rotation
                best_energy = current_energy

        # Once we checked all the angles, we rotate the water molecule to the best angle
        # But also we have to consider how much we rotated the water molecule before
        best_angle = (360. - current_rotation) + best_angle
        water.rotate(best_angle, ref_id=ref_id)
        energy_profile = np.array(energy_profile)

        return best_angle, energy_profile

    def optimize_shell(self, waters, connections, distance=2.9, angle=145, cutoff=0):
        """ Optimize position of water molecules """
        df = {}
        rotation = 10
        cluster_distance = 2.

        shell_id = self._water_box.get_number_of_shells()
        ad_map = self._water_box.get_map(shell_id, False)

        if shell_id == 0:
            receptor = self._water_box.get_molecules_in_shell(0)[0]
            # Start first by optimizing the water on rotatable bonds
            self._optimize_rotatable_waters(receptor, waters, connections, ad_map, rotation)

        angles = []
        energies = []
        profiles = []
        to_be_removed = []

        # And now we optimize all water individually
        for i, water in enumerate(waters):
            if ad_map.is_in_map(water.get_coordinates(0)[0]):
                # Optimize the position of the spherical water
                self._optimize_position(water, ad_map, distance, angle)
                # Use the energy from the OW map
                energy = water.get_energy(ad_map, 0)

                # Before going further we check the energy
                if energy <= cutoff:
                    # ... and we build the TIP5
                    water.build_tip5p()
                    # ... and optimize the rotation
                    angle, profile = self._optimize_rotation_pairwise(water, rotation)

                    energies.append(energy)
                    #profiles.append(profile)
                    angles.append(angle)
                    water.energy = energy

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

        # Add water shell informations
        columns = ['shell_id', 'energy', 'angle']
        data = [(shell_id + 1, energy, angle) for energy, angle in zip(energies, angles)]
        df_shell = pd.DataFrame(data, columns=columns)
        df['shells'] = df_shell

        # Add water profiles
        df_profile = pd.DataFrame(profiles)
        df['profiles'] = df_profile

        return (waters, connections, df)

    def select_waters(self, waters, df, how='best'):
        """ Select water molecules from the shell """
        clusters = []
        cluster_distance = 2.0

        # All water molecules are inactive by default
        df['active'] = False

        if how == 'best':
            if len(waters) > 1:
                # Identify clusters of waters
                clusters = self._cluster_waters(waters, distance=cluster_distance)
            elif len(waters) == 1:
                clusters = [1]

            df['cluster_id'] = clusters
            # Activate only the best one in each cluster
            index = df.groupby('cluster_id', sort=False)['energy'].idxmin()
            df.loc[index, 'active'] = True

        return df
