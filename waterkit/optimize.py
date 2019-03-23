#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# WaterKit
#
# Class for water network optimizer
#

import time
import os
import uuid
import warnings

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, fcluster

import utils
from autogrid import AutoGrid

class WaterOptimizer():

    def __init__(self, water_box, how='best', min_distance=2.5, max_distance=3.4, angle=90, rotation=10,
                 orientation=100, energy_cutoff=0, temperature=298.15):
        self._water_box = water_box
        self._how = how
        self._min_distance = min_distance
        self._max_distance = max_distance
        self._angle = angle
        self._rotation = rotation
        self._orientation = orientation
        self._temperature = temperature
        self._energy_cutoff = energy_cutoff
        # Boltzmann constant (kcal/mol)
        self._kb = 0.0019872041

        # Generate n orientation quaternions
        coordinates = np.random.random(size=(self._orientation, 3))
        self._quaternions = utils.shoemake(coordinates)

    def _cluster(self, waters, distance=2., method='single'):
        """ Cluster water molecule based on their position using hierarchical clustering """
        coordinates = np.array([w.coordinates([0])[0] for w in waters])

        # Clustering
        Z = linkage(coordinates, method=method, metric='euclidean')
        clusters = fcluster(Z, distance, criterion='distance')
        return clusters

    def _boltzmann_choice(self, energies, all_choices=False):
        """Choose state i based on boltzmann probability."""
        energies = np.array(energies)
        
        d = np.exp(-energies / (self._kb * self._temperature))
        # We ignore divide by zero warning
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            p = d / np.sum(d)

        if all_choices:
            # If some prob. in p are zero, ValueError: size of nonzero p is lower than size
            size = np.count_nonzero(p)
            i = np.random.choice(d.shape[0], size, False, p)
        else:
            i = np.random.choice(d.shape[0], p=p)

        return i

    def _optimize_disordered_waters(self, receptor, waters, connections, ad_map):
        """Optimize water molecules on rotatable bonds."""
        disordered_energies = []

        # Number of rotation necessary to do a full spin
        n_rotation = np.int(np.floor((360 / self._rotation))) - 1
        rotation = np.radians(self._rotation)

        # Iterate through every disordered bonds
        for index, row in receptor.rotatable_bonds.iterrows():
            energies = []
            angles = []
            rot_waters = []

            # Get index of all the waters attached
            # to a disordered group by looking at the connections
            tmp = connections['atom_i'].isin(row[['atom_i', 'atom_j']])
            molecule_j = connections.loc[tmp]["molecule_j"].values
            rot_waters.extend([waters[j] for j in molecule_j])

            # Get energy of the favorable disordered waters
            energy_waters = np.array([ad_map.energy(w.atom_informations()) for w in rot_waters])
            energy_waters[energy_waters > 0] = 0
            energies.append(np.sum(energy_waters))
            # Current angle of the disordered group
            current_angle = utils.dihedral(row[['atom_i_xyz', 'atom_j_xyz', 'atom_k_xyz', 'atom_l_xyz']].values)
            angles.append(current_angle)

            """ Find all the atoms that depends on these atoms. This
            will be useful when we will want to rotate a whole sidechain."""
            # Atom children has to be initialized before
            # molecule._OBMol.FindChildren(atom_children, match[2], match[3])
            # print np.array(atom_children)

            # Atoms 3 and 2 define the rotation axis
            p1 = row['atom_k_xyz']
            p2 = row['atom_j_xyz']

            # Scan all the angles
            for i in range(n_rotation):
                """TODO: Performance wise, we shouldn't update water
                coordinates everytime. Coordinates should be extracted
                before doing the optimization and only at the end
                we update the coordinates of the water molecules."""
                for rot_water in rot_waters:
                    p0 = rot_water.coordinates([0])[0]
                    p_new = utils.rotate_point(p0, p1, p2, rotation)
                    rot_water.update_coordinates(p_new, atom_id=0)

                # Get energy and update the current angle (increment rotation)
                energy_waters = np.array([ad_map.energy(w.atom_informations()) for w in rot_waters])
                energy_waters[energy_waters > 0] = 0
                energies.append(np.sum(energy_waters))
                current_angle += rotation
                angles.append(current_angle)

            # Choose the best or the best-boltzmann state
            if self._how == 'best':
                i = np.argmin(energies)
            elif self._how == 'boltzmann':
                i = self._boltzmann_choice(energies)

            disordered_energies.append(energies[i])

            # Calculate the best angle, based on how much we rotated
            best_angle = np.radians((360. - np.degrees(current_angle)) + np.degrees(angles[i]))
            # Update coordinates to the choosen state
            for rot_water in rot_waters:
                p0 = rot_water.coordinates([0])[0]
                p_new = utils.rotate_point(p0, p1, p2, best_angle)
                rot_water.update_coordinates(p_new, atom_id=0)
                # Update also the anchor point
                anchor = rot_water._anchor
                anchor[0] = utils.rotate_point(anchor[0], p1, p2, best_angle)
                anchor[1] = utils.rotate_point(anchor[1], p1, p2, best_angle)

        return disordered_energies

    def _neighbor_points_grid(self, water, ad_map, add_noise=False):
        oxygen_type = water.atom_types([0])[0]
        """This is how we select the allowed positions:
        1. Get all the point coordinates on the grid around the anchor (sphere). If the anchor type 
        is donor, we have to reduce the radius by 1 angstrom. Because the hydrogen atom is closer
        to the water molecule than the heavy atom.
        2. Compute angles between all the coordinates and the anchor
        3. Select coordinates with an angle superior or equal to the choosen angle
        4. Get their energy"""
        if water._anchor_type == 'donor':
            coord_sphere = ad_map.neighbor_points(water._anchor[0], self._min_distance - 1., self._max_distance - 1.)
        else:
            coord_sphere = ad_map.neighbor_points(water._anchor[0], self._min_distance, self._max_distance)

        if add_noise:
            limit = ad_map._spacing / 2.
            coord_sphere += np.random.uniform(-limit, limit, coord_sphere.shape)

        angle_sphere = utils.get_angle(coord_sphere, water._anchor[0], water._anchor[1])

        coord_sphere = coord_sphere[angle_sphere >= self._angle]
        energy_sphere = ad_map.energy_coordinates(coord_sphere, atom_type=oxygen_type)

        return coord_sphere, energy_sphere

    def _optimize_placement_order_grid(self, waters, ad_map, add_noise=False):
        energies = []

        for water in waters:
            _, energy_sphere = self._neighbor_points_grid(water, ad_map, add_noise)

            if energy_sphere.size:
                energies.append(np.min(energy_sphere))
            else:
                energies.append(np.inf)

        if self._how == 'best':
            order = np.argsort(energies)
        elif self._how == 'boltzmann':
            order = self._boltzmann_choice(energies, True)

        return order

    def _optimize_position_grid(self, water, ad_map, add_noise=False):
        """Optimize the position of the spherical water molecule. 

        The movement of the water is contrained by the distance and 
        the angle with the anchor."""
        coord_sphere, energy_sphere = self._neighbor_points_grid(water, ad_map, add_noise)

        if energy_sphere.size:
            if self._how == 'best':
                i = energy_sphere.argmin()
            elif self._how == 'boltzmann':
                i = self._boltzmann_choice(energy_sphere)

            # Update the coordinates
            water.translate(utils.vector(water.coordinates(0), coord_sphere[i]))

            return energy_sphere[i]
        else:
            """If we don't find anything, at least we return the energy
            of the current water molecule. """
            return ad_map.energy_coordinates(water.coordinates(0), atom_type=oxygen_type)

    def _optimize_orientation_grid(self, water):
        """Optimize the orientation of the TIP5P water molecule using the grid. """
        energies = []
        coordinates = []

        ad_map = self._water_box.map
        coordinates_water = water.coordinates()
        xyz_oxygen = water.coordinates(0)

        # Translate the water to the origin for the rotation
        coordinates_water -= xyz_oxygen

        for x, q in enumerate(self._quaternions):
            coor_tmp = np.zeros(shape=(4, 3))

            # Rotate each atoms by the quaternion
            for i, u in enumerate(coordinates_water[1:]):
                coor_tmp[i] = utils.rotate_vector_by_quaternion(u, q)

            # Change coordinates by the new ones and
            # we translate it back the original oxygen position
            coor_tmp += xyz_oxygen

            # Get energy from the grid and save the coordinates
            [water.update_coordinates(coor_tmp[i - 1], i) for i in range(1,5)]
            info_water = water.atom_informations(range(1, 5))

            energies.append(ad_map.energy(info_water))
            coordinates.append(coor_tmp)

        if self._how == 'best':
            i = energies.argmin()
        elif self._how == 'boltzmann':
            i = self._boltzmann_choice(energies)

        # Update the coordinates with the selected orientation
        [water.update_coordinates(coordinates[i][j - 1], j) for j in range(1,5)]

        return energies[i]

    def _orient_disordered_groups(self, p, atoms, hb_vectors, disordered_groups):
        """Point disordered HB vectors toward water molecule."""
        acceptor_types = ['OA', 'NA', 'SA']
        angle = np.radians(109.5)
        anchor_name = 'H_O_001'
        anchor_type = 'donor'
        atom_type = 'HD'
        distance_hydrogen = 1.
        distance_vector = 2.8
        partial_charge = 0.200

        for _, disordered_atoms in disordered_groups.iterrows():
            if disordered_atoms[['atom_i', 'atom_j']].isin(atoms['atom_i']).any():
                atom_i_xyz = disordered_atoms['atom_i_xyz']
                atom_j_xyz = disordered_atoms['atom_j_xyz']
                atom_k_xyz = disordered_atoms['atom_k_xyz']

                # 1. Get rotation point
                r = np.cross(utils.vector(atom_i_xyz, p), utils.vector(atom_i_xyz, atom_k_xyz)) + atom_j_xyz
                h = utils.rotate_point(atom_k_xyz, atom_j_xyz, r, angle)
                new_hd = utils.resize_vector(h, distance_hydrogen, atom_j_xyz)
                new_vector = utils.resize_vector(h, distance_vector, atom_j_xyz)

                """2. Change hydrogen atom coordinate and HB vector associated.
                If there is no hydrogen atom, we create it and HB vector also."""
                try:
                    i = atoms.index[atoms['atom_i'] == disordered_atoms['atom_i']].tolist()[0]
                    atoms.at[i, 'atom_xyz'] = new_hd
                    i = hb_vectors.index[hb_vectors['atom_i'] == disordered_atoms['atom_i']].tolist()[0]
                    hb_vectors.at[i, 'vector_xyz'] = new_vector
                except:
                    atom_data = {'atom_i' : disordered_atoms['atom_i'],
                            'atom_xyz': new_hd,
                            'atom_q': partial_charge,
                            'atom_type': atom_type}
                    atoms = atoms.append(atom_data, ignore_index=True)

                    hb_data = {'atom_i' : disordered_atoms['atom_i'],
                            'vector_xyz': new_vector,
                            'anchor_type': anchor_type,
                            'anchor_name': anchor_name}
                    hb_vectors = hb_vectors.append(hb_data, ignore_index=True)

                """3. Change HB vector associated to the OA/NA/SA.And keep 
                only one HB vector if there is more than one. If the oxygen 
                atom is not here, we do nothing because it means that it is 
                too far from the water molecule anyway."""
                try:
                    i = hb_vectors.index[hb_vectors['atom_i'] == disordered_atoms['atom_j']].tolist()
                    if len(i) == 2:
                        hb_vectors.drop(i[-1], inplace=True)
                    hb_vectors.at[i[0], 'vector_xyz'] = new_vector
                except:
                    # We do nothing
                    pass

                # 4. If hydrogen atom attached to OA/NA/SA, we change the type from HD to Hd
                if atoms.loc[atoms['atom_i'] == disordered_atoms['atom_j']]['atom_type'].isin(acceptor_types).any():
                    atoms.loc[atoms['atom_i'] == disordered_atoms['atom_i'], 'atom_type'] = 'Hd'

        return atoms, hb_vectors

    def _optimize_orientation_pairwise(self, water):
        """Optimize the orientation of the water molecule."""
        angles = []
        atoms = []
        disordered_groups = []
        energies = []
        hb_vectors = []
        distance_cutoff = 4.0

        if water._anchor_type == 'donor':
            ref_id = 3
        else:
            ref_id = 1

        pairwise_energy = self._water_box._ad4_forcefield.intermolecular_energy

        water_xyz = water.coordinates(0)[0]
        # Number of rotation necessary to do a full spin
        n_rotation = np.int(np.floor((360 / self._rotation))) - 1

        # Get all the neighborhood atoms (active molecules)
        closest_atom_ids = self._water_box.closest_atoms(water_xyz, distance_cutoff)
        se = closest_atom_ids.groupby('molecule_i')['atom_i'].apply(list)

        for molecule_i, atom_ids in se.iteritems():
            molecule = self._water_box.molecules[molecule_i]
            """We have to add the molecule id because the atom id alone
            is not enough to identify an atom."""
            mol_prefix = str(molecule_i) + '_'

            # Retrieve all the informations about the closest atoms
            tmp_atoms = molecule.atom_informations(atom_ids)
            tmp_atoms['atom_i'] = mol_prefix + tmp_atoms['atom_i'].astype(str)
            atoms.append(tmp_atoms)

            # Retrieve all the hydrogen bond anchors
            tmp = molecule.hydrogen_bond_anchors['atom_i'].isin(atom_ids)
            tmp_hb = molecule.hydrogen_bond_anchors.loc[tmp]
            if not tmp_hb.empty:
                tmp_hb['atom_i'] = mol_prefix + tmp_hb[['atom_i']].astype(str)
                hb_vectors.append(tmp_hb)

            # Retrieve all the disordered atoms
            if molecule.rotatable_bonds is not None:
                tmp = molecule.rotatable_bonds[['atom_i', 'atom_j']].isin(atom_ids)
                tmp = tmp['atom_i'] | tmp['atom_j']
                tmp_disordered = molecule.rotatable_bonds.loc[tmp]
                if not tmp_disordered.empty:
                    tmp_disordered['atom_i'] = mol_prefix + tmp_disordered[['atom_i']].astype(str)
                    tmp_disordered['atom_j'] = mol_prefix + tmp_disordered[['atom_j']].astype(str)
                    disordered_groups.append(tmp_disordered)

        # Concatenate all the results
        atoms = pd.concat(atoms, ignore_index=True)

        try:
            hb_vectors = pd.concat(hb_vectors, ignore_index=True)
        except ValueError:
            hb_vectors = None
        
        try:
            disordered_groups = pd.concat(disordered_groups, ignore_index=True)
        except ValueError:
            disordered_groups = None

        # We remove all the oxygen water molecule to speed up the computation
        atoms.drop(atoms[atoms['atom_type'] == 'Ow'].index, inplace=True)
        """We want that all the disordered hydrogen atoms point toward the water.
        Otherwise the interaction are not correctly modelized."""
        if disordered_groups is not None:
            atoms, hb_vectors = self._orient_disordered_groups(water_xyz, atoms, hb_vectors, disordered_groups)

        # Get the energy of the current orientation
        energy_water = pairwise_energy(water.atom_informations(atom_ids=[1, 2, 3, 4]), atoms,
                                       water.hydrogen_bond_anchors, hb_vectors)
        energies.append(energy_water)
        # Set the current to 0, there is no angle reference
        current_angle = 0.
        angles.append(current_angle)

        # Rotate the water molecule and get its energy
        for i in range(n_rotation):
            water.rotate(self._rotation, ref_id=ref_id)

            # Get energy 
            energies.append(pairwise_energy(water.atom_informations(atom_ids=[1, 2, 3, 4]), atoms,
                                            water.hydrogen_bond_anchors, hb_vectors))
            # ...and update the current angle (increment rotation)
            current_angle += self._rotation
            angles.append(current_angle)

        # Choose the orientation
        if self._how == 'best':
            i = np.argmin(energies)
        elif self._how == 'boltzmann':
            i = self._boltzmann_choice(energies)

        # Once we checked all the angles, we rotate the water molecule to the best angle
        # But also we have to consider how much we rotated the water molecule before
        best_angle = (360. - current_angle) + angles[i]
        water.rotate(best_angle, ref_id=ref_id)

        return energies[i]

    def optimize_pairwise(self, waters, connections=None, opt_position=True, opt_rotation=True, opt_disordered=True):
        """Optimize position of water molecules."""
        df = {}
        data = []
        profiles = []
        to_be_removed = []

        shell_id = self._water_box.number_of_shells(ignore_xray=True)
        ad_map = self._water_box.map

        if opt_disordered and connections is not None:
            receptor = self._water_box.molecules_in_shell(0)[0]
            # Start first by optimizing the disordered water molecules
            self._optimize_disordered_waters(receptor, waters, connections, ad_map)

        """And now we optimize all water individually. All the
        water molecules are outside the box or with a positive
        energy are considered as bad and are removed."""
        for i, water in enumerate(waters):
            if ad_map.is_in_map(water.coordinates(0)[0]):
                # Optimize the position of the spherical water
                if opt_position:
                    energy_position = self._optimize_position_grid(water, ad_map)
                else:
                    energy_position = ad_map.energy(water.atom_informations())

                """Before going further we check the energy.
                If the spherical water has already a bad energy
                there is no point of going further and try to
                orient it..."""
                if energy_position <= self._energy_cutoff:
                    # Build the TIP5
                    water.build_tip5p()

                    # Optimize the rotation
                    if opt_rotation:
                        energy_orientation = self._optimize_orientation_pairwise(water)
                    else:
                        # Make sure we pass the energy filter
                        energy_orientation = self._energy_cutoff - 1.

                    # Last energy filter
                    if energy_orientation <= self._energy_cutoff:
                        # TODO: Doublon, all the information should be stored in waterbox df
                        water.energy = energy_orientation
                        data.append((shell_id + 1, energy_position, energy_orientation))
                    else:
                        to_be_removed.append(i)
                else:
                    to_be_removed.append(i)
            else:
                to_be_removed.append(i)

        # Keep only the good waters
        waters = [water for i, water in enumerate(waters) if not i in to_be_removed]
        # Keep connections of the good waters
        if connections is not None:
            index = connections.loc[connections['molecule_j'].isin(to_be_removed)].index
            connections.drop(index, inplace=True)
            # Renumber the water molecules
            connections['molecule_j'] = range(0, len(waters))
            df['connections'] = connections

        # Add water shell informations
        columns = ['shell_id', 'energy_position', 'energy_orientation']
        df_shell = pd.DataFrame(data, columns=columns)
        df['shells'] = df_shell

        return (waters, df)

    def optimize_grid(self, waters, connections=None, opt_disordered=True):
        """Optimize position of water molecules."""
        df = {}

        atom_types = ['Oa', 'Od', 'HD', 'Lp']
        atom_types_replaced = ['Ow', 'HD', 'Lp']
        best_energy_spherical_waters = []
        data = []
        npts = (19, 19, 19)
        profiles = []
        to_be_removed = []

        type_lp = 'Lp'
        type_hd = 'HD'
        type_oa = 'Oa'
        type_od = 'Od'
        type_w = 'Ow'
        type_e = 'Electrostatics'

        ad_map = self._water_box.map
        receptor = self._water_box.molecules_in_shell(0)[0]
        shell_id = self._water_box.number_of_shells(ignore_xray=True)

        ag = AutoGrid()

        if opt_disordered and connections is not None:
            self._optimize_disordered_waters(receptor, waters, connections, ad_map)

        # The placement order is based on the best energy around each hydrogen anchor point
        water_orders = self._optimize_placement_order_grid(waters, ad_map)
        to_be_removed.extend(set(np.arange(len(waters))) - set(water_orders))
        # ... or the placement order is random. So the starting point will be always different.
        #water_orders = np.arange(len(waters))
        #np.random.shuffle(water_orders)

        """And now we optimize all water individually. All the
        water molecules are outside the box or with a positive
        energy are considered as bad and are removed."""
        for i in water_orders:
            water = waters[i]

            energy_position = self._optimize_position_grid(water, ad_map, add_noise=True)

            """Before going further we check the energy.
            If the spherical water has already a bad energy
            there is no point of going further and try to
            orient it..."""
            if energy_position < self._energy_cutoff:
                # Build the TIP5
                water.build_tip5p()
                # Optimize the orientation
                energy_orientation = self._optimize_orientation_grid(water)

                # The last great energy filter
                if energy_orientation < self._energy_cutoff:
                    # TODO: Doublon, all the information should be stored in waterbox df
                    water.energy = energy_orientation
                    data.append((shell_id + 1, energy_position, energy_orientation))

                    # We don't want name overlap between different replicates
                    short_uuid = str(uuid.uuid4())[0:8]
                    receptor_file = '%s.pdbqt' % short_uuid
                    center = water.coordinates(0)[0]

                    # Dirty hack to write the receptor with all the water molecules
                    receptor.add_molecule(water.tip3p())
                    receptor.to_file(receptor_file, 'pdbqt', 'rcp')

                    water_map = ag.run(receptor_file, atom_types, center, npts, clean=True)
                    water_map.combine(type_w, [type_oa, type_od], how='add')

                    # And we update the receptor map
                    for atom_type in atom_types_replaced:
                        ad_map.combine(atom_type, atom_type, 'replace', water_map)

                    os.remove(receptor_file)

                else:
                    to_be_removed.append(i)
            else:
                to_be_removed.append(i)

        # Keep only the good waters
        waters = [waters[i] for i in water_orders if not i in to_be_removed]
        # Keep connections of the good waters
        if connections is not None:
            index = connections.loc[connections['molecule_j'].isin(to_be_removed)].index
            connections.drop(index, inplace=True)
            # Renumber the water molecules
            connections['molecule_j'] = range(0, len(waters))
            df['connections'] = connections

        # Add water shell informations
        columns = ['shell_id', 'energy_position', 'energy_orientation']
        df_shell = pd.DataFrame(data, columns=columns)
        df['shells'] = df_shell

        return (waters, df)

    def activate_molecules_in_shell(self, shell_id):
        """Activate waters in the shell."""
        clusters = []
        cluster_distance = 2.7
        minimal_distance = 2.5

        waters = self._water_box.molecules_in_shell(shell_id, active_only=False)
        df = self._water_box.molecule_informations_in_shell(shell_id)

        # The dataframe and the waters list must have the same index
        df.reset_index(drop=True, inplace=True)

        if self._how == 'best' or self._how == 'boltzmann':
            if len(waters) > 1:
                # Identify clusters of waters
                clusters = self._cluster(waters, distance=cluster_distance)
            elif len(waters) == 1:
                clusters = [1]

            df['cluster_id'] = clusters

            for i, cluster in df.groupby('cluster_id', sort=False):
                to_activate = []

                cluster = cluster.copy()

                """This is how we cluster water molecules:
                1. We identify the best or the bolzmann-best water molecule in the 
                cluster, by taking first X-ray water molecules, if not the best 
                water molecule in term of energy. 
                2. Calculate the distance with the best(s) and all the
                other water molecules. The water molecules too close are removed 
                and are kept only the ones further than 2.4 A. 
                3. We removed the best and the water that are clashing from the dataframe.
                4. We continue until there is nothing left in the dataframe."""
                while cluster.shape[0] > 0:
                    to_drop = []

                    if True in cluster['xray'].values:
                        best_water_ids = cluster[cluster['xray'] == True].index.values
                    else:
                        if self._how == 'best':
                            best_water_ids = [cluster['energy_orientation'].idxmin()]
                        elif self._how == 'boltzmann':
                            i = self._boltzmann_choice(cluster['energy_orientation'].values)
                            best_water_ids = [cluster.index.values[i]]

                    water_ids = cluster.index.difference(best_water_ids).values

                    if water_ids.size > 0:
                        waters_xyz = np.array([waters[x].coordinates(0)[0] for x in water_ids])

                        for best_water_id in best_water_ids:
                            best_water_xyz = waters[best_water_id].coordinates(0)
                            d = utils.get_euclidean_distance(best_water_xyz, waters_xyz)
                            to_drop.extend(water_ids[np.argwhere(d < minimal_distance)].flatten())

                    to_activate.extend(best_water_ids)
                    cluster.drop(best_water_ids, inplace=True)
                    cluster.drop(to_drop, inplace=True)

                # The best water identified are activated
                df.loc[to_activate, 'active'] = True

        elif how == 'all':
            df['active'] = True

        # We update the information to able to build the next hydration shell
        self._water_box.update_informations_in_shell(df['active'].values, shell_id, 'active')
