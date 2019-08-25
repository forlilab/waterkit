#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# WaterKit
#
# Class for water network optimizer
#

import imp
import os
import uuid

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, fcluster

import utils
from autogrid import AutoGrid
from forcefield import AutoDockForceField


class WaterSampler():

    def __init__(self, water_box, how="best", min_distance=2.4, max_distance=3.2, angle=110,
                 energy_cutoff=0, temperature=298.15):
        self._water_box = water_box
        self._water_model = water_box._water_model
        self._ad_map = water_box.map
        self._receptor = water_box.molecules_in_shell(0)[0]

        self._how = how
        self._min_distance = min_distance
        self._max_distance = max_distance
        self._angle = angle
        self._temperature = temperature
        self._energy_cutoff = energy_cutoff
        # Boltzmann constant (kcal/mol)
        self._kb = 0.0019872041

        """ Load pre-generated water molecules (only hydrogens)
        We will use those to sample the orientation."""
        n_atoms = 2
        usecols = [0, 1, 2, 3, 4, 5]
        if self._water_model == "tip5p":
            usecols += [6, 7, 8, 9, 10, 11]
            n_atoms += 2

        d = imp.find_module("waterkit")[1]
        w_orientation_file = os.path.join(d, "data/water_orientations.txt")
        water_orientations = np.loadtxt(w_orientation_file, usecols=usecols)
        shape = (water_orientations.shape[0], n_atoms, 3)
        self._water_orientations = water_orientations.reshape(shape)

        # AutoGrid initialization and ADFF
        ad_parameters_file = os.path.join(d, "data/AD4_parameters.dat")
        #gpf_file = os.path.join(d, "data/nbp_r_eps.gpf")
        #self._ag = AutoGrid(param_file=ad_parameters_file, gpf_file=gpf_file)
        self._ag = AutoGrid(param_file=ad_parameters_file)
        self._adff = water_box._adff

    def _boltzmann_choice(self, energies, size=None):
        """Choose state i based on boltzmann probability."""
        energies = np.array(energies)
        
        d = np.exp(-energies / (self._kb * self._temperature))
        d_sum = np.sum(d)

        if d_sum > 0:
            p = d / np.sum(d)
        else:
            # It means that energies are too high
            return None

        if size > 1:
            # If some prob. in p are zero, ValueError: size of nonzero p is lower than size
            non_zero = np.count_nonzero(p)
            size = non_zero if non_zero < size else size

        i = np.random.choice(d.shape[0], size, False, p)
        
        return i

    def _waters_on_disordered_group(self, heavy_atom_id, waters, connections):
        receptor = self._receptor
        rot_bonds = receptor.rotatable_bonds

        # Get all the disordered hydrogen attached to heavy atom (atom_id)
        hydrogen_ids = rot_bonds.loc[rot_bonds["atom_j"] == heavy_atom_id]["atom_i"].values
        # Get the indices of the water attached to the heavy atom and disordered hydrogens
        tmp = connections["atom_i"].isin([heavy_atom_id] + hydrogen_ids.tolist())
        water_ids = connections.loc[tmp]["molecule_j"].values
        # Retrieve water molecules
        rot_waters = [waters[i] for i in water_ids]

        return rot_waters, hydrogen_ids

    def _sample_disordered_group(self, heavy_atom_id, waters, connections):
        receptor = self._receptor
        rot_bonds = receptor.rotatable_bonds
        energies = []
        water_xyzs = []
        hydrogen_xyzs = []
        distance = 10.

        # Get all the water attached
        rot_waters, hydrogen_ids = self._waters_on_disordered_group(heavy_atom_id, waters, connections)

        print heavy_atom_id

        # Get all the receptor atoms around for pairwise interactions
        heavy_atom_xyz = receptor.atom_informations(heavy_atom_id)["xyz"]
        excluded_atom_ids = pd.DataFrame(data=[], columns=("molecule_i", "atom_i"))
        excluded_atom_ids['atom_i'] = [heavy_atom_id] + hydrogen_ids.tolist()
        excluded_atom_ids['molecule_i'] = 0 # molecule 0 = receptor
        ids = self._water_box.closest_atoms(heavy_atom_xyz, distance, excluded_atom_ids)
        receptor_infos = receptor.atom_informations(ids["atom_i"].values.tolist())

        # Get the current positions
        water_infos = np.array([w.atom_informations() for w in rot_waters])
        hydrogen_infos = receptor.atom_informations(hydrogen_ids)
        rot_infos = rot_bonds.loc[rot_bonds["atom_j"] == heavy_atom_id].iloc[0]

        print excluded_atom_ids
        print water_infos
        print receptor_infos
        print hydrogen_infos

        # Atoms 3 and 2 define the rotation axis
        p1 = rot_infos["atom_k_xyz"]
        p2 = rot_infos["atom_j_xyz"]

        rotamers = rot_infos["rotamers"]
        current_angle = utils.dihedral(rot_infos[["atom_i_xyz", "atom_j_xyz", "atom_k_xyz", "atom_l_xyz"]].values, degree=True)
        rotations = np.sort((rotamers - np.round(current_angle) + 360.) % 360.)
        rotations[1:] = np.diff(rotations)
        rotations = np.radians(rotations)

        #print current_angle
        #print np.degrees(rotations)
        tmp = 0

        # Scan all the angles
        for rotation in rotations:
            if rotation > 0:
                # Get the new positions
                water_infos["xyz"] = utils.rotate_point(water_infos["xyz"], p1, p2, rotation)
                hydrogen_infos["xyz"] = utils.rotate_point(hydrogen_infos["xyz"], p1, p2, rotation)

            # Get the new energy
            energy_hydrogens = self._adff.intermolecular_energy(hydrogen_infos, receptor_infos)
            energy_waters = self._ad_map.energy(water_infos, ignore_electrostatic=True, 
                                                ignore_desolvation=True, sum_energies=False)
            energy_waters[energy_waters > 0] = 0
            energy_waters = np.sum(energy_waters)
            # Store the new state
            energies.append(energy_waters)
            water_xyzs.append(water_infos["xyz"].copy())
            hydrogen_xyzs.append(hydrogen_infos["xyz"].copy())

            tmp += rotation

        # Choose the best or the best-boltzmann state
        if self._how == "best":
            idx = np.argmin(energies)
        elif self._how == "boltzmann":
            idx = self._boltzmann_choice(energies)

        # Update water coordinates to the choosen state
        new_angle = rotamers[idx]

        if new_angle > 0:
            for i, rot_water in enumerate(rot_waters):
                rot_water.update_coordinates(water_xyzs[idx][i], atom_id=1)
                #rot_water.to_file("water_rotation_%s_%s.pdbqt" % (heavy_atom_id, i), "pdbqt")
                # Update the anchor point
                anchor = rot_water._anchor
                anchor[0] = utils.rotate_point(anchor[0], p1, p2, new_angle)
                anchor[1] = utils.rotate_point(anchor[1], p1, p2, new_angle)

        # Update also the position of disordered hydrogen
        for i, hydrogen_id in enumerate(hydrogen_ids):
            receptor.update_coordinates(hydrogen_xyzs[idx][i], atom_id=hydrogen_id)

        print idx, energies[idx], hydrogen_xyzs[idx]
        print energies
        print ""

        return rot_waters

    def _sample_disordered_groups(self, waters, connections):
        """Optimize water molecules on rotatable bonds."""
        ad_map = self._ad_map
        receptor = self._receptor
        rot_bonds = receptor.rotatable_bonds
        disordered_energies = []
        distance = 2.
        method = "single"

        # Get all the heavy atoms with a disordered hydrogens in the map
        unique_rot_bonds = rot_bonds.drop_duplicates('atom_j')
        coordinates = np.vstack(unique_rot_bonds["atom_j_xyz"])
        in_map = ad_map.is_in_map(coordinates)
        heavy_atom_ids = unique_rot_bonds.loc[in_map]['atom_j']

        # Cluster them
        Z = linkage(coordinates[in_map], method=method, metric="euclidean")
        clusters = fcluster(Z, distance, criterion="distance")
        disordered_clusters = [[] for i in range(np.max(clusters))]
        for i, cluster in zip(heavy_atom_ids, clusters):
            disordered_clusters[cluster - 1].append(i)

        # Sampling!!!
        for disordered_cluster in disordered_clusters:
            if len(disordered_cluster) > 1:
                ok = 1
            else:
                heavy_atom_id = disordered_cluster[0]
                rot_waters = self._sample_disordered_group(heavy_atom_id, waters, connections)

        short_uuid = str(uuid.uuid4())[0:8]
        receptor.to_file("receptor_disordered_%s.pdbqt" % short_uuid, "pdbqt")

    def _optimize_disordered_waters(self, waters, connections):
        """Optimize water molecules on rotatable bonds."""
        ad_map = self._ad_map
        receptor = self._receptor
        disordered_energies = []
        self._rotation = 10

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
            tmp = connections["atom_i"].isin(row[["atom_i", "atom_j"]])
            molecule_j = connections.loc[tmp]["molecule_j"].values
            rot_waters.extend([waters[j] for j in molecule_j])

            if rot_waters:
                # Get energy of the favorable disordered waters
                energy_waters = np.array([ad_map.energy(w.atom_informations(), ignore_electrostatic=True, 
                                                        ignore_desolvation=True) for w in rot_waters])
                energy_waters[energy_waters > 0] = 0
                energies.append(np.sum(energy_waters))
                # Current angle of the disordered group
                current_angle = utils.dihedral(row[["atom_i_xyz", "atom_j_xyz", "atom_k_xyz", "atom_l_xyz"]].values)
                angles.append(current_angle)

                """ Find all the atoms that depends on these atoms. This
                will be useful when we will want to rotate a whole sidechain."""
                # Atom children has to be initialized before
                # molecule._OBMol.FindChildren(atom_children, match[2], match[3])
                # print np.array(atom_children)

                # Atoms 3 and 2 define the rotation axis
                p1 = row["atom_k_xyz"]
                p2 = row["atom_j_xyz"]

                # Scan all the angles
                for i in range(n_rotation):
                    """TODO: Performance wise, we should not update water
                    coordinates everytime. Coordinates should be extracted
                    before doing the optimization and only at the end
                    we update the coordinates of the water molecules."""
                    for rot_water in rot_waters:
                        p0 = rot_water.coordinates(1)[0]
                        p_new = utils.rotate_point(p0, p1, p2, rotation)
                        rot_water.update_coordinates(p_new, atom_id=1)

                    # Get energy and update the current angle (increment rotation)
                    energy_waters = np.array([ad_map.energy(w.atom_informations(), ignore_electrostatic=True, 
                                                            ignore_desolvation=True) for w in rot_waters])

                    energy_waters[energy_waters > 0] = 0
                    energies.append(np.nansum(energy_waters))
                    current_angle += rotation
                    angles.append(current_angle)

                # Choose the best or the best-boltzmann state
                if self._how == "best":
                    i = np.argmin(energies)
                elif self._how == "boltzmann":
                    i = self._boltzmann_choice(energies)

                disordered_energies.append(energies[i])

                # Calculate the best angle, based on how much we rotated
                best_angle = np.radians((360. - np.degrees(current_angle)) + np.degrees(angles[i]))
                # Update coordinates to the choosen state
                for rot_water in rot_waters:
                    p0 = rot_water.coordinates(1)[0]
                    p_new = utils.rotate_point(p0, p1, p2, best_angle)
                    rot_water.update_coordinates(p_new, atom_id=1)
                    # Update also the anchor point
                    anchor = rot_water._anchor
                    anchor[0] = utils.rotate_point(anchor[0], p1, p2, best_angle)
                    anchor[1] = utils.rotate_point(anchor[1], p1, p2, best_angle)

        return disordered_energies

    def _neighbor_points_grid(self, water, from_edges=None):
        ad_map = self._ad_map
        oxygen_type = water.atom_types(0)
        """This is how we select the allowed positions:
        1. Get all the point coordinates on the grid around the anchor (sphere). If the anchor type 
        is donor, we have to reduce the radius by 1 angstrom. Because the hydrogen atom is closer
        to the water molecule than the heavy atom.
        2. Compute angles between all the coordinates and the anchor
        3. Select coordinates with an angle superior or equal to the choosen angle
        4. Get their energy"""
        if water._anchor_type == "donor":
            coord_sphere = ad_map.neighbor_points(water._anchor[0], self._max_distance - 1., self._min_distance - 1.)
        else:
            coord_sphere = ad_map.neighbor_points(water._anchor[0], self._max_distance, self._min_distance)

        if from_edges is not None:
            is_close = ad_map.is_close_to_edge(coord_sphere, from_edges)
            coord_sphere = coord_sphere[~is_close]

        angle_sphere = utils.get_angle(coord_sphere, water._anchor[0], water._anchor[1])

        coord_sphere = coord_sphere[angle_sphere >= self._angle]
        energy_sphere = ad_map.energy_coordinates(coord_sphere, atom_type=oxygen_type)

        return coord_sphere, energy_sphere

    def _optimize_placement_order_grid(self, waters, from_edges=None):
        energies = []

        for water in waters:
            _, energy_sphere = self._neighbor_points_grid(water, from_edges)

            if energy_sphere.size:
                energies.append(np.min(energy_sphere))
            else:
                energies.append(np.inf)

        if self._how == "best":
            order = np.argsort(energies)
        elif self._how == "boltzmann":
            order = self._boltzmann_choice(energies, len(energies))

        return order

    def _optimize_position_grid(self, water, add_noise=False, from_edges=None):
        """Optimize the position of the spherical water molecule. 
        
        The movement of the water is contrained by the distance and 
        the angle with the anchor."""
        ad_map = self._ad_map
        oxygen_type = water.atom_types([0])
        
        coord_sphere, energy_sphere = self._neighbor_points_grid(water, from_edges)

        if energy_sphere.size:
            if self._how == "best":
                idx = energy_sphere.argmin()
            elif self._how == "boltzmann":
                idx = self._boltzmann_choice(energy_sphere)

            if idx is not None:
                new_coord = coord_sphere[idx]

                if add_noise:
                    limit = ad_map._spacing / 2.
                    new_coord += np.random.uniform(-limit, limit, new_coord.shape[0])

                # Update the coordinates
                water.translate(utils.vector(water.coordinates(1), new_coord))
                return energy_sphere[idx]

        """If we do not find anything, at least we return the energy
        of the current water molecule. """
        return ad_map.energy_coordinates(water.coordinates(1), atom_type=oxygen_type)

    def _optimize_orientation_grid(self, water):
        """Optimize the orientation of the TIP5P water molecule using the grid. """
        ad_map = self._ad_map
        oxygen_xyz = water.coordinates(1)
        water_info = water.atom_informations()
        energies = np.zeros(self._water_orientations.shape[0])

        # Translate the coordinates
        water_orientations = self._water_orientations + oxygen_xyz
        # Get the energies for each atom
        # Oxygen first
        energies += ad_map.energy_coordinates(oxygen_xyz, water_info["t"][0])
        # ... and then hydrogens/lone-pairs
        for i, atom_type in enumerate(water_info["t"][1:]):
            energies += ad_map.energy_coordinates(water_orientations[:,i], atom_type)

        # Pick one orientation based the energy
        if self._how == "best":
            idx = np.argmin(energies)
        elif self._how == "boltzmann":
            idx = self._boltzmann_choice(energies)

        if idx is not None:
            # Update the coordinates with the selected orientation, except oxygen (i + 2)
            new_orientation = water_orientations[idx]
            for i, xyz in enumerate(new_orientation):
                water.update_coordinates(xyz, i + 2)
            return energies[idx]

        """If we do not find anything, at least we return the energy
        of the current water molecule. """
        return ad_map.energy(water.atom_informations(), ignore_electrostatic=True, 
                                   ignore_desolvation=True)

    def _update_maps(self, receptor_file, center, npts):
        dielectric = self._water_box._dielectric
        smooth = self._water_box._smooth
        spacing = self._ad_map._spacing

        e_type = "Electrostatics"
        sw_type = "OD"
        atom_types_replaced = [sw_type]
        ligand_types = [sw_type]

        if self._water_model == "tip3p":
            ow_type = "OW"
            hw_type = "HW"
            ow_q = -0.834
            hw_q = 0.417
            atom_types_replaced += [ow_type, hw_type]
            ligand_types += [ow_type]
        elif self._water_model == "tip5p":
            ot_type = "OT"
            hw_type = "HT"
            lw_type = "LP"
            hw_q = 0.241
            lw_q = -0.241
            atom_types_replaced += [ot_type, hw_type, lw_type]
            ligand_types += [ot_type]

        # Fire off AutoGrid
        water_map = self._ag.run(receptor_file, ligand_types, center, npts, 
                           spacing, smooth, dielectric, clean=True)

        # For the TIP3P and TIP5P models
        water_map.apply_operation_on_maps(hw_type, e_type, "x * %f" % hw_q)
        if self._water_model == "tip3p":
            water_map.apply_operation_on_maps(e_type, e_type, "x * %f" % ow_q)
            water_map.combine(ow_type, [ow_type, e_type], how="add")
        elif self._water_model == "tip5p":
            water_map.apply_operation_on_maps(lw_type, e_type, "x * %f" % lw_q)

        # And we update the receptor map
        for atom_type in atom_types_replaced:
            self._ad_map.combine(atom_type, atom_type, "add", water_map)

    def sample_grid(self, waters, connections=None, opt_disordered=True):
        """Optimize position of water molecules."""
        shell_id = self._water_box.number_of_shells()

        df = {}
        data = []
        to_be_removed = []
        spacing = self._ad_map._spacing
        boxsize = np.array([8, 8, 8])
        npts = np.round(boxsize / spacing).astype(np.int)

        if self._how == "best":
            add_noise = False
        else:
            add_noise = True

        if opt_disordered and connections is not None:
            self._optimize_disordered_waters(waters, connections)
        #    #self._sample_disordered_groups(waters, connections)

        #sys.exit(0)

        # The placement order is based on the best energy around each hydrogen anchor point
        water_orders = self._optimize_placement_order_grid(waters, from_edges=1.)
        to_be_removed.extend(set(np.arange(len(waters))) - set(water_orders))

        # We do not want name overlap between different replicates
        short_uuid = str(uuid.uuid4())[0:8]
        water_file = "%s.pdbqt" % short_uuid

        """ And now we optimize all water individually. All the
        water molecules are outside the box or with a positive
        energy are considered as bad and are removed.
        """
        for i in water_orders:
            water = waters[i]

            energy_position = self._optimize_position_grid(water, add_noise, from_edges=1.)

            """ Before going further we check the energy. If the spherical water 
            has already a bad energy there is no point of going further and try to
            orient it.
            """
            if energy_position < self._energy_cutoff:
                # Build the explicit water
                water.build_explicit_water(self._water_model)

                # Optimize the orientation
                energy_orientation = self._optimize_orientation_grid(water)

                # The last great energy filter
                if energy_orientation < self._energy_cutoff:
                    data.append((shell_id + 1, energy_position, energy_orientation))

                    """ If we choose the closest point in the grid and not the coordinates of the
                    oxygen as the center of the grid, it is because we want to avoid any edge effect
                    when we will combine the small box to the bigger box, and also the energy is
                    not interpolated but it is coming from the grid directly.
                    """
                    center = self._ad_map.neighbor_points(water.coordinates(1)[0], spacing)[0]

                    water.to_file(water_file, "pdbqt")
                    self._update_maps(water_file, center, npts)
                else:
                    to_be_removed.append(i)
            else:
                to_be_removed.append(i)

        os.remove(water_file)

        # Keep only the good waters
        waters = [waters[i] for i in water_orders if not i in to_be_removed]
        # Keep connections of the good waters
        if connections is not None:
            index = connections.loc[connections["molecule_j"].isin(to_be_removed)].index
            connections.drop(index, inplace=True)
            # Renumber the water molecules
            connections["molecule_j"] = range(0, len(waters))
            df["connections"] = connections

        # Add water shell informations
        columns = ["shell_id", "energy_position", "energy_orientation"]
        df_shell = pd.DataFrame(data, columns=columns)
        df["shells"] = df_shell

        return (waters, df)
