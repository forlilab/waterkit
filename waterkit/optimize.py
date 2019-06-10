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

    def __init__(self, water_box, how="best", min_distance=2.5, max_distance=3.4, angle=90, rotation=10,
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
            tmp = connections["atom_i"].isin(row[["atom_i", "atom_j"]])
            molecule_j = connections.loc[tmp]["molecule_j"].values
            rot_waters.extend([waters[j] for j in molecule_j])

            # Get energy of the favorable disordered waters
            energy_waters = np.array([ad_map.energy(w.atom_informations()) for w in rot_waters])
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
            if self._how == "best":
                i = np.argmin(energies)
            elif self._how == "boltzmann":
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

    def _neighbor_points_grid(self, water, ad_map, add_noise=False, from_edges=None):
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

        if add_noise:
            limit = ad_map._spacing / 2.
            coord_sphere += np.random.uniform(-limit, limit, coord_sphere.shape)

        if from_edges is not None:
            is_close = ad_map.is_close_to_edge(coord_sphere, from_edges)
            coord_sphere = coord_sphere[~is_close]

        angle_sphere = utils.get_angle(coord_sphere, water._anchor[0], water._anchor[1])

        coord_sphere = coord_sphere[angle_sphere >= self._angle]
        energy_sphere = ad_map.energy_coordinates(coord_sphere, atom_type=oxygen_type)

        return coord_sphere, energy_sphere

    def _optimize_placement_order_grid(self, waters, ad_map, add_noise=False, from_edges=None):
        energies = []

        for water in waters:
            _, energy_sphere = self._neighbor_points_grid(water, ad_map, add_noise, from_edges)

            if energy_sphere.size:
                energies.append(np.min(energy_sphere))
            else:
                energies.append(np.inf)

        if self._how == "best":
            order = np.argsort(energies)
        elif self._how == "boltzmann":
            order = self._boltzmann_choice(energies, True)

        return order

    def _optimize_position_grid(self, water, ad_map, add_noise=False, from_edges=None):
        """Optimize the position of the spherical water molecule. 
        
        The movement of the water is contrained by the distance and 
        the angle with the anchor."""
        oxygen_type = water.atom_types([0])
        
        coord_sphere, energy_sphere = self._neighbor_points_grid(water, ad_map, add_noise, from_edges)

        if energy_sphere.size:
            if self._how == "best":
                idx = energy_sphere.argmin()
            elif self._how == "boltzmann":
                idx = self._boltzmann_choice(energy_sphere)

            # Update the coordinates
            water.translate(utils.vector(water.coordinates(0), coord_sphere[idx]))

            return energy_sphere[idx]
        else:
            """If we do not find anything, at least we return the energy
            of the current water molecule. """
            return ad_map.energy_coordinates(water.coordinates(0), atom_type=oxygen_type)

    def _optimize_orientation_grid(self, water):
        """Optimize the orientation of the TIP5P water molecule using the grid. """
        energies = []
        coordinates = []

        ad_map = self._water_box.map
        oxygen_xyz = water.coordinates(0)
        water_xyz = water.coordinates()
        num_atoms = water_xyz.shape[0]
        atom_ids = list(range(0, num_atoms))
        water_info = water.atom_informations()

        # Translate the water to the origin for the rotation
        water_xyz -= oxygen_xyz

        for q in self._quaternions:
            # num_atoms - 1, because oxygen does not rotate
            tmp_xyz = np.zeros(shape=(num_atoms - 1, 3))

            # Rotate each atoms by the quaternion, except oxygen
            for i, u in enumerate(water_xyz[1:]):
                tmp_xyz[i] = utils.rotate_vector_by_quaternion(u, q)

            # Change coordinates by the new ones and
            # we translate it back the original oxygen position
            tmp_xyz += oxygen_xyz
            # Instead of updating the Water object and loosing time, 
            # we just update the coordinates of water_info, except oxygen
            water_info[1:]["xyz"] = tmp_xyz

            energy = ad_map.energy(water_info, ignore_electrostatic=True, ignore_desolvation=True)

            energies.append(energy)
            coordinates.append(tmp_xyz)

        if self._how == "best":
            idx = np.argmin(energies)
        elif self._how == "boltzmann":
            idx = self._boltzmann_choice(energies)

        # Update the coordinates with the selected orientation, except oxygen
        for i, j in enumerate(atom_ids[1:]):
            water.update_coordinates(coordinates[idx][i], j)

        return energies[idx]

    def optimize_grid(self, waters, connections=None, opt_disordered=True):
        """Optimize position of water molecules."""
        ad_map = self._water_box.map
        dielectric = self._water_box._dielectric
        smooth = self._water_box._smooth
        water_model = self._water_box._water_model
        receptor = self._water_box.molecules_in_shell(0)[0]
        shell_id = self._water_box.number_of_shells()

        df = {}
        data = []
        to_be_removed = []
        spacing = ad_map._spacing
        boxsize = np.array([7, 7, 7])
        npts = np.round(boxsize / spacing).astype(np.int)

        e_type = "Electrostatics"
        ow_q = -0.834
        ow_type = "OW"

        if water_model == "tip3p":
            hw_type = "HW"
            hw_q = 0.417
            atom_types_replaced = ["OW", "HW"]
        elif water_model == "tip5p":
            hw_type = "HT"
            lpw_type = "LP"
            hw_q = 0.241
            lpw_q = -0.241
            atom_types_replaced = ["OW", "HT", "LP"]

        if self._how == "best":
            add_noise = False
        else:
            add_noise = True

        # We do not want name overlap between different replicates
        short_uuid = str(uuid.uuid4())[0:8]
        receptor_file = "%s.pdbqt" % short_uuid
        self._water_box.to_pdbqt(receptor_file)

        ag = AutoGrid()

        if opt_disordered and connections is not None:
            self._optimize_disordered_waters(receptor, waters, connections, ad_map)

        # The placement order is based on the best energy around each hydrogen anchor point
        water_orders = self._optimize_placement_order_grid(waters, ad_map, add_noise, from_edges=1.)
        to_be_removed.extend(set(np.arange(len(waters))) - set(water_orders))

        """ And now we optimize all water individually. All the
        water molecules are outside the box or with a positive
        energy are considered as bad and are removed.
        """
        for i in water_orders:
            water = waters[i]

            energy_position = self._optimize_position_grid(water, ad_map, add_noise, from_edges=1.)

            """ Before going further we check the energy. If the spherical water 
            has already a bad energy there is no point of going further and try to
            orient it.
            """
            if energy_position < self._energy_cutoff:
                # Build the TIP5
                water.build_explicit_water(water_model)

                # Optimize the orientation
                energy_orientation = self._optimize_orientation_grid(water)

                # The last great energy filter
                if energy_orientation < self._energy_cutoff:
                    # TODO: Doublon, all the information should be stored in waterbox df
                    water.energy = energy_orientation
                    data.append((shell_id + 1, energy_position, energy_orientation))

                    # Add water water to the receptor before calculating the new map
                    water.to_file(receptor_file, "pdbqt", append=True)

                    """ If we choose the closest point in the grid and not the coordinates of the
                    oxygen as the center of the grid, it is because we want to avoid any edge effect
                    when we will combine the small box to the bigger box, and also the energy is
                    not interpolated but it is coming from the grid directly.
                    """
                    center = ad_map.neighbor_points(water.coordinates(0)[0], spacing)[0]

                    # Fire off AutoGrid
                    water_map = ag.run(receptor_file, ow_type, center, npts, 
                                       spacing, smooth, dielectric, clean=False)

                    # Modify electrostatics map and add it
                    # For the TIP3P and TIP5P models
                    water_map.apply_operation_on_maps(hw_type, e_type, "x * %f" % hw_q)
                    if water_model == "tip5p":
                        water_map.apply_operation_on_maps(lpw_type, e_type, "x * %f" % lpw_q)
                    # For the spherical model and TIP3P model
                    water_map.apply_operation_on_maps(e_type, e_type, "-np.abs(x * %f)" % ow_q)
                    water_map.combine(ow_type, [ow_type, e_type], how="add")

                    # And we update the receptor map
                    for atom_type in atom_types_replaced:
                        ad_map.combine(atom_type, atom_type, "replace", water_map)

                else:
                    to_be_removed.append(i)
            else:
                to_be_removed.append(i)

        os.remove(receptor_file)

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
