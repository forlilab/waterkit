#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# WaterKit
#
# Class for water network optimizer
#

import imp
import os
import time

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.transform import Rotation

import utils
from autodock_map import Map
from molecule import Molecule
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

        d = imp.find_module("waterkit")[1]

        # Water orientations set
        """ Load pre-generated water molecules (only hydrogens)
        We will use those to sample the orientation."""
        n_atoms = 2
        usecols = [0, 1, 2, 3, 4, 5]
        if self._water_model == "tip5p":
            usecols += [6, 7, 8, 9, 10, 11]
            n_atoms += 2

        w_orientation_file = os.path.join(d, "data/water_orientations.txt")
        water_orientations = np.loadtxt(w_orientation_file, usecols=usecols)
        shape = (water_orientations.shape[0], n_atoms, 3)
        self._water_orientations = water_orientations.reshape(shape)

        # Water map reference
        if self._water_model == "tip3p":
            atom_types = ["SW", "OW", "HW"]
            map_files = ["data/water/tip3p/water_VW.map",
                         "data/water/tip3p/water_OW.map",
                         "data/water/tip3p/water_HW.map"]
            map_files = [os.path.join(d, map_file) for map_file in map_files]
            water_ref_file = os.path.join(d, "data/water/tip3p/water.pdbqt")
        elif self._water_model == "tip5p":
            atom_types = ["SW", "OT", "HT", "LP"]
            map_files = ["data/water/tip5p/water_VW.map",
                         "data/water/tip5p/water_OT.map",
                         "data/water/tip5p/water_HT.map",
                         "data/water/tip5p/water_LP.map"]
            map_files = [os.path.join(d, map_file) for map_file in map_files]
            water_ref_file = os.path.join(d, "data/water/tip5p/water.pdbqt")

        self._water_map = Map(map_files, atom_types)
        self._water_ref = Molecule.from_file(water_ref_file)

        # Pairwise forcefield
        #self._adff = water_box._adff

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
                    i = utils.boltzmann_choices(energies, self._temperature)

                disordered_energies.append(energies[i])

                # Calculate the best angle, based on how much we rotated
                best_angle = np.radians((360. - np.degrees(current_angle)) + np.degrees(angles[i]))
                # Update coordinates to the choosen state
                for rot_water in rot_waters:
                    p0 = rot_water.coordinates(1)[0]
                    p_new = utils.rotate_point(p0, p1, p2, best_angle)
                    rot_water.update_coordinates(p_new, atom_id=1)
                    # Update also the anchor point
                    rot_water.hb_anchor = utils.rotate_point(rot_water.hb_anchor, p1, p2, best_angle)
                    rot_water.hb_vector = utils.rotate_point(rot_water.hb_vector, p1, p2, best_angle)

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
        if water.hb_type == "donor":
            coord_sphere = ad_map.neighbor_points(water.hb_anchor, self._max_distance - 1., self._min_distance - 1.)
        else:
            coord_sphere = ad_map.neighbor_points(water.hb_anchor, self._max_distance, self._min_distance)

        if from_edges is not None:
            is_close = ad_map.is_close_to_edge(coord_sphere, from_edges)
            coord_sphere = coord_sphere[~is_close]

        # It is mandatory to normalize the hb_vector, otherwise you don't get the angle right
        hb_vector = water.hb_anchor + utils.normalize(utils.vector(water.hb_vector, water.hb_anchor))
        angle_sphere = utils.get_angle(coord_sphere, water.hb_anchor, hb_vector)

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
            order = utils.boltzmann_choices(energies, self._temperature, len(energies))

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
                idx = utils.boltzmann_choices(energy_sphere, self._temperature)

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
            idx = utils.boltzmann_choices(energies, self._temperature)

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

    def _update_maps(self, water):

        """ If we choose the closest point in the grid and not the coordinates of the
        oxygen as the center of the grid, it is because we want to avoid any edge effect
        when we will combine the small box to the bigger box, and also the energy is
        not interpolated but it is coming from the grid directly.
        """
        spacing = self._ad_map._spacing
        ad_map = self._ad_map
        boxsize = np.array([8, 8, 8])
        npts = np.round(boxsize / spacing).astype(np.int) // 2 * 2 + 1
        water_xyz = water.coordinates()

        # The center is the closest grid point from the water molecule
        center_xyz = self._ad_map.neighbor_points(water.coordinates(1)[0], spacing)[0]
        center_index = self._ad_map._cartesian_to_index(center_xyz)

        # Get grid indices in the receptor box
        # Necessary, in order to add the water map to the receptor map
        size_half_box = ((npts - 1) / 2)
        i_min = np.clip(center_index - size_half_box, [0, 0, 0], ad_map._npts)
        i_max = np.clip(center_index + size_half_box + 1, [0, 0, 0], ad_map._npts)
        indices = np.index_exp[i_min[0]:i_max[0], i_min[1]:i_max[1], i_min[2]:i_max[2]]

        # Convert the indices in xyz coordinates, and translate it in order
        # that the water molecule is at the origin
        x = np.arange(i_min[0], i_max[0])
        y = np.arange(i_min[1], i_max[1])
        z = np.arange(i_min[2], i_max[2])
        X, Y, Z = np.meshgrid(x, y, z)
        grid_index = np.stack((X.ravel(), Y.ravel(), Z.ravel()), axis=-1)
        grid_xyz = ad_map._index_to_cartesian(grid_index)
        grid_xyz -= water_xyz[0]

        # Get the quaternion between the current water and the water reference
        # The water is translating to the origin before
        q = utils.quaternion_rotate(water_xyz - water_xyz[0], self._water_ref.coordinates())

        # Rotate the grid!
        r = Rotation.from_quat(q)
        rotated_grid_xyz = r.apply(grid_xyz)

        atom_types = ["SW"]
        if self._water_model == "tip3p":
            atom_types += ["OW", "HW"]
        elif self._water_model == "tip5p":
            atom_types += ["OT", "HT", "LP"]

        # Interpolation baby!
        for atom_type in atom_types:
            energy = self._water_map.energy_coordinates(rotated_grid_xyz, atom_type)
            energy = np.swapaxes(energy.reshape((y.shape[0], x.shape[0], z.shape[0])), 0, 1)

            ad_map._maps[atom_type][indices] += energy
            ad_map._maps_interpn[atom_type] = ad_map._generate_affinity_map_interpn(ad_map._maps[atom_type])

    def sample_grid(self, waters, connections=None, tmp_dir=".", opt_disordered=True):
        """Optimize position of water molecules."""
        df = {}
        data = []
        to_be_removed = []
        shell_id = self._water_box.number_of_shells()

        if self._how == "best":
            add_noise = False
        else:
            add_noise = True

        if opt_disordered and connections is not None:
            self._optimize_disordered_waters(waters, connections)
            #self._sample_disordered_groups(waters, connections)

        # The placement order is based on the best energy around each hydrogen anchor point
        water_orders = self._optimize_placement_order_grid(waters, from_edges=1.)
        to_be_removed.extend(set(np.arange(len(waters))) - set(water_orders))

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

                #if energy_orientation > self._energy_cutoff:
                #    delta_e = np.min([1.0, np.exp(-energy_orientation / (self._kb * self._temperature))])
                #    p = np.random.rand()

                # The last great energy filter
                # Accept the water molecule if energy is favorable
                # Or apply Metropolis criterion for bad orientation only (energy > energy_cutoff)
                if energy_orientation < self._energy_cutoff: # or p < delta_e:
                    data.append((shell_id + 1, energy_position, energy_orientation))
                    self._update_maps(water)
                else:
                    to_be_removed.append(i)
            else:
                to_be_removed.append(i)

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
