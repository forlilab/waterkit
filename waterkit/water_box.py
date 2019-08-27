#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# WaterKit
#
# Class to manage water box
#

import copy

import numpy as np
import openbabel as ob
import pandas as pd
from scipy import spatial

import utils
from water import Water
from optimize import WaterSampler


class WaterBox():

    def __init__(self, receptor, ad_map, ad_forcefield, water_model="tip3p", how="boltzmann", temperature=300.):
        self.df = {}
        self.molecules = {}
        self.map = None
        self._kdtree = None
        self._water_model = water_model

        # All the informations are stored into a dict of df
        columns = ["molecule_i", "atom_i", "molecule_j", "atom_j"]
        self.df["connections"] = pd.DataFrame(columns=columns)
        columns = ["shell_id"]
        self.df["shells"] = pd.DataFrame(columns=columns)
        columns = ["molecule_i", "atom_i"]
        self.df["kdtree_relations"] = pd.DataFrame(columns=columns)

        self._add_map(ad_map)
        self._add_receptor(receptor)

        # Forcefields, forcefield parameters and water model
        self._how = how
        self._temperature = temperature
        self._dielectric = 1.
        self._smooth = 0.
        self._adff = ad_forcefield
        # Initialize the sampling method
        self._wopt = WaterSampler(self, self._how, angle=110, temperature=self._temperature)

    def copy(self):
        """Return deepcopy of WaterBox."""
        return copy.deepcopy(self)

    def _add_map(self, ad_map):
        """Add the receptor and the corresponding ad_map to the waterbox."""
        self.map = ad_map.copy()

        e_type = "Electrostatics"
        sw_type = "OD"

        """In TIP3P and TIP5P models, hydrogen atoms and lone-pairs does not
        have VdW radius, so their interactions with the receptor are purely
        based on electrostatics. So the HD and Lp maps are just the electrostatic 
        map. Each map is multiplied by the partial charge. So it is just a
        look-up table to get the energy for each water molecule.
        """
        if self._water_model == "tip3p":
            ow_type = "OW"
            hw_type = "HW"
            ow_q = -0.834
            hw_q = 0.417
        elif self._water_model == "tip5p":
            ot_type = "OT"
            hw_type = "HT"
            lw_type = "LP"
            hw_q = 0.241
            lw_q = -0.241
        else:
            print "Error: water model %s unknown." % self._water_model
            return False

        # For the TIP3P and TIP5P models
        self.map.apply_operation_on_maps(hw_type, e_type, "x * %f" % hw_q)

        if self._water_model == "tip3p":
            self.map.apply_operation_on_maps(e_type, e_type, "x * %f" % ow_q)
            self.map.combine(ow_type, [ow_type, e_type], how="add")
        elif self._water_model == "tip5p":
            self.map.apply_operation_on_maps(lw_type, e_type, "x * %f" % lw_q)

        return True

    def _add_receptor(self, receptor):
        """Add the receptor and the corresponding ad_map to the waterbox."""
        if not 0 in self.molecules:
            # Add the receptor to the waterbox
            self._add_molecules(receptor)
            # Add informations about the receptor
            data = pd.DataFrame([[0]], columns=["shell_id"])
            self._add_informations(data, "shells")

            return True
        else:
            # The receptor was already added to the waterbox
            return False

    def _add_molecules(self, molecules, connections=None, add_KDTree=True):
        """ Add a new molecule to the waterbox """
        if not isinstance(molecules, (list, tuple)):
            molecules = [molecules]

        try:
            last_key = np.max(self.molecules.keys())
        except:
            # We initliaze at -1, make first molecule at index 0
            last_key = -1

        # Add molecules to the dictionary
        new_keys = range(last_key + 1, len(molecules) + last_key + 1)
        d = {key: molecule for key, molecule in zip(new_keys, molecules)}
        self.molecules.update(d)

        if connections is not None:
            self._add_connections(connections)
        if add_KDTree:
            self._add_molecules_to_kdtree(molecules)

    def _add_connections(self, connections):
        """ Add connections between molecules """
        try:
            last_connections = self.df["connections"].tail(1)
            last_molecule_i = last_connections["molecule_i"].values[0]
            last_molecule_j = last_connections["molecule_j"].values[0]
        except:
            last_molecule_i = -1
            last_molecule_j = 0

        connections["molecule_i"] += last_molecule_i + 1
        connections["molecule_j"] += last_molecule_j + 1
        self._add_informations(connections, "connections")

    def _add_molecules_to_kdtree(self, molecules):
        """ Build or update the cKDTree of all the atoms in
        the water box for quick nearest-neighbor lookup
        """
        if not isinstance(molecules, (list, tuple)):
            molecules = [molecules]

        try:
            last_kdtree_relations = self.df["kdtree_relations"].tail(1)
            last_molecule_i = last_kdtree_relations["molecule_i"].values[0]
        except:
            # We initliaze at -1, make first molecule at index 0
            last_molecule_i = -1

        data = []
        relations = []

        for index, molecule in enumerate(molecules):
            coordinates = molecule.coordinates()
            mol_i = index + last_molecule_i + 1
            relations.append([[mol_i, i + 1] for i in range(coordinates.shape[0])])
            data.append(coordinates)

        # Update the KDTree relation database
        columns = ["molecule_i", "atom_i"]
        relations = np.vstack(relations)
        relations = pd.DataFrame(relations, columns=columns)
        self._add_informations(relations, "kdtree_relations")

        # Update the KDTree
        data = np.vstack(data)
        try:
            data = np.concatenate((self._kdtree.data, data))
        except:
            pass
        self._kdtree = spatial.cKDTree(data)

    def _add_informations(self, data, where):
        """ Append DF to the existing information DF """
        try:
            self.df[where] = self.df[where].append(data, sort=False)
            self.df[where].reset_index(drop=True, inplace=True)
        except:
            print "Error: Cannot add informations to %s dataframe." % where

    def _update_informations_in_shell(self, data, shell_id, key):
        """Update shell information."""
        index = self.df["shells"]["shell_id"] == shell_id
        self.df["shells"].loc[index, key] = data

    def number_of_shells(self):
        """Total number of shells in the WaterBox.

        Returns:
            int: number of shells

        """
        # df["column"].max() faster than np.max(df["column"])
        return self.df["shells"]["shell_id"].max()

    def molecules_in_shell(self, shell_ids=None):
        """Get all the molecule in shell.

        Args:
            shell_ids (list): ids of the shell(s) (default: None)

        Returns:
            list: list of all the molecules in the selected shell(s)

        """
        if shell_ids is not None:
            if not isinstance(shell_ids, (list, tuple)):
                shell_ids = [shell_ids]
            df = self.df["shells"][self.df["shells"]["shell_id"].isin(shell_ids)]
        else:
            df = self.df["shells"]

        molecules = [self.molecules[i] for i in df.index.tolist()]

        return molecules

    def molecule_informations_in_shell(self, shell_id):
        """Get information of shell.

        Args:
            shell_id (int): id the shell

        Returns:
            DataFrame: informations concerning the shell

        """
        df = self.df["shells"]
        # Return a copy to avoid a SettingWithCopyWarning flag
        return df.loc[df["shell_id"] == shell_id].copy()

    def closest_atoms(self, xyz, radius, exclude=None):
        """Retrieve indices of the closest atoms around x 
        at a certain radius.

        Args:
            xyz (array_like): array of 3D coordinates
            raidus (float): radius
            exclude (DataFrame): contains molecule to exclude (columns: molecule_i and atom_i)

        Returns:
            DataFrame: contains all the molecules/atoms at XX angstrom from xyz

        """
        if self._kdtree is None:
            print "Warning: KDTree is empty."
            return pd.DataFrame(columns=["molecule_i", "atom_i"])

        index = self._kdtree.query_ball_point(xyz, radius, p=2)
        df = self.df["kdtree_relations"].loc[index]

        if isinstance(exclude, pd.DataFrame):
            df = df.merge(exclude, indicator=True, how="outer")
            df = df[df["_merge"] == "left_only"]
            df.drop(columns="_merge", inplace=True)

        return df

    def closest_hydrogen_bond_anchor(self, xyz, radius, exclude=None):
        """Find the closest hydrogen bond anchors.

        Args:
            xyz (array_like): array of 3D coordinates
            radius (float): max radius in Angström
            exclude (list): list of index of molecules to exclude

        Returns:
            best_hba 
            best_hbv_id

        """
        best_hba = None
        best_hbv_id = None
        best_hbv_distance = np.inf

        df = self.closest_atoms(xyz, radius, exclude)

        if not df.empty:
            for index, row in df.iterrows():
                try:
                    hydrogen_bonds = self.molecules[row["molecule_i"]].hydrogen_bonds
                    hba = hydrogen_bonds.loc[hydrogen_bonds["atom_i"] == row["atom_i"]]
                    hba_xyz = self.molecules[row["molecule_i"]].coordinates(row["atom_i"])

                    hba_distance = utils.get_euclidean_distance(xyz, hba_xyz)[0]
                    hbv_distances = utils.get_euclidean_distance(xyz, hba.vectors)
                    hbv_min_distance = np.min(hbv_distances)
                    hbv_min_id = np.argmin(hbv_distances)

                    # We add 1 A to interpolate the distance to the heavy atom
                    if hba.type == "donor":
                        hba_distance += 1.

                    # Select the closest HBV and make sure that the heavy atom is close enough
                    if hbv_min_distance < best_hbv_distance and hba_distance <= radius:
                        best_hba = hba
                        best_hbv_id = hbv_min_id
                        best_hbv_distance = hbv_min_distance
                except KeyError:
                    continue

        return best_hba, best_hbv_id

    def _place_optimal_spherical_waters(self, molecules, atom_type="W", partial_charge=0.):
        """Place spherical water molecules in the optimal position.

        Args:
            molecules (Molecule): molecules on which spherical water molecules will be placed
            atom_type (str): atom type of the spherical water molecules (default: OW)
            partial_charges (float): partial charges of the spherical water molecules (default: -0.834)

        Returns:
            list: list of water molecules
            DataFrame: contains connections between molecules

        """
        waters = []
        data = []

        for i, molecule in enumerate(molecules):
            for index, row in molecule.hydrogen_bonds.iterrows():
                # Add water molecule only if it's in the map
                anchor_xyz = molecule.coordinates(row.atom_i)[0]
                vector_xyz = anchor_xyz + utils.normalize(utils.vector(row.vector_xyz, anchor_xyz))

                if self.map.is_in_map(anchor_xyz):
                    w = Water(row.vector_xyz, atom_type, partial_charge, anchor_xyz, vector_xyz, row.anchor_type)
                    
                    waters.append(w)
                    data.append((i, row.atom_i, len(waters) - 1, None))

        # Convert list of tuples into dataframe
        columns = ["molecule_i", "atom_i", "molecule_j", "atom_j"]
        connections = pd.DataFrame(data, columns=columns)

        return (waters, connections)

    def build_next_shell(self):
        """Build the next hydration shell.

        Returns:
            bool: True if water molecules were added or False otherwise

        """
        type_sw = "OD"
        partial_charge = 0.0
        shell_id = self.number_of_shells()
        molecules = self.molecules_in_shell(shell_id)

        waters, connections = self._place_optimal_spherical_waters(molecules, type_sw, partial_charge)

        # Only the receptor contains disordered hydrogens
        if shell_id == 0:
            waters, df = self._wopt.sample_grid(waters, connections, opt_disordered=True)
        else:
            """After the first hydration layer, we don't care anymore about 
            connections. It was only useful for the disordered hydrogen atoms.
            """
            waters, df = self._wopt.sample_grid(waters, opt_disordered=False)

        if len(waters):
            self._add_molecules(waters, add_KDTree=False)
            # Add informations about the new shell
            if "shells" in df.keys():
                self._add_informations(df["shells"], "shells")

            return True
        else:
            return False

    def to_pdbqt(self, fname):
        """Write all the content of the water box in a PDBQT file.

        We cannot use OpenBabel to write the PDBQT file of water molecules 
        because it is using the default AutoDock atom types (OA, HD, ...).
        But it is used for the receptor.

        Args:
            fname (str): name of the output file

        Returns:
            None

        """
        output_str = ""
        pdbqt_str = "ATOM  %5d %-4s %-3s  %4d    %8.3f%8.3f%8.3f  0.00 0.00     %6.3f %-2s\n"

        atoms = self.molecules[0].atoms
        for atom in atoms:
            x, y, z = atom["xyz"]
            output_str += pdbqt_str % (atom["i"], atom["name"], atom["resname"], atom["resnum"],
                                       x, y, z, atom["q"], atom["t"])

        # Get the index of the nex atom and residue
        i = atom["i"] + 1
        j = atom["resnum"] + 1

        # And we do it manually for the water molecules
        for key in self.molecules.keys()[1:]:
            atoms = self.molecules[key].atoms

            for atom in atoms:
                x, y, z = atom["xyz"]
                output_str += pdbqt_str % (i, atom["name"], atom["resname"], j,
                                           x, y, z, atom["q"], atom["t"])
                i += 1

            j += 1

        # ... but we add it again at the end
        output_str += "TER\n"

        with open(fname, "w") as w:
            w.write(output_str)
