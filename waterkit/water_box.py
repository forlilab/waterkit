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
from optimize import WaterOptimizer


class WaterBox():

    def __init__(self, hb_forcefield, water_model="tip3p", smooth=0.5, dielectric=-0.1465):
        self.df = {}
        self._kdtree = None
        self.molecules = {}
        self.map = None

        # Forcefields, forcefield parameters and water model
        self._hb_forcefield = hb_forcefield
        self._water_model = water_model
        self._dielectric = dielectric
        self._smooth = smooth

        # All the informations are stored into a dict of df
        columns = ['molecule_i', 'atom_i', 'molecule_j', 'atom_j']
        self.df['connections'] = pd.DataFrame(columns=columns)
        columns = ['shell_id']
        self.df['shells'] = pd.DataFrame(columns=columns)
        columns = ['molecule_i', 'atom_i']
        self.df['kdtree_relations'] = pd.DataFrame(columns=columns)
        self.df['profiles'] = pd.DataFrame()

    def copy(self):
        """Return deepcopy of WaterBox."""
        return copy.deepcopy(self)

    def add_receptor(self, receptor, ad_map):
        """Add the receptor and the corresponding ad_map to the waterbox."""
        if not 0 in self.molecules:
            # Find all the HBA and disordered atoms if necessary
            if receptor.hydrogen_bond_anchors is None:
                receptor.guess_hydrogen_bond_anchors(self._hb_forcefield)

            if receptor.rotatable_bonds is None:
                receptor.guess_rotatable_bonds()

            # Add the receptor/map to the waterbox
            self.add_molecules(receptor)
            self.map = ad_map.copy()
            # Add informations about the receptor
            data = pd.DataFrame([[0]], columns=['shell_id'])
            self.add_informations(data, 'shells')

            return True
        else:
            # The receptor was already added to the waterbox
            return False

    def add_molecules(self, molecules, connections=None, add_KDTree=True):
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
            last_connections = self.df['connections'].tail(1)
            last_molecule_i = last_connections['molecule_i'].values[0]
            last_molecule_j = last_connections['molecule_j'].values[0]
        except:
            last_molecule_i = -1
            last_molecule_j = 0

        connections['molecule_i'] += last_molecule_i + 1
        connections['molecule_j'] += last_molecule_j + 1
        self.add_informations(connections, 'connections')

    def _add_molecules_to_kdtree(self, molecules):
        """ Build or update the cKDTree of all the atoms in
        the water box for quick nearest-neighbor lookup
        """
        if not isinstance(molecules, (list, tuple)):
            molecules = [molecules]

        try:
            last_kdtree_relations = self.df['kdtree_relations'].tail(1)
            last_molecule_i = last_kdtree_relations['molecule_i'].values[0]
        except:
            # We initliaze at -1, make first molecule at index 0
            last_molecule_i = -1

        data = []
        relations = []

        for index, molecule in enumerate(molecules):
            coordinates = molecule.coordinates()
            mol_i = index + last_molecule_i + 1
            relations.append([[mol_i, i] for i in range(coordinates.shape[0])])
            data.append(coordinates)

        # Update the KDTree relation database
        columns = ['molecule_i', 'atom_i']
        relations = np.vstack(relations)
        relations = pd.DataFrame(relations, columns=columns)
        self.add_informations(relations, 'kdtree_relations')

        # Update the KDTree
        data = np.vstack(data)
        try:
            data = np.concatenate((self._kdtree.data, data))
        except:
            pass
        self._kdtree = spatial.cKDTree(data)

    def add_informations(self, data, where):
        """ Append DF to the existing information DF """
        try:
            self.df[where] = self.df[where].append(data, sort=False)
            self.df[where].reset_index(drop=True, inplace=True)
        except:
            print "Error: Cannot add informations to %s dataframe." % where

    def update_informations_in_shell(self, data, shell_id, key):
        """Update shell information."""
        index = self.df['shells']['shell_id'] == shell_id
        self.df['shells'].loc[index, key] = data

    def molecules_in_shell(self, shell_ids=None):
        """ Get all the molecule in shell """
        if shell_ids is not None:
            if not isinstance(shell_ids, (list, tuple)):
                shell_ids = [shell_ids]
            df = self.df['shells'][self.df['shells']['shell_id'].isin(shell_ids)]
        else:
            df = self.df['shells']

        molecules = [self.molecules[i] for i in df.index.tolist()]

        return molecules

    def closest_atoms(self, xyz, radius, exclude=None):
        """ Retrieve indices of the closest atoms around x 
        at a certain radius """
        if self._kdtree is None:
            print "Warning: KDTree is empty."
            return pd.DataFrame(columns=['molecule_i', 'atom_i'])

        index = self._kdtree.query_ball_point(xyz, radius)
        df = self.df['kdtree_relations'].loc[index]

        if exclude is not None:
            if not isinstance(exclude, (list, tuple)):
                exclude = [exclude]
            df = df[-df['molecule_i'].isin(exclude)]

        return df

    def molecule_informations_in_shell(self, shell_id):
        """Get information of shell."""
        df = self.df['shells']
        # Return a copy to avoid a SettingWithCopyWarning flag
        return df.loc[df['shell_id'] == shell_id].copy()

    def number_of_shells(self):
        """Total number of shells in the WaterBox."""
        # df['column'].max() faster than np.max(df['column'])
        return self.df['shells']['shell_id'].max()

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
                    hba = self.molecules[row["molecule_i"]].hydrogen_bond_anchors[row["atom_i"]]
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

    def place_optimal_spherical_waters(self, molecules, atom_type='OW', partial_charge=-0.834):
        """Place spherical water molecules in the optimal position.

        Args:
            molecules (Molecule): molecules on which spherical water molecules will be placed
            atom_type (str): atom type of the spherical water molecules (default: OW)
            partial_charges (float): partial charges of the spherical water molecules (default: -0.834)

        Returns:
            waters (list): list of water molecules
            connections (DataFrame): contains connections between molecules

        """
        waters = []
        data = []

        for i, molecule in enumerate(molecules):
            if molecule.hydrogen_bond_anchors is None:
                molecule.guess_hydrogen_bond_anchors(self._hb_forcefield)

            for index, row in molecule.hydrogen_bond_anchors.iterrows():
                # Add water molecule only if it's in the map
                if self.map.is_in_map(row.vector_xyz):
                    anchor_xyz = molecule.coordinates(row.atom_i)[0]
                    w = Water(row.vector_xyz, atom_type, partial_charge, anchor_xyz, row.vector_xyz, row.anchor_type)
                    
                    waters.append(w)
                    data.append((i, row.atom_i, len(waters) - 1, None))

        # Convert list of tuples into dataframe
        columns = ['molecule_i', 'atom_i', 'molecule_j', 'atom_j']
        connections = pd.DataFrame(data, columns=columns)

        return (waters, connections)

    def build_next_shell(self, how='boltzmann', temperature=300.):
        """Build the next hydration shell.
        
        Args:
            how (str): method used to place water molecules (choice: best, boltzmann)(default: boltzmann)
            temperature (float): sampling temperature in Kelvin (default: 300)

        Returns:
            bool: True if water molecules were added or False otherwise

        """
        type_ow = 'OW'
        partial_charge = -0.834
        shell_id = self.number_of_shells()
        molecules = self.molecules_in_shell(shell_id)

        # Test if we have all the material to continue
        assert len(molecules) > 0, "There is molecule(s) in the shell %s" % shell_id

        wopt = WaterOptimizer(self, how, angle=110, temperature=temperature)

        waters, connections = self.place_optimal_spherical_waters(molecules, type_ow, partial_charge)

        # Only the receptor contains disordered hydrogens
        if shell_id == 0:
            waters, df = wopt.optimize_grid(waters, connections, opt_disordered=True)
        else:
            """After the first hydration layer, we don't care anymore about 
            connections. It was only useful for the disordered hydrogen atoms.
            """
            waters, df = wopt.optimize_grid(waters, opt_disordered=False)

        if len(waters):
            self.add_molecules(waters, add_KDTree=False)
            # Add informations about the new shell
            if "shells" in df.keys():
                self.add_informations(df["shells"], "shells")

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
        i = 1
        j = 1
        pdbqt_options = "rcp"
        output_str = ""
        pdbqt_str = "ATOM  %5d  %-3s HOH  %4d    %8.3f%8.3f%8.3f  0.00 0.00     %6.3f %2s\n"

        obconv = ob.OBConversion()
        obconv.SetOutFormat("pdbqt")

        for option in pdbqt_options:
            obconv.AddOption(option)

        # We use OpenBabel for the receptor
        # By default we remove the TER keyword...
        output_str = obconv.WriteString(self.molecules[0]._OBMol)[:-5]

        # And we do it manually for the water molecules
        for key in self.molecules.keys()[1:]:
            df = self.molecules[key].atom_informations()

            for row in df.itertuples():
                output_str += pdbqt_str % (i, row.t[0], j, row.x, row.y, row.z, row.q, row.t)
                i += 1

            j += 1

        # ... but we add it again at the end
        output_str += 'TER\n'

        with open(fname, 'w') as w:
            w.write(output_str)
