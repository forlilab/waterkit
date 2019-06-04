#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# WaterKit
#
# Class to manage autodock maps
#

import collections
import os
import re
import copy

import numpy as np
from scipy import spatial
from scipy.interpolate import RegularGridInterpolator

import utils


class Map():

    def __init__(self, map_files=None, labels=None):
        """Initialize a Map object by reading AutoDock map files.

        Args:
            map_files (list): list of the autodock map files
            labels (list): list of the atom types corresponding to each maps

        """
        maps = {}
        prv_grid_information = None

        if map_files is not None and labels is not None:
            if not isinstance(map_files, (list, tuple)):
                map_files = [map_files]
            if not isinstance(labels, (list, tuple)):
                labels = [labels]

            assert (len(map_files) == len(labels)), 'map files and labels must have the same number of elements'

            # Get information (center, spacing, nelements) from each grid
            # and make sure they are all identical
            for map_file, label in zip(map_files, labels):
                grid_information = self._grid_information_from_map(map_file)

                if prv_grid_information is not None:
                    if not cmp(prv_grid_information, grid_information):
                        raise Exception('grid %s is different from the previous one.' % label)

                    prv_grid_information = grid_information

                maps[label] = map_file

            self._maps = {}
            self._maps_interpn = {}
            self._spacing = grid_information['spacing']
            self._npts = grid_information['nelements']
            self._center = grid_information['center']

            # Compute min and max coordinates
            # Half of each side
            l = (self._spacing * (self._npts - 1)) / 2.
            # Minimum and maximum coordinates
            self._xmin, self._ymin, self._zmin = self._center - l
            self._xmax, self._ymax, self._zmax = self._center + l
            # Generate the kdtree and bin edges of the grid
            self._kdtree, self._edges = self._build_kdtree_from_grid()

            # Read all the affinity maps
            for label, map_file in maps.items():
                affinity_map = self._read_affinity_map(map_file)

                self._maps[label] = affinity_map
                self._maps_interpn[label] = self._generate_affinity_map_interpn(affinity_map)

    def __str__(self):
        """Print basic information about the maps"""
        try:
            info = 'SPACING %s\n' % self._spacing
            info += 'NELEMENTS %s\n' % ' '.join(self._npts.astype(str))
            info += 'CENTER %s\n' % ' '.join(self._center.astype(str))
            info += 'MAPS %s\n' % ' '.join(self._maps.iterkeys())
        except AttributeError:
            info = 'AutoDock Map object is not defined.'

        return info

    def copy(self):
        """Create a deep copy the current state of the Map object.

        Returns:
            Map: Deep copy of the map

        """
        return copy.deepcopy(self)

    @classmethod
    def from_fld(cls, fld_file):
        """Read a fld file.

        The AutoDock map files are read using the information contained
        into the fld file. The fld file is created by AutoGrid.

        Args:
            fld_file (str): pathname of the AutoGrid fld file.

        Returns:
            Map: Instance of Map object.

        """
        map_files = []
        labels = []

        path = os.path.dirname(fld_file)
        # If there is nothing, it means we are in the current directory
        if path == '':
            path = '.'

        with open(fld_file) as f:
            for line in f:
                if re.search('^label=', line):
                    labels.append(line.split('=')[1].split('#')[0].split('-')[0].strip())
                elif re.search('^variable', line):
                    map_files.append(path + os.sep + line.split(' ')[2].split('=')[1].split('/')[-1])

        if map_files and labels:
            return cls(map_files, labels)

    def _grid_information_from_map(self, map_file):
        """Read grid information in the map file"""
        grid_information = {'spacing': None,
                            'nelements': None,
                            'center': None}

        with open(map_file) as f:
            for line in f:
                if re.search('^SPACING', line):
                    grid_information['spacing'] = np.float(line.split(' ')[1])
                elif re.search('^NELEMENTS', line):
                    nelements = np.array(line.split(' ')[1:4], dtype=np.int)
                    # Transform even numbers to the nearest odd integer
                    nelements = nelements // 2 * 2 + 1
                    grid_information['nelements'] = nelements
                elif re.search('CENTER', line):
                    grid_information['center'] = np.array(line.split(' ')[1:4], dtype=np.float)
                elif re.search('^[0-9]', line):
                    # If the line starts with a number, we stop
                    break

        return grid_information

    def _read_affinity_map(self, map_file):
        """
        Take a grid file and extract gridcenter, spacing and npts info
        """
        with open(map_file) as f:
            # Read all the lines directly
            lines = f.readlines()

            npts = np.array(lines[4].split(' ')[1:4], dtype=np.int) + 1

            # Get the energy for each grid element
            affinity = [np.float(line) for line in lines[6:]]
            # Some sorceries happen here --> swap x and z axes
            affinity = np.swapaxes(np.reshape(affinity, npts[::-1]), 0, 2)

        return affinity

    def _generate_affinity_map_interpn(self, affinity_map):
        """
        Return a interpolate function from the grid and the affinity map.
        This helps to interpolate the energy of coordinates off the grid.
        """
        return RegularGridInterpolator(self._edges, affinity_map, bounds_error=False, fill_value=np.inf)

    def _build_kdtree_from_grid(self):
        """
        Return the kdtree and bin edges (x, y, z) of the AutoDock map
        """
        x = np.linspace(self._xmin, self._xmax, self._npts[0])
        y = np.linspace(self._ymin, self._ymax, self._npts[1])
        z = np.linspace(self._zmin, self._zmax, self._npts[2])

        # We use a tuple of numpy arrays and not a complete numpy array
        # because otherwise the output will be different if the grid is cubic or not
        edges = tuple([x, y, z])

        X, Y, Z = np.meshgrid(x, y, z)
        xyz = np.stack((X.ravel(), Y.ravel(), Z.ravel()), axis=-1)
        kdtree = spatial.cKDTree(xyz)

        return kdtree, edges

    def _index_to_cartesian(self, idx):
        """
        Return the cartesian grid coordinates associated to the grid index
        """
        return self._kdtree.mins + idx * self._spacing

    def _cartesian_to_index(self, xyz):
        """
        Return the closest grid index of the cartesian grid coordinates
        """
        idx = np.rint((xyz - self._kdtree.mins) / self._spacing).astype(np.int)
        # All the index values outside the grid are clipped (limited) to the nearest index
        np.clip(idx, [0, 0, 0], self._npts, idx)

        return idx

    def size(self):
        """Return the number of grid points in the grid.

        Returns:
            int: number of grid points

        """
        return self._npts.prod()

    def create_empty_map(self, name, fill_value=None):
        """Initialize an empty map

        Args:
            name (str): name of the new map
            fill_value (float): fill value (default: None)

        Returns:
            bool: True if succeeded or False otherwise

        """
        if not name in self._maps:
            if value is None:
                new_map = np.zeros(self._npts)
            else:
                new_map = np.full(self._npts, fill_value)

            self._maps[label] = new_map
            self._maps_interpn[label] = self._generate_affinity_map_interpn(new_map)

            return True
        else:
            print "Error: map %s already exists." % name
            return False

    def atoms_in_map(self, molecule):
        """List of index of all the atoms in the map.

        Args:
            molecule (molecule): Molecule object

        Returns:
            list: atom indexes

        """
        idx = []
        OBMol = molecule._OBMol

        for ob_atom in ob.OBMolAtomIter(OBMol):
            x, y, z = ob_atom.GetX(), ob_atom.GetY(), ob_atom.GetZ()

            if self.is_in_map([x, y, z]):
                idx.append(ob_atom.GetIdx())

        return idx

    def residues_in_map(self, molecule):
        """List of index of all the residues in the map.

        Args:
            molecule (Molecule): Molecule object

        Returns:
            list: residue indexes

        """
        idx = []
        OBMol = molecule._OBMol

        for ob_residue in ob.OBResidueIter(OBMol):
            for ob_atom in ob.OBResidueAtomIter(ob_residue):
                x, y, z = ob_atom.GetX(), ob_atom.GetY(), ob_atom.GetZ()

                # If at least one atom (whatever the type) is in the grid, add the residue
                if self.is_in_map([x, y, z]):
                    idx.append(ob_residue.GetIdx())
                    break

        return idx

    def is_in_map(self, xyz):
        """Check if coordinates are in the map.

        Args:
            xyz (ndarray): 2d Numpy array of the coordinates

        Returns:
            ndarray: 1d Numpy array of boolean

        """
        xyz = np.atleast_2d(xyz)
        x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]

        x_in = np.logical_and(self._xmin <= x, x <= self._xmax)
        y_in = np.logical_and(self._ymin <= y, y <= self._ymax)
        z_in = np.logical_and(self._zmin <= z, z <= self._zmax)
        all_in = np.all((x_in, y_in, z_in), axis=0)
        # all_in = np.logical_and(np.logical_and(x_in, y_in), z_in)

        return all_in

    def is_close_to_edge(self, xyz, distance):
        """Check if the points xyz is at a certain distance of the edge of the box.

        Args:
            xyz (ndarray): 2n Numpy array of the coordinates
            distance (float): distance

        Returns:
            ndarray: 1d Numpy array of boolean

        """
        xyz = np.atleast_2d(xyz)
        x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]

        x_close = np.logical_or(np.abs(self._xmin - x) <= distance, np.abs(self._xmax - x) <= distance)
        y_close = np.logical_or(np.abs(self._ymin - y) <= distance, np.abs(self._ymax - y) <= distance)
        z_close = np.logical_or(np.abs(self._zmin - z) <= distance, np.abs(self._zmax - z) <= distance)
        close_to = np.any((x_close, y_close, z_close), axis=0)

        return close_to

    def energy_coordinates(self, xyz, atom_type, method='linear'):
        """Grid energy of each coordinates xyz.

        Args:
            xyz (ndarray): 2d Numpy array of coordinates
            atom_type (str): name of the atom type
            method (str): Interpolate method (default: linear)

        Returns:
            ndarray: 1d Numpy array of the energy values

        """
        return self._maps_interpn[atom_type](xyz, method=method)

    def energy(self, df, ignore_atom_types=None, ignore_electrostatic=False, 
               ignore_desolvation=False, method='linear'):
        """Get energy interaction of a molecule based of the grid.

        Args:
            df (DataFrame): Pandas DatFrame with columns ("i", "x", "y", "z", "q", "t")
            ignore_atom_types (list): list of atom types/terms to ignore (default: None)
            ignore_electrostatic (bool): to ignore electrostatic term (default: False)
            ignore_desolvation (bool): to ignore desolvation term (default: False)
            method (str): Interpolate method (default: linear)

        Returns:
            float: Grid energy interaction

        """
        energy = 0.
        elec = 0.
        desolv = 0.

        if ignore_atom_types is None:
            ignore_atom_types = []

        if not isinstance(ignore_atom_types, (list, tuple)):
            ignore_atom_types = [ignore_atom_types]

        se = df.groupby('t', as_index=False)['x', 'y', 'z', 'q'].agg(lambda x: list(x)).values

        for atom_type, x, y, z, q in se:
            xyz = np.stack([x, y, z], axis=1)

            if not atom_type in ignore_atom_types:
                vdw_hb = self._maps_interpn[atom_type](xyz, method=method)

            if not ignore_electrostatic:
                elec = self._maps_interpn['Electrostatics'](xyz, method=method) * np.array(q)

            if not ignore_desolvation:
                desolv = self._maps_interpn['Desolvation'](xyz, method=method)

            energy += np.sum(vdw_hb + elec + desolv)

        return energy

    def neighbor_points(self, xyz, radius, min_radius=0):
        """Grid coordinates around a point at a certain distance.

        Args:
            xyz (array_like): 3D coordinates of a point
            radius (float): max radius
            min_radius (float): min radius (default: 0)

        Returns:
            ndarray: 2D Numpy array of coordinates

        """
        coordinates = self._kdtree.data[self._kdtree.query_ball_point(xyz, radius)]
        
        if min_radius > 0:
            distance = spatial.distance.cdist([xyz], coordinates, 'euclidean')[0]
            coordinates = coordinates[distance >= min_radius]

        return coordinates

    def apply_operation_on_maps(self, names, atom_types, expression):
        """Apply mathematical expression on affinity grids.

        Args:
            names (list): name of the new or existing grid
            atom_types (list): list of atom types
            expression (str): maths expression that must contains x (x is the grid value)

        Returns:
            None

        """
        if not isinstance(names, (list, tuple)):
            names = [names]

        if not isinstance(atom_types, (list, tuple)):
            atom_types = [atom_types]

        assert len(atom_types) == len(names), "Names and atom_types lengths are not matching."

        if not 'x' in expression:
            print "Error: operation cannot be applied, x is not defined."
            return None

        for name, atom_type in zip(names, atom_types):
            try:
                x = self._maps[atom_type]
                # When 'eval' a new copy of the map is created
                x = eval(expression)

                # Update map and interpolator
                self._maps[name] = x
                self._maps_interpn[name] = self._generate_affinity_map_interpn(x)
            except:
                print "Warning: This map %s does not exist." % (atom_type)
                continue

    def combine(self, name, atom_types, how='best', ad_map=None):
        """Funtion to combine Autoock map together.

        Args:
            name (str): name of the new or existing grid
            atom_types (list): list of atom types combined
            how (str)): combination methods: best, add, replace (default: best)
            ad_map (Map): another Map object (default: None)

        Returns:
            bool: True if succeeded or False otherwise

        """
        selected_maps = []
        same_grid = True
        indices = np.index_exp[:, :, :]

        if not isinstance(atom_types, (list, tuple)):
            atom_types = [atom_types]

        if how == 'replace':
            assert ad_map is not None, "Another map has to be specified for the replace mode."
            assert len(atom_types) == 1, "Multiple atom types cannot replace the same atom type."

        if name not in self._maps:
            self._maps[name] = np.zeros(self._npts)

        """ Get the common maps, the requested ones and the actual ones that we have 
        and check maps that we cannot process because there are not present in one
        of the ad_map.
        """
        selected_types = set(self._maps.keys()) & set(atom_types)
        if ad_map is not None:
            selected_types = set(selected_types) & set(ad_map._maps.keys())
        unselected_types = set(selected_types) - set(atom_types)

        if not selected_types:
            print "Warning: no maps were selected from %s list." % atom_types
            return False
        if unselected_types:
            print "Warning: %s maps can't be combined." % ' '.join(unselected_types)
        
        """ Check if the grid are the same between the two ad_maps.
        And we do it like this because grid are tuples of numpy array.
        """
        if ad_map is not None:
            same_grid = all([np.array_equal(x, y) for x, y in zip(self._edges, ad_map._edges)])

        if ad_map is not None and same_grid is False:
            """ If the grids are not the same between the two ad_maps, we have to
            retrieve the indices of the self map (reference) that corresponds to the
            cordinates of the ad_map.
            """
            ix_min, iy_min, iz_min = self._cartesian_to_index([ad_map._xmin, ad_map._ymin, ad_map._zmin])
            ix_max, iy_max, iz_max = self._cartesian_to_index([ad_map._xmax, ad_map._ymax, ad_map._zmax])

            x = self._edges[0][ix_min:ix_max]
            y = self._edges[1][iy_min:iy_max]
            z = self._edges[2][iz_min:iz_max]

            X, Y, Z = np.meshgrid(x, y, z)
            grid = np.stack((X.ravel(), Y.ravel(), Z.ravel()), axis=-1)
            
            """ All the coordinates outside ad_map are clipped, to avoid inf value
            during the interpolation. This is not the best way of doing that 
            because because we lose the exact correspondance between the index 
            and the coordinates in the self map.
            """
            np.clip(grid, [ad_map._xmin, ad_map._ymin, ad_map._zmin], 
                    [ad_map._xmax, ad_map._ymax, ad_map._zmax], grid)

            indices = np.index_exp[ix_min:ix_max, iy_min:iy_max, iz_min:iz_max]

        # Select maps
        for selected_type in selected_types:
            if how != 'replace':
                selected_maps.append(self._maps[selected_type][indices])

            if ad_map is not None:
                if same_grid:
                    selected_maps.append(ad_map._maps[selected_type])
                else:
                    energy = ad_map.energy_coordinates(grid, selected_type)
                    # Reshape and swap x and y axis, right? Easy.
                    # Thank you Diogo Santos Martins!!
                    energy = np.reshape(energy, (y.shape[0], x.shape[0], z.shape[0]))
                    energy = np.swapaxes(energy, 0, 1)

                    selected_maps.append(energy)

        if selected_types:
            # Combine all the maps
            if how == 'best':
                self._maps[name][indices] = np.nanmin(selected_maps, axis=0)
            elif how == 'add':
                self._maps[name][indices] = np.nansum(selected_maps, axis=0)
            elif how == 'replace':
                self._maps[name][indices] = selected_maps[0]

            # Update the interpolate energy function
            self._maps_interpn[name] = self._generate_affinity_map_interpn(self._maps[name])

        return True

    def to_pdb(self, fname, map_type, max_energy=None):
        """Export AutoDock map in PDB format.

        Args:
            fname (str): PDB file pathname
            mapt_type (str): atom type name
            max_energy (float): max limit energy (default: None)

        Returns:
            None

        """
        idx = np.array(np.where(self._maps[map_type] <= max_energy)).T

        i = 0
        line = "ATOM  %5d  D   DUM Z%4d    %8.3f%8.3f%8.3f  1.00%6.2f           D\n"

        with open(fname, 'w') as w:
            for j in range(idx.shape[0]):

                v = self._maps[atom_type][idx[j][0], idx[j][1], idx[j][2]]

                if v > 999.99:
                    v = 999.99

                w.write(line % (i, i, self._edges[0][idx[j][0]], self._edges[1][idx[j][1]], self._edges[2][idx[j][2]], v))
                i += 1

    def to_map(self, map_types=None, prefix=None, grid_parameter_file='grid.gpf',
               grid_data_file='maps.fld', macromolecule='molecule.pdbqt'):
        """Export AutoDock maps.

        Args:
            map_types (list): list of atom types to export
            prefix (str): prefix name file (default: None)
            grid_parameter_file (str): name of the gpf file (default: grid.gpf)
            grid_data_file (str): name of the fld file (default: maps.fld)
            macromolecule (str): name of the receptor (default: molecule.pdbqt)

        Returns:
            None

        """
        if map_types is None:
            map_types = self._maps.keys()
        elif not isinstance(map_types, (list, tuple)):
            map_types = [map_types]

        for map_type in map_types:
            if map_type in self._maps:
                filename = '%s.map' % map_type
                if prefix is not None:
                    filename = '%s.%s' % (prefix, filename)

                with open(filename, 'w') as w:
                    npts = np.array([n if not n % 2 else n - 1 for n in self._npts])

                    # Write header
                    w.write('GRID_PARAMETER_FILE %s\n' % grid_parameter_file)
                    w.write('GRID_DATA_FILE %s\n' % grid_data_file)
                    w.write('MACROMOLECULE %s\n' % macromolecule)
                    w.write('SPACING %s\n' % self._spacing)
                    w.write('NELEMENTS %s\n' % ' '.join(npts.astype(str)))
                    w.write('CENTER %s\n' % ' '.join(self._center.astype(str)))
                    # Write grid (swap x and z axis before)
                    m = np.swapaxes(self._maps[map_type], 0, 2).flatten()
                    w.write('\n'.join(m.astype(str)))
            else:
                print 'Error: Map %s does not exist.' % map_type
