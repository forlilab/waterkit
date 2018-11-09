#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# WaterKit
#
# AutoDock ForceField
#

import argparse
import imp
import os
import re

import pandas as pd
import numpy as np

import utils


class AutoDockForceField():
    def __init__(self, parameter_file=None, hb_cutoff=8, elec_cutoff=20, weighted=True):
        if parameter_file is None:
            d = imp.find_module('waterkit')[1]
            parameter_file = os.path.join(d, 'data/AD4_parameters.dat')

        self.weights = self._set_weights(parameter_file, weighted)
        self.atom_par = self._set_atom_parameters(parameter_file)
        self.pairwise = self._build_pairwise_table()

        # Parameters
        self.hb_cutoff = hb_cutoff
        self.elec_cutoff = elec_cutoff

        # VdW and hydrogen bond
        self.smooth = 0.5

        # Desolvation constants
        self.desolvation_k = 0.01097

        # Dielectric constants
        self.dielectric_epsilon = 78.4
        self.dielectric_A = -8.5525
        self.dielectric_B = self.dielectric_epsilon - self.dielectric_A
        self.dielectric_lambda = 0.003627
        self.dielectric_k = 7.7839
        self.elec_scale = 332.06363

    def _set_weights(self, parameter_file, weighted=True):
        columns = []
        data = []

        with open(parameter_file) as f:
            for line in f.readlines():
                if re.search('^FE_coeff', line):
                    line = line.split()
                    columns.append(line[0].split('_')[2])

                    if weighted:
                        data.append(np.float(line[-1]))
                    else:
                        data.append(1.0)

        weights = pd.DataFrame(data=[data], columns=columns)
        return weights

    def _set_atom_parameters(self, parameter_file):
        columns = ['type', 'rii', 'epsii', 'vol', 'solpar', 'rij_hb',
                   'epsij_hb', 'hbond', 'rec_index', 'map_index',
                   'bond_index']
        data = []

        with open(parameter_file) as f:
            for line in f.readlines():
                if re.search('^atom_par', line):
                    line = line.split()
                    data.append((line[1], np.float(line[2]), np.float(line[3]), 
                                 np.float(line[4]), np.float(line[5]), np.float(line[6]), 
                                 np.float(line[7]), np.int(line[8]), np.int(line[9]), 
                                 np.int(line[10]), np.int(line[11])))
        
        atom_par = pd.DataFrame(data=data, columns=columns)
        atom_par.set_index('type', inplace=True)
        return atom_par

    def load_intnbp_r_eps_from_dpf(self, dp_file):
        """Load intnbp_r_eps from dpf file."""
        pass

    def _coefficient(self, epsilon, reqm, a, b):
        """Compute coefficients."""
        return (a / np.abs(a - b)) * epsilon * reqm**b

    def _build_pairwise_table(self):
        """Pre-compute all pairwise interactions."""
        columns = ['i', 'j', 'vdw_rij', 'vdw_epsij', 'A', 
                   'B', 'hb_rij', 'hb_epsij', 'C', 'D']
        data = []

        for i in self.atom_par.index:
            for j in self.atom_par.index:
                # vdW
                vdw_rij = (self.atom_par.loc[i]['rii'] + self.atom_par.loc[j]['rii']) / 2.
                vdw_epsij = np.sqrt(self.atom_par.loc[i]['epsii'] * self.atom_par.loc[j]['epsii'])
                a = self._coefficient(vdw_epsij, vdw_rij, 6, 12)
                b = self._coefficient(vdw_epsij, vdw_rij, 12, 6)

                # Hydrogen bond
                hbond_i = self.atom_par.loc[i]['hbond']
                hbond_j = self.atom_par.loc[j]['hbond']

                if hbond_i in [1, 2] and hbond_j in [3, 4, 5]:
                    hb_rij = self.atom_par.loc[j]['rij_hb']
                    hb_epsij = self.atom_par.loc[j]['epsij_hb']
                elif hbond_i in [3, 4, 5] and hbond_j in [1, 2]:
                    hb_rij = self.atom_par.loc[i]['rij_hb']
                    hb_epsij = self.atom_par.loc[i]['epsij_hb']
                else:
                    hb_rij = 0
                    hb_epsij = 0

                c = self._coefficient(hb_epsij, hb_rij, 10, 12)
                d = self._coefficient(hb_epsij, hb_rij, 12, 10)

                data.append((i, j, vdw_rij, vdw_epsij, a, b, hb_rij, hb_epsij, c, d))

        pairwise = pd.DataFrame(data=data, columns=columns)
        pairwise.set_index(['i', 'j'], inplace=True)
        return pairwise

    def smooth_distance(self, r, reqm, smooth=0.5):
        """Smooth distance."""
        # Otherwise we changed r in-place.
        r = r.copy()
        sf = .5 * smooth
        r[((reqm - sf) < r) & (r < (reqm + sf))] = reqm
        r[r >= reqm + sf] -= sf
        r[r <= reqm - sf] += sf
        return r

    def van_der_waals(self, r, reqm, A, B):
        """Compute VdW interaction."""
        r = self.smooth_distance(r, reqm, self.smooth)
        return np.sum((A / r**12) - (B / r**6))

    def hydrogen_bond(self, r, reqm, C, D):
        """Compute hydrogen bond interaction."""
        if r <= self.hb_cutoff:
            r = self.smooth_distance(r, reqm, self.smooth)
            return np.sum((C / r**12) - (D / r**10))
        else:
            return 0.

    def distance_dependent_dielectric(self, r):
        """Distance dependent dielectric, mehler and solmajer."""
        lambda_B = -self.dielectric_lambda * self.dielectric_B
        return self.dielectric_A + self.dielectric_B / (1. + self.dielectric_k * np.exp(lambda_B * r))

    def electrostatic(self, r, qi, qj):
        """Compute electrostatic interaction."""
        if r <= self.elec_cutoff:
            r_ddd = self.distance_dependent_dielectric(r)
            return np.sum(self.elec_scale * qi * qj / (r * r_ddd))
        else:
            return 0.

    def desolvation(self, r, qi, qj, ai, aj, vi, vj, sigma=3.6):
        """Compute desolvatation interaction."""
        desolv = (ai * vj) + (aj * vi)
        desolv += (self.desolvation_k * np.abs(qi) * vj) + (self.desolvation_k * np.abs(qj) * vi)
        desolv *= np.exp(-0.5 * (r**2 / sigma**2))
        return np.sum(desolv)

    def intermolecular_energy(self, atoms_i, atoms_j, details=False):
        """Compute total interaction energy."""
        total = 0.
        total_hb = 0.
        total_vdw = 0.
        total_elec = 0.
        total_desolv = 0.

        for index_i, row_i in atoms_i.iterrows():
            hb = 0.
            vdw = 0.
            elec = 0.
            desolv = 0.

            for index_j, row_j in atoms_j.iterrows():
                r = utils.get_euclidean_distance(np.array(row_i['xyz']), np.array([row_j['xyz']]))
                pairwise = self.pairwise.loc[(index_i, index_j)]

                vdw += self.van_der_waals(r, pairwise['vdw_rij'], pairwise['A'], pairwise['B'])
                hb += self.hydrogen_bond(r, pairwise['hb_rij'], pairwise['C'], pairwise['D'])
                elec += self.electrostatic(r, row_i['q'], row_j['q'])
                desolv += self.desolvation(r, row_i['q'], row_j['q'],
                                           self.atom_par.loc[index_i]['solpar'],
                                           self.atom_par.loc[index_j]['solpar'],
                                           self.atom_par.loc[index_i]['vol'],
                                           self.atom_par.loc[index_j]['vol'])

            total_hb += self.weights['hbond'].values[0] * hb
            total_elec += self.weights['estat'].values[0] * elec
            total_vdw += self.weights['vdW'].values[0] * vdw
            total_desolv += self.weights['desolv'].values[0] * desolv
            total += total_hb + total_vdw + total_elec + total_desolv

        print "Total: ", total
        print "HB: ", total_hb
        print "VDW: ", total_vdw
        print "ELEC: ", total_elec
        print "DESOLV: ", total_desolv

        if details:
            return (total, total_hb, total_vdw, total_elec, total_desolv)
        else:
            return total

    def intermolecular_energy_rf(self, atoms_i, atoms_j, rfs, details=False):
        """Compute total interaction energy."""
        total = 0.
        total_hb = 0.
        total_vdw = 0.
        total_elec = 0.
        total_desolv = 0.

        for index_i, row_i in atoms_i.iterrows():
            hb = 0.
            vdw = 0.
            elec = 0.
            desolv = 0.

            i = 0

            for index_j, row_j in atoms_j.iterrows():
                r = utils.get_euclidean_distance(np.array(row_i['xyz']), np.array([row_j['xyz']]))
                pairwise = self.pairwise.loc[(index_i, index_j)]

                vdw += self.van_der_waals(r, pairwise['vdw_rij'], pairwise['A'], pairwise['B']) * rfs[i]
                hb += self.hydrogen_bond(r, pairwise['hb_rij'], pairwise['C'], pairwise['D']) * rfs[i]
                elec += self.electrostatic(r, row_i['q'], row_j['q'])
                desolv += self.desolvation(r, row_i['q'], row_j['q'],
                                           self.atom_par.loc[index_i]['solpar'],
                                           self.atom_par.loc[index_j]['solpar'],
                                           self.atom_par.loc[index_i]['vol'],
                                           self.atom_par.loc[index_j]['vol'])

                i += 1

            total_hb += self.weights['hbond'].values[0] * hb
            total_elec += self.weights['estat'].values[0] * elec
            total_vdw += self.weights['vdW'].values[0] * vdw
            total_desolv += self.weights['desolv'].values[0] * desolv
            total += total_hb + total_vdw + total_elec + total_desolv

        print "Total: ", total
        print "HB: ", total_hb
        print "VDW: ", total_vdw
        print "ELEC: ", total_elec
        print "DESOLV: ", total_desolv

        if details:
            return (total, total_hb, total_vdw, total_elec, total_desolv)
        else:
            return total


def cmd_lineparser():
    parser = argparse.ArgumentParser(description='waterkit')
    parser.add_argument("-i", "--atoms_i", dest="atoms_i_file", required=True,
                        action="store", help="atoms i pdbqt file")
    parser.add_argument("-j", "--atom_j", dest="atoms_j_file", required=True,
                        action="store", help="atoms j pdbqt file")
    parser.add_argument("-p", "--par", dest="parameters_file", required=True,
                        action="store", help="AutoDock parameters file")
    return parser.parse_args()

def parse_pdbqt_file(pdbqt_file):
    columns = ['xyz', 'q', 'type']
    data = []

    with open(pdbqt_file) as f:
        lines = f.readlines()
        for line in lines:
            if re.search('^HETATM', line) or re.search('^ATOM', line):
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
                chrg = float(line[68:76])
                atype = line[77:79].strip()

                data.append(([x, y, z], chrg, atype))
    
    df = pd.DataFrame(data=data, columns=columns)
    return df

def main():
    args = cmd_lineparser()
    atoms_i_file = args.atoms_i_file
    atoms_j_file = args.atoms_j_file
    parameters_file = args.parameters_file

    atoms_i = parse_pdbqt_file(atoms_i_file)
    atoms_j = parse_pdbqt_file(atoms_j_file)

    atoms_i.set_index('type', inplace=True)
    atoms_j.set_index('type', inplace=True)

    ff = AutoDockForceField(parameters_file, weighted=False)
    ff.intermolecular_energy(atoms_i, atoms_j)
    ff.pairwise.to_csv('pairwise.csv')

if __name__ == '__main__':
    main()
