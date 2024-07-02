#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# get GAFF2 parameters for a molecule
#

import argparse
import os
import shutil
import sys
from pathlib import Path

import numpy as np
from rdkit import Chem

from waterkit import utils


TLEAP_TEMPLATE = """
source leaprc.protein.ff14SB
source leaprc.DNA.bsc1
source leaprc.water.tip3p
source leaprc.%(gaff_version)s
loadamberparams out.frcmod
%(resname)s = loadmol2 out.mol2
check %(resname)s
saveoff %(resname)s out.lib
quit
"""


def _transfer_coordinates_from_pdb_to_mol2(pdb_filename, mol2_filename, new_mol2_filename=None):
    """ Transfer coordinates from pdb to mol2 filename. 
    I neither trust RDKit or OpenBabel for doing that...
    """
    i = 0
    pdb_coordinates = []
    output_mol2 = ""
    coordinate_flag = False

    # Get all the coordinates from the pdb
    with open(pdb_filename) as f:
        lines = f.readlines()

        for line in lines:
            x = line[31:39].strip()
            y = line[39:47].strip()
            z = line[47:54].strip()
            pdb_coordinates.append([x, y, z])

    pdb_coordinates = np.array(pdb_coordinates, dtype=float)

    # We read the mol2 file and modify each atom line
    with open(mol2_filename) as f:
        lines = f.readlines()

        for line in lines:
            # Stop reading coordinates
            # '@<TRIPOS>SUBSTRUCTURE' in case there is only one atom...
            # But who would do this?!
            if '@<TRIPOS>BOND' in line or '@<TRIPOS>SUBSTRUCTURE' in line:
                coordinate_flag = False

            if coordinate_flag:
                x, y, z = pdb_coordinates[i]
                new_line = line[0:17] + "%10.4f %10.4f %10.4f " % (x, y, z) + line[50:]
                output_mol2 += new_line
                i += 1
            else:
                output_mol2 += line

            # It's time to read the coordinates
            if '@<TRIPOS>ATOM' in line:
                coordinate_flag = True

    # Write the new mol2 file
    if new_mol2_filename is None:
        with open(mol2_filename, 'w') as w:
            w.write(output_mol2)
    else:
        with open(new_mol2_filename, 'w') as w:
            w.write(output_mol2)


def run_antechamber(mol_filename, molecule_name, resname, charge=0, charge_method="bcc", gaff_version="gaff2", output_directory='.'):
    """Run antechamber.
    """
    original_mol_filename = os.path.abspath(mol_filename)
    local_mol_filename = os.path.basename(mol_filename)
    cwd_dir = os.path.abspath(output_directory)
    output_mol2_filename = cwd_dir + os.path.sep + '%s.mol2' % molecule_name
    output_frcmod_filename = cwd_dir + os.path.sep + '%s.frcmod' % molecule_name
    output_lib_filename = cwd_dir + os.path.sep + '%s.lib' % molecule_name

    with utils.temporary_directory(prefix=molecule_name, dir='.') as tmp_dir:
        shutil.copy2(original_mol_filename, local_mol_filename)

        # Run Antechamber
        # We output a mol2 (and not a mol file) because the mol2 contains the GAFF atom types (instead of SYBYL)
        cmd = "antechamber -i %s -fi mdl -o out.mol2 -fo mol2 -s 2 -at %s -c %s -nc %d -rn %s"
        cmd = cmd % (local_mol_filename, gaff_version, charge_method, charge, resname)
        ante_outputs, ante_errors = utils.execute_command(cmd)

        # Run parmchk2 for the additional force field file
        cmd = 'parmchk2 -i out.mol2 -f mol2 -o out.frcmod -s %s' 
        cmd = cmd % (gaff_version)
        parm_outputs, parm_errors = utils.execute_command(cmd)

        # Run tleap for the library file
        with open('tleap.cmd', 'w') as w:
            w.write(TLEAP_TEMPLATE % {'gaff_version': gaff_version, 
                                      'molecule_name': molecule_name,
                                      'resname': resname
                                      })
        cmd = 'tleap -s -f tleap.cmd'
        tleap_outputs, tleap_errors = utils.execute_command(cmd)

        try:
            # The final mol2 file from antechamber does not contain
            # the optimized geometry from sqm. Why?!
            _transfer_coordinates_from_pdb_to_mol2('sqm.pdb', 'out.mol2')
        except FileNotFoundError:
            # Antechamber is the primary source of error due to weird atomic valence (mol2 BONDS or charge)
            print("ERROR: Parametrization of %s failed. Check atomic valence, bonds and charge." % molecule_name)
            print("ANTECHAMBER ERROR LOG: ")
            print(ante_errors)
            print("PARMCHK2 ERROR LOG: ")
            print(parm_errors)
            print("TLEAP ERROR LOG: ")
            print(tleap_errors)
            sys.exit(1)

        # Copy back all we need
        shutil.copy('out.mol2', output_mol2_filename)
        shutil.copy('out.frcmod', output_frcmod_filename)
        shutil.copy('out.lib', output_lib_filename)

    return output_mol2_filename, output_frcmod_filename, output_lib_filename


def cmd_lineparser():
    parser = argparse.ArgumentParser(description='Generate GAFF2 parameters.')
    parser.add_argument('-i', '--mol', required=True,
        dest='mol_filename', help='MOL/SDF input file with explicit hydrogen atoms and desired protonation state.')
    parser.add_argument('-n', '--resname', default=True,
        dest='resname', help='resname of the molecule (3-letters code) as seen in the input PDB file (receptor + ligand).')
    parser.add_argument('-o', '--out', default='.',
        dest='output_directory', help='output directory (default: \'.\')')
    return parser.parse_args()


def main():
    args = cmd_lineparser()
    mol_filename = args.mol_filename
    output_directory = args.output_directory
    resname = args.resname

    assert len(resname) == 3, "Resname of the molecule must be a 3-letters code."

    molecule_name = Path(mol_filename).stem

    mol = Chem.MolFromMolFile(mol_filename)
    charge = Chem.rdmolops.GetFormalCharge(mol)

    run_antechamber(mol_filename, molecule_name, resname, charge, output_directory=output_directory)


if __name__ == '__main__':
    main()
