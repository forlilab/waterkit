#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# prepare receptor
#

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import argparse
import copy
import logging
import math
import os
import string
import sys

import parmed as pmd
from pdb4amber import AmberPDBFixer
from pdb4amber.utils import easy_call


# Change ff14SB to ff19SB
default_force_field = """
source leaprc.protein.ff19SB
source leaprc.DNA.OL15
source leaprc.RNA.OL3
source leaprc.water.tip3p
source leaprc.gaff2
"""

leap_template = """
{force_fields}
{more_force_fields}
x = loadpdb {input_pdb}
{box_info}
{more_leap_cmds}
set default nocenter on
saveAmberParm x {prmtop} {rst7}
quit
"""

# Added CYM residue
HEAVY_ATOM_DICT = {
    'ALA': 5,
    'ARG': 11,
    'ASN': 8,
    'ASP': 8,
    'CYS': 6,
    'GLN': 9,
    'GLU': 9,
    'GLY': 4,
    'HIS': 10,
    'ILE': 8,
    'LEU': 8,
    'LYS': 9,
    'MET': 8,
    'PHE': 11,
    'PRO': 7,
    'SER': 6,
    'THR': 7,
    'TRP': 14,
    'TYR': 12,
    'VAL': 7,
    'HID': 10,
    'HIE': 10,
    'HIN': 10,
    'HIP': 10,
    'CYX': 6,
    'CYM': 6,
    'ASH': 8,
    'GLH': 9,
    'LYH': 9
}

# Global constants
# Added CYM residue
RESPROT = ('ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 'HIS',
           'ILE', 'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP',
           'TYR', 'VAL', 'HID', 'HIE', 'HIN', 'HIP', 'CYX', 'CYM', 'ASH', 'GLH',
           'LYH', 'ACE', 'NME', 'GL4', 'AS4')

RESNA = ('C', 'G', 'U', 'A', 'DC', 'DG', 'DT', 'DA', 'OHE', 'C5', 'G5', 'U5',
         'A5', 'C3', 'G3', 'U3', 'A3', 'DC5', 'DG5', 'DT5', 'DA5', 'DC3',
         'DG3', 'DT3', 'DA3' )

RESSOLV = ('WAT', 'HOH', 'AG', 'AL', 'Ag', 'BA', 'BR', 'Be', 'CA', 'CD', 'CE',
           'CL', 'CO', 'CR', 'CS', 'CU', 'CU1', 'Ce', 'Cl-', 'Cr', 'Dy', 'EU',
           'EU3', 'Er', 'F', 'FE', 'FE2', 'GD3', 'HE+', 'HG', 'HZ+', 'Hf',
           'IN', 'IOD', 'K', 'K+', 'LA', 'LI', 'LU', 'MG', 'MN', 'NA', 'NH4',
           'NI', 'Na+', 'Nd', 'PB', 'PD', 'PR', 'PT', 'Pu', 'RB', 'Ra', 'SM',
           'SR', 'Sm', 'Sn', 'TB', 'TL', 'Th', 'Tl', 'Tm', 'U4+', 'V2+', 'Y',
           'YB2', 'ZN', 'Zr')

AMBER_SUPPORTED_RESNAMES = RESPROT + RESNA + RESSOLV


def write_pdb_file(output_name, molecule,  overwrite=True, **kwargs):
    """Write PDB file

    Args:
        output_name (str): pdbqt output filename
        molecule (parmed): parmed molecule object

    """
    molecule.save(output_name, format="pdb", overwrite=overwrite, **kwargs)


def write_pdbqt_file(output_name, molecule):
    """Write PDBQT file

    Args:
        output_name (str): pdbqt output filename
        molecule (parmed): parmed molecule object

    """
    pdbqt_str = "%-6s%5d %-4s %3s %s%4d    %8.3f%8.3f%8.3f  1.00  1.00    %6.3f %-2s\n"
    output_str = ""
    chain_id = 0

    for atom in molecule.atoms:
        if len(atom.name) < 4:
            name = " %s" % atom.name
        else:
            name = atom.name

        resname = atom.residue.name
        resid = atom.residue.idx + 1

        # OpenBabel does not like atom types starting with a number
        if atom.type[0].isdigit():
            atom_type = atom.type[::-1]
        else:
            atom_type = atom.type

        # AutoGrid does not accept atom type name of length > 2
        atom_type = atom_type[:2]

        if resname in RESSOLV:
            atype = "HETATM"
        else:
            atype = "ATOM"

        output_str += pdbqt_str % (atype, atom.idx + 1, name, resname, string.ascii_uppercase[chain_id],
                                   resid, atom.xx, atom.xy, atom.xz, atom.charge, atom_type)

        if name.strip() == "OXT":
            chain_id += 1
            output_str += "TER\n"

    if name.strip() != "OXT":
        output_str += "TER\n"
    output_str += "END\n"

    with open(output_name, "w") as w:
        w.write(output_str)


def convert_amber_to_autodock_types(molecule):
    molecule = copy.deepcopy(molecule)

    amber_autodock_dict = {
        'N3': 'N',
        'H': 'HD',
        'CX': 'C',
        'HP': 'H',
        'CT': 'C',
        'HC': 'H',
        'C': 'C',
        'O': 'OA',
        'N': 'N',
        'H1': 'H',
        'C3': 'C',
        '3C': 'C',
        'C2': 'C',
        '2C': 'C',
        'CO': 'C',
        'O2': 'OA',
        'OH': 'OA',
        'HO': 'HD',
        'SH': 'SA',
        'HS': 'HD',
        'CA': 'A',
        'HA': 'H',
        'S': 'SA',
        'C8': 'C',
        'N2': 'N',
        'CC': 'A',
        'NB': 'NA',
        'CR': 'A',
        'CV': 'A',
        'H5': 'H',
        'NA': 'N',
        'CW': 'A',
        'H4': 'H',
        'C*': 'A',
        'CN': 'A',
        'CB': 'A',
        'Zn2+': 'Zn',
        'XC': 'C'
    }

    for atom in molecule.atoms:
        if atom.residue.name == 'TYR' and atom.name == 'CZ' and atom.type == 'C':
            atom.type = 'A'
        elif atom.residue.name == 'ARG' and atom.name == 'CZ' and atom.type == 'CA':
            atom.type = 'C'
        else:
            atom.type = amber_autodock_dict[atom.type]

    return molecule


def _make_leap_template(parm, ns_names, gaplist, sslist, input_pdb,
                        prmtop='prmtop', rst7='rst7'):
    # box
    box = parm.box
    if box is not None:
        box_info = 'set x box { %s  %s  %s }' % (box[0], box[1], box[2])
    else:
        box_info = ''

    # Now we can assume that we are dealing with AmberTools16:
    more_force_fields = ''

    for res in ns_names:
        more_force_fields += '%s = loadmol2 %s.mol2\n' % (res, res)
        more_force_fields += 'loadAmberParams %s.frcmod\n' % res

    #  more_leap_cmds
    more_leap_cmds = ''
    if gaplist:
        for d, res1, resid1, res2, resid2 in gaplist:
            more_leap_cmds += 'deleteBond x.%d.C x.%d.N\n' % (resid1 + 1, resid2 + 1)

    #  process sslist
    if sslist:
        for resid1, resid2 in sslist:
            more_leap_cmds += 'bond x.%d.SG x.%d.SG\n' % (resid1+1, resid2+1)

    leap_string = leap_template.format(
        force_fields=default_force_field,
        more_force_fields=more_force_fields,
        box_info=box_info,
        input_pdb=input_pdb,
        prmtop=prmtop,
        rst7=rst7,
        more_leap_cmds=more_leap_cmds)
    return leap_string


def find_alt_residues(molecule):
    alt_residues = set()
        
    for residue in molecule.residues:
        chains.add(residue.chain)
        for atom in residue.atoms:
            if atom.other_locations:
                alt_residues.add(residue)

    return alt_residues


def remove_alt_residues(molecule):
    # remove altlocs label
    residue_collection = []

    for residue in molecule.residues:
        for atom in residue.atoms:
            atom.altloc = ''
            for oatom in atom.other_locations.values():
                oatom.altloc = ''
                residue_collection.append(residue)

    residue_collection = list(set(residue_collection))
    return residue_collection


def find_gaps(molecule, resprot):
    # TODO: doc
    # report original resnum?
    CA_atoms = []
    C_atoms = []
    N_atoms = []
    gaplist = []

    # N.B.: following only finds gaps in protein chains!
    # H.N: Assume that residue has all 3 atoms: CA, C, and N
    respro_nocap = set(resprot) - {'ACE', 'NME'}
    for i, atom in enumerate(molecule.atoms):
        # TODO: if using 'CH3', this will be failed with
        # ACE ALA ALA ALA NME system
        # if atom.name in ['CA', 'CH3'] and atom.residue.name in RESPROT:
        if atom.name in [
                'CA',
        ] and atom.residue.name in respro_nocap:
            CA_atoms.append(i)
        if atom.name == 'C' and atom.residue.name in respro_nocap:
            C_atoms.append(i)
        if atom.name == 'N' and atom.residue.name in respro_nocap:
            N_atoms.append(i)

    nca = len(CA_atoms)

    for i in range(nca - 1):
        is_ter = molecule.atoms[CA_atoms[i]].residue.ter
        if is_ter:
            continue
        # Changed here to look at the C-N peptide bond distance:
        C_atom = molecule.atoms[C_atoms[i]]
        N_atom = molecule.atoms[N_atoms[i + 1]]

        dx = float(C_atom.xx) - float(N_atom.xx)
        dy = float(C_atom.xy) - float(N_atom.xy)
        dz = float(C_atom.xz) - float(N_atom.xz)
        gap = math.sqrt(dx * dx + dy * dy + dz * dz)

        if gap > 2.0:
            gaprecord = (gap, C_atom.residue.name, C_atom.residue.idx,
                         N_atom.residue.name, N_atom.residue.idx)
            gaplist.append(gaprecord)

    return gaplist


def find_non_standard_resnames(molecule, amber_supported_resname):
    ns_names = set()
    for residue in molecule.residues:
        if len(residue.name) > 3:
            rname = residue.name[:3]
        else:
            rname = residue.name
        if rname.strip() not in amber_supported_resname:
            ns_names.add(rname)
    return ns_names


def cmd_lineparser():
    parser = argparse.ArgumentParser(description="prepare receptor")
    parser.add_argument("-i", "--in", required=True,
        dest="pdbin", help="PDB input file (default: stdin)",
        default='stdin')
    parser.add_argument("-o", "--out", default='protein_prepared',
        dest="out_filename", help="output filename (default: protein)")
    parser.add_argument("-y", "--nohyd", action="store_true", default=False,
        dest="nohyd", help="remove all hydrogen atoms (default: no)")
    parser.add_argument("--nodisu", action="store_false", default=True,
        dest="nodisu", help="ignore difsulfide bridges (default: no)")
    parser.add_argument("-d", "--dry", action="store_true", default=False,
        dest="dry", help="remove all water molecules (default: no)")
    parser.add_argument("--most-populous", action="store_true",
        dest="mostpop", help="keep most populous alt. conf. (default is to keep 'A')")
    parser.add_argument("--model", type=int, default=1,
        dest="model",
        help="Model to use from a multi-model pdb file (integer).  (default: use 1st model). "
        "Use a negative number to keep all models")
    parser.add_argument("--pdb", dest="make_pdb", default=False,
        action="store_true", help="convert to pdb also")
    parser.add_argument("--pdbqt", dest="make_pdbqt", default=False,
        action="store_true", help="convert to pdbqt also")
    parser.add_argument("--ADtype", dest="ad_type", default=False,
        action="store_true", help="Amber types are convert to AD types in the PDBQT file")
    return parser.parse_args()


def main():
    args = cmd_lineparser()
    pdbin_filename = args.pdbin
    out_filename = args.out_filename
    nohyd = args.nohyd
    nodisu = args.nodisu
    dry = args.dry
    mostpop = args.mostpop
    model = args.model - 1
    make_pdb = args.make_pdb
    make_pdbqt = args.make_pdbqt
    ad_type = args.ad_type

    logger = logging.getLogger('WaterKit receptor preparation')
    logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))

    base_filename, extension = os.path.splitext(out_filename)
    pdb_clean_filename = "%s_clean.pdb" % base_filename
    pdb_prepared_filename = "%s.pdb" % base_filename
    pdbqt_prepared_filename = "%s.pdbqt" % base_filename
    prmtop_filename = "%s.prmtop" % base_filename
    rst7_filename = "%s.rst7" % base_filename

    try:
        receptor = pmd.load_file(pdbin_filename)
    except OSError:
        logger.error("Receptor file (%s) cannot be found." % pdbin_filename)
        sys.exit(0)

    pdbfixer = AmberPDBFixer(receptor)

    # Remove box and symmetry
    pdbfixer.parm.box = None
    pdbfixer.parm.symmetry = None

    # Find all the gaps
    gaplist = find_gaps(pdbfixer.parm, RESPROT)
    if gaplist:
        cformat = "gap of %lf A between %s %d and %s %d"
        for _, (d, resname0, resid0, resname1, resid1) in enumerate(gaplist):
            # convert to 1-based
            logger.info(cformat % (d, resname0, resid0 + 1, resname1, resid1 + 1))

    # Find missing heavy atoms
    missing_atoms = pdbfixer.find_missing_heavy_atoms(HEAVY_ATOM_DICT)
    if missing_atoms:
        logger.warning("Found residue(s) with missing heavy atoms: %s" % ', '.join([str(m) for m in missing_atoms]))

    # Remove all the hydrogens
    if nohyd:
        pdbfixer.parm.strip('@/H')
        logger.info("Removed all hydrogen atoms")

    # Remove water molecules
    if dry:
        pdbfixer.remove_water()
        logger.info("Removed all water molecules")

    # Keep only standard-Amber residues
    ns_names = find_non_standard_resnames(pdbfixer.parm, AMBER_SUPPORTED_RESNAMES)
    if ns_names:
        pdbfixer.parm.strip('!:' + ','.join(AMBER_SUPPORTED_RESNAMES))
        logger.info("Removed all non-standard Amber residues: %s" % ', '.join(ns_names))

    # Assign histidine protonations
    pdbfixer.assign_histidine()

    # Find all the disulfide bonds
    if nodisu:
        sslist, cys_cys_atomidx_set = pdbfixer.find_disulfide()
        if sslist:
            pdbfixer.rename_cys_to_cyx(sslist)
            logger.info("Found disulfide bridges between residues %s" % ', '.join(['%s-%s' % (ss[0], ss[1]) for ss in sslist]))
    else:
        sslist = None

    # Remove all the aternate residue sidechains
    alt_residues = remove_alt_residues(pdbfixer.parm)
    if alt_residues:
        logger.info("Removed all alternatives residue sidechains")

    # Write cleaned PDB file
    final_coordinates = pdbfixer.parm.get_coordinates()[model]
    write_kwargs = dict(coordinates=final_coordinates)
    write_kwargs["increase_tercount"] = False # so CONECT record can work properly
    write_kwargs["altlocs"] = "occupancy"

    try:
        write_pdb_file(pdb_clean_filename, pdbfixer.parm, **write_kwargs)
    except:
        logger.error("Could not write pdb file %s"  % pdb_clean_filename)
        sys.exit(0)

    # Generate topology/coordinates files
    with open('leap.template.in', 'w') as w:
        final_ns_names = []
        
        content = _make_leap_template(pdbfixer.parm, final_ns_names, gaplist,
                                      sslist, input_pdb=pdb_clean_filename, 
                                      prmtop=prmtop_filename,
                                      rst7=rst7_filename)
        w.write(content)

    try:
        easy_call("tleap -s -f leap.template.in > leap.template.out", shell=True)
    except RuntimeError:
        logger.error("Could not generate topology/coordinates files with tleap")
        sys.exit(0)

    if make_pdb or make_pdbqt:
        # Write PDBQT file
        try:
            molecule = pmd.load_file(prmtop_filename, rst7_filename)
        except:
            logger.error("Cannot load topology and coordinates Amber files")
            sys.exit(0)

        if make_pdb:
            write_pdb_file(pdb_prepared_filename, molecule)

        if make_pdbqt:
            if ad_type:
                molecule = convert_amber_to_autodock_types(molecule)

            # the PDBQT file for WaterKit
            write_pdbqt_file(pdbqt_prepared_filename, molecule)


if __name__ == '__main__':
    main()