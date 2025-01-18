import meeko
import rust_waterkit

def to_xyz(traj, step_size, fname):
    with open(f"{fname}.xyz", 'w') as fo:
        fo.write(f"{len(traj)}\n")
        fo.write("\n")
        for t in traj:
            t_v = t
            fo.write(f"He {t_v[0]} {t_v[1]} {t_v[2]}\n")
    return 

def get_data_form_meeko(wanted_residues, pdb_file):
    surface_atoms = list()
    with open(pdb_file) as fi:
        pdbstring = fi.read()
    mk_prep = meeko.MoleculePreparation(
        merge_these_atom_types=[],
        load_atom_params=["openff"],
        charge_model="espaloma",
    )
    templates = meeko.ResidueChemTemplates.create_from_defaults()
    polymer = meeko.Polymer.from_pdb_string(pdb_string=pdbstring, 
                                            chem_templates=templates, 
                                            mk_prep=mk_prep, 
                                            allow_bad_res=True, 
                                            default_altloc="A")
    for res_id, monomer in polymer.get_valid_monomers().items():
        unique_id = f"{res_id.split(':')[0]}:{monomer.input_resname}:{res_id.split(':')[-1]}"
        if unique_id in wanted_residues:

            for atom in monomer.molsetup.atoms:
                if atom.is_ignore:
                    continue
                rmin_half = monomer.molsetup.atom_params["rmin_half"][atom.index]
                epsilon = monomer.molsetup.atom_params["epsilon"][atom.index]
                charge = atom.charge
                atom_type = atom.pdbinfo.name
                new_atom = rust_waterkit.Atom(atom_type=atom_type, 
                            atom_id=f"{unique_id}:{atom_type}", 
                            coords_point=atom.coord, 
                            rmin_half=rmin_half, 
                            epsilon=epsilon, 
                            charge=charge)
                surface_atoms.append(new_atom)
    return surface_atoms

def xyz_to_pdb_with_temp(xyz_file, pdb_file, energies):
    """
    Convert an XYZ file to a PDB file and add temperature data.
    
    Parameters:
    - xyz_file: str, path to the input XYZ file.
    - pdb_file: str, path to the output PDB file.
    - temperature_data: list of float, temperature factors for each atom.
    """
    pdb_lines = []
    
    with open(xyz_file, 'r') as f:
        lines = f.readlines()
    
    # The first two lines of an XYZ file are metadata (atom count and comment).
    atom_count = int(lines[0].strip())
    comment = lines[1].strip()
    
    # Check if temperature_data matches the number of atoms
    if len(energies) != atom_count:
        raise ValueError("Temperature data does not match the number of atoms in the XYZ file.")
    
    # Parse XYZ atoms and create PDB lines
    for i, line in enumerate(lines[2:], start=1):
        parts = line.split()
        atom_type = parts[0]
        x, y, z = map(float, parts[1:4])
        temp_factor = energies[i - 1]
        
        # Format according to PDB specifications
        pdb_line = (
            f"ATOM  {i:5d} {atom_type:<4} MOL     1    "  # Atom serial, name, residue
            f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00{temp_factor:6.2f}           {atom_type:>2}"
        )
        pdb_lines.append(pdb_line)
    
    # Write to PDB file
    with open(pdb_file, 'w') as f:
        f.write(f"REMARK Converted from XYZ file: {comment}\n")
        f.write("\n".join(pdb_lines))
        f.write("\nEND\n")
    return

def pdb_with_temp(pdb_file, traj, energies):
    import prody
    capped_energies = list()
    for e in energies:
        if e > 100:
            capped_energies.append(100)
        else:
            capped_energies.append(e)
    ag = prody.AtomGroup('Surface')
    ag.setCoords(traj)
    ag.setNames(["He" for _ in traj])
    ag.setResnames(["MOL" for _ in traj])
    ag.setResnums([1 for _ in traj])
    ag.setBetas(capped_energies)
    prody.writePDB(pdb_file, ag)
    return

if __name__ == "__main__":
    # select as, i. 111+107+103+162+150+98+97+184+96+93+55+52+51+138+139+136+135
    import time
    
    surface_points = []
    wanted_residues = ["A:ASN:51", "A:SER:52", "A:ALA:55",
                       "A:ASP:93", "A:ILE:96", "A:GLY:97",
                       "A:MET:98", "A:LEU:103", "A:LEU:107", 
                       "A:ALA:111", "A:GLY:135", "A:VAL:136", 
                       "A:PHE:138", "A:TYR:139", "A:VAL:150", 
                       "A:TRP:162", "A:THR:184"]
    parametrized_atoms = get_data_form_meeko(wanted_residues, "/data/phd/waterkit/example/1uyg.pdb")
    start = time.time()
    step_size = 1.4
    energies, trajectories = rust_waterkit.roll_sphere_and_compute_energies(parametrized_atoms, step_size)
    print(f"Time to grid: {time.time() - start}")
    to_xyz(trajectories, step_size, "trajectory")
    pdb_with_temp("surface_distribution.pdb", trajectories, energies)
    