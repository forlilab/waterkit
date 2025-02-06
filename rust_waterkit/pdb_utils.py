import sys

import numpy as np
import prody

def print_differences_in_energies(f1, f2):
    energies1 = list()
    coords1 = list()

    energies2 = list()
    coords2 = list()

    with open(f1) as fi:
        lines = fi.readlines()

    for line in lines:
        line = line.strip()
        l = line.split(" ")
        e = float(l[0])
        c = [round(float(l[1].split(',')[0]), 2), round(float(l[2].split(',')[0]), 2), round(float(l[3].split(',')[0]), 2)]
        energies1.append(e)
        coords1.append(c)
    # pdb_with_temp(f"{f1.split('.')[0]}.pdb", coords1, energies1)

    with open(f2) as fi:
        lines = fi.readlines()

    for line in lines:
        line = line.strip()
        l = line.split(" ")
        e = float(l[0])
        c = [round(float(l[1].split(',')[0]), 2), round(float(l[2].split(',')[0]), 2), round(float(l[3].split(',')[0]), 2)]
        energies2.append(e)
        coords2.append(c)
    
    diff = np.abs(np.array(energies1) - np.array(energies2))
    indices = np.where(diff > 0.3)[0]
    energies = diff[indices]
    coords = np.array(coords2)[indices]  
    pdb_with_temp(f"energies_difference.pdb", coords, energies)
    return

def f_to_pdb(f):
    energies = list()
    coords = list()
    with open(f) as fi:
        lines = fi.readlines()

    for line in lines:
        line = line.strip()
        l = line.split(" ")
        e = float(l[0])
        c = [float(l[1].split(',')[0]), float(l[2].split(',')[0]), float(l[3].split(',')[0])]
        energies.append(e)
        coords.append(c)
    pdb_with_temp(f"{f.split('.')[0]}.pdb", coords, energies)
    return

def pdb_with_temp(pdb_file, traj, energies, atom_type="He"):
    capped_energies = list()
    for e in energies:
        if e > 100:
            capped_energies.append(100)
        else:
            capped_energies.append(e)
    ag = prody.AtomGroup('Surface')
    ag.setCoords(traj)
    ag.setNames([atom_type for _ in traj])
    ag.setResnames(["MOL" for _ in traj])
    ag.setResnums([1 for _ in traj])
    ag.setBetas(capped_energies)
    prody.writePDB(pdb_file, ag)
    return

if __name__ == "__main__":
    if len(sys.argv) < 3 or sys.argv[1] == "--pdb":
        f_to_pdb(sys.argv[2])

    elif sys.argv[1] == "--diff":
        print_differences_in_energies(sys.argv[2], sys.argv[3]) 