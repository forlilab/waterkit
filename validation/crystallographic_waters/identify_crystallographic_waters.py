import os
import sys
import prody
import numpy as np
from scipy.spatial import KDTree

# Load identified hydration sites
# Load original structure and select waters within 5A from the ligand atoms
# Load the structure used to run MCSwell
# align the two structures
# for each identified water in the crystal look if there's a water within 1.5A


def load_structure(path_to_structure):
    if path_to_structure.endswith(".cif"):
        structure = prody.parseMMCIF(path_to_structure)
    else:
        structure = prody.parsePDB(path_to_structure)
    return structure

def align_structures(crystal, reference):
    lig = crystal.select('not protein and not water')
    # print(f"Before aligining: {lig.getCoords()}")
    crystal = prody.matchAlign(crystal, reference)[0]
    lig = crystal.select('not protein and not water')
    # print(f"After aligining: {lig.getCoords()}")
    return crystal

def extract_key_waters(aligned_crystal, aligned_holo_ligand=None, cutoff=5.0):
    positions = []
    prody.defSelectionMacro('ligand', 'not protein and not water')
    if aligned_holo_ligand is None:
        key_waters = aligned_crystal.select('within 5 of ligand and water')
        for atom in key_waters:
            positions.append(atom.getCoords())
    else:
        waters = aligned_crystal.select('water')
        tree = KDTree(np.array(aligned_holo_ligand.getCoords()))
        for water in waters:
            water_coords = water.getCoords()
            distance, index = tree.query(np.array(water_coords))
            if distance <= 5.0:
                positions.append(water_coords)
    return positions

def identify_crystal_waters(key_waters_positions, discrete_waters, th):
    matching_waters = []
    tree = KDTree(np.array(key_waters_positions))
    for water in discrete_waters:
        distance, index = tree.query(np.array(water))
        if distance < th:
            matching_waters.append([key_waters_positions[index], water])
    return matching_waters

if __name__ == "__main__":
    crystal = load_structure("/data/phd/waterkit/example/5j80.cif")
    holo = load_structure("/data/phd/waterkit/example/1uyg.cif")
    reference = load_structure("/data/phd/waterkit/validation/hsp90_target/1uyg_compatible.pdb")
    aligned_crystal = align_structures(crystal, reference)
    aligned_holo = align_structures(holo, reference)
    aligned_holo_ligand = aligned_holo.select("not protein and not water")
    key_waters_positions = extract_key_waters(aligned_crystal, aligned_holo_ligand)
    # with open("key_waters_hsp90.xyz", "w") as fo:
    #     for p in key_waters_positions:
    #         fo.write(f"O {p[0]} {p[1]} {p[2]}\n")
    # Don't need the protein structures anymore
    del(crystal)
    del(reference)

    # Load the discrete waters
    discrete_waters = load_structure("/data/phd/waterkit/validation/hsp90_target/GIST_MD_NO_HMR/hydration_sites_dG_smoothed_MD.pdb")
    discrete_waters_positions = discrete_waters.getCoords()
    # print(discrete_waters_positions)
    th_cutoffs = [0.51, 1.01, 1.51]
    print("Results for MD:")
    for th in th_cutoffs:
        n_identified_waters = identify_crystal_waters(key_waters_positions, discrete_waters_positions, th)
        print(f"{len(n_identified_waters)} discrete waters identified as matching with crystal waters ot ouf {len(key_waters_positions)}. Success rate: {(len(n_identified_waters)*100)/len(key_waters_positions)}% with threshold {th}")

    # Load the discrete waters
    discrete_waters = load_structure("/data/phd/waterkit/validation/hsp90_target/TIP3P/200000_steps/hydration_sites_dG_smoothed_MC.pdb")
    discrete_waters_positions = discrete_waters.getCoords()
    # print(discrete_waters_positions)
    th_cutoffs = [0.51, 1.01, 1.51]
    print("Results for MCSwell:")
    for th in th_cutoffs:
        n_identified_waters = identify_crystal_waters(key_waters_positions, discrete_waters_positions, th)
        print(f"{len(n_identified_waters)} discrete waters identified as matching with crystal waters ot ouf {len(key_waters_positions)}. Success rate: {(len(n_identified_waters)*100)/len(key_waters_positions)}% with threshold {th}")
    
    # Load the discrete waters
    discrete_waters = load_structure("/data/phd/waterkit/validation/hsp90_target/OG_WK/hydration_sites_dG_smoothed_WK.pdb")
    discrete_waters_positions = discrete_waters.getCoords()
    # print(discrete_waters_positions)
    th_cutoffs = [0.51, 1.01, 1.51]
    print("Results for WaterKit:")
    for th in th_cutoffs:
        n_identified_waters = identify_crystal_waters(key_waters_positions, discrete_waters_positions, th)
        print(f"{len(n_identified_waters)} discrete waters identified as matching with crystal waters ot ouf {len(key_waters_positions)}. Success rate: {(len(n_identified_waters)*100)/len(key_waters_positions)}% with threshold {th}")
