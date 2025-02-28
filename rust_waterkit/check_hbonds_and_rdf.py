import sys
import os

from itertools import combinations
import mdtraj as md
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    if len(sys.argv) < 5:
        print("Wrong input parameters.")
        print("Usage: python check_hbonds_and_rdf.py path_to_traj path_to_topology path_and_name_of_rdf_plot path_and_name_of_hbonds_plot")
    traj = md.load(sys.argv[1], top=sys.argv[2])
    oxygen_indices = traj.topology.select("water and name O")
    pairs = np.array(list(combinations(oxygen_indices, 2)))
    r, rdf = md.compute_rdf(traj, pairs)
    
    plt.figure(figsize=(6, 4))
    plt.plot(r, rdf, label="Water-Water RDF")
    plt.xlabel("Distance (nm)")
    plt.ylabel("g(r)")
    plt.title("Radial Distribution Function of Water")
    plt.legend()
    plt.savefig(sys.argv[3], dpi=300)
    plt.close()
    
    
    hbonds = md.wernet_nilsson(traj)  # Returns a list of hydrogen bonds per frame
    # Count the number of hydrogen bonds per frame
    hb_counts = np.array([len(frame) for frame in hbonds])
    
    plt.figure(figsize=(6, 4))
    # Plot histogram of hydrogen bond counts
    plt.hist(hb_counts, bins=20, alpha=0.7, color='b', label="H-bonds per frame")
    plt.xlabel("Number of H-bonds")
    plt.ylabel("Frequency")
    plt.title("Hydrogen Bond Distribution")
    plt.legend()
    plt.savefig(sys.argv[4], dpi=300)
    plt.close()
    
    # Print average number of hydrogen bonds
    print(f"Average number of hydrogen bonds: {np.mean(hb_counts):.2f}")