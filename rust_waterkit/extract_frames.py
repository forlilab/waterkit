#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# extract_frames_dcd
#

import argparse
import MDAnalysis as mda
import numpy as np

def extract_frames(input_dcd, input_topology, output_dcd, n_frames):
    """
    Load a DCD trajectory, extract n evenly spaced frames, and save to a new DCD file.

    Args:
        input_dcd (str): Input DCD trajectory filename
        input_topology (str): Input topology file (e.g., PDB, PSF)
        output_dcd (str): Output DCD trajectory filename
        n_frames (int): Number of frames to extract
    """
    # Load the universe with topology and trajectory
    u = mda.Universe(input_topology, input_dcd)
    
    # Calculate frame indices to extract (evenly spaced)
    total_frames = len(u.trajectory)
    if n_frames > total_frames:
        raise ValueError(f"Requested {n_frames} frames, but trajectory only has {total_frames} frames")
        
    # Select all atoms
    atoms = u.select_atoms("all")
    
    # Create a new writer for the output DCD
    with mda.Writer(output_dcd, atoms.n_atoms) as writer:
        # Iterate through selected frames
        frame_indices = list(range(40000, 50000))
        for frame_idx in frame_indices:
            # Go to the specified frame
            u.trajectory[frame_idx]
            # Write the current frame to the new DCD
            writer.write(atoms)

def cmd_lineparser():
    parser = argparse.ArgumentParser(description="Extract evenly spaced frames from a DCD trajectory")
    parser.add_argument("-t", "--topology", dest="topology_file", required=True,
                        action="store", help="Topology file (e.g., PDB, PSF)")
    parser.add_argument("-i", "--input", dest="input_dcd", required=True,
                        action="store", help="Input DCD trajectory file")
    parser.add_argument("-o", "--output", dest="output_dcd", default="output.dcd",
                        action="store", help="Output DCD trajectory file (default: output.dcd)")
    parser.add_argument("-n", "--nframes", dest="n_frames", type=int, required=True,
                        action="store", help="Number of frames to extract")
    return parser.parse_args()

def main():
    args = cmd_lineparser()
    extract_frames(args.input_dcd, args.topology_file, args.output_dcd, args.n_frames)

if __name__ == "__main__":
    main()