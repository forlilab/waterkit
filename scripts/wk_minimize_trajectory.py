#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# minimize trajectory
#

import argparse

from waterkit import WaterMinimizer


def cmd_lineparser():
    parser = argparse.ArgumentParser(description="make_trajectory")
    parser.add_argument("-p", "--prmtop", dest="prmtop_filename", required=True,
                        action="store", help="amber topology file")
    parser.add_argument("-t", "--trj", dest="traj_filename", required=True,
                        action="store", help="netcdf trajectory file")
    parser.add_argument("-s", "--steps", dest="n_steps", default=100, type=int,
                        action="store", help="number of minimization steps")
    parser.add_argument("-r", "--restraint", dest="restraint", default=2.5, type=float,
                        action="store", help="harmonic restraint on protein heavy atoms")
    parser.add_argument("-o", "--output", dest="output_filename", default="protein_min.nc",
                        action="store", help="netcdf output trajectory name")
    parser.add_argument("--platform", dest="platform", default="CUDA", choices=["CUDA", "OpenCL", "CPU"],
                        action="store", help="choice of the platform (default: CUDA)")
    return parser.parse_args()



def main():
    args = cmd_lineparser()
    prmtop_filename = args.prmtop_filename
    traj_filename = args.traj_filename
    output_filename = args.output_filename
    n_steps = args.n_steps
    restraint = args.restraint
    platform = args.platform

    m = WaterMinimizer(n_steps, restraint, platform)
    m.minimize_trajectory(prmtop_filename, traj_filename, output_filename)
    

if __name__ == '__main__':
    main()
