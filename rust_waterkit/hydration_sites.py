#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
import argparse
from gridData import Grid
from waterkit.analysis import HydrationSites
from waterkit.analysis import blur_map

def cmd_lineparser():
    parser = argparse.ArgumentParser(description='Hydration sites analysis')
    parser.add_argument('--o_dens', dest='grid_o', required=True,
                        action='store', help='Oxygen density grid')
    parser.add_argument('--esw', dest='esw', required=True,
                        action='store', help='Energy sw density')
    parser.add_argument('--eww', dest='eww', required=True,
                        action='store', help='Energy ww density')
    parser.add_argument('--dtstrans', dest='dtstrans', required=True,
                        action='store', help='dTStrans')
    parser.add_argument('--dtsorient', dest='dtsorient', required=True,
                        action='store', help='dTSorient')
    return parser.parse_args()


def main():
    args = cmd_lineparser()
    gO = Grid(args.grid_o)
    esw = Grid(args.esw)
    eww = Grid(args.eww)
    tst = Grid(args.dtstrans)
    tso = Grid(args.dtsorient)
    dg = (esw + 2 * eww) - (tst + tso)

    # Identification of hydration site positions using gO
    hs = HydrationSites(gridsize=0.5, water_radius=1.4, min_water_distance=2.5, min_density=2.0)
    hydration_sites = hs.find(gO) # can pass "gist-gO.dx" directly also

    # Get Gaussian smoothed energy for hydration sites only
    dg_energy = hs.hydration_sites_energy(dg, water_radius=1.4)
    hs.export_to_pdb("hydration_sites_dG_smoothed.pdb", hydration_sites, dg_energy)

    # ... or get the whole Gaussian smoothed map
    map_smooth = blur_map(dg, radius=1.4)
    map_smooth.export("gist-dG-dens_smoothed.dx")

if __name__ == "__main__":
   main()