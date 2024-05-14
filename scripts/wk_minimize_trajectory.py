#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# minimize trajectory
#

import argparse
import sys
from packaging.version import Version

import numpy as np
import parmed as pmd
from parmed.amber import NetCDFTraj
from openmm.unit import Quantity, picoseconds, kilocalories_per_mole, angstroms, nanometer, kelvin, kilocalories, mole
from openmm import CustomExternalForce, LangevinIntegrator, Platform
from openmm.app import AmberPrmtopFile, HBonds, CutoffNonPeriodic, Simulation


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


def _box_information(traj_filename):
    box = None

    try:
        traj = NetCDFTraj.open_old(traj_filename)

        if traj.hasbox:
            box = traj.box[0]

        traj.close()
    except FileNotFoundError:
        print("Cannot find trajectory file: %s" % traj_filename)
        sys.exit(0)

    return box


class WaterMinimizer:
    def __init__(self, n_steps=100, restraint=None, platform="OpenCL", verbose=True):
        self._n_steps = n_steps
        self._restraint = restraint
        self._platform = platform
        self._verbose = verbose

    def minimize_trajectory(self, prmtop_filename, traj_filename, output_filename):
        nonbondedMethod = CutoffNonPeriodic
        nonbondedCutoff = 9 * angstroms
        rigidWater = True
        constraints = HBonds
        dt = 0.002 * picoseconds
        temperature = 300 * kelvin
        friction = 1.0 / picoseconds
        K = self._restraint * kilocalories_per_mole / angstroms**2

        # If someone still uses an old version of OpenMM
        if Version(Platform.getOpenMMVersion()) > Version("7.5.0"):
            tolerance = 1.0 * kilocalories / (nanometer * mole)
        else:
            tolerance = 1.0 * kilocalories_per_mole

        box = _box_information(traj_filename)

        platform = Platform.getPlatformByName(self._platform)
        platformProperties = {'Precision': 'single'}

        prmtop = AmberPrmtopFile(prmtop_filename)
        parmedtop = pmd.load_file(prmtop_filename)
        old_traj = NetCDFTraj.open_old(traj_filename)

        n_atom = old_traj.atom
        n_frame = old_traj.frame

        new_trj = NetCDFTraj.open_new(output_filename, natom=n_atom, box=True, crds=True)

        system = prmtop.createSystem(nonbondedMethod=nonbondedMethod, nonbondedCutoff=nonbondedCutoff, constraints=constraints)

        for i, coordinates in enumerate(old_traj.coordinates):
            old_positions = Quantity(coordinates.tolist(), unit=angstroms)

            if self._restraint > 0 or self._restraint is not None:
                # Add harmonic constraints on protein
                force = CustomExternalForce("k * ((x-x0)^2 + (y-y0)^2 + (z-z0)^2)")
                force.addGlobalParameter("k", K)
                force.addPerParticleParameter("x0")
                force.addPerParticleParameter("y0")
                force.addPerParticleParameter("z0")
                for atom in parmedtop.view['!@H= & !:WAT']:
                    force.addParticle(atom.idx, old_positions[atom.idx])
                force_idx = system.addForce(force)

            # Create simulation
            integrator = LangevinIntegrator(temperature, friction, dt)
            simulation = Simulation(prmtop.topology, system, integrator, platform)
            simulation.context.setPositions(old_positions)

            # Minimize the water molecules
            simulation.minimizeEnergy(maxIterations=self._n_steps, tolerance=tolerance)
            # Get new positions and store in the new trajectory
            new_positions = simulation.context.getState(getPositions=True).getPositions(asNumpy=True)

            new_trj.add_coordinates(new_positions)
            new_trj.add_box(box)
            new_trj.add_time(i + 1)

            if self._restraint > 0 or self._restraint is not None:
                system.removeForce(force_idx)

            if (i % 100 == 0) and self._verbose:
                sys.stdout.write("\rConformations minimized:  %5.2f / 100 %%" % (float(i) / n_frame * 100.))
                sys.stdout.flush()

        new_trj.close()
        old_traj.close()

        print("")


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