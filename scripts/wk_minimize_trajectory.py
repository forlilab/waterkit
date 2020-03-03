#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# minimize trajectory
#

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import argparse
import multiprocessing as mp
import os
import sys
import time

import numpy as np
from parmed.amber import NetCDFTraj

from waterkit import utils


def cmd_lineparser():
    parser = argparse.ArgumentParser(description="make_trajectory")
    parser.add_argument("-p", "--prmtop", dest="prmtop_filename", required=True,
                        action="store", help="amber topology file")
    parser.add_argument("-c", "--rst7", dest="rst7_filename", required=True,
                        action="store", help="amber rst7 file")
    parser.add_argument("-t", "--trj", dest="traj_filename", required=True,
                        action="store", help="netcdf trajectory file")
    parser.add_argument("-j", "--job", dest="n_jobs", default=-1, type=int,
                        action="store", help="number of jobs")
    parser.add_argument("-s", "--steps", dest="n_steps", default=50, type=int,
                        action="store", help="number of minimization steps")
    parser.add_argument("--restraint", dest="restraint", default=2.5, type=float,
                        action="store", help="harmonic restraint on protein heavy atoms")
    parser.add_argument("-o", "--output", dest="output_filename", default="protein_min.nc",
                        action="store", help="netcdf output trajectory name")
    return parser.parse_args()


def _minimize_single(input_filename, log_filename, prmtop_filename,
                     rst7_filename, trajin_filename, trajout_filename, 
                     n_steps=50, cutoff=35, nsnb=25, restraint=None):
    if restraint is None:
        restraint = 0

    mininp = ("Minimization of solvent orientation\n"
              " &cntrl\n"
              "  imin = 5,                            ! Apply to every frame in the input traj\n"
              "  maxcyc = %d,                         ! (maxcyc - ncyc) steps of CONJ\n"
              "  ntmin = 0,                           ! ONLY CONJ\n"
              "  ntb = 0,                             ! No periodic box, PME is off\n"
              "  igb = 6,                             ! In vacuum electrostatics\n"
              "  cut = %d,                            ! Non-bonded cutoff\n"
              "  nsnb = %d,                           ! Non-bonded list update every x steps\n"
              "  ntc = 1,                             ! No SHAKE\n"
              "  ntf = 1,                             ! Force evaluation, complete interaction is calculated\n"
              "  jfastw = 4,                          ! Do not use the fast SHAKE routines for waters\n")

    if restraint > 0:
        tmp = ("  ntr = 1,                             ! Use harmonic constraints\n"
               "  restraint_wt = %.2f,                 ! Constraints of 2.5 kcal/mol\n"
               "  restraintmask = '!@H= & !:WAT',      ! Constraints of heavy atoms except WAT\n")
        mininp += tmp % restraint
    else:
        mininp += "  ntr = 0,                             ! No harmonic constraints\n"

    mininp += ("  ntx = 1,                             ! No velocities in the inpcrd file\n"
               "  ntwx = 1,                            ! Every ntwx steps, the coordinates will be written to the mdcrd file\n"
               "  ioutfm = 1                           ! Binary NetCDF trajectory\n"
               "/\n")

    mininp = mininp % (n_steps, cutoff, nsnb)

    with open(input_filename, "w") as w:
        w.write(mininp)

    command_line = "sander -O -i %s -o %s " % (input_filename, log_filename)
    command_line += "-p %s -c %s -ref %s " % (prmtop_filename, rst7_filename, rst7_filename)
    command_line += "-y %s -x %s" % (trajin_filename, trajout_filename)

    output, errors = utils.execute_command(command_line)

    if errors:
        print("Error: minimization of the trajectory failed (%s)!" % log_filename)
        print(errors)
        return False

    return True


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


def _trajectory_n_frames(traj_filename):
    # Get number of frames in the trajectory
    try:
        traj = NetCDFTraj.open_old(traj_filename)
    except FileNotFoundError:
        print("Cannot find trajectory file: %s" % traj_filename)
        sys.exit(0)

    n_frames = traj.frame
    traj.close()

    return n_frames


def _longest_distance_trajectory(traj_filename):
    # Get number of frames in the trajectory
    try:
        traj = NetCDFTraj.open_old(traj_filename)
    except FileNotFoundError:
        print("Cannot find trajectory file: %s" % traj_filename)
        sys.exit(0)

    xyz = traj.coordinates
    r = np.reshape(xyz, (xyz.shape[0] * xyz.shape[1], 3))
    s = np.sum(r, axis=1)

    xyz_min = r[np.argmin(s), :]
    xyz_max = r[np.argmax(s), :]

    distance = np.sqrt(np.sum((xyz_min - xyz_max)**2))

    return distance


def _split_trajectory(prmtop_filename, traj_filename, n_chunk=2, prefix=None):
    trajout_filenames = []
    split_filename = "traj_split.inp"
    trajout_name = "traj_split"
    if prefix is not None:
        trajout_name = prefix + "_" + trajout_name

    # Get number of frames in the trajectory
    n_frames = _trajectory_n_frames(traj_filename)

    # Get start and stop for each part
    chunks = utils.split_list_in_chunks(n_frames, n_chunk)

    splitin = "parm %s\n" % prmtop_filename
    splitin += "trajin %s\n" % traj_filename
    for i, chunk in enumerate(chunks):
        trajout_filename = trajout_name + "_%s.nc" % i
        splitin += "trajout %s onlyframes %d-%d\n" % (trajout_filename,  chunk[0] + 1, chunk[1] + 1)
        trajout_filenames.append(trajout_filename)
    splitin += "go\n"
    splitin += "quit\n"

    with open(split_filename, "w") as w:
        w.write(splitin)

    # Split!!
    command_line = "cpptraj -i %s" % split_filename
    output, errors = utils.execute_command(command_line)

    if errors:
        print("Error: spliting of the trajectory failed!")
        print(errors)
        return []

    return trajout_filenames


def _concatenate_trajectories(prmtop_filename, trajin_filenames, trajout_filename, box=None):
    conc_filename = "traj_concatenate.inp"

    concin = "parm %s\n" % prmtop_filename
    if box is not None:
        concin += "box x %f y %f z %s alpha %d beta %d gamma %d\n" % (box[0], box[1], box[2], box[3], box[4], box[5])
    for trajin_filename in trajin_filenames:
        concin += "trajin %s\n" % trajin_filename
    concin += "trajout %s\n" % trajout_filename
    concin += "go\n"
    concin += "quit\n"

    with open(conc_filename, "w") as w:
        w.write(concin)

    command_line = "cpptraj -i %s" % conc_filename
    output, errors = utils.execute_command(command_line)

    if errors:
        print("Error: concatenation of the trajectories failed!")
        print(errors)
        return False

    return True


class WaterMinimizer:
    def __init__(self, n_steps=50, restraint=None, n_jobs=-1):
        self._n_steps = n_steps
        self._restraint = restraint

        if n_jobs == -1:
            self._n_jobs = mp.cpu_count()
        else:
            self._n_jobs = n_jobs

    def minimize_trajectory(self, prmtop_filename, rst7_filename, traj_filename, output_filename, clean=False):
        jobs = []
        files_to_remove = ["restrt"]

        box = _box_information(traj_filename)
        cutoff = np.ceil(_longest_distance_trajectory(traj_filename)).astype(np.int)

        if self._n_jobs > 1:
            # Split trajectory
            traj_split_filenames = _split_trajectory(prmtop_filename, traj_filename, self._n_jobs)
        else:
            traj_split_filenames = [traj_filename]

        # Dirty trick to be sure cpptraj finished writing
        time.sleep(15)

        if traj_split_filenames:
            input_filenames = ["traj_min_%d.inp" % i for i in range(self._n_jobs)]
            log_filenames = ["traj_min_%d.out" % i for i in range(self._n_jobs)]
            traj_min_filenames = ["traj_min_%d.nc" % i for i in range(self._n_jobs)]
            nsnb = (int(_trajectory_n_frames(traj_filename) / self._n_jobs) * self._n_steps) + self._n_steps

            # Fire off!!
            for i in range(self._n_jobs):
                args = (input_filenames[i], log_filenames[i],
                        prmtop_filename, rst7_filename, 
                        traj_split_filenames[i], traj_min_filenames[i],
                        self._n_steps, cutoff, nsnb, self._restraint)
                job = mp.Process(target=_minimize_single, args=args)
                job.start()
                jobs.append(job)

            for job in jobs:
                job.join()

            # Concatenate trajectory
            # Even if the trajectory was not splitted, we still have to add the box informations
            conc_status = _concatenate_trajectories(prmtop_filename, traj_min_filenames, output_filename, box)

            if self._n_jobs > 1:
                # Cleaning...
                if clean and conc_status:
                    files_to_remove.append("traj_split.inp")
                    files_to_remove.append("traj_concatenate.inp")
                    files_to_remove.extend(traj_split_filenames)
                    files_to_remove.extend(input_filenames)
                    files_to_remove.extend(log_filenames)
                    files_to_remove.extend(traj_min_filenames)
            else:
                if clean:
                    files_to_remove.extend(input_filenames)
                    files_to_remove.extend(log_filenames)

        if clean:
            for file_to_remove in files_to_remove:
                os.remove(file_to_remove)


def main():
    args = cmd_lineparser()
    prmtop_filename = args.prmtop_filename
    rst7_filename = args.rst7_filename
    traj_filename = args.traj_filename
    output_filename = args.output_filename
    n_steps = args.n_steps
    restraint = args.restraint
    n_jobs = args.n_jobs

    m = WaterMinimizer(n_steps, restraint, n_jobs)
    m.minimize_trajectory(prmtop_filename, rst7_filename, traj_filename, output_filename, True)
    

if __name__ == '__main__':
    main()