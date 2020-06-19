#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# WaterKit
#
# Launch waterkit
#

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import argparse

from waterkit import SphericalWaterMap
from waterkit import Map
from waterkit import utils

def cmd_lineparser():
    parser = argparse.ArgumentParser(description="waterkit")
    parser.add_argument("-m", "--fld", dest="fld_file", required=True,
                        action="store", help="autodock fld file")
    parser.add_argument("-w", "--water", dest="water_model", default="tip3p",
                        choices=["tip3p", "tip5p"], action="store",
                        help="water model used (tip3p or tip5p)")
    parser.add_argument("-t", "--temperature", dest="temperature", default=300., type=float,
                        action="store", help="temperature")
    parser.add_argument("-d", "--dieletric", dest="dieletric", default=1., type=float,
                        action="store", help="dieletric constant")
    parser.add_argument("-j", "--n_jobs", dest="n_jobs", default=1., type=int,
                        action="store", help="number of jobs to run in parallel")
    parser.add_argument("-o", "--output", dest="output_dir", default=".",
                        action="store", help="output directory")
    return parser.parse_args()


def main():
    args = cmd_lineparser()
    fld_file = args.fld_file
    water_model = args.water_model
    temperature = args.temperature
    dieletric = args.dieletric
    n_jobs = args.n_jobs
    output_dir = args.output_dir

    ad_map = Map.from_fld(fld_file)
    utils.prepare_water_map(ad_map, water_model, dieletric)

    s = SphericalWaterMap(water_model, temperature, n_jobs, verbose=True)
    s.run(ad_map, "spherical_water")

    ad_map.to_map("spherical_water", "%s/" % output_dir)


if __name__ == "__main__":
    main()
