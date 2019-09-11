#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
from setuptools import setup, find_packages

PROJECT_ROOT = os.path.dirname(os.path.realpath(__file__))
REQUIREMENTS_FILE = os.path.join(PROJECT_ROOT, "requirement.txt")

with open(REQUIREMENTS_FILE) as f:
   install_reqs = f.read().splitlines()

install_reqs.append("setuptools")

setup(name="waterkit",
      version=0.3,
      description="WaterKit",
      author="Jerome Eberhardt",
      author_email="jerome@scripps.edu",
      url="https://github.com/jeeberhardt/waterkit",
      packages=find_packages(),
      scripts=["scripts/amber2pdbqt.py",
               "scripts/make_trajectory.py",
               "scripts/run_waterkit.py"],
      data_files = [("", ["waterkit/data/AD4_parameters.dat",
                          "waterkit/data/disordered_hydrogens.par",
                          "waterkit/data/waterfield.par",
                          "waterkit/data/water_orientations.txt"])],
      install_requires=install_reqs,
      include_package_data=True,
      zip_safe=False,
      license="MIT",
      keywords=["molecular modeling", "drug design",
               "docking", "autodock"],
      classifiers=[
            "Programming Language :: Python :: 2.7",
            "Operating System :: Unix",
            "Operating System :: MacOS",
            "Topic :: Scientific/Engineering"
      ]
)
