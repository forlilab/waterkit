#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import glob
import fnmatch
from setuptools import setup, find_packages


def find_files(directory):
    try:
        # Python 3
        return glob.glob(os.path.join(directory, "**/*"), recursive=True)
    except TypeError:
        # Python 2
        matches = []

        for root, dirnames, filenames in os.walk(directory):
            for filename in fnmatch.filter(filenames, '*'):
                matches.append(os.path.join(root, filename))

        return matches


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
      data_files=[("", find_files("waterkit/data"))],
      install_requires=install_reqs,
      include_package_data=True,
      zip_safe=False,
      license="MIT",
      keywords=["molecular modeling", "drug design",
                "docking", "autodock"],
      classifiers=["Programming Language :: Python :: 2.7",
                   "Programming Language :: Python :: 3.7",
                   "Operating System :: Unix",
                   "Operating System :: MacOS",
                   "Topic :: Scientific/Engineering"]
)
