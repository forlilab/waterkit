#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import fnmatch


def find_files(directory):
    matches = []

    for root, dirnames, filenames in os.walk(directory):
        for filename in fnmatch.filter(filenames, '*'):
            matches.append(os.path.join(root, filename))

    return matches


setup(name="waterkit",
      version=0.3,
      description="WaterKit",
      author="Jerome Eberhardt",
      author_email="jerome@scripps.edu",
      url="https://github.com/jeeberhardt/waterkit",
      scripts=["scripts/amber2pdbqt.py",
               "scripts/make_trajectory.py",
               "scripts/run_waterkit.py",
               "scripts/create_grid_protein_file.py"],
      data_files=[("", find_files("waterkit/data"))],
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
