#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
from setuptools import setup, find_packages

PROJECT_ROOT = os.path.dirname(os.path.realpath(__file__))
REQUIREMENTS_FILE = os.path.join(PROJECT_ROOT, 'requirement.txt')

with open(REQUIREMENTS_FILE) as f:
   install_reqs = f.read().splitlines()

install_reqs.append('setuptools')

setup(name='waterkit',
      version=0.1,
      description='WaterKit',
      author='Jerome Eberhardt',
      author_email='jerome@scripps.edu',
      url='https://gitlab.com/jeeberhardt/waterkit',
      packages=find_packages(),
      package_data={'': ['requirement.txt']},
      install_requires=install_reqs,
      include_package_data=True,
      zip_safe=False,
      license='MIT',
      keywords=['molecular modeling', 'drug design',
               'docking', 'autodock'],
      classifiers=[
            'Programming Language :: Python :: 2.7',
            'Operating System :: Unix',
            'Operating System :: MacOS',
            'Topic :: Scientific/Engineering'
      ]
)
