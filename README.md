[![made-with-python](https://img.shields.io/badge/Made%20with-Python-1f425f.svg)](https://www.python.org/) [![GitHub license](https://img.shields.io/badge/License-GPL%20v3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0) [![PyPI version fury.io](https://img.shields.io/badge/version-0.5.1-green.svg)](https://pypi.python.org/pypi/ansicolortags/) 

# Waterkit
Tool to predict water molecules placement and energy in ligand binding sites

## Prerequisites

You need, at a minimum (requirements):
* Python (>=3.7)
* OpenBabel
* Numpy 
* Scipy
* Pandas
* tqdm (progress bar)
* AutoDock Vina and autogrid (for generating maps)
* AmberTools (protein preparation and gist calculations)
* OpenMM (minimization)
* ParmED (files conversion)
* gridData (read dx files from GIST)
* Sphinx (documentation)
* Sphinx_rtd_theme (documentation)

## Installation

I highly recommand you to install the Anaconda distribution (https://www.continuum.io/downloads) if you want a clean python environnment with nearly all the prerequisites already installed. To install everything properly, you just have to do this:
```bash
$ conda create -n waterkit -c conda-forge python=3 mkl numpy scipy pandas \
    openbabel parmed ambertools openmm netcdf4 griddataformats tqdm \
    sphinx sphinx_rtd_theme
$ conda activate waterkit
$ pip install vina
```

We can now install the `WaterKit` package
```bash
$ git clone https://github.com/jeeberhardt/waterkit
$ cd waterkit
$ python setup.py build install
```

Finally we will need to compile a version of `autogrid`
```bash
$ cd autogrid
$ autoreconf -i
$ mkdir x86_64Linux2 # for x86_64architecture
$ cd x86_64Linux2
$ ../configure
$ make
$ make install (optional)
```

## Quick tutorial

### Receptor preparation

Conversion to PDBQT using AmberTools and `wk_prepare_receptor.py` script
```bash
$ wk_prepare_receptor.py -i protein.pdb -o protein_prepared --pdb --amber_pdbqt
```

The following protein coordinate files will be generated:```protein_prepared.pdb``` and ```protein_prepared_amber.pdbqt```. The PDBQT file will be used by WaterKit and the PDB file will be used to create the trajectory file at the end.

### Sample water molecule positions with WaterKit

Run WaterKit
```bash
$ mkdir traj
# Generate 10.000 frames using 16 cpus (tip3p, 300 K, 3 hydration layers)
# X Y Z define the center and SX SY SZ the size (in Angstrom) of the box
# If it has issue locating autogrid, specify the path with the argument --autogrid_exec_path
$ run_waterkit.py -i protein_prepared_amber.pdbqt -c X Y Z -s SX SY SZ -n 10000 -j 16 -o traj
# Create ensemble trajectory
$ wk_make_trajectory.py -r protein_prepared.pdb -w traj -o protein
# Minimize each conformation (100 steps, 2.5 kcal/mol/A**2 restraints on heavy atoms, CUDA)
$ wk_minimize_trajectory.py -p protein_system.prmtop -t protein_system.nc -o protein_min.nc
```

### Run Grid Inhomogeneous Solvation Theory (GIST)

1. Create input file for cpptraj
```
# gist.inp
parm protein_system.prmtop
trajin protein.nc
# X Y Z define the center and GX GY GZ the size (in gridpoints) of the box
# Example: if SX = SY = SZ = 24 Angstrom and gridspacn = 0.5, then GX = GY = GZ = 48
gist gridspacn 0.5 gridcntr X Y Z griddim GX GY GZ
go
quit
```

Usually you would choose the same parameters as the AutoGrid maps (```npts``` and ```gridcenter```). Unlike AutoGrid, the default the grid spacing in GIST is 0.5 A, so you will have to choose box dimension accordingly to match the Autogrid maps dimensions. More informations on GIST are available here: https://amber-md.github.io/cpptraj/CPPTRAJ.xhtml#magicparlabel-4672

2. Run GIST

```bash
$ cpptraj -i gist.inp
```

### Identification of hydration sites
Hydration sites are identified based on the oxygen density map (gO) in an iterative way, by selecting the voxel with the highest density, then the second highest and so on, whil keeping a minimum distance of 2.5 A between them. Energies for each of those identified hydration sites are then computed by adding all the surrounding voxels and Gaussian weighted by their distances from the hydration site.

```python
#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
from gridData import Grid
from waterkit.analysis import HydrationSites
from waterkit.analysis import blur_map

gO = Grid("gist-gO.dx")
esw = Grid('gist-Esw-dens.dx')
eww = Grid('gist-Eww-dens.dx')
tst = Grid('gist-dTStrans-dens.dx')
tso = Grid('gist-dTSorient-dens.dx')
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
```

### ????
### PROFIT!!!
