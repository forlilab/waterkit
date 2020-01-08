[![made-with-python](https://img.shields.io/badge/Made%20with-Python-1f425f.svg)](https://www.python.org/) [![GitHub license](https://img.shields.io/github/license/Naereen/StrapDown.js.svg)](https://github.com/Naereen/StrapDown.js/blob/master/LICENSE) [![PyPI version fury.io](https://img.shields.io/badge/version-0.3-green.svg)](https://pypi.python.org/pypi/ansicolortags/) 

# Waterkit
Tool to predict water molecules placement and energy in ligand binding sites

## Prerequisites

You need, at a minimum (requirements):
* Python (=2.7 || =3.7)
* OpenBabel
* Numpy 
* Scipy
* Pandas
* AmberTools (protein preparation and gist calculations)
* ParmED (files conversion)
* Sphinx (documentation)
* Sphinx_rtd_theme (documentation)

## Installation

I highly recommand you to install the Anaconda distribution (https://www.continuum.io/downloads) if you want a clean python environnment with nearly all the prerequisites already installed. To install everything properly, you just have to do this:
```bash
$ conda create -n waterkit python=3.7
$ conda activate waterkit
$ conda install -c conda-forge -c ambermd -c omnia mkl numpy scipy pandas openbabel=2.4.1 \
    parmed ambertools sphinx sphinx_rtd_theme
```

Finally, we can install the `WaterKit` package
```bash
$ git clone https://github.com/jeeberhardt/waterkit
$ cd waterkit
$ python setup.py build install
```

## Documentation

Build documentation with Sphinx
```bash
$ cd docs
$ make html
```

Open the file ```build/html/index.html``` with your favorite browser (Google Chrome is evil).

## Quick tutorial

### Receptor preparation

Conversion to PDBQT using AmberTools19 and `wk_prepare_receptor.py` script
```bash
$ wk_prepare_receptor.py -i protein.pdb -o protein_prepared --dry --nohyd --pdb --pdbqt
```

The following protein coordinate files will be generated: ```protein_prepared.pdbqt``` and ```protein_prepared.pdb```. The PDBQT file will be used by WaterKit and the PDB file will be used to create the trajectory file at the end.

### Sample water molecule positions with WaterKit

1. Create Grid Protein File (GPF)
```bash
$ create_grid_protein_file.py -r protein_prepared.pdbqt -c 0 0 0 -s 24 24 24 -o protein.gpf
```

2. Pre-calculate grid maps with autogrid4
```bash
$ autogrid4 -p protein_grid.gpf -l protein_grid.glg
```

The AutoDock parameters (```AD4_parameters.dat```) are provided and located in the ```data``` directory of the waterkit module.

3. Run WaterKit
```bash
$ mkdir traj
# Generate 10.000 frames using 16 cpus
$ run_waterkit.py -i protein_prepared.pdbqt -m protein_maps.fld -n 10000 -j 16 -o traj
```

### Run Grid Inhomogeneous Solvation Theory (GIST)

1. Create Amber trajectory with `make_trajectory.py` script
```bash
$ make_trajectory.py -r protein_prepared.pdb -w traj -o protein
$ wk_prepare_receptor.py -i protein_system.pdb -o protein_system
```

2. Create input file for cpptraj
```
# gist.inp
parm protein_system.prmtop
trajin protein.nc
gist gridspacn 0.5 gridcntr 0.0 0.0 0.0 griddim 48 48 48
go
quit
```

Usually you would choose the same parameters as the AutoGrid maps (```npts``` and ```gridcenter```). Unlike AutoGrid, the default the grid spacing in GIST is 0.5 A, so you will have to choose box dimension accordingly to match the Autogrid maps dimensions. More informations on GIST are available here: https://amber-md.github.io/cpptraj/CPPTRAJ.xhtml#magicparlabel-4672

3. Run GIST

```bash
$ cpptraj -i gist.inp
```

### ????
### PROFIT!!!
