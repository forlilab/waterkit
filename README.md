[![made-with-python](https://img.shields.io/badge/Made%20with-Python-1f425f.svg)](https://www.python.org/) [![GitHub license](https://img.shields.io/github/license/Naereen/StrapDown.js.svg)](https://github.com/Naereen/StrapDown.js/blob/master/LICENSE) [![PyPI version fury.io](https://img.shields.io/badge/version-0.3-green.svg)](https://pypi.python.org/pypi/ansicolortags/) 

# Waterkit
Tool to predict water molecules placement and energy in ligand binding sites

## Prerequisites

You need, at a minimum (requirements):
* Python (=2.7)
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
$ conda install -c conda-forge openbabel parmed
$ conda install -c ambermd ambertools
```

To install the `WaterKit` package
```bash
$ python setup.py install
```

For the documentation only
```bash
$ conda install sphinx sphinx_rtd_theme
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

Conversion to PDBQT using AmberTools19 (http://ambermd.org/GetAmber.php) and `amber2pdbqt.py` script
```bash
$ pdb4amber -i protein.pdb -o protein_clean.pdb --dry --leap-template --nohyd
$ tleap -s -f leap.template.in > leap.template.out
$ python amber2pdbqt.py -t prmtop -c rst7 -o protein
```

The following protein coordinate files will be generated: ```protein_prepared.pdbqt``` and ```protein_prepared.pdb```. The PDBQT file will be used by WaterKit and the PDB file will be used to create the trajectory file at the end.

### Grid calculation with autogrid4

1. Create Grid Protein File (GPF)
```
# protein_grid.gpf
npts 64 64 64
parameter_file AD4_parameters.dat
gridfld protein_maps.fld
spacing 0.375
receptor_types HP HO C3 HC HA O2 C* NA NB C8 CB C CO CN CC H CA O N S CX C2 CR N2 N3 CW CT OH H1 H4 H5
ligand_types SW OW OT
receptor protein_prepared.pdbqt
gridcenter 0.0 0.0 0.0
smooth 0
map protein_SW.map
map protein_OW.map
map protein_OT.map
elecmap protein_e.map
dsolvmap protein_d.map
dielectric 1
```

Depending of your system, you would have at least to modify the grid parameters (```npts```, ```gridcenter```) and the receptor atom types list (```receptor_types```). An example of GPF file (```protein_grid.gpf```) as well as the AutoDock parameters (```AD4_parameters.dat```) are provided. Those files are located in the ```data``` waterkit module's directory.

2. Run autogrid4
```bash
$ autogrid4 -p protein_grid.gpf -l protein_grid.glg
```

### Sample water molecule positions with WaterKit

```bash
$ mkdir traj
# Generate 10.000 frames using 16 cpus
$ python run_waterkit.py -i protein_prepared.pdbqt -m protein_maps.fld -n 10000 -j 16 -o traj
```

### Run Grid Inhomogeneous Solvation Theory (GIST) with SSTMap

1. Create Amber trajectory with `make_trajectory.py` script
```bash
$ python make_trajectory.py -r protein_prepared.pdb -w traj -o protein
$ tleap -s -f protein.leap.in > protein.leap.out
```

2. Run GIST
```
# gist.inp
parm protein_system.prmtop
trajin protein.nc
gist gridspacn 0.5 gridcntr 0.0 0.0 0.0 griddim 48 48 48
go
quit
```

Usually you would choose the same parameters as the AutoGrid maps (```npts``` and ```gridcenter```). Unlike AutoGrid, the default the grid spacing in GIST is 0.5 A, so you will have to choose box dimension accordingly to match the Autogrid maps dimensions. More informations on GIST are available here: https://amber-md.github.io/cpptraj/CPPTRAJ.xhtml#magicparlabel-4672

```bash
$ cpptraj -i gist.inp
```

### ????
### PROFIT!!!
