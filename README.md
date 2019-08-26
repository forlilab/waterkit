# Waterkit
Tool to predict water molecules placement and energy in ligand binding sites

## Prerequisites

You need, at a minimum (requirements):
* Python (=2.7)
* OpenBabel
* Numpy 
* Scipy
* AmberTools (protein preparation)
* ParmED (conversion to PDBQT file)
* parallel (multicore)
* mdtraj (SSTMap dependency)
* SSTMap (solvation calculation)
* Sphinx (documentation)
* Sphinx_rtd_theme (documentation)

## Installation

I highly recommand you to install the Anaconda distribution (https://www.continuum.io/downloads) if you want a clean python environnment with nearly all the prerequisites already installed. To install everything properly, you just have to do this:
```bash
conda install -c conda-forge openbabel parmed parallel mdtraj
conda install -c ambermd ambertools
conda install -c solvationtools sstmap
```

To install the `WaterKit` package
```bash
python setup.py install
```

For the documentation only
```bash
conda install sphinx sphinx_rtd_theme
```

## Documentation

<<<<<<< HEAD
### Receptor preparation
1. Add hydrogen atoms and optimize side-chains using REDUCE
```bash
reduce -FLIP protein.pdb > protein_reduce.pdb
```
2. Conversion to PDBQT using AutoDock Tools
```bash
pythonsh prepare_receptor4.py -r protein_reduce.pdb -A 'None' -o protein.pdbqt
```
3. ... and now to mol2 format using OpenBabel
=======
Build documentation with Sphinx
```bash
cd docs
make html
```

Open the file ```build/html/index.html``` with your favorite browser (Google Chrome is evil).

## Quick tutorial

### Receptor preparation

Conversion to PDBQT using AmberTools19 (http://ambermd.org/GetAmber.php) and `amber2pdbqt.py` script
>>>>>>> dev
```bash
pdb4amber -i protein.pdb -o protein_clean.pdb --dry --leap-template --nohyd
tleap -s -f leap.template.in > leap.template.out
amber2pdbqt.py -t prmtop -c rst7 -o protein
```

The following protein coordinate files will be generated: ```protein_prepared.pdbqt``` and ```protein_prepared.pdb```. The PDBQT file will be used by WaterKit and the PDB file will be used to create the trajectory file at the end.

### Grid calculation with autogrid4

1. Create Grid Protein File (GPF)
```
npts 64 64 64
parameter_file AD4_parameters.dat
gridfld protein_maps.fld
spacing 0.375
receptor_types HP HO C3 HC HA O2 C* NA NB C8 CB C CO CN CC H CA O N S CX C2 CR N2 N3 CW CT OH H1 H4 H5
ligand_types OD OW OT
receptor protein_prepared.pdbqt
gridcenter 0.0 0.0 0.0
smooth 0
map protein_OD.map
map protein_OW.map
map protein_OT.map
elecmap protein_e.map
dsolvmap protein_d.map
<<<<<<< HEAD
dielectric -0.1465
nbp_r_eps 2.8 0.315 12 10 OD OA
nbp_r_eps 2.8 0.315 12 10 OD NA
nbp_r_eps 2.8 0.315 12 10 OD SA
nbp_r_eps 2.8 0.315 12 10 OD NS
nbp_r_eps 2.8 0.315 12 10 OD OS
=======
dielectric 1
>>>>>>> dev
```

Depending of your system, you would have at least to modify the grid parameters (```npts```, ```gridcenter```) and the receptor atom types list (```receptor_types```). An example of GPF file (```protein_grid.gpf```) as well as the AutoDock parameters (```AD4_parameters.dat```) are provided. Those files are located in the ```data``` waterkit module's directory.

2. Run autogrid4
```bash
autogrid4 -p protein_grid.gpf -l protein_grid.glg
```

### Sample water molecule positions with WaterKit

```bash
mkdir traj
# Generate 10.000 frames using 16 cpus
seq -f "water_%05g" 1 10000 | parallel --jobs 16 python run_waterkit.py -i protein_prepared.pdbqt -m protein_maps.fld -o traj/{}
```

### Run Grid Inhomogeneous Solvation Theory (GIST) with SSTMap

1. Create Amber trajectory with `make_trajectory.py` script
```bash
python make_trajectory.py -r protein_prepared.pdb -w traj -o protein
tleap -s -f protein.tleap.in > protein.tleap.out
```

2. Run GIST
```bash
run_gist -i protein_system.prmtop -t protein.nc -l ligand.pdb -g 48 48 48 -f 10000
```
The PDB coordinate file ```ligand.pdb``` is a simply PDB file containing a dummy atom that will define the center of the box. Usually you would choose the same parameters as the AutoGrid maps (```npts``` and ```gridcenter```). Unlike AutoGrid, the default the grid spacing in SSTMap is 0.5 A, so you will have to choose box dimension accordingly to match the Autogrid maps dimensions.
