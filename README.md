# Waterkit
Tool to predict water molecules placement in ligand binding sites

## Prerequisites

You need, at a minimum (requirements):
* Python (=2.7)
* OpenBabel
* Numpy 
* Scipy

## Installation

I highly recommand you to install the Anaconda distribution (https://www.continuum.io/downloads) if you want a clean python environnment with nearly all the prerequisites already installed. For OpenBabel, you just have to do this:
```bash
conda install openbabel
```

## Documentation

Build documentation with Sphinx
```bash
cd docs
make html
```

Open file ```build/html/index.html``` with your favorite browser (not Google Chrome).

## Quick tutorial

### Receptor preparation
Conversion to PDBQT using AutoDock Tools
```bash
pythonsh prepare_receptor4.py -r protein.pdb -C -o protein.pdbqt
```

### Grid calculation with autogrid4
1. Create Grid Protein File (GPF)
```
npts 20 20 20                        
parameter_file AD4_parameters.dat
gridfld protein_maps.fld        
spacing 0.375
disorder_h 
receptor_types A N NA C OA SA HD
ligand_types OW
receptor protein.pdbqt          
gridcenter 0.0 0.0 0.0      
smooth 0.5                           
map protein_OW.map
elecmap protein_e.map
dsolvmap protein_d.map
dielectric -0.1465
```

Depending of your system, you would have at least to modify the grid parameters (```npts```, ```gridcenter```) and the receptor atom types list (```receptor_types```). The ```disorder_h``` and ```nbp_r_eps``` options are mandatory in order to optimize the hydroxyl position and to place the spherical water (OD) around acceptor atoms (OA). An example of GPF file (```protein_grid.gpf```) as well as the AutoDock parameters (```AD4_parameters.dat```) are provided. Those files are located in the ```data``` waterkit module's directory.

2. Run autogrid4
```bash
autogrid4 -p protein_grid.gpf -l protein_grid.glg
```

### Predict water molecules position with WaterKit

1. Run WaterKit:
```bash
python run_waterkit.py -i protein.pdbqt -m protein_maps.fld -o waters
```

2. ???
3. PROFIT
