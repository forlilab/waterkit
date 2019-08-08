# Waterkit
Tool to predict water molecules placement in ligand binding sites

## Prerequisites

You need, at a minimum (requirements):
* Python (=2.7)
* OpenBabel
* Numpy 
* Scipy
* Sphinx (documentation)
* Sphinx_rtd_theme (documentation)
* ParmED (conversion to PDBQT file)
* parallel (multicore)

## Installation

I highly recommand you to install the Anaconda distribution (https://www.continuum.io/downloads) if you want a clean python environnment with nearly all the prerequisites already installed. For OpenBabel, you just have to do this:
```bash
conda install openbabel
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

Build documentation with Sphinx
```bash
cd docs
make html
```

Open the file ```build/html/index.html``` with your favorite browser (Google Chrome is evil).

## Quick tutorial

### Receptor preparation
Conversion to PDBQT using AmberTools19 (http://ambermd.org/GetAmber.php) and `amber2pdbqt.py` script
```bash
pdb4amber -i protein.pdb -o protein_clean.pdb --dry --leap-template --nohyd
tleap -s -f leap.template.in > leap.template.out
amber2pdbqt.py -t prmtop -c rst7 -o protein.pdbqt
```

### Grid calculation with autogrid4
1. Create Grid Protein File (GPF)
```
npts 20 20 20
parameter_file AD4_parameters.dat
gridfld protein_maps.fld
spacing 0.375
receptor_types HP HO C3 HC HA O2 C* NA NB C8 CB C CO CN CC H CA O N S CX C2 CR N2 N3 CW CT OH H1 H4 H5
ligand_types OD OW OT
receptor protein.pdbqt
gridcenter 0.0 0.0 0.0
smooth 0
map protein_OD.map
map protein_OW.map
map protein_OT.map
elecmap protein_e.map
dsolvmap protein_d.map
dielectric 1
```

Depending of your system, you would have at least to modify the grid parameters (```npts```, ```gridcenter```) and the receptor atom types list (```receptor_types```). An example of GPF file (```protein_grid.gpf```) as well as the AutoDock parameters (```AD4_parameters.dat```) are provided. Those files are located in the ```data``` waterkit module's directory.

2. Run autogrid4
```bash
autogrid4 -p protein_grid.gpf -l protein_grid.glg
```

### Sample water molecule positions with WaterKit

```bash
mkdir traj
# Generate 10.000 frames with 16 cpu
seq -f "water_%05g" 1 10000 | parallel --jobs 16 python run_waterkit.py -i protein.pdbqt -m protein_maps.fld -o traj/{}
```
