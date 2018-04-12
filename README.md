# Waterkit
Tool to predict hydration of molecules

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

### Usage

* Finger in the nose:
```bash
python run_waterkit.py -i protein.mol2 -m protein_maps.fld -o water.pdbqt
```

* Raccoon mode:
```bash
python run_waterkit.py -i protein.mol2 -m protein_maps.fld -o water.pdbqt -f waterfield.par -w water/maps.fld
```
