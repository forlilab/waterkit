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

## How-To
```bash
python waterkit/waterkit.py --pdbqt protein.mol2 --map protein_maps.fld -f waterkit/waterfield_0.1.par -w docs/water/water_maps.fld -o water.pdbqt
```
