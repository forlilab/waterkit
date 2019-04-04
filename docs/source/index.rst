WaterKit: fast method for estimating receptor desolvation free energy
=====================================================================

Water desolvation is one the driving force of the free energy binding of small molecules to receptor. Thus, understanding the energetic effects of solvation and desolvation of individual water molecules can be crucial when evaluating ligands poses and to improve outcome of high-throughput virtual screening (HTVS). Over the last decades, several methods were developed to tackle this problem, ranging from fast approximate methods (based on empirical functions using either discrete atom-atom pairwise interactions, or continuum solvent models), to more computationally expensive and accurate ones (mostly based on Molecular Dynamics (MD) simulations, such as GIST or Double Decoupling)

On one hand, MD-based methods are prohibitive to use in HTVS to estimate the role of waters on the fly for each ligand. On the other hand, fast and approximate methods show very poor agreement with the results obtained with the more expensive ones.

Here we show a new grid-based sampling method using explicit water molecules, called Waterkit, based on the AutoDock forcefield and can then be integrated directly in the AutoDock docking software. The WaterKit method is able to sample specific regions on the receptor surface, such as the binding site of a receptor, without having to hydrate and simulate  the whole receptor structure. For these hydrated regions thermodynamics properties can be computed using Grid Inhomogeneous Solvation Theory (GIST).

Our results show that the discrete placement of water molecules is successful in reproducing the position of crystallographic waters with very high accuracy. Moreover, preliminary results calculated on the first hydration shell show that WaterKit can produce results comparable with more expensive results from fully-atomistic MD simulations (i.e., GIST), providing an accurate identification of single entropically unfavorable water molecules. Together, those early results show the feasibility of a general and approximated fast method to compute thermodynamic properties of water molecules, and as a first step for a subsequent integration during dockings.

.. _dev-docs:

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: Developer Documentation

   modules

