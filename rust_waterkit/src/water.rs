use pyo3::prelude::*;
use crate::{atom::Atom, grid::GridPoint, utils::RMIN_HALF_WATER};

#[derive(Clone, Debug)]
// #[pyclass]
/// Water Molecule defined by three atoms.
/// Electrostatics and charges are assigned
/// according to TIP3P forcefield.
pub struct WaterMolecule {
    oxygen: Atom,
    hydrogen_1: Atom,
    hydrogen_2: Atom,
}

// #[pymethods]
impl WaterMolecule {
    // #[new]
    pub fn new(hydrogen_1_coords: [f64; 3],
        hydrogen_2_coords: [f64; 3],
        oxygen_coords: [f64; 3]) -> Self {
            let oxygen = Atom::new("OW".to_string(),
                "0".to_string(),
                oxygen_coords,
                RMIN_HALF_WATER,
                0.6364,
                -0.8340,
            );
            let hydrogen_1 = Atom::new("HW".to_string(),
                "1".to_string(),
                hydrogen_1_coords,
                0.0,
                0.0,
                0.4170,
            );
            let hydrogen_2 = Atom::new("HW".to_string(),
                "2".to_string(),
                hydrogen_2_coords,
                0.0,
                0.0,
                0.4170,
            );
            WaterMolecule {
                oxygen: oxygen,
                hydrogen_1: hydrogen_1,
                hydrogen_2: hydrogen_2,
            }
        }
    
    pub fn from_oxygen_atom(oxygen_atom: GridPoint, hydrogen_1_coords: [f64; 3], hydrogen_2_coords: [f64; 3]) -> Self {
        let oxygen = Atom::new("OW".to_string(),
                "0".to_string(),
                oxygen_atom.coords,
                RMIN_HALF_WATER,
                0.6364,
                -0.8340,
            );
            let hydrogen_1 = Atom::new("HW".to_string(),
                "1".to_string(),
                hydrogen_1_coords,
                0.0,
                0.0,
                0.4170,
            );
            let hydrogen_2 = Atom::new("HW".to_string(),
                "2".to_string(),
                hydrogen_2_coords,
                0.0,
                0.0,
                0.4170,
            );
            WaterMolecule {
                oxygen: oxygen,
                hydrogen_1: hydrogen_1,
                hydrogen_2: hydrogen_2,
            }
    }

    pub fn as_vec(&self) -> Vec<Atom> {
        let atoms = vec![self.oxygen.clone(), self.hydrogen_1.clone(), self.hydrogen_2.clone()];
        atoms
    }
}

