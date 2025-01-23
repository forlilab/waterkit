use pyo3::prelude::*;

#[derive(Clone, Debug)]
#[pyclass]
pub struct Atom {
    // Atom type for the forcefield
    atom_type: String,

    // Atom id with chain:residue:atom
    atom_id: String,

    // 3D coordinates of the atom
    coords: [f64; 3],

    // sigma value of the VdW
    rmin_half: f64,

    // epsilon value of the VdW
    epsilon: f64,

    // partial charge for coulomb
    charge: f64,
}

#[pymethods]
impl Atom {
    #[new]
    pub fn new(atom_type: String, atom_id: String, coords_point: [f64; 3], rmin_half: f64, epsilon: f64, charge: f64) -> Atom {
        let atom = Self {
            atom_type: atom_type,
            atom_id: atom_id,
            coords: coords_point,
            rmin_half: rmin_half,
            epsilon: epsilon,
            charge: charge
        };

        atom
    }

    pub fn atom_type(&self) -> &String {
        &self.atom_type
    }

    pub fn atom_id(&self) -> &String {
        &self.atom_id
    }

    pub fn coords(&self) -> [f64; 3] {
        self.coords
    }

    pub fn rmin_half(&self) -> &f64 {
        &self.rmin_half
    }

    pub fn epsilon(&self) -> &f64 {
        &self.epsilon
    }

    pub fn charge(&self) -> &f64 {
        &self.charge
    }

    pub fn is_heavy_atom(&self) -> bool {
        let atom_id = self.atom_id();
        atom_id != "H" && atom_id != "HW"
    }
}