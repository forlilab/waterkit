use pyo3::prelude::*;

#[derive(Clone, Debug, PartialEq)]
#[pyclass]
pub struct Atom {
    // Atom type for the forcefield
    atom_type: String,

    // Atom id with chain:residue:atom
    atom_id: String,

    // Atom number 
    pub residue_number: i32,

    // 3D coordinates of the atom
    coords: [f64; 3],

    // sigma value of the VdW
    rmin_half: f64,

    // epsilon value of the VdW
    epsilon: f64,

    // partial charge for coulomb
    charge: f64,

    // vina parameters
    vina_rij: f64,

    vina_donor: bool,

    vina_acceptor: bool
}

#[pymethods]
impl Atom {
    #[new]
    pub fn new(atom_type: String, atom_id: String, coords_point: [f64; 3], rmin_half: f64, epsilon: f64, charge: f64, vina_rij: f64, vina_donor: bool, vina_acceptor: bool) -> Atom {
        let mut atom = Self {
            atom_type: atom_type,
            atom_id: atom_id,
            residue_number: 0,
            coords: coords_point,
            rmin_half: rmin_half,
            epsilon: epsilon,
            charge: charge,
            vina_rij: vina_rij,
            vina_donor: vina_donor,
            vina_acceptor: vina_acceptor
        };
        atom.set_atom_number();
        atom
    }

    pub fn atom_type(&self) -> &String {
        &self.atom_type
    }

    pub fn atom_id(&self) -> &String {
        &self.atom_id
    }

    pub fn set_atom_number(&mut self) {
        let splitted: Vec<&str> = self.atom_id().split(":").collect();
        self.residue_number = splitted[2]
            .parse::<i32>()
            .expect("Failed to parse atom number")
    }

    pub fn coords(&self) -> [f64; 3] {
        self.coords
    }

    pub fn set_coords(&mut self, new_coords: [f64; 3]) {
        self.coords = new_coords;
    }

    pub fn rmin_half(&self) -> f64 {
        self.rmin_half
    }

    pub fn epsilon(&self) -> f64 {
        self.epsilon
    }

    pub fn charge(&self) -> f64 {
        self.charge
    }

    pub fn is_heavy_atom(&self) -> bool {
        !self.atom_type.starts_with("H") 
    }

    pub fn is_vina_donor(&self) -> bool {
        self.vina_donor
    }

    pub fn is_vina_acceptor(&self) -> bool {
        self.vina_acceptor
    }

    pub fn vina_rij(&self) -> &f64 {
        &self.vina_rij
    }
}