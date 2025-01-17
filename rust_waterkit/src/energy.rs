use pyo3::prelude::*;
use crate::{atom::Atom, geometry::euclidean_distance, spheric_probe::Sphere};


/// Calculate the Lennard-Jones interaction energy
/// 
/// Parameters:
///     r (&f64): Distance between two atoms (in angstroms).
///     epsilon (&f64): Depth of the potential well (in kcal/mol).
///     sigma (&f64): Distance at which the potential is zero (in angstroms).
/// 
/// Returns:
///     f64: Lennard-Jones energy (in kcal/mol).
pub fn lennard_jones(r: &f64, epsilon: &f64, sigma: &f64) -> f64 {
    let term = (sigma / r).powi(6);
    let lj = 4.0 * epsilon * (term.powi(2) - term);
    lj
}

/// Calculate the Coulomb interaction energy.
/// Parameters:
///     q1 (&f64): Charge of the first atom (in e).
///     q2 (&f64): Charge of the second atom (in e).
///     r (&f64): Distance between two atoms (in angstroms).
/// 
/// Returns:
///     f64: Coulomb energy (in kcal/mol).
pub fn coulomb_energy(q1: &f64, q2: &f64, r: &f64) -> f64 {
    let k_e = 332.0636; // Electrostatic constant in kcal·Å/(mol·e^2)
    let dielectric = 1.0; // Dielectric constant of the medium (default: 1.0)
    let coulomb = k_e * (q1 * q2) / (dielectric * r);
    coulomb
}


#[pyfunction]
pub fn energy(atoms_1: Vec<Atom>, atoms_2: Vec<Atom>) -> f64 {
    let mut total_energy = 0.0;
    for atom_1 in atoms_1.iter() {
        for atom_2 in atoms_2.iter() {
            let atom_1_coords = atom_1.coords();
            let atom_2_coords = atom_2.coords();
            
            // Calculate distance
            let r = euclidean_distance(&atom_1_coords, &atom_2_coords);
            
            // Combine parameters using Lorentz-Berthelot rules
            let epsilon = (atom_1.epsilon() * atom_2.epsilon()).sqrt();
            let sigma = (atom_1.sigma() + atom_2.sigma()) / 2.0;
            
            let lj_energy = lennard_jones(&r, &epsilon, &sigma);
            let coulomb_energy = coulomb_energy(atom_1.charge(), atom_2.charge(), &r);
            
            // Add to total energy
            total_energy += lj_energy + coulomb_energy;
        }
    }
    total_energy
}

pub fn spheric_energy(atoms_1: Vec<Atom>, sphere: Sphere) -> f64 {
    let mut total_energy = 0.0;
    for atom_1 in atoms_1.iter() {
        let atom_1_coords = atom_1.coords();
        let sphere_coords = sphere.coords();

        // Calculate distance
        let r = euclidean_distance(&atom_1_coords, &sphere_coords);
        
        if r < (atom_1.sigma() + sphere.radius()) {
            let epsilon = (atom_1.epsilon() * sphere.epsilon()).sqrt();
            let sigma = (atom_1.sigma() + sphere.radius()) / 2.0;
            let lj_energy = lennard_jones(&r, &epsilon, &sigma);
            total_energy += lj_energy; 
        }
    }
    total_energy
}