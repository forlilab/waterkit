use crate::atom::Atom;
use crate::geometry;
use crate::consts;


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

pub fn lennard_jones_rmin_half(epsilon_1: &f64, epsilon_2: &f64, dist: &f64, rmin_half1: &f64, rmin_half2: &f64) -> f64 {
    let rmin = rmin_half1 + rmin_half2;
    let epsilon = (epsilon_1 * epsilon_2).sqrt();
    let lj = epsilon * ((rmin / dist).powi(12) - (2.0 * (rmin / dist).powi(6)));
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
    // let dielectric = 1.0; // Dielectric constant of the medium (default: 1.0)
    let coulomb = k_e * ((q1 * q2) / r);
    coulomb
}


pub fn energy(atoms_1: &Vec<Atom>, atoms_2: &Vec<Atom>) -> f64 {
    let mut total_energy = 0.0;
    for atom_1 in atoms_1.iter() {
        for atom_2 in atoms_2.iter() {
            let atom_1_coords = atom_1.coords();
            let atom_2_coords = atom_2.coords();

            // Calculate distance avoiding division by 0
            let r = f64::max(geometry::euclidean_distance(&atom_1_coords,
                &atom_2_coords), 1e-8_f64.sqrt());

            if r < consts::ELECTROSTATICS_CUTOFF {
                let mut lj_energy = 0.0;

                if atom_1.atom_type() != &"HW".to_string() && atom_2.atom_type() != &"HW".to_string() {
                    lj_energy = lennard_jones_rmin_half(atom_1.epsilon(),
                        atom_2.epsilon(), &r,
                        atom_1.rmin_half(),
                        atom_2.rmin_half());
                }

                let coulomb_energy = coulomb_energy(atom_1.charge(), atom_2.charge(), &r);

                // Add to total energy
                total_energy += lj_energy + coulomb_energy;
            }
        }
    }
    total_energy
}

pub fn energy_for_real_water(atoms_1: &Vec<Atom>, atoms_2: &Vec<Atom>) -> f64 {
    let mut total_energy = 0.0;
    for atom_1 in atoms_1.iter() {
        let atom_1_coords = atom_1.coords().clone();
        // let mut cnt = 0;
        // Atoms2 are the water's atoms
        for atom_2 in atoms_2.iter() {
            let atom_2_coords = atom_2.coords().clone();
            // Calculate distance avoiding division by 0
            let r = f64::max(geometry::euclidean_distance(&atom_1_coords,
                &atom_2_coords), 1e-8_f64);

            if r < consts::ELECTROSTATICS_CUTOFF {
                if atom_1.atom_type() != &"HW".to_string() && atom_2.atom_type() != &"HW".to_string() {
                    // println!("Atom1: {}; Atom2: {}", atom_1.atom_type(), atom_2.atom_type());
                    let lj_energy = lennard_jones_rmin_half(atom_1.epsilon(),
                        atom_2.epsilon(), &r,
                        atom_1.rmin_half(),
                        atom_2.rmin_half());
                    total_energy += lj_energy;
                    // println!("LJ: {}", lj_energy);
                }

                let electrostatics_energy = coulomb_energy(atom_1.charge(), atom_2.charge(), &r);
                // println!("Coulomb: {}", electrostatics_energy);
                // Add to total energy
                total_energy += electrostatics_energy;
            }
        }
    }
    total_energy
}


/// Grids region
pub fn get_ow_energy(atoms_1: &Vec<Atom>, sphere_center: &[f64; 3]) -> f64{
    let mut total_energy = 0.0;
    for atom_1 in atoms_1.iter() {
        let atom_1_coords = atom_1.coords();

        // Calculate distance avoiding division by 0
        let distance = f64::max(geometry::euclidean_distance(&atom_1_coords,
            sphere_center), 1e-8_f64);
        if distance < consts::ELECTROSTATICS_CUTOFF {
            if atom_1.atom_type() != &"HW" {
                let lj_energy = lennard_jones_rmin_half(atom_1.epsilon(),
                &consts::TIP3P_EPSILON,
                &distance,
                atom_1.rmin_half(),
                &consts::RMIN_HALF_WATER);
                total_energy += lj_energy;
            }
        }
    }
    total_energy
}

pub fn get_q_energy(atoms_1: &Vec<Atom>, sphere_center: &[f64; 3]) -> f64 {
    let mut total_energy = 0.0;
    for atom_1 in atoms_1.iter() {
        let atom_1_coords = atom_1.coords();

        // Calculate distance avoiding division by 0
        let distance = f64::max(geometry::euclidean_distance(&atom_1_coords,
            sphere_center), 1e-8_f64);

        if distance < consts::ELECTROSTATICS_CUTOFF {
            let electrostatics = coulomb_energy(atom_1.charge(), &1.0, &distance);
            total_energy += electrostatics;
        }
    }
    total_energy
}