use crate::atom::Atom;
use crate::geometry;
use crate::consts;
use crate::utils;
use crate::vina_ff;
use crate::water;
use crate::water::WaterMolecule;

use rayon::prelude::*;


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

pub fn lennard_jones_rmin_half(epsilon_1: f64, epsilon_2: f64, dist: f64, rmin_half1: f64, rmin_half2: f64) -> f64 {
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
pub fn coulomb_energy(q1: f64, q2: f64, r: f64) -> f64 {
    let k_e = 332.0636; // Electrostatic constant in kcal·Å/(mol·e^2)
    // let dielectric = 1.0; // Dielectric constant of the medium (default: 1.0)
    let coulomb = k_e * (q1 * q2) / r;
    coulomb
}

pub fn dielectric(q1: f64, q2: f64, r: f64) -> f64{
    // self.diel_A = -8.5525
    // self.diel_wat = 78.4
    // self.diel_B = self.diel_wat - self.diel_A
    // self.diel_lambda = 0.003627
    // self.diel_k = 7.7839 
    // self.diel_lB = self.diel_lambda * self.diel_B
    // diels = self.diel_A + (self.diel_B / (1 + self.diel_k * np.exp(
    //     -self.diel_lB * distances)))
    // energies = np.array(qij / distances / diels)
    let k_e = 332.0636;
    let diel_A = -8.5525;
    let diel_wat = 78.4;
    let diel_B = diel_wat - diel_A;
    let diel_lambda = 0.003627;
    let diel_k = 7.7839;
    let diel_lB = diel_lambda * diel_B;
    let dielectric = diel_A + (diel_B / (1.0 + diel_k * (-diel_lB * r).exp()));
    let coulomb = k_e * (q1 * q2) / r / dielectric;
    coulomb
}

pub fn energy(atoms_1: &Vec<Atom>, atoms_2: &Vec<Atom>) -> f64 {
    let mut total_energy = 0.0;
    for atom_1 in atoms_1.iter() {
        for atom_2 in atoms_2.iter() {
            let atom_1_coords = atom_1.coords();
            let atom_2_coords = atom_2.coords();

            // Calculate distance avoiding division by 0
            let distance = f64::max(geometry::euclidean_distance(&atom_1_coords,
                &atom_2_coords), 1e-8_f64.sqrt());

            if distance < consts::ELECTROSTATICS_CUTOFF {
                let mut lj_energy = 0.0;

                if atom_1.atom_type() != &"HW".to_string() && atom_2.atom_type() != &"HW".to_string() {
                    lj_energy = lennard_jones_rmin_half(atom_1.epsilon(),
                        atom_2.epsilon(), 
                        distance,
                        atom_1.rmin_half(),
                        atom_2.rmin_half());
                }

                let coulomb_energy = coulomb_energy(atom_1.charge(), atom_2.charge(), distance);

                // Add to total energy
                total_energy += lj_energy + coulomb_energy;
            }
        }
    }
    total_energy
}

pub fn energy_for_real_water(atoms_1: &Vec<Atom>, atoms_2: &Vec<Atom>) -> f64 {
    let mut e_lj = 0.0;
    let mut e_elec = 0.0;
    for atom_1 in atoms_1.iter() {
        let atom_1_coords = atom_1.coords();
        // Atoms2 are the water's atoms
        for atom_2 in atoms_2.iter() {
            if atom_1.residue_number == atom_2.residue_number {
                continue
            }

            let atom_2_coords = atom_2.coords();
            // Calculate distance avoiding division by 0
            let distance = f64::max(geometry::euclidean_distance(&atom_1_coords,
                &atom_2_coords), 1e-8_f64);
            // if distance < consts::ELECTROSTATICS_CUTOFF {
                if atom_1.atom_type() != &"HW".to_string() && atom_2.atom_type() != &"HW".to_string() {
                    let lj_energy = lennard_jones_rmin_half(atom_1.epsilon(),
                        atom_2.epsilon(), 
                        distance,
                        atom_1.rmin_half(),
                        atom_2.rmin_half());
                    e_lj += lj_energy;
                }
                let mut electrostatics_energy = 0.0;
                if !consts::USE_DIELECTRIC {
                    electrostatics_energy = coulomb_energy(atom_1.charge(), atom_2.charge(), distance);
                } else {
                    electrostatics_energy = dielectric(atom_1.charge(), atom_2.charge(), distance);
                }
                e_elec += electrostatics_energy;
            }
        }
    // }
    e_elec + e_lj
}


/// Grids region
pub fn get_ow_energy(atoms_1: &Vec<Atom>, sphere_center: &[f64; 3]) -> f64{
    let mut total_energy = 0.0;
    for atom_1 in atoms_1.iter() {
        let atom_1_coords = atom_1.coords();

        // Calculate distance avoiding division by 0
        let distance = f64::max(geometry::euclidean_distance(&atom_1_coords,
            sphere_center), 1e-8_f64);
        // if distance < consts::ELECTROSTATICS_CUTOFF {
            // TIP3P
            // if atom_1.atom_type() != &"HW" {
            //     let lj_energy = lennard_jones_rmin_half(atom_1.epsilon(),
            //     consts::TIP3P_EPSILON,
            //     distance,
            //     atom_1.rmin_half(),
            //     consts::RMIN_HALF_WATER);
            //     total_energy += lj_energy;
            // }
            // TIP3PFB
            if atom_1.atom_type() != &"HW" {
                let lj_energy = lennard_jones_rmin_half(atom_1.epsilon(),
                consts::TIP3P_EPSILON,
                distance,
                atom_1.rmin_half(),
                consts::RMIN_HALF_WATER);
                total_energy += lj_energy;
            }
        // }
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

        // if distance < consts::ELECTROSTATICS_CUTOFF {
        let mut electrostatics = 0.0;
        if !consts::USE_DIELECTRIC {
            electrostatics = coulomb_energy(atom_1.charge(), 1.0, distance);
        } else {
            electrostatics = dielectric(atom_1.charge(), 1.0, distance);
        }    
        total_energy += electrostatics;
        // }
    }
    total_energy
}


pub fn update_grid_energies(atoms_1: &Vec<Atom>, sphere_center: &[f64; 3]) -> (f64, f64, f64) {
    let mut total_oda_energy = 0.0;
    let mut total_ow_energy = 0.0;
    let mut total_q_energy = 0.0;

    for atom_1 in atoms_1.iter() {
        let atom_1_coords = atom_1.coords();
        let distance = f64::max(geometry::euclidean_distance(&atom_1_coords, sphere_center), 1e-8_f64);

        if atom_1.atom_type() != &"HW" {
            let lj_energy = lennard_jones_rmin_half(atom_1.epsilon(),
            consts::TIP3P_EPSILON,
            distance,
            atom_1.rmin_half(),
            consts::RMIN_HALF_WATER);
            total_ow_energy += lj_energy;
        }
        let mut electrostatics = 0.0;
        if !consts::USE_DIELECTRIC {
            let electrostatics = coulomb_energy(atom_1.charge(), 1.0, distance);
        } else {
            let electrostatics = dielectric(atom_1.charge(), 1.0, distance);
        }
        total_q_energy += electrostatics;

        if distance < consts::VINA_DISTANCE_CUTOFF {
            if atom_1.is_heavy_atom() {
                let rijs = atom_1.vina_rij() + consts::VINA_O_RIJ;
                let vg1 = vina_ff::vina_gauss1(&distance, &rijs) * consts::VINA_GAUSS1_W;
                let vg2 = vina_ff::vina_gauss2(&distance, &rijs) * consts::VINA_GAUSS2_W;
                let rep = vina_ff::vina_repulsion(&distance, &rijs) * consts::VINA_REPULSION_W;
                // println!()
                total_oda_energy += vg1 + vg2 + rep;
                
                // only donors or acceptors contribute to this term
                if atom_1.is_vina_acceptor() || atom_1.is_vina_donor() {
                    let hb = vina_ff::vina_hb(&distance, &rijs) * consts::VINA_HB_W;
                    total_oda_energy += hb;
                }
            }
        }
    }
    (total_oda_energy, total_ow_energy, total_q_energy)
}


pub fn set_waters_energies(water_molecules: &mut Vec<WaterMolecule>, receptor_atoms: &Vec<Atom>) {
    let waters_clone = water_molecules.clone();
    for (water_idx, water) in water_molecules.iter_mut().enumerate() {
        let other_waters: Vec<&WaterMolecule> = waters_clone.iter().filter(|x| *x != water).collect();
        let mut water_energy = 0.0;
        let water_atoms = water.as_vec();
        water_energy += energy_for_real_water(receptor_atoms, &water_atoms);
        for other_water in other_waters {
            water_energy += energy_for_real_water(&water_atoms, &other_water.as_vec());
        }
        water.set_energy(water_energy);
    }
}

pub fn get_system_energy(water_molecules: &Vec<WaterMolecule>, receptor_atoms: &Vec<Atom>) -> (f64, f64) {
    let mut total_energy = 0.0;
    let mut water_energy = 0.0;
    let mut receptor_energy = 0.0;
    let n_waters = water_molecules.len();
    for index_1 in 0..n_waters {
        let water_1_atoms = &water_molecules[index_1].as_vec();
        for index_2 in index_1+1..n_waters {
            let water_2_atoms = &water_molecules[index_2].as_vec();
            water_energy += energy_for_real_water(&water_1_atoms, &water_2_atoms);
        }
        receptor_energy += energy_for_real_water(&water_1_atoms, &receptor_atoms);
    }
    (utils::round(water_energy), utils::round(receptor_energy))
}

// pub fn get_system_energy(water_molecules: &Vec<WaterMolecule>, receptor_atoms: &Vec<Atom>) -> (f64, f64) {
//     let (water_energy, receptor_energy): (f64, f64) = water_molecules
//         .par_iter()
//         .enumerate()
//         .map(|(index_1, water_1)| {
//             let water_1_atoms = water_1.as_vec();
//             let mut local_water_energy = 0.0;
//             let mut local_receptor_energy = 0.0;

//             // Water-water interactions
//             for water_2 in water_molecules.iter().skip(index_1 + 1) {
//                 local_water_energy += energy_for_real_water(&water_1_atoms, &water_2.as_vec());
//             }
//             // Water-receptor interactions
//             local_receptor_energy += energy_for_real_water(&receptor_atoms, &water_1_atoms);

//             (local_water_energy, local_receptor_energy)
//         })
//         .reduce(
//             || (0.0, 0.0),
//             |(w1, r1), (w2, r2)| (w1 + w2, r1 + r2),
//         );

//     // Round to three decimal places
//     ((water_energy * 1000.0).round() / 1000.0, (receptor_energy * 1000.0).round() / 1000.0)
// }