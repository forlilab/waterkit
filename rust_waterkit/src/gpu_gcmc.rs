use core::f32;

use crate::atom::Atom;
use crate::{consts, gpu_gcmc_moves, gpu_geometry};
use crate::water::WaterMolecule;
use cubecl::std::tensor::TensorHandle;
use cubecl::{compute, prelude::*};
use cubecl_random::random_normal;
use fixed::types::extra::Unsigned;
use nalgebra::Scalar;
use rand::distributions::Uniform;
use rand::prelude::Distribution;
use rand::rngs::ThreadRng;
use rand::{random, Rng};

const MAX_N_WATERS: u32 = 300;
const ATOM_FEATURES: u32 = 7;
const MOVE_TYPE_IDX: u32 = 0;
const TRANSLATION_X_IDX: u32 = 1;
const TRANSLATION_Y_IDX: u32 = 2;
const TRANSLATION_Z_IDX: u32 = 3;
const ROTATION_IDX: u32 = 4;
const ROT_AXIS_X_IDX: u32 = 5;
const ROT_AXIS_Y_IDX: u32 = 6;
const ROT_AXIS_Z_IDX: u32 = 7;
const INSERTION_X_IDX: u32 = 8;
const INSERTION_Y_IDX: u32 = 9;
const INSERTION_Z_IDX: u32 = 10;
const ACCEPTANCE_IDX: u32 = 11;

#[cube]
fn compute_interactions<F: Float>(
    receptor_atoms: &Array<F>,
    water_atoms: &Array<F>,
    target_water: &Array<F>,
    energies: &mut Array<F>,
    sim_id: u32
) {
    let n_receptor_atoms = receptor_atoms.len() / ATOM_FEATURES;
    let n_water_atoms = water_atoms.len() / ATOM_FEATURES;
    let base_energy_idx = sim_id * (n_receptor_atoms + n_water_atoms);

    for idx in 0..n_receptor_atoms+n_water_atoms {
        let cutoff_sq = F::new(12.0*12.0);
        let mut energy = F::new(0.0);

        // Fixed: multiply idx by NUM_FEATURES to get correct base index
        let base_idx = idx * ATOM_FEATURES;
        let x = receptor_atoms[base_idx];
        let y = receptor_atoms[base_idx + 1];
        let z = receptor_atoms[base_idx + 2];
        let charge = receptor_atoms[base_idx + 3];
        let epsilon = receptor_atoms[base_idx + 4];
        let rmin_half = receptor_atoms[base_idx + 5];
        let resnum = receptor_atoms[base_idx + 6];
        
        for j in 0..(target_water.len() / ATOM_FEATURES) {
            let start_idx = j * ATOM_FEATURES;
            // Fixed: Read from target_water instead of atoms
            let wat_x = target_water[start_idx];
            let wat_y = target_water[start_idx + 1];
            let wat_z = target_water[start_idx + 2];
            let wat_charge = target_water[start_idx + 3];
            let wat_epsilon = target_water[start_idx + 4];
            let wat_rmin_half = target_water[start_idx + 5];
            let wat_resnum = target_water[start_idx + 6];
            
            if resnum != wat_resnum {
                let dx = x - wat_x;
                let dy = y - wat_y;
                let dz = z - wat_z;
                let r2 = dx * dx + dy * dy + dz * dz;

                // if r2 < cutoff_sq {
                    let r = F::sqrt(r2);
                    let rmin_half1 = rmin_half;
                    let rmin_half2 = wat_rmin_half;
                    let epsilon1 = epsilon;
                    let epsilon2 = wat_epsilon;
                    let charge1 = charge;
                    let charge2 = wat_charge;
                    let rmin = rmin_half1 + rmin_half2;
                    // let sigma_mixed = (sigma1 + sigma2) * F::new(0.5);
                    let epsilon_mixed = F::sqrt(epsilon1 * epsilon2);
                    
                    let mut lj = F::new(0.0);
                    
                    // No HW in VdW
                    if j % 3 == 0 {
                        lj = epsilon * (F::powf((rmin / r), F::new(12.0)) - (F::new(2.0) * F::powf((rmin / r), F::new(6.0))));
                        // lj = F::new(4.0) * epsilon_mixed * (sigma_over_r12 - sigma_over_r6);
                    }
                    let coulomb = F::new(332.0636) * charge1 * charge2 / r;
                    energy += lj + coulomb;
                // }
            }
        }

        // // Now process the waters
        // // Fixed: multiply idx by NUM_FEATURES to get correct base index
        // let base_idx = idx * ATOM_FEATURES;
        // let x = water_atoms[base_idx];
        // let y = water_atoms[base_idx + 1];
        // let z = water_atoms[base_idx + 2];
        // let charge = water_atoms[base_idx + 3];
        // let epsilon = water_atoms[base_idx + 4];  // Make sure this matches your packing order
        // let rmin_half = water_atoms[base_idx + 5];
        // let resnum = water_atoms[base_idx + 6];
        
        // for j in 0..(target_water.len() / ATOM_FEATURES) {
        //     let start_idx = j * ATOM_FEATURES;
        //     // Fixed: Read from target_water instead of atoms
        //     let wat_x = target_water[start_idx];
        //     let wat_y = target_water[start_idx + 1];
        //     let wat_z = target_water[start_idx + 2];
        //     let wat_charge = target_water[start_idx + 3];
        //     let wat_epsilon = target_water[start_idx + 4];
        //     let wat_rmin_half = target_water[start_idx + 5];
        //     let wat_resnum = target_water[start_idx + 6];
            
        //     if resnum != wat_resnum {
        //         let dx = x - wat_x;
        //         let dy = y - wat_y;
        //         let dz = z - wat_z;
        //         let r2 = dx * dx + dy * dy + dz * dz;

        //         // if r2 < cutoff_sq {
        //             let r = F::sqrt(r2);
        //             let rmin_half1 = rmin_half;
        //             let rmin_half2 = wat_rmin_half;
        //             let epsilon1 = epsilon;
        //             let epsilon2 = wat_epsilon;
        //             let charge1 = charge;
        //             let charge2 = wat_charge;
        //             let rmin = rmin_half1 + rmin_half2;
        //             // let sigma_mixed = (sigma1 + sigma2) * F::new(0.5);
        //             let epsilon_mixed = F::sqrt(epsilon1 * epsilon2);
                    
        //             let mut lj = F::new(0.0);
                    
        //             // No HW in VdW
        //             if j % 3 == 0 && base_idx % 3 == 0 {
        //                 lj = epsilon * (F::powf((rmin / r), F::new(12.0)) - (F::new(2.0) * F::powf((rmin / r), F::new(6.0))));
        //             }
        //             let coulomb = F::new(332.0636) * charge1 * charge2 / r;
        //             energy += lj + coulomb;
        //     }
        // }
        energies[base_energy_idx + idx] = energy;
    }
}

// Pre-generating these random numbers to avoid overhead during the simulation
// These numbers will be copied to GPU memory before starting the simulation
pub fn prepare_random_numbers(n_epochs: usize, 
    boundaries: &Vec<f32>, 
    translation_lower_bound: f32,
    translation_upper_bound: f32, 
    rotation_lower_bound: f32,
    rotation_upper_bound: f32,) -> Vec<f32> {
    
    // Get boundary values
    let x_min = boundaries[0];
    let x_max = boundaries[1];
    let y_min = boundaries[2];
    let y_max = boundaries[3];
    let z_min = boundaries[4];
    let z_max = boundaries[5];

    let mut random_numbers = Vec::with_capacity(n_epochs * 12); // 1 for move type, 3 for translation, 1 for rotation, 3 for rotation axis, 3 for insertion, 1 for acceptance probability

    for _ in 0..n_epochs {
        let mut rng = rand::thread_rng();
        let delta = [
            Uniform::from(translation_lower_bound..translation_upper_bound).sample(&mut rng),
            Uniform::from(translation_lower_bound..translation_upper_bound).sample(&mut rng),
            Uniform::from(translation_lower_bound..translation_upper_bound).sample(&mut rng),
        ];
        let insertion = [
            Uniform::from(x_min..x_max).sample(&mut rng),
            Uniform::from(y_min..y_max).sample(&mut rng),
            Uniform::from(z_min..z_max).sample(&mut rng),
        ];

        let axis = [rng.gen(), rng.gen(), rng.gen()];

        let angle= rng.gen_range(-180_f64..180_f64).to_radians() as f32;
        let probability_acc = Uniform::from(0.0..1.0).sample(&mut rng);
        
        // random_numbers.push(rng.gen_range(0..3) as f32); // move type
        random_numbers.push(1.0);
        random_numbers.push(delta[0]); // translation x
        random_numbers.push(delta[1]); // translation y
        random_numbers.push(delta[2]); // translation z
        random_numbers.push(angle); // rotation
        random_numbers.push(axis[0]); // rotation axis x
        random_numbers.push(axis[1]); // rotation axis y
        random_numbers.push(axis[2]); // rotation axis z
        random_numbers.push(insertion[0]); // insertion x
        random_numbers.push(insertion[1]); // insertion y
        random_numbers.push(insertion[2]); // insertion z
        random_numbers.push(probability_acc); // acceptance probability
    }
    random_numbers
}

// #[cube]
// fn propose_perturbation<F: Float>(
//     boundaries: &Array<F>,
//     old_water: &Array<F>, // Flattened water coordinates [ox, oy, oz, charge_o, epsilon_o, rmin_half_o, resnum_o, h1x, h1y, h1z, charge_h1, epsilon_h1, rmin_half_h1, resnum_h1, h2x, h2y, h2z, charge_h2, epsilon_h2, rmin_half_h2, resnum_h2]
//     rng_array: &Array<F>,
//     epoch: u32) -> Array<F> {

//     let mut new_water = Array::new(ATOM_FEATURES * 3);
//     // for idx in 0..old_water.len() {
//     //     new_water[idx] = old_water[idx];
//     // }
    
//     new_water[0] = old_water[0];
//     new_water[1] = old_water[1];
//     new_water[2] = old_water[2];
//     new_water[3] = old_water[3];
//     new_water[4] = old_water[4];
//     new_water[5] = old_water[5];
//     new_water[6] = old_water[6];
//     new_water[7] = old_water[7];
//     new_water[8] = old_water[8];
//     new_water[9] = old_water[9];
//     new_water[10] = old_water[10];
//     new_water[11] = old_water[11];
//     new_water[12] = old_water[12];
//     new_water[13] = old_water[13];
//     new_water[14] = old_water[14];
//     new_water[15] = old_water[15];
//     new_water[16] = old_water[16];
//     new_water[17] = old_water[17];
//     new_water[18] = old_water[18];
//     new_water[19] = old_water[19];
//     new_water[20] = old_water[20];

//     let rnd_idx = (CUBE_POS_X + epoch) * 12;
//     let delta_x = rng_array[rnd_idx + TRANSLATION_X_IDX];
//     let delta_y = rng_array[rnd_idx + TRANSLATION_Y_IDX];
//     let delta_z = rng_array[rnd_idx + TRANSLATION_Z_IDX];
//     let angle_rnd = rng_array[rnd_idx + ROTATION_IDX];
//     let axis_x = rng_array[rnd_idx + ROT_AXIS_X_IDX];
//     let axis_y = rng_array[rnd_idx + ROT_AXIS_Y_IDX];
//     let axis_z = rng_array[rnd_idx + ROT_AXIS_Z_IDX];

//     // Get boundary values
//     let x_min = boundaries[0];
//     let x_max = boundaries[1];
//     let y_min = boundaries[2];
//     let y_max = boundaries[3];
//     let z_min = boundaries[4];
//     let z_max = boundaries[5];
    
//     // Get current oxygen position
//     let ox = old_water[0];
//     let oy = old_water[1];
//     let oz = old_water[2];
    
//     // Calculate new oxygen position
//     let new_ox = ox + delta_x;
//     let new_oy = oy + delta_y;
//     let new_oz = oz + delta_z;
    
//     // Check boundaries
//     let in_bounds = (new_ox >= x_min) && (new_ox <= x_max) &&
//                    (new_oy >= y_min) && (new_oy <= y_max) &&
//                    (new_oz >= z_min) && (new_oz <= z_max);
    
//     if in_bounds {
//         // Apply translation to all atoms
//         new_water[0] = new_ox; // oxygen x
//         new_water[1] = new_oy; // oxygen y
//         new_water[2] = new_oz; // oxygen z
        
        
//         // Simple rotation around oxygen (optional)
//         let angle = angle_rnd; // Small rotation
//         let cos_a = F::cos(angle);
//         let sin_a = F::sin(angle);
        
//         // Rotate hydrogen atoms around oxygen (z-axis rotation for simplicity)
//         let mut h1 = Array::new(3);
//         h1[0] = new_water[7];
//         h1[1] = new_water[8];
//         h1[2] = new_water[9];

//         let mut h2 = Array::new(3);
//         h2[0] = new_water[14];
//         h2[1] = new_water[15];
//         h2[2] = new_water[16];

//         let mut new_oxygen = Array::new(3);
//         new_oxygen[0] = new_ox;
//         new_oxygen[1] = new_oy;
//         new_oxygen[2] = new_oz; 

//         let mut axis_array = Array::new(3);
//         axis_array[0] = axis_x;
//         axis_array[1] = axis_y;
//         axis_array[2] = axis_z; 

//         axis_array = gpu_geometry::normalize(&mut axis_array);

//         let mut new_h1: Array::<F> = Array::new(3);
//         rodrigues_rotation(&h1, &axis_array, angle, &new_oxygen, &mut new_h1);
//         new_water[7] = new_h1[0] + delta_x; // hydrogen1 x
//         new_water[8] = new_h1[1] + delta_y; // hydrogen1 y
//         new_water[9] = new_h1[2] + delta_z; // hydrogen1 z

//         let mut new_h2: Array::<F> = Array::new(3);
//         rodrigues_rotation(&h2, &axis_array, angle, &new_oxygen, &mut new_h2);
//         new_water[14] = new_h2[0] + delta_x; // hydrogen2 x
//         new_water[15] = new_h2[1] + delta_y; // hydrogen2 y
//         new_water[16] = new_h2[2] + delta_z; // hydrogen2 z
//     } 
//     // else {
//     //     // Copy original coordinates if out of bounds
//     //     for idx in 0..old_water.len() {
//     //         new_water[idx] = old_water[idx];
//     //     }
//     // }
//     new_water
// }


// Propose an insertion move, generating a position within user-specified bounds
#[cube]
fn propose_insertion(
    boundaries: &Array<f32>,
    rng_array: &Array<f32>,
    new_water: &mut Array<f32>,
    resnum: f32,
    epochs: u32,
    epoch: u32) {

    let rnd_idx = (CUBE_POS_X * epochs + epoch) * 12;
    let delta_x = rng_array[rnd_idx + INSERTION_X_IDX];
    let delta_y = rng_array[rnd_idx + INSERTION_Y_IDX];
    let delta_z = rng_array[rnd_idx + INSERTION_Z_IDX];
    let angle_rnd = rng_array[rnd_idx + ROTATION_IDX];
    let axis_x = rng_array[rnd_idx + ROT_AXIS_X_IDX];
    let axis_y = rng_array[rnd_idx + ROT_AXIS_Y_IDX];
    let axis_z = rng_array[rnd_idx + ROT_AXIS_Z_IDX];
    
    let x_min = boundaries[0];
    let x_max = boundaries[1];
    let y_min = boundaries[2];
    let y_max = boundaries[3];
    let z_min = boundaries[4];
    let z_max = boundaries[5];
    
    // Get current oxygen position
    let ox = new_water[0];
    let oy = new_water[1];
    let oz = new_water[2];
    
    // Check boundaries
    // No need to check bounds here, as we are generating a position within the specified bounds
    // let in_bounds = (new_ox >= x_min) && (new_ox <= x_max) &&
    //             (new_oy >= y_min) && (new_oy <= y_max) &&
    //             (new_oz >= z_min) && (new_oz <= z_max);

    // if in_bounds {
    // Apply translation to all atoms
    new_water[0] = ox; // oxygen x
    new_water[1] = oy; // oxygen y
    new_water[2] = oz; // oxygen z
    new_water[3] = -0.8340;     // oxygen charge
    new_water[4] = 0.15210325;  // oxygen epsilon
    new_water[5] = 1.7682;      // oxygen rmin_half
    new_water[6] = resnum;              // oxygen resnum
    new_water[7] = new_water[7]; // hydrogen1 x
    new_water[8] = new_water[8]; // hydrogen1 y
    new_water[9] = new_water[9]; // hydrogen1 z
    new_water[10] = 0.4170;     // hydrogen1 charge
    new_water[11] = 0.0;        // hydrogen1 epsilon
    new_water[12] = 0.0;        // hydrogen1 rmin_half
    new_water[13] = resnum;             // hydrogen1 resnum
    new_water[14] = new_water[14]; // hydrogen2 x
    new_water[15] = new_water[15]; // hydrogen2 y
    new_water[16] = new_water[16]; // hydrogen2 z
    new_water[17] = 0.4170;     // hydrogen2 charge
    new_water[18] = 0.0;        // hydrogen2 epsilon
    new_water[19] = 0.0;        // hydrogen2 rmin_half
    new_water[20] = resnum;             // hydrogen2 resnum
    
    // Simple rotation around oxygen (optional)
    let angle = angle_rnd; // Small rotation
    
    // Rotate hydrogen atoms around oxygen (z-axis rotation for simplicity)
    let mut h1 = Array::new(3);
    h1[0] = new_water[7];
    h1[1] = new_water[8];
    h1[2] = new_water[9];

    let mut h2 = Array::new(3);
    h2[0] = new_water[14];
    h2[1] = new_water[15];
    h2[2] = new_water[16];

    let mut oxygen = Array::new(3);
    oxygen[0] = new_water[0];
    oxygen[1] = new_water[1];
    oxygen[2] = new_water[2];


    let mut new_oxygen = Array::new(3);
    // Calculate new oxygen position
    let new_ox = delta_x;
    let new_oy = delta_y;
    let new_oz = delta_z;
    new_oxygen[0] = new_ox;
    new_oxygen[1] = new_oy;
    new_oxygen[2] = new_oz; 

    let mut axis_array = Array::new(3);
    axis_array[0] = axis_x;
    axis_array[1] = axis_y;
    axis_array[2] = axis_z; 

    axis_array = gpu_geometry::normalize(&mut axis_array);

    let mut new_h1: Array::<f32> = Array::new(3);
    gpu_geometry::rodrigues_rotation(&h1, &axis_array, angle, &oxygen, &mut new_h1);
    let translated_h1 = gpu_geometry::add(&new_h1, &new_oxygen);
    new_water[7] = translated_h1[0]; // hydrogen1 x
    new_water[8] = translated_h1[1]; // hydrogen1 y
    new_water[9] = translated_h1[2]; // hydrogen1 z

    let mut new_h2: Array::<f32> = Array::new(3);
    gpu_geometry::rodrigues_rotation(&h2, &axis_array, angle, &oxygen, &mut new_h2);
    let translated_h2 = gpu_geometry::add(&new_h2, &new_oxygen);
    new_water[14] = translated_h2[0]; // hydrogen2 x
    new_water[15] = translated_h2[1]; // hydrogen2 y
    new_water[16] = translated_h2[2]; // hydrogen2 z

    new_water[0] = new_ox;
    new_water[1] = new_oy;
    new_water[2] = new_oz;
}

// // Propose a deletion move
// #[cube]
// fn propose_deletion<F: Float>(water: &Array<F>, new_water: &mut Array<F>) {
//     // Set everything to 0 to indicate dead water
//     new_water[0] = F::new(0.0);
//     new_water[1] = F::new(0.0);
//     new_water[2] = F::new(0.0);
//     new_water[3] = F::new(0.0);
//     new_water[4] = F::new(0.0);
//     new_water[5] = F::new(0.0);
//     new_water[6] = F::new(0.0);
//     new_water[7] = F::new(0.0);
//     new_water[8] = F::new(0.0);
//     new_water[9] = F::new(0.0);
//     new_water[10] = F::new(0.0);
//     new_water[11] = F::new(0.0);
//     new_water[12] = F::new(0.0);
//     new_water[13] = F::new(0.0);
//     new_water[14] = F::new(0.0);
//     new_water[15] = F::new(0.0);
//     new_water[16] = F::new(0.0);
//     new_water[17] = F::new(0.0);
//     new_water[18] = F::new(0.0);
//     new_water[19] = F::new(0.0);
//     new_water[20] = F::new(0.0);
// }

// pub fn generate_particle_index(rng_state: i32, num_particles: i32) -> i32 {
//     // XORShift random number generator
//     let mut r= rng_state; 
//     let mut retvalue = 0;
//     r ^= r << 13;
//     r ^= r >> 17;
//     r ^= r << 5;
    
//     // Return index in range [0, num_particles)
//     if num_particles > 0 {
//         retvalue = r % num_particles as i32;
//     }
//     retvalue 
// }

// // Helper functions for water management
// #[cube]
// fn copy_water(source: &Array<f32>, dest: &mut Array<f32>, dest_base_idx: u32) {
//     for i in 0..7*3 {
//         dest[dest_base_idx + i] = source[i];
//     }
// }

// #[cube]
// fn load_water(source: &Array<f32>, source_base_idx: u32, dest: &mut Array<f32>) {
//     for i in 0..7*3 {
//         dest[i] = source[source_base_idx + i];
//     }
// }

// #[cube]
// fn move_water(waters: &mut Array<f32>, from_idx: u32, to_idx: u32) {
//     for i in 0..7*3 {
//         waters[to_idx + i] = waters[from_idx + i];
//         waters[from_idx + i] = 0.0; // Clear old position
//     }
// }

// #[cube(launch_unchecked)]
// fn run_gcmc(boundaries: &Array<f32>,
//             receptor_atoms: &Array<f32>,
//             water_atoms: &mut Array<f32>,
//             random_numbers: &Array<f32>,
//             energies: &mut Array<f32>,
//             seeds: &Array<u32>,
//             last_resnum: u32,
//             B: f32,
//             volume: f32,
//             epochs: u32,
//         energies_debug: &mut Array<f32>) {
//     let n_receptor_atoms = receptor_atoms.len() / ATOM_FEATURES;
//     let n_water_atoms = water_atoms.len() / ATOM_FEATURES;
//     let sim_id = CUBE_POS_X; // Simulation ID (0..499)
//     let thread_id = UNIT_POS; // Thread within workgroup
    
//     let seeds_handle_idx = sim_id * epochs; 
//     let rng_handle_idx = sim_id * epochs * 12;
//     let waters_handle_idx = sim_id * MAX_N_WATERS * ATOM_FEATURES * 3;
    
//     let mut waters_in_the_system: i32 = 0;

//     let receptor_idx = sim_id * (n_receptor_atoms + n_water_atoms);
    
//     for epoch in 0..epochs {
//         let base_idx_for_rng = (sim_id * epochs + epoch) * 12;
//         let base_water_idx = waters_handle_idx + (waters_in_the_system as u32 * ATOM_FEATURES * 3);
//         let move_type = random_numbers[base_idx_for_rng + MOVE_TYPE_IDX];

//         // if move_type == 0.0 && waters_in_the_system > 0 {
//         //     // Move is translation
//         //     // Step 0 - Select water to perturb
//         //     let seed = seeds[seeds_handle_idx + epoch];
            
//         //     // Inline the random number generation
//         //     let mut r = seed;
//         //     r ^= r << 13;
//         //     r ^= r >> 17;
//         //     r ^= r << 5;
            
//         //     let water_idx_to_perturb = if waters_in_the_system > 0 {
//         //         r % waters_in_the_system
//         //     } else {
//         //         u32::new(0)
//         //     };

//         //     let mut old_water = Array::<f32>::new(ATOM_FEATURES*3);
//         //     old_water[0] = water_atoms[base_water_idx];
//         //     old_water[1] = water_atoms[base_water_idx+1];
//         //     old_water[2] = water_atoms[base_water_idx+2];
//         //     old_water[3] = water_atoms[base_water_idx+3];
//         //     old_water[4] = water_atoms[base_water_idx+4];
//         //     old_water[5] = water_atoms[base_water_idx+5];
//         //     old_water[6] = water_atoms[base_water_idx+6];
//         //     old_water[7] = water_atoms[base_water_idx+7];
//         //     old_water[8] = water_atoms[base_water_idx+8];
//         //     old_water[9] = water_atoms[base_water_idx+9];
//         //     old_water[10] = water_atoms[base_water_idx+10];
//         //     old_water[11] = water_atoms[base_water_idx+11];
//         //     old_water[12] = water_atoms[base_water_idx+12];
//         //     old_water[13] = water_atoms[base_water_idx+13];
//         //     old_water[14] = water_atoms[base_water_idx+14];
//         //     old_water[15] = water_atoms[base_water_idx+15];
//         //     old_water[16] = water_atoms[base_water_idx+16];
//         //     old_water[17] = water_atoms[base_water_idx+17];
//         //     old_water[18] = water_atoms[base_water_idx+18];
//         //     old_water[19] = water_atoms[base_water_idx+19];
//         //     old_water[20] = water_atoms[base_water_idx+20];

//         //     // Step 1 - calculate water's energy
//         //     compute_interactions(receptor_atoms, 
//         //         water_atoms, 
//         //         &old_water, 
//         //         energies,
//         //         sim_id);

//         //     let mut old_energy = 0.0;
//         //     // Map by simulation ID
//         //     for energy in receptor_idx..receptor_idx + (n_receptor_atoms + n_water_atoms) {
//         //         old_energy += energies[energy];
//         //         // reset to calculate again after;
//         //         energies[energy] = 0.0;
//         //     }

//         //     // Step 2 - perturb
//         //     let new_water = propose_perturbation::<f32>(boundaries, &old_water, random_numbers, epoch);
            
//         //     // Step 3 - calculate new water's energy
//         //     let mut new_energy = 0.0;
//         //     // Map by simulation ID
//         //     for energy in receptor_idx..receptor_idx + (n_receptor_atoms + n_water_atoms) {
//         //         new_energy += energies[energy];
//         //         // reset to calculate again after;
//         //         energies[energy] = 0.0;
//         //     }

//         //     // Step 4 - acceptance criteria
//         //     let deltaE = new_energy - old_energy;
//         //     let acceptance_prob = f32::min(f32::exp(-consts::BETA * deltaE), f32::new(1.0));
//             // let rnd_acceptance = random_numbers[base_idx_for_rng + ACCEPTANCE_IDX];
//         //     // Step 5 - overwrite if accepted revert if not
//         //     if rnd_acceptance < acceptance_prob {
//         //         // Accept
//         //         water_atoms[base_water_idx] = new_water[0];
//         //         water_atoms[base_water_idx+1] = new_water[1];
//         //         water_atoms[base_water_idx+2] = new_water[2];
//         //         water_atoms[base_water_idx+3] = new_water[3];
//         //         water_atoms[base_water_idx+4] = new_water[4];
//         //         water_atoms[base_water_idx+5] = new_water[5];
//         //         water_atoms[base_water_idx+6] = new_water[6];
//         //         water_atoms[base_water_idx+7] = new_water[7];
//         //         water_atoms[base_water_idx+8] = new_water[8];
//         //         water_atoms[base_water_idx+9] = new_water[9];
//         //         water_atoms[base_water_idx+10] = new_water[10];
//         //         water_atoms[base_water_idx+11] = new_water[11];
//         //         water_atoms[base_water_idx+12] = new_water[12];
//         //         water_atoms[base_water_idx+13] = new_water[13];
//         //         water_atoms[base_water_idx+14] = new_water[14];
//         //         water_atoms[base_water_idx+15] = new_water[15];
//         //         water_atoms[base_water_idx+16] = new_water[16];
//         //         water_atoms[base_water_idx+17] = new_water[17];
//         //         water_atoms[base_water_idx+18] = new_water[18];
//         //         water_atoms[base_water_idx+19] = new_water[19];
//         //         water_atoms[base_water_idx+20] = new_water[20];
//         //     }

//         // } else if move_type == 1.0 {
//         if move_type == 1.0 {
//             // Move is insertion
//             let mut new_water = Array::<f32>::new(ATOM_FEATURES*3);
//             new_water[0] = water_atoms[base_water_idx];
//             new_water[1] = water_atoms[base_water_idx+1];
//             new_water[2] = water_atoms[base_water_idx+2];
//             new_water[3] = water_atoms[base_water_idx+3];
//             new_water[4] = water_atoms[base_water_idx+4];
//             new_water[5] = water_atoms[base_water_idx+5];
//             new_water[6] = water_atoms[base_water_idx+6];
//             new_water[7] = water_atoms[base_water_idx+7];
//             new_water[8] = water_atoms[base_water_idx+8];
//             new_water[9] = water_atoms[base_water_idx+9];
//             new_water[10] = water_atoms[base_water_idx+10];
//             new_water[11] = water_atoms[base_water_idx+11];
//             new_water[12] = water_atoms[base_water_idx+12];
//             new_water[13] = water_atoms[base_water_idx+13];
//             new_water[14] = water_atoms[base_water_idx+14];
//             new_water[15] = water_atoms[base_water_idx+15];
//             new_water[16] = water_atoms[base_water_idx+16];
//             new_water[17] = water_atoms[base_water_idx+17];
//             new_water[18] = water_atoms[base_water_idx+18];
//             new_water[19] = water_atoms[base_water_idx+19];
//             new_water[20] = water_atoms[base_water_idx+20];

//             let possible_resnum = last_resnum + waters_in_the_system as u32 + 1;

//             propose_insertion(boundaries, random_numbers, &mut new_water, possible_resnum as f32, epoch);
            
//             compute_interactions(receptor_atoms, 
//                 water_atoms, 
//                 &new_water, 
//                 energies,
//                 sim_id);

//             let mut energies_sum = 0.0;
//             // Map by simulation ID
//             for energy in receptor_idx..receptor_idx + (n_receptor_atoms + n_water_atoms) {
//                 energies_sum = energies_sum + energies[energy];
//                 // energies[energy] = 0.0;
//             }
//             energies_debug[sim_id * epochs + epoch] = energies_sum;

//             let deltaE = energies_sum + consts::CHEMICAL_POTENTIAL;

//             let acceptance_prob = f32::min((1.0/(waters_in_the_system+1) as f32) * f32::exp(B) * f32::exp(-consts::BETA * deltaE), 1.0);
//             let rnd_acceptance = random_numbers[base_idx_for_rng + ACCEPTANCE_IDX];
//             if rnd_acceptance < acceptance_prob {
//                 // Accept
//                 water_atoms[base_water_idx] = new_water[0];
//                 water_atoms[base_water_idx+1] = new_water[1];
//                 water_atoms[base_water_idx+2] = new_water[2];
//                 water_atoms[base_water_idx+3] = new_water[3];
//                 water_atoms[base_water_idx+4] = new_water[4];
//                 water_atoms[base_water_idx+5] = new_water[5];
//                 water_atoms[base_water_idx+6] = new_water[6];
//                 water_atoms[base_water_idx+7] = new_water[7];
//                 water_atoms[base_water_idx+8] = new_water[8];
//                 water_atoms[base_water_idx+9] = new_water[9];
//                 water_atoms[base_water_idx+10] = new_water[10];
//                 water_atoms[base_water_idx+11] = new_water[11];
//                 water_atoms[base_water_idx+12] = new_water[12];
//                 water_atoms[base_water_idx+13] = new_water[13];
//                 water_atoms[base_water_idx+14] = new_water[14];
//                 water_atoms[base_water_idx+15] = new_water[15];
//                 water_atoms[base_water_idx+16] = new_water[16];
//                 water_atoms[base_water_idx+17] = new_water[17];
//                 water_atoms[base_water_idx+18] = new_water[18];
//                 water_atoms[base_water_idx+19] = new_water[19];
//                 water_atoms[base_water_idx+20] = new_water[20];
//                 waters_in_the_system = waters_in_the_system + 1;
//             }
//         } 
//         // else if move_type == 2.0 && waters_in_the_system > 0 {
//         //     // Move is deletion
//         //     // Step 0 - Select water to perturb
//         //     let seed = seeds[seeds_handle_idx + epoch];
            
//         //     // Inline the random number generation
//         //     let mut r = seed;
//         //     r ^= r << 13;
//         //     r ^= r >> 17;
//         //     r ^= r << 5;
            
//         //     let water_idx_to_perturb = if waters_in_the_system > 0 {
//         //         r % waters_in_the_system
//         //     } else {
//         //         u32::new(0)
//         //     };

//         //     let mut old_water = Array::<f32>::new(ATOM_FEATURES*3);
//         //     old_water[0] = water_atoms[base_water_idx];
//         //     old_water[1] = water_atoms[base_water_idx+1];
//         //     old_water[2] = water_atoms[base_water_idx+2];
//         //     old_water[3] = water_atoms[base_water_idx+3];
//         //     old_water[4] = water_atoms[base_water_idx+4];
//         //     old_water[5] = water_atoms[base_water_idx+5];
//         //     old_water[6] = water_atoms[base_water_idx+6];
//         //     old_water[7] = water_atoms[base_water_idx+7];
//         //     old_water[8] = water_atoms[base_water_idx+8];
//         //     old_water[9] = water_atoms[base_water_idx+9];
//         //     old_water[10] = water_atoms[base_water_idx+10];
//         //     old_water[11] = water_atoms[base_water_idx+11];
//         //     old_water[12] = water_atoms[base_water_idx+12];
//         //     old_water[13] = water_atoms[base_water_idx+13];
//         //     old_water[14] = water_atoms[base_water_idx+14];
//         //     old_water[15] = water_atoms[base_water_idx+15];
//         //     old_water[16] = water_atoms[base_water_idx+16];
//         //     old_water[17] = water_atoms[base_water_idx+17];
//         //     old_water[18] = water_atoms[base_water_idx+18];
//         //     old_water[19] = water_atoms[base_water_idx+19];
//         //     old_water[20] = water_atoms[base_water_idx+20];

//         //     // Step 1 - calculate water's energy
//         //     compute_interactions(receptor_atoms, 
//         //         water_atoms, 
//         //         &old_water, 
//         //         energies,
//         //         sim_id);

//         //     let mut removed_water_energy = 0.0;
//         //     // Map by simulation ID
//         //     for energy in receptor_idx..receptor_idx + (n_receptor_atoms + n_water_atoms) {
//         //         removed_water_energy += energies[energy];
//         //         // reset to calculate again after;
//         //         energies[energy] = 0.0;
//         //     }

//         //     let new_energy = -removed_water_energy;
//         //     let deltaE = new_energy - consts::CHEMICAL_POTENTIAL;
            
//         //     // Step 2 - acceptance criteria
//         //     let acceptance_prob = f32::min(f32::exp(waters_in_the_system as f32 * -B) * f32::exp(-consts::BETA * deltaE), 1.0);
            
//         //     // Step 3 - overwrite if accepted revert if not
//         //     let rnd_acceptance = random_numbers[base_idx_for_rng + ACCEPTANCE_IDX];
//         //     if rnd_acceptance < acceptance_prob {
//         //         // Accept
//         //         // This is not gonna work if the amount of insertiion + deletion
//         //         // is higher than the max_number of waters allowed
//         //         water_atoms[base_water_idx] = 0.0;
//         //         water_atoms[base_water_idx+1] = 0.0;
//         //         water_atoms[base_water_idx+2] = 0.0;
//         //         water_atoms[base_water_idx+3] = 0.0;
//         //         water_atoms[base_water_idx+4] = 0.0;
//         //         water_atoms[base_water_idx+5] = 0.0;
//         //         water_atoms[base_water_idx+5] = 0.0;
//         //         water_atoms[base_water_idx+5] = 0.0;
//         //         water_atoms[base_water_idx+6] = 0.0;
//         //         water_atoms[base_water_idx+7] = 0.0;
//         //         water_atoms[base_water_idx+8] = 0.0;
//         //         water_atoms[base_water_idx+9] = 0.0;
//         //         water_atoms[base_water_idx+10] = 0.0;
//         //         water_atoms[base_water_idx+11] = 0.0;
//         //         water_atoms[base_water_idx+12] = 0.0;
//         //         water_atoms[base_water_idx+13] = 0.0;
//         //         water_atoms[base_water_idx+14] = 0.0;
//         //         water_atoms[base_water_idx+15] = 0.0;
//         //         water_atoms[base_water_idx+16] = 0.0;
//         //         water_atoms[base_water_idx+17] = 0.0;
//         //         water_atoms[base_water_idx+18] = 0.0;
//         //         water_atoms[base_water_idx+19] = 0.0;
//         //         water_atoms[base_water_idx+20] = 0.0;
//         //         waters_in_the_system = waters_in_the_system - 1;
//         //     }

//         // }
//     } 

// }

#[cube(launch_unchecked)]
fn run_gcmc(
    boundaries: &Array<f32>,
    receptor_atoms: &Array<f32>,
    water_atoms: &mut Array<f32>,
    random_numbers: &Array<f32>,
    // energies: &mut Array<f32>,
    seeds: &Array<u32>,
    // waters_in_the_system: &mut Array<u32>,
    last_resnum: u32,
    B: f32,
    volume: f32,
    epochs: u32,
    // energies_debug: &mut Array<f32>
) {
    let sim_id = CUBE_POS_X;
    let n_receptor_atoms = receptor_atoms.len() / ATOM_FEATURES;
    let n_water_atoms = water_atoms.len() / ATOM_FEATURES;
    
    // Each simulation has its own compact water array starting at this offset
    let waters_base_idx = sim_id * MAX_N_WATERS * gpu_gcmc_moves::WATER_SIZE;
    
    // Track number of active waters for this simulation
    // let mut active_waters: u32 = waters_in_the_system[sim_id];
    let mut active_waters = 0;
    let receptor_idx = sim_id * (n_receptor_atoms + n_water_atoms);

    for epoch in 0..epochs {
        // energies_debug[epoch] = active_waters as f32;
        let base_idx_for_rng = (sim_id * epochs + epoch) * 12;
        let move_type = 1.0;
        let base_water_idx = waters_base_idx + active_waters * gpu_gcmc_moves::WATER_SIZE;
        // if move_type == 1.0 {
        //     // debug
        //     // water_atoms[base_water_idx] = epoch as f32;
        //     // water_atoms[base_water_idx+1] = sim_id as f32;
        //     // water_atoms[base_water_idx+2] = active_waters as f32;
        //     // Move is insertion
        //     let mut new_water = Array::<f32>::new(ATOM_FEATURES*3);
        //     new_water[0] = water_atoms[base_water_idx];
        //     new_water[1] = water_atoms[base_water_idx+1];
        //     new_water[2] = water_atoms[base_water_idx+2];
        //     new_water[3] = water_atoms[base_water_idx+3];
        //     new_water[4] = water_atoms[base_water_idx+4];
        //     new_water[5] = water_atoms[base_water_idx+5];
        //     new_water[6] = water_atoms[base_water_idx+6];
        //     new_water[7] = water_atoms[base_water_idx+7];
        //     new_water[8] = water_atoms[base_water_idx+8];
        //     new_water[9] = water_atoms[base_water_idx+9];
        //     new_water[10] = water_atoms[base_water_idx+10];
        //     new_water[11] = water_atoms[base_water_idx+11];
        //     new_water[12] = water_atoms[base_water_idx+12];
        //     new_water[13] = water_atoms[base_water_idx+13];
        //     new_water[14] = water_atoms[base_water_idx+14];
        //     new_water[15] = water_atoms[base_water_idx+15];
        //     new_water[16] = water_atoms[base_water_idx+16];
        //     new_water[17] = water_atoms[base_water_idx+17];
        //     new_water[18] = water_atoms[base_water_idx+18];
        //     new_water[19] = water_atoms[base_water_idx+19];
        //     new_water[20] = water_atoms[base_water_idx+20];

        //     let possible_resnum = last_resnum + active_waters as u32 + 1;

        //     propose_insertion(boundaries, random_numbers, &mut new_water, possible_resnum as f32, epoch, epochs);
            
        //     compute_interactions(receptor_atoms, 
        //         water_atoms, 
        //         &new_water, 
        //         energies,
        //         sim_id);

        //     let mut energies_sum = 0.0;
        //     // Map by simulation ID
        //     for energy in receptor_idx..receptor_idx + (n_receptor_atoms + n_water_atoms) {
        //         energies_sum = energies_sum + energies[energy];
        //         // energies[energy] = 0.0;
        //     }
        //     let deltaE = energies_sum + consts::CHEMICAL_POTENTIAL;

        //     let acceptance_prob = f32::min((1.0/(active_waters+1) as f32) * f32::exp(B) * f32::exp(-consts::BETA * deltaE), 1.0);
        //     let rnd_acceptance = random_numbers[base_idx_for_rng + ACCEPTANCE_IDX];
        //     // if rnd_acceptance < acceptance_prob {
        //         // Accept
        //         water_atoms[base_water_idx] = new_water[0];
        //         water_atoms[base_water_idx+1] = new_water[1];
        //         water_atoms[base_water_idx+2] = new_water[2];
        //         water_atoms[base_water_idx+3] = new_water[3];
        //         water_atoms[base_water_idx+4] = new_water[4];
        //         water_atoms[base_water_idx+5] = new_water[5];
        //         water_atoms[base_water_idx+6] = new_water[6];
        //         water_atoms[base_water_idx+7] = new_water[7];
        //         water_atoms[base_water_idx+8] = new_water[8];
        //         water_atoms[base_water_idx+9] = new_water[9];
        //         water_atoms[base_water_idx+10] = new_water[10];
        //         water_atoms[base_water_idx+11] = new_water[11];
        //         water_atoms[base_water_idx+12] = new_water[12];
        //         water_atoms[base_water_idx+13] = new_water[13];
        //         water_atoms[base_water_idx+14] = new_water[14];
        //         water_atoms[base_water_idx+15] = new_water[15];
        //         water_atoms[base_water_idx+16] = new_water[16];
        //         water_atoms[base_water_idx+17] = new_water[17];
        //         water_atoms[base_water_idx+18] = new_water[18];
        //         water_atoms[base_water_idx+19] = new_water[19];
        //         water_atoms[base_water_idx+20] = new_water[20];
        //         active_waters = active_waters + 1;
        //     // }
        // } 
        // let move_type = random_numbers[base_idx_for_rng + MOVE_TYPE_IDX];
        // let move_type = 1.0;
        // if move_type == 1.0 { // INSERTION
        if gpu_gcmc_moves::insertion_move(
                boundaries,
                receptor_atoms,
                water_atoms,
                random_numbers,
                sim_id,
                epoch,
                active_waters,
                n_receptor_atoms,
                last_resnum,
                B,
                epochs){
                active_waters += 1;
            }
        // } else if move_type == 2.0 && active_waters > 0 { // DELETION
        //     gpu_gcmc_moves::deletion_move(
        //         receptor_atoms,
        //         water_atoms,
        //         random_numbers,
        //         energies,
        //         seeds,
        //         sim_id,
        //         epoch,
        //         &mut active_waters,
        //         waters_base_idx,
        //         n_receptor_atoms,
        //         B,
        //         epochs
        //     );
            
        // } else if move_type == 0.0 && active_waters > 0 { // TRANSLATION
        //     gpu_gcmc_moves::translation_move(
        //         boundaries,
        //         receptor_atoms,
        //         water_atoms,
        //         random_numbers,
        //         energies,
        //         seeds,
        //         sim_id,
        //         epoch,
        //         active_waters,
        //         waters_base_idx,
        //         n_receptor_atoms,
        //         epochs
        //     );
        // }
    }
}

pub fn simulate<R: Runtime>(
    n_simulations: usize,
    device: &R::Device,
    receptor_atoms: Vec<Atom>,
    water_configuration: WaterMolecule,
    cutoff: f32,
    boundaries: Vec<f32>,
    volume: f32,
    num_steps: usize) -> Vec<f32> {
    let client: ComputeClient<<R as Runtime>::Server, <R as Runtime>::Channel> = R::client(device);
    let random_buffer = prepare_random_numbers(n_simulations * num_steps, 
        &boundaries, 
        -0.5, 
        0.5, 
        -180.0, 
    180.0);

    // Try to use cubecl-random to create random seeds to pass to the kernels
    // These are the seeds that will be used to generate random numbers for the simulation
    // One seed per simulation per epoch -> random molecule to affect will be picked with xorshift function + the seed
    let seed_tensor = TensorHandle::<R, u32>::empty(&client, [n_simulations*num_steps].to_vec());
    random_normal::<R, u32>(&client, 42, 2, seed_tensor.as_ref());

    println!("# Atoms in the receptor: {}", receptor_atoms.len());

    // Move the receptor to the global memory on the GPU
    let mut receptor_atoms_buffer = Vec::with_capacity(receptor_atoms.len() * ATOM_FEATURES as usize);
    let n_receptor_atoms = receptor_atoms.len();

    // // Here I populate the array of waters with the water_configuration so that I always have it
    for atom in receptor_atoms {
        let coords = atom.coords();
        receptor_atoms_buffer.push(coords[0] as f32);
        receptor_atoms_buffer.push(coords[1] as f32);
        receptor_atoms_buffer.push(coords[2] as f32);
        receptor_atoms_buffer.push(atom.charge() as f32);
        receptor_atoms_buffer.push(atom.epsilon() as f32);
        receptor_atoms_buffer.push(atom.rmin_half() as f32);
        receptor_atoms_buffer.push(atom.residue_number as f32);
    }
    println!("Last element of the receptor before going to GPU: {}", receptor_atoms_buffer[receptor_atoms_buffer.len() -1]);

    let max_n_waters = MAX_N_WATERS as usize;
    println!("N FRAMES: {}", n_simulations);
    let mut waters_buffer = Vec::with_capacity(n_simulations * max_n_waters * ATOM_FEATURES as usize * 3);

    for idx in 0..(n_simulations * max_n_waters) {
        let water = water_configuration.as_vec();
        let o_c = water[0].coords();
        let h1_c = water[1].coords();
        let h2_c = water[2].coords();
        waters_buffer.push(o_c[0] as f32);
        waters_buffer.push(o_c[1] as f32);
        waters_buffer.push(o_c[2] as f32);
        waters_buffer.push(0.);
        waters_buffer.push(0.);
        waters_buffer.push(0.);
        waters_buffer.push(0.);
        waters_buffer.push(h1_c[0] as f32);
        waters_buffer.push(h1_c[1] as f32);
        waters_buffer.push(h1_c[2] as f32);
        waters_buffer.push(0.);
        waters_buffer.push(0.);
        waters_buffer.push(0.);
        waters_buffer.push(0.);
        waters_buffer.push(h2_c[0] as f32);
        waters_buffer.push(h2_c[1] as f32);
        waters_buffer.push(h2_c[2] as f32);
        waters_buffer.push(0.);
        waters_buffer.push(0.);
        waters_buffer.push(0.);
        waters_buffer.push(0.);
    }
    println!("N WATERS: {}", waters_buffer.len());
    let boundaries_handle = client.create(f32::as_bytes(&boundaries));
    let receptor_atoms_handle = client.create(f32::as_bytes(&receptor_atoms_buffer));
    let water_atoms_handle = client.create(f32::as_bytes(&waters_buffer));
    let random_numbers_handle = client.create(f32::as_bytes(&random_buffer));
    
    let volume_var = volume / consts::STANDARD_VOLUME;
    let B = consts::CHEMICAL_POTENTIAL * consts::BETA + volume_var.ln();

    // let threads_per_workgroup: u32 = 256;
    // let num_workgroups = (n_receptor_atoms as u32 + threads_per_workgroup - 1) / threads_per_workgroup;

    unsafe {
        run_gcmc::launch_unchecked::<R>(
            &client, 
            CubeCount::Static(n_simulations as u32, 1, 1),
            CubeDim::new(1,1, 1),
            ArrayArg::from_raw_parts::<f32>(&boundaries_handle, 6, 1), 
            ArrayArg::from_raw_parts::<f32>(&receptor_atoms_handle, receptor_atoms_buffer.len(), 1), 
            ArrayArg::from_raw_parts::<f32>(&water_atoms_handle, waters_buffer.len(), 1), 
            ArrayArg::from_raw_parts::<f32>(&random_numbers_handle, n_simulations * num_steps * 12, 1), 
            ArrayArg::from_raw_parts::<u32>(&seed_tensor.handle, n_simulations*num_steps, 1), 
            ScalarArg {elem: receptor_atoms_buffer[receptor_atoms_buffer.len() -1] as u32}, 
            ScalarArg {elem: B},
            ScalarArg {elem: volume}, 
            ScalarArg {elem: num_steps as u32},
        );
    }

    let bytes = client.read_one(water_atoms_handle.clone().binding());
    let output: Vec<f32> = f32::from_bytes(&bytes).to_vec();
    // println!("Water handle after simulation: {}", output.len());
    // let mut retvalue = Vec::new();
    // for s_idx in 0..n_simulations {
    //     for w_idx in 0..max_n_waters {
    //         let base_idx = w_idx * ATOM_FEATURES as usize;
    //         retvalue.push(output[s_idx * base_idx as usize]);
    //         retvalue.push(output[s_idx * base_idx as usize + 1]);
    //         retvalue.push(output[s_idx * base_idx as usize + 2]);
    //     }
    // } 
    // retvalue

    // let energies_bytes = client.read_one(energies_debug.clone().binding());
    // let edo: Vec<f32> = f32::from_bytes(&energies_bytes).to_vec();
    // println!("active waters: {:?}", edo);

    // for sim in 0..n_simulations {
    //     let waters_base_idx = sim * MAX_N_WATERS as usize * gpu_gcmc_moves::WATER_SIZE as usize;
    //     for epoch in 0..num_steps {
    //         let base_water_idx = waters_base_idx + (epoch * ATOM_FEATURES as usize * 3);
    //         println!("{} - {} - {}\n", output[base_water_idx], output[base_water_idx+1], output[base_water_idx+2]);
    //     } 
    // }

    output
}

// pub fn simulate<R: Runtime>(
//     n_simulations: usize,
//     device: &R::Device,
//     receptor_atoms: Vec<Atom>,
//     water_configuration: WaterMolecule,
//     cutoff: f32,
//     boundaries: Vec<f32>,
//     volume: f32,
//     num_steps: usize,
//     batch_size: Option<usize>) -> Vec<f32> {
    
//     let client: ComputeClient<<R as Runtime>::Server, <R as Runtime>::Channel> = R::client(device);
    
//     // Determine optimal batch size based on available memory or user preference
//     let effective_batch_size = batch_size.unwrap_or_else(|| {
//         // Conservative default: limit to 1000 steps per batch to avoid memory issues
//         // You can tune this based on your GPU memory
//         std::cmp::min(100, num_steps)
//     });
    
//     println!("Processing {} steps in batches of {}", num_steps, effective_batch_size);
//     println!("# Atoms in the receptor: {}", receptor_atoms.len());

//     // Prepare static data that doesn't change between batches
//     let mut receptor_atoms_buffer = Vec::with_capacity(receptor_atoms.len() * ATOM_FEATURES as usize);
//     let n_receptor_atoms = receptor_atoms.len();

//     for atom in receptor_atoms {
//         let coords = atom.coords();
//         receptor_atoms_buffer.push(coords[0] as f32);
//         receptor_atoms_buffer.push(coords[1] as f32);
//         receptor_atoms_buffer.push(coords[2] as f32);
//         receptor_atoms_buffer.push(atom.charge() as f32);
//         receptor_atoms_buffer.push(atom.epsilon() as f32);
//         receptor_atoms_buffer.push(atom.rmin_half() as f32);
//         receptor_atoms_buffer.push(atom.residue_number as f32);
//     }
    
//     println!("Last element of the receptor before going to GPU: {}", 
//              receptor_atoms_buffer[receptor_atoms_buffer.len() - 1]);

//     let max_n_waters = MAX_N_WATERS as usize;
//     println!("N FRAMES: {}", n_simulations);

//     // Initialize waters buffer (this remains the same size regardless of batch size)
//     let mut waters_buffer = Vec::with_capacity(n_simulations * max_n_waters * ATOM_FEATURES as usize * 3);
//     for idx in 0..(n_simulations * max_n_waters) {
//         let water = water_configuration.as_vec();
//         let o_c = water[0].coords();
//         let h1_c = water[1].coords();
//         let h2_c = water[2].coords();
//         waters_buffer.push(o_c[0] as f32);
//         waters_buffer.push(o_c[1] as f32);
//         waters_buffer.push(o_c[2] as f32);
//         waters_buffer.push(0.); waters_buffer.push(0.); waters_buffer.push(0.); waters_buffer.push(0.);
//         waters_buffer.push(h1_c[0] as f32);
//         waters_buffer.push(h1_c[1] as f32);
//         waters_buffer.push(h1_c[2] as f32);
//         waters_buffer.push(0.); waters_buffer.push(0.); waters_buffer.push(0.); waters_buffer.push(0.);
//         waters_buffer.push(h2_c[0] as f32);
//         waters_buffer.push(h2_c[1] as f32);
//         waters_buffer.push(h2_c[2] as f32);
//         waters_buffer.push(0.); waters_buffer.push(0.); waters_buffer.push(0.); waters_buffer.push(0.);
//     }

//     // Create static GPU handles that will be reused across batches
//     let boundaries_handle = client.create(f32::as_bytes(&boundaries));
//     let receptor_atoms_handle = client.create(f32::as_bytes(&receptor_atoms_buffer));
//     let water_atoms_handle = client.empty(waters_buffer.len() * core::mem::size_of::<f32>());
//     let waters_in_the_system_handle = client.create(u32::as_bytes(&vec![0u32; n_simulations]));
    
//     // Calculate volume variables
//     let volume_var = volume / consts::STANDARD_VOLUME;
//     let B = consts::CHEMICAL_POTENTIAL * consts::BETA + volume_var.ln();
    
//     // Result accumulator
//     let mut all_energies_debug = Vec::new();
    
//     // Process in batches
//     let num_batches = (num_steps + effective_batch_size - 1) / effective_batch_size;
    
//     for batch_idx in 0..num_batches {
//         let batch_start = batch_idx * effective_batch_size;
//         let current_batch_size = std::cmp::min(effective_batch_size, num_steps - batch_start);
        
//         println!("Processing batch {}/{} (steps {}-{})", 
//                  batch_idx + 1, num_batches, batch_start, batch_start + current_batch_size - 1);
        
//         // Create batch-specific buffers
//         let batch_random_buffer = prepare_random_numbers(
//             n_simulations * current_batch_size, 
//             &boundaries, 
//             -0.5, 0.5, -180.0, 180.0
//         );
        
//         // Create seed tensor for this batch
//         let batch_seed_tensor = TensorHandle::<R, u32>::empty(
//             &client, 
//             [n_simulations * current_batch_size].to_vec()
//         );
//         random_normal::<R, u32>(&client, 42 + batch_idx as u32, 2, batch_seed_tensor.as_ref());
        
//         // Create GPU handles for this batch
//         let batch_random_numbers_handle = client.create(f32::as_bytes(&batch_random_buffer));
//         let batch_energies_debug = client.empty(n_simulations * current_batch_size * core::mem::size_of::<f32>());
        
//         // Launch kernel for this batch
//         unsafe {
//             run_gcmc::launch_unchecked::<R>(
//                 &client, 
//                 CubeCount::Static(n_simulations as u32, 1, 1),
//                 CubeDim::new(1, 1, 1),
//                 ArrayArg::from_raw_parts::<f32>(&boundaries_handle, 6, 1), 
//                 ArrayArg::from_raw_parts::<f32>(&receptor_atoms_handle, receptor_atoms_buffer.len(), 1), 
//                 ArrayArg::from_raw_parts::<f32>(&water_atoms_handle, waters_buffer.len(), 1), 
//                 ArrayArg::from_raw_parts::<f32>(&batch_random_numbers_handle, n_simulations * current_batch_size * 12, 1), 
//                 ArrayArg::from_raw_parts::<u32>(&batch_seed_tensor.handle, n_simulations * current_batch_size, 1), 
//                 ArrayArg::from_raw_parts::<u32>(&waters_in_the_system_handle, n_simulations, 1),
//                 ScalarArg { elem: receptor_atoms_buffer[receptor_atoms_buffer.len() - 1] as u32 },
//                 ScalarArg { elem: B },
//                 ScalarArg { elem: volume }, 
//                 ScalarArg { elem: current_batch_size as u32 },
//                 ArrayArg::from_raw_parts::<f32>(&batch_energies_debug, n_simulations * current_batch_size, 1)
//             );
//         }
        
//         // Read results from this batch
//         let batch_energies_bytes = client.read_one(batch_energies_debug.clone().binding());
//         let batch_energies: Vec<f32> = f32::from_bytes(&batch_energies_bytes).to_vec();
        
//         // Accumulate results
//         all_energies_debug.extend(batch_energies);
        
//         println!("Completed batch {}/{}", batch_idx + 1, num_batches);
//     }
    
//     // Read final water configuration
//     let bytes = client.read_one(water_atoms_handle.clone().binding());
//     let output: Vec<f32> = f32::from_bytes(&bytes).to_vec();
    
//     println!("Total active waters across all batches: {:?}", all_energies_debug.len());
    
//     output
// }