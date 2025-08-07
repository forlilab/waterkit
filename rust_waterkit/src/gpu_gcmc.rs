use core::f32;
use std::f32::consts::LN_10;

use crate::atom::Atom;
use crate::{consts, gpu_geometry};
use crate::water::WaterMolecule;
use cubecl::std::tensor::TensorHandle;
use cubecl::{compute, prelude::*};
use cubecl_random::random_normal;
use nalgebra::Scalar;
use rand::distributions::Uniform;
use rand::prelude::Distribution;
use rand::rngs::ThreadRng;
use rand::{random, Rng};

const ATOM_FEATURES: u32 = 7;

#[cube]
fn compute_interactions<F: Float>(
    receptor_atoms: &Array<F>,
    water_atoms: &Array<F>,
    target_water: &Array<F>,
    energies: &mut Array<F>,
    sim_id: u32
) {
    
    for idx in 0..receptor_atoms.len() / ATOM_FEATURES {
        let cutoff_sq = F::new(12.0*12.0);
        let mut energy = F::new(0.0);

        // Fixed: multiply idx by NUM_FEATURES to get correct base index
        let base_idx = idx * ATOM_FEATURES;
        let x = receptor_atoms[base_idx];
        let y = receptor_atoms[base_idx + 1];
        let z = receptor_atoms[base_idx + 2];
        let charge = receptor_atoms[base_idx + 3];
        let epsilon = receptor_atoms[base_idx + 4];  // Make sure this matches your packing order
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
        // Now process the waters
        // Fixed: multiply idx by NUM_FEATURES to get correct base index
        let base_idx = idx * ATOM_FEATURES;
        let x = water_atoms[base_idx];
        let y = water_atoms[base_idx + 1];
        let z = water_atoms[base_idx + 2];
        let charge = water_atoms[base_idx + 3];
        let epsilon = water_atoms[base_idx + 4];  // Make sure this matches your packing order
        let rmin_half = water_atoms[base_idx + 5];
        let resnum = water_atoms[base_idx + 6];
        
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
                    if j % 3 == 0 && base_idx % 3 == 0 {
                        lj = epsilon * (F::powf((rmin / r), F::new(12.0)) - (F::new(2.0) * F::powf((rmin / r), F::new(6.0))));
                    }
                    let coulomb = F::new(332.0636) * charge1 * charge2 / r;
                    energy += lj + coulomb;
            }
        }
        energies[idx] = energy;
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

    let mut random_numbers = Vec::with_capacity(n_epochs * 11); // 1 for move type, 3 for translation, 1 for rotation, 3 for rotation axis, 3 for insertion, 1 for acceptance probability
    let mut rng = rand::thread_rng();

    for _ in 0..n_epochs {
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

#[cube]
fn propose_perturbation<R: Runtime, F: Float>(
    boundaries: &Array<F>,
    water_coords: &Array<F>, // Flattened water coordinates [ox, oy, oz, charge_o, epsilon_o, rmin_half_o, resnum_o, h1x, h1y, h1z, charge_h1, epsilon_h1, rmin_half_h1, resnum_h1, h2x, h2y, h2z, charge_h2, epsilon_h2, rmin_half_h2, resnum_h2]
    rng_array: &Array<F>,
    new_water_coords: &mut Array<F>,
    epoch: u32) {
    
    let rnd_idx = (CUBE_POS_X + epoch) * 11;
    let delta_x = rng_array[rnd_idx + 1];
    let delta_y = rng_array[rnd_idx + 2];
    let delta_z = rng_array[rnd_idx + 3];
    let angle_rnd = rng_array[rnd_idx + 4];
    let axis_x = rng_array[rnd_idx + 5];
    let axis_y = rng_array[rnd_idx + 6];
    let axis_z = rng_array[rnd_idx + 7];

    // Get boundary values
    let x_min = boundaries[0];
    let x_max = boundaries[1];
    let y_min = boundaries[2];
    let y_max = boundaries[3];
    let z_min = boundaries[4];
    let z_max = boundaries[5];
    
    // Get current oxygen position
    let ox = water_coords[0];
    let oy = water_coords[1];
    let oz = water_coords[2];
    
    // Calculate new oxygen position
    let new_ox = ox + delta_x;
    let new_oy = oy + delta_y;
    let new_oz = oz + delta_z;
    
    // Check boundaries
    let in_bounds = (new_ox >= x_min) && (new_ox <= x_max) &&
                   (new_oy >= y_min) && (new_oy <= y_max) &&
                   (new_oz >= z_min) && (new_oz <= z_max);
    
    if in_bounds {
        // Apply translation to all atoms
        new_water_coords[0] = new_ox; // oxygen x
        new_water_coords[1] = new_oy; // oxygen y
        new_water_coords[2] = new_oz; // oxygen z
        
        
        // Simple rotation around oxygen (optional)
        let angle = angle_rnd; // Small rotation
        let cos_a = F::cos(angle);
        let sin_a = F::sin(angle);
        
        // Rotate hydrogen atoms around oxygen (z-axis rotation for simplicity)
        let mut h1 = Array::new(3);
        h1[0] = new_water_coords[7];
        h1[1] = new_water_coords[8];
        h1[2] = new_water_coords[9];

        let mut h2 = Array::new(3);
        h2[0] = new_water_coords[14];
        h2[1] = new_water_coords[15];
        h2[2] = new_water_coords[16];

        let mut new_oxygen = Array::new(3);
        new_oxygen[0] = new_ox;
        new_oxygen[1] = new_oy;
        new_oxygen[2] = new_oz; 

        let mut axis_array = Array::new(3);
        axis_array[0] = axis_x;
        axis_array[1] = axis_y;
        axis_array[2] = axis_z; 

        axis_array = gpu_geometry::normalize(&mut axis_array);

        let new_h1 = rodrigues_rotation(&h1, &new_oxygen, &axis_array, cos_a, sin_a);
        new_water_coords[7] = new_h1[0] + delta_x; // hydrogen1 x
        new_water_coords[8] = new_h1[1] + delta_y; // hydrogen1 y
        new_water_coords[9] = new_h1[2] + delta_z; // hydrogen1 z

        let new_h2 = rodrigues_rotation(&h2, &new_oxygen, &axis_array, cos_a, sin_a);
        new_water_coords[14] = new_h2[0] + delta_x; // hydrogen1 x
        new_water_coords[15] = new_h2[1] + delta_y; // hydrogen1 y
        new_water_coords[16] = new_h2[2] + delta_z; // hydrogen1 z
    } else {
        // Copy original coordinates if out of bounds
        for i in 0u32..9u32 {
            new_water_coords[i] = water_coords[i];
        }
    }
}

#[cube]
fn rodrigues_rotation<F: Float>(
    point: &Array<F>,
    center: &Array<F>, 
    axis: &Array<F>,
    cos_theta: F,
    sin_theta: F,
) -> Array<F> {
    let v_rel = gpu_geometry::subtract_points(&point, &center);
    let k_cross_v = gpu_geometry::cross(&axis, &v_rel);
    let k_dot_v = gpu_geometry::dot(&axis, &v_rel);
    
    let term1 = gpu_geometry::scale_point(&v_rel, cos_theta);
    let term2 = gpu_geometry::scale_point(&k_cross_v, sin_theta);
    let term3 = gpu_geometry::scale_point(&axis, k_dot_v * (F::new(1.0) - cos_theta));
    
    let rotated = gpu_geometry::sum_points(&gpu_geometry::sum_points(&term1, &term2), &term3);
    gpu_geometry::sum_points(&rotated, &center)
}


// Propose an insertion move, generating a position within user-specified bounds
#[cube]
fn propose_insertion(
    boundaries: &Array<f32>,
    rng_array: &Array<f32>,
    new_water_coords: &mut Array<f32>,
    resnum: f32,
    epoch: u32) {

    let rnd_idx = (CUBE_POS_X + epoch) * 11;
    let delta_x = rng_array[rnd_idx + 8];
    let delta_y = rng_array[rnd_idx + 9];
    let delta_z = rng_array[rnd_idx + 10];
    let angle_rnd = rng_array[rnd_idx + 4];
    let axis_x = rng_array[rnd_idx + 5];
    let axis_y = rng_array[rnd_idx + 6];
    let axis_z = rng_array[rnd_idx + 7];
    
    let x_min = boundaries[0];
    let x_max = boundaries[1];
    let y_min = boundaries[2];
    let y_max = boundaries[3];
    let z_min = boundaries[4];
    let z_max = boundaries[5];
    
    // Get current oxygen position
    let ox = new_water_coords[0];
    let oy = new_water_coords[1];
    let oz = new_water_coords[2];
    

    // Calculate new oxygen position
    let new_ox = ox + delta_x;
    let new_oy = oy + delta_y;
    let new_oz = oz + delta_z;
    
    // Check boundaries
    // No need to check bounds here, as we are generating a position within the specified bounds
    // let in_bounds = (new_ox >= x_min) && (new_ox <= x_max) &&
    //             (new_oy >= y_min) && (new_oy <= y_max) &&
    //             (new_oz >= z_min) && (new_oz <= z_max);

    // if in_bounds {
    // Apply translation to all atoms
    new_water_coords[0] = new_ox; // oxygen x
    new_water_coords[1] = new_oy; // oxygen y
    new_water_coords[2] = new_oz; // oxygen z
    new_water_coords[3] = -0.8340;     // oxygen charge
    new_water_coords[4] = 0.15210325;  // oxygen epsilon
    new_water_coords[5] = 1.7682;      // oxygen rmin_half
    new_water_coords[6] = resnum;              // oxygen resnum
    new_water_coords[7] = new_water_coords[7] + delta_x; // hydrogen1 x
    new_water_coords[8] = new_water_coords[8] + delta_y; // hydrogen1 y
    new_water_coords[9] = new_water_coords[9] + delta_z; // hydrogen1 z
    new_water_coords[10] = 0.4170;     // hydrogen1 charge
    new_water_coords[11] = 0.0;        // hydrogen1 epsilon
    new_water_coords[12] = 0.0;        // hydrogen1 rmin_half
    new_water_coords[13] = resnum;             // hydrogen1 resnum
    new_water_coords[14] = new_water_coords[14] + delta_x; // hydrogen2 x
    new_water_coords[15] = new_water_coords[15] + delta_y; // hydrogen2 y
    new_water_coords[16] = new_water_coords[16] + delta_z; // hydrogen2 z
    new_water_coords[17] = 0.4170;     // hydrogen2 charge
    new_water_coords[18] = 0.0;        // hydrogen2 epsilon
    new_water_coords[19] = 0.0;        // hydrogen2 rmin_half
    new_water_coords[20] = resnum;             // hydrogen2 resnum
    
    // Simple rotation around oxygen (optional)
    let angle = angle_rnd; // Small rotation
    let cos_a = f32::cos(angle);
    let sin_a = f32::sin(angle);
    
    // Rotate hydrogen atoms around oxygen (z-axis rotation for simplicity)
    let mut h1 = Array::new(3);
    h1[0] = new_water_coords[7];
    h1[1] = new_water_coords[8];
    h1[2] = new_water_coords[9];

    let mut h2 = Array::new(3);
    h2[0] = new_water_coords[14];
    h2[1] = new_water_coords[15];
    h2[2] = new_water_coords[16];

    let mut new_oxygen = Array::new(3);
    new_oxygen[0] = new_ox;
    new_oxygen[1] = new_oy;
    new_oxygen[2] = new_oz; 

    let mut axis_array = Array::new(3);
    axis_array[0] = axis_x;
    axis_array[1] = axis_y;
    axis_array[2] = axis_z; 

    axis_array = gpu_geometry::normalize(&mut axis_array);

    let new_h1 = rodrigues_rotation(&h1, &new_oxygen, &axis_array, cos_a, sin_a);
    new_water_coords[7] = new_h1[0] + delta_x; // hydrogen1 x
    new_water_coords[8] = new_h1[1] + delta_y; // hydrogen1 y
    new_water_coords[9] = new_h1[2] + delta_z; // hydrogen1 z

    let new_h2 = rodrigues_rotation(&h2, &new_oxygen, &axis_array, cos_a, sin_a);
    new_water_coords[14] = new_h2[0] + delta_x; // hydrogen1 x
    new_water_coords[15] = new_h2[1] + delta_y; // hydrogen1 y
    new_water_coords[16] = new_h2[2] + delta_z; // hydrogen1 z
    // }
}

#[cube]
pub fn generate_particle_index(rng_state: &mut u32, #[comptime]num_particles: u32) -> u32 {
    // XORShift random number generator
    *rng_state ^= *rng_state << u32::new(13);
    *rng_state ^= *rng_state >> u32::new(17);
    *rng_state ^= *rng_state << u32::new(5);
    
    // Return index in range [0, num_particles)
    if num_particles > u32::new(0) {
        *rng_state % num_particles
    } else {
        u32::new(0)
    }
}

// Propose a deletion move
#[cube]
fn propose_deletion<F: Float>(water: &Array<F>, new_water: &mut Array<F>) {
    // Set everything to 0 to indicate dead water
    new_water[0] = F::new(0.0);
    new_water[1] = F::new(0.0);
    new_water[2] = F::new(0.0);
    new_water[3] = F::new(0.0);
    new_water[4] = F::new(0.0);
    new_water[5] = F::new(0.0);
    new_water[6] = F::new(0.0);
    new_water[7] = F::new(0.0);
    new_water[8] = F::new(0.0);
    new_water[9] = F::new(0.0);
    new_water[10] = F::new(0.0);
    new_water[11] = F::new(0.0);
    new_water[12] = F::new(0.0);
    new_water[13] = F::new(0.0);
    new_water[14] = F::new(0.0);
    new_water[15] = F::new(0.0);
    new_water[16] = F::new(0.0);
    new_water[17] = F::new(0.0);
    new_water[18] = F::new(0.0);
    new_water[19] = F::new(0.0);
    new_water[20] = F::new(0.0);
}


#[cube(launch_unchecked)]
fn run_gcmc(boundaries: &Array<f32>,
            receptor_atoms: &Array<f32>,
            water_atoms: &mut Array<f32>,
            random_numbers: &Array<f32>,
            energies: &mut Array<f32>,
            last_resnum: u32,
            B: f32,
            volume: f32,
            epochs: u32) {
    let sim_id = CUBE_POS_X; // Simulation ID (0..499)
    let thread_id = UNIT_POS; // Thread within workgroup
    let waters_handle_idx = sim_id * 1000 * 7 * 3;
    let mut waters_in_the_system = 0;

    for epoch in 0..epochs {
        let base_idx_for_rng = sim_id * epoch * 11;
        let base_water_idx = waters_handle_idx + (waters_in_the_system * ATOM_FEATURES * 3);
        let move_type = random_numbers[base_idx_for_rng];

        if move_type == 0.0 && waters_in_the_system > 0 {
            // Move is translation
            // Step 0 - Select water to perturb
            let random_number = rng.gen_range(0..waters_in_the_system);
            let water_to_perturb = generate_particle_index(&mut (random_numbers[base_idx_for_rng] as u32), 
                waters_in_the_system as u32);
            // Step 1 - calculate water's energy
            // Step 2 - perturb
            // Step 3 - calculate new water's energy
            // Step 4 - acceptance criteria
            // Step 5 - overwrite if accepted revert if not

        } else if move_type == 1.0 {
            // Move is insertion
            let mut new_water = Array::<f32>::new(ATOM_FEATURES*3);
            new_water[0] = water_atoms[base_water_idx];
            new_water[1] = water_atoms[base_water_idx+1];
            new_water[2] = water_atoms[base_water_idx+2];
            new_water[3] = water_atoms[base_water_idx+3];
            new_water[4] = water_atoms[base_water_idx+4];
            new_water[5] = water_atoms[base_water_idx+5];
            new_water[6] = water_atoms[base_water_idx+6];
            new_water[7] = water_atoms[base_water_idx+7];
            new_water[8] = water_atoms[base_water_idx+8];
            new_water[9] = water_atoms[base_water_idx+9];
            new_water[10] = water_atoms[base_water_idx+10];
            new_water[11] = water_atoms[base_water_idx+11];
            new_water[12] = water_atoms[base_water_idx+12];
            new_water[13] = water_atoms[base_water_idx+13];
            new_water[14] = water_atoms[base_water_idx+14];
            new_water[15] = water_atoms[base_water_idx+15];
            new_water[16] = water_atoms[base_water_idx+16];
            new_water[17] = water_atoms[base_water_idx+17];
            new_water[18] = water_atoms[base_water_idx+18];
            new_water[19] = water_atoms[base_water_idx+19];
            new_water[20] = water_atoms[base_water_idx+20];

            let possible_resnum = last_resnum + waters_in_the_system + 1;

            propose_insertion(boundaries, random_numbers, &mut new_water, possible_resnum as f32, epoch);
            
            compute_interactions(receptor_atoms, 
                water_atoms, 
                &new_water, 
                energies,
                sim_id);

            let mut energies_sum = 0.0;
            // Map by simulation ID
            for energy in sim_id..receptor_atoms.len() / ATOM_FEATURES {
                energies_sum += energies[energy];
            }

            let delta_e = energies_sum + consts::CHEMICAL_POTENTIAL;

            let acceptance_prob = f32::min((1.0/(waters_in_the_system+1) as f32) * f32::exp(B) * f32::exp(-consts::BETA * delta_e), 1.0);
            let rnd_acceptance = random_numbers[base_idx_for_rng+8];
            if rnd_acceptance < acceptance_prob {
                // Accept
                water_atoms[base_water_idx] = new_water[0];
                water_atoms[base_water_idx+1] = new_water[1];
                water_atoms[base_water_idx+2] = new_water[2];
                water_atoms[base_water_idx+3] = new_water[3];
                water_atoms[base_water_idx+4] = new_water[4];
                water_atoms[base_water_idx+5] = new_water[5];
                water_atoms[base_water_idx+5] = new_water[5];
                water_atoms[base_water_idx+5] = new_water[5];
                water_atoms[base_water_idx+6] = new_water[6];
                water_atoms[base_water_idx+7] = new_water[7];
                water_atoms[base_water_idx+8] = new_water[8];
                water_atoms[base_water_idx+9] = new_water[9];
                water_atoms[base_water_idx+10] = new_water[10];
                water_atoms[base_water_idx+11] = new_water[11];
                water_atoms[base_water_idx+12] = new_water[12];
                water_atoms[base_water_idx+13] = new_water[13];
                water_atoms[base_water_idx+14] = new_water[14];
                water_atoms[base_water_idx+15] = new_water[15];
                water_atoms[base_water_idx+16] = new_water[16];
                water_atoms[base_water_idx+17] = new_water[17];
                water_atoms[base_water_idx+18] = new_water[18];
                water_atoms[base_water_idx+19] = new_water[19];
                water_atoms[base_water_idx+20] = new_water[20];
                waters_in_the_system = waters_in_the_system + 1;
            }
        } else if move_type == 2.0 && waters_in_the_system > 0 {
            
        }
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
    random_normal::<R, u32>(&client, 42, 5, seed_tensor.as_ref());

    println!("# Atoms in the receptor: {}", receptor_atoms.len());

    let mut receptor_atoms_buffer = Vec::with_capacity(receptor_atoms.len() * ATOM_FEATURES as usize);
    let n_receptor_atoms = receptor_atoms.len();

    // Here I populate the array of waters with the water_configuration so that I always have it
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

    let max_n_waters = 1000;
    println!("N FRAMES: {}", n_simulations);
    let mut waters_buffer = Vec::with_capacity(n_simulations * ATOM_FEATURES as usize * 3 * max_n_waters);

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
    let energies_handle = client.empty(((n_receptor_atoms + max_n_waters) * n_simulations) * core::mem::size_of::<f32>());
    
    let volume_var = volume / consts::STANDARD_VOLUME;
    let B = consts::CHEMICAL_POTENTIAL * consts::BETA + volume_var.ln();

    let threads_per_workgroup: u32 = 256;
    let num_workgroups = (n_receptor_atoms as u32 + threads_per_workgroup - 1) / threads_per_workgroup;

    unsafe {
        run_gcmc::launch_unchecked::<R>(
            &client, 
            CubeCount::Static(n_simulations as u32, 1, 1),
            CubeDim::new(1,1, 1),
            ArrayArg::from_raw_parts::<f32>(&boundaries_handle, 6, 1), 
            ArrayArg::from_raw_parts::<f32>(&receptor_atoms_handle, n_receptor_atoms, 1), 
            ArrayArg::from_raw_parts::<f32>(&water_atoms_handle, n_simulations * ATOM_FEATURES as usize * 3 * max_n_waters, 1), 
            ArrayArg::from_raw_parts::<f32>(&random_numbers_handle, n_simulations * num_steps * 6, 1), 
            ArrayArg::from_raw_parts::<f32>(&energies_handle, (n_receptor_atoms + max_n_waters) * n_simulations, 1),  
            ScalarArg {elem: receptor_atoms_buffer[receptor_atoms_buffer.len() -1] as u32}, 
            ScalarArg {elem: B},
            ScalarArg {elem: volume}, 
            ScalarArg {elem: num_steps as u32});
    }

    let bytes = client.read_one(water_atoms_handle.clone().binding());
    let output: Vec<f32> = f32::from_bytes(&bytes).to_vec();
    // let mut retvalue = Vec::new();
    // for i in 0..300 {
    //     let base_idx = i * ATOM_FEATURES;
    //     retvalue.push(output[base_idx as usize]);
    //     retvalue.push(output[base_idx as usize + 1]);
    //     retvalue.push(output[base_idx as usize + 2]);
    // } 
    // retvalue
    output
}