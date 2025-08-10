use std::sync::atomic::ATOMIC_ISIZE_INIT;

use cubecl::prelude::*;
use crate::{consts, gpu_energy, gpu_geometry::{self, rodrigues_rotation}};

pub const MAX_N_WATERS: u32 = 300;
pub const ATOM_FEATURES: u32 = 7;
pub const WATER_SIZE: u32 = ATOM_FEATURES * 3;

pub const MOVE_TYPE_IDX: u32 = 0;
pub const TRANSLATION_X_IDX: u32 = 1;
pub const TRANSLATION_Y_IDX: u32 = 2;
pub const TRANSLATION_Z_IDX: u32 = 3;
pub const ROTATION_IDX: u32 = 4;
pub const ROT_AXIS_X_IDX: u32 = 5;
pub const ROT_AXIS_Y_IDX: u32 = 6;
pub const ROT_AXIS_Z_IDX: u32 = 7;
pub const INSERTION_X_IDX: u32 = 8;
pub const INSERTION_Y_IDX: u32 = 9;
pub const INSERTION_Z_IDX: u32 = 10;
pub const ACCEPTANCE_IDX: u32 = 11;

// INSERTION: Add water at the end of active waters array
#[cube]
pub fn insertion_move(
    boundaries: &Array<f32>,
    receptor_atoms: &Array<f32>,
    water_atoms: &mut Array<f32>,
    random_numbers: &Array<f32>,
    // energies: &mut Array<f32>,
    sim_id: u32,
    epoch: u32,
    active_waters: u32,
    // waters_base_idx: u32,
    n_receptor_atoms: u32,
    last_resnum: u32,
    B: f32,
    epochs: u32,
    // debug_energies: &mut Array<f32>
) -> bool {
    let mut accepted = false;
    let waters_handle_idx = sim_id  * MAX_N_WATERS * ATOM_FEATURES * 3;
    // Check if we have space for another water
    if active_waters >= MAX_N_WATERS {
        terminate!(); // No space available
        // return false;
    }
    // Calculate position for new water (at end of active waters)
    let new_water_idx = waters_handle_idx + active_waters * WATER_SIZE;
    let base_idx_for_rng = (sim_id * epochs + epoch) * 12;

    // Create temporary water for testing
    let possible_resnum = last_resnum + active_waters + 1;
    let mut new_water = create_water_std(possible_resnum);

    // Generate new water configuration
    propose_insertion_compact(boundaries, random_numbers, &mut new_water, possible_resnum as f32, epochs, epoch);
    
    // Calculate energy of new water interacting with receptor and existing waters
    let new_energy = energy_for_real_water_kernel(receptor_atoms,
        water_atoms,
        &new_water,
        n_receptor_atoms,
        active_waters,
        sim_id);
        
    // Calculate acceptance probability
    let deltaE = new_energy + consts::CHEMICAL_POTENTIAL;
    let acceptance_prob = f32::min(
        (1.0 / (active_waters + 1) as f32) * f32::exp(B) * f32::exp(-consts::BETA * deltaE),
        1.0
    );

    // Check acceptances
    let rnd_acceptance = random_numbers[base_idx_for_rng + ACCEPTANCE_IDX];
    if rnd_acceptance < acceptance_prob {
    // if new_enexrgy <= 0.0 {
        // ACCEPT: Copy new water to the active position
        copy_water_to_array(&new_water, water_atoms, new_water_idx);
        accepted = true;
    }
    // If rejected, do nothing - the slot remains empty
    accepted
}

// DELETION: Remove water and compact array by moving last water to deleted position
#[cube]
pub fn deletion_move(
    receptor_atoms: &Array<f32>,
    water_atoms: &mut Array<f32>,
    random_numbers: &Array<f32>,
    energies: &mut Array<f32>,
    seeds: &Array<u32>,
    sim_id: u32,
    epoch: u32,
    active_waters: &mut u32,
    waters_base_idx: u32,
    n_receptor_atoms: u32,
    B: f32,
    epochs: u32
) {
    if *active_waters == 0 {
        terminate!(); // No waters to delete
    }
    
    // Select random water to delete (0 to active_waters-1)
    let seed = seeds[sim_id * epochs + epoch];
    let water_to_delete = xorshift_random(seed) % (*active_waters);
    
    // Calculate position of water to delete
    let delete_water_idx = waters_base_idx + water_to_delete * WATER_SIZE;
    
    // Load water to be deleted for energy calculation
    let mut water_to_remove = Array::<f32>::new(WATER_SIZE);
    load_water_from_array(water_atoms, delete_water_idx, &mut water_to_remove);
    
    // Calculate energy of removing this water
    compute_interactions_for_deletion(
        receptor_atoms,
        water_atoms,
        &water_to_remove,
        energies,
        sim_id,
        *active_waters,
        waters_base_idx,
        n_receptor_atoms,
        water_to_delete
    );
    
    // Sum up all interaction energies (this is the energy we're removing from the system)
    let removed_energy = get_deletion_energy_sum(
        energies,
        sim_id,
        n_receptor_atoms,
        *active_waters,
        water_to_delete
    );
    
    // Calculate acceptance probability for deletion
    // When we remove a water, the energy change is -removed_energy (energy decreases)
    // We also lose the chemical potential contribution
    let deltaE = -removed_energy - consts::CHEMICAL_POTENTIAL;
    let acceptance_prob = f32::min(
        (*active_waters as f32) * f32::exp(-B) * f32::exp(-consts::BETA * deltaE),
        1.0
    );
    
    // Check acceptance
    let base_idx_for_rng = (sim_id * epochs + epoch) * 12;
    let rnd_acceptance = random_numbers[base_idx_for_rng + ACCEPTANCE_IDX];
    
    if rnd_acceptance < acceptance_prob {
        // ACCEPT DELETION
        *active_waters -= 1; // Decrement count first
        
        // If we're not deleting the last water, move the last water to the deleted position
        // This keeps the array compact
        if water_to_delete < *active_waters {
            let last_water_idx = waters_base_idx + (*active_waters) * WATER_SIZE;
            move_water_in_array(water_atoms, last_water_idx, delete_water_idx);
        }
        
        // Clear the last position (where we moved the water from, or where deleted water was if it was last)
        let clear_idx = waters_base_idx + (*active_waters) * WATER_SIZE;
        clear_water_in_array(water_atoms, clear_idx);
    }
    // If rejected, do nothing - water stays in place
}


// TRANSLATION: Move existing water
#[cube]
pub fn translation_move(
    boundaries: &Array<f32>,
    receptor_atoms: &Array<f32>,
    water_atoms: &mut Array<f32>,
    random_numbers: &Array<f32>,
    energies: &mut Array<f32>,
    seeds: &Array<u32>,
    sim_id: u32,
    epoch: u32,
    active_waters: u32,
    waters_base_idx: u32,
    n_receptor_atoms: u32,
    epochs: u32
) {
    if active_waters == 0 {
        terminate!();
    }
    
    // Select random active water to move
    let seed = seeds[sim_id * epochs + epoch];
    let water_to_move = xorshift_random(seed) % active_waters;
    let move_water_idx = waters_base_idx + water_to_move * WATER_SIZE;
    
    // Load current water
    let mut old_water = Array::<f32>::new(WATER_SIZE);
    load_water_from_array(water_atoms, move_water_idx, &mut old_water);
    
    // Calculate old energy
    // ... (similar to your current code)
    
    // Propose new position
    let new_water = propose_perturbation::<f32>(boundaries, &old_water, random_numbers, epoch);
    
    // Calculate new energy
    // ... (similar to your current code)
    
    // Accept/reject and update if accepted
    let base_idx_for_rng = (sim_id * epochs + epoch) * 12;
    let rnd_acceptance = random_numbers[base_idx_for_rng + ACCEPTANCE_IDX];
    
    let deltaE = 0.0; // Calculate actual energy difference
    let acceptance_prob = f32::min(f32::exp(-consts::BETA * deltaE), 1.0);
    
    if rnd_acceptance < acceptance_prob {
        copy_water_to_array(&new_water, water_atoms, move_water_idx);
    }
}

// Helper functions for compact array operations
#[cube]
fn copy_water_to_array(source: &Array<f32>, dest: &mut Array<f32>, dest_base_idx: u32) {
    dest[dest_base_idx] = source[0];
    dest[dest_base_idx + 1] = source[1];
    dest[dest_base_idx + 2] = source[2];
    dest[dest_base_idx + 3] = source[3];
    dest[dest_base_idx + 4] = source[4];
    dest[dest_base_idx + 5] = source[5];
    dest[dest_base_idx + 6] = source[6];
    dest[dest_base_idx + 7] = source[7];
    dest[dest_base_idx + 8] = source[8];
    dest[dest_base_idx + 9] = source[9];
    dest[dest_base_idx + 10] = source[10];
    dest[dest_base_idx + 11] = source[11];
    dest[dest_base_idx + 12] = source[12];
    dest[dest_base_idx + 13] = source[13];
    dest[dest_base_idx + 14] = source[14];
    dest[dest_base_idx + 15] = source[15];
    dest[dest_base_idx + 16] = source[16];
    dest[dest_base_idx + 17] = source[17];
    dest[dest_base_idx + 18] = source[18];
    dest[dest_base_idx + 19] = source[19];
    dest[dest_base_idx + 20] = source[20];
}

#[cube]
fn load_water_from_array(source: &Array<f32>, source_base_idx: u32, dest: &mut Array<f32>) {
    dest[0] = source[source_base_idx];
    dest[1] = source[source_base_idx + 1];
    dest[2] = source[source_base_idx + 2];
    dest[3] = source[source_base_idx + 3];
    dest[4] = source[source_base_idx + 4];
    dest[5] = source[source_base_idx + 5];
    dest[6] = source[source_base_idx + 6];
    dest[7] = source[source_base_idx + 7];
    dest[8] = source[source_base_idx + 8];
    dest[9] = source[source_base_idx + 9];
    dest[10] = source[source_base_idx + 10];
    dest[11] = source[source_base_idx + 11];
    dest[12] = source[source_base_idx + 12];
    dest[13] = source[source_base_idx + 13];
    dest[14] = source[source_base_idx + 14];
    dest[15] = source[source_base_idx + 15];
    dest[16] = source[source_base_idx + 16];
    dest[17] = source[source_base_idx + 17];
    dest[18] = source[source_base_idx + 18];
    dest[19] = source[source_base_idx + 19];
    dest[20] = source[source_base_idx + 20];
}

#[cube]
fn move_water_in_array(waters: &mut Array<f32>, from_idx: u32, to_idx: u32) {
    waters[to_idx] = waters[from_idx];
    waters[to_idx + 1] = waters[from_idx + 1];
    waters[to_idx + 2] = waters[from_idx + 2];
    waters[to_idx + 3] = waters[from_idx + 3];
    waters[to_idx + 4] = waters[from_idx + 4];
    waters[to_idx + 5] = waters[from_idx + 5];
    waters[to_idx + 6] = waters[from_idx + 6];
    waters[to_idx + 7] = waters[from_idx + 7];
    waters[to_idx + 8] = waters[from_idx + 8];
    waters[to_idx + 9] = waters[from_idx + 9];
    waters[to_idx + 10] = waters[from_idx + 10];
    waters[to_idx + 11] = waters[from_idx + 11];
    waters[to_idx + 12] = waters[from_idx + 12];
    waters[to_idx + 13] = waters[from_idx + 13];
    waters[to_idx + 14] = waters[from_idx + 14];
    waters[to_idx + 15] = waters[from_idx + 15];
    waters[to_idx + 16] = waters[from_idx + 16];
    waters[to_idx + 17] = waters[from_idx + 17];
    waters[to_idx + 18] = waters[from_idx + 18];
    waters[to_idx + 19] = waters[from_idx + 19];
    waters[to_idx + 20] = waters[from_idx + 20];
}

#[cube]
fn clear_water_in_array(waters: &mut Array<f32>, base_idx: u32) {
    waters[base_idx] = 0.0;
    waters[base_idx+1] = 0.0;
    waters[base_idx+2] = 0.0;
    waters[base_idx+3] = 0.0;
    waters[base_idx+4] = 0.0;
    waters[base_idx+5] = 0.0;
    waters[base_idx+5] = 0.0;
    waters[base_idx+5] = 0.0;
    waters[base_idx+6] = 0.0;
    waters[base_idx+7] = 0.0;
    waters[base_idx+8] = 0.0;
    waters[base_idx+9] = 0.0;
    waters[base_idx+10] = 0.0;
    waters[base_idx+11] = 0.0;
    waters[base_idx+12] = 0.0;
    waters[base_idx+13] = 0.0;
    waters[base_idx+14] = 0.0;
    waters[base_idx+15] = 0.0;
    waters[base_idx+16] = 0.0;
    waters[base_idx+17] = 0.0;
    waters[base_idx+18] = 0.0;
    waters[base_idx+19] = 0.0;
    waters[base_idx+20] = 0.0;
}

#[cube]
fn xorshift_random(seed: u32) -> u32 {
    let mut r = seed;
    r ^= r << 13;
    r ^= r >> 17;
    r ^= r << 5;
    r
}

// Modified insertion proposal for compact layout
#[cube]
fn propose_insertion_compact(
    boundaries: &Array<f32>,
    rng_array: &Array<f32>,
    new_water: &mut Array<f32>,
    resnum: f32,
    epochs: u32,
    epoch: u32,
) {
    let rnd_idx = (CUBE_POS_X * epochs + epoch) * 12;
    let delta_x = rng_array[rnd_idx + INSERTION_X_IDX];
    let delta_y = rng_array[rnd_idx + INSERTION_Y_IDX];
    let delta_z = rng_array[rnd_idx + INSERTION_Z_IDX];
    let angle_rnd = rng_array[rnd_idx + ROTATION_IDX];
    let axis_x = rng_array[rnd_idx + ROT_AXIS_X_IDX];
    let axis_y = rng_array[rnd_idx + ROT_AXIS_Y_IDX];
    let axis_z = rng_array[rnd_idx + ROT_AXIS_Z_IDX];
    
    // Get current oxygen position
    let ox = new_water[0];
    let oy = new_water[1];
    let oz = new_water[2];

    // Calculate new oxygen position
    let new_ox = ox + delta_x;
    let new_oy = oy + delta_y;
    let new_oz = oz + delta_z;
    
    // Apply rotation around oxygen center
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

    let mut new_oxygen = Array::new(3);
    new_oxygen[0] = new_ox;
    new_oxygen[1] = new_oy;
    new_oxygen[2] = new_oz; 

    let mut axis_array = Array::new(3);
    axis_array[0] = axis_x;
    axis_array[1] = axis_y;
    axis_array[2] = axis_z; 

    let mut original_oxygen = Array::new(3);
    original_oxygen[0] = new_water[0];
    original_oxygen[1] = new_water[1];
    original_oxygen[2] = new_water[2];

    let normalized_axis_array = gpu_geometry::normalize(&mut axis_array);

    let mut new_h1: Array::<f32> = Array::new(3);
    rodrigues_rotation(&h1, &normalized_axis_array, angle, &original_oxygen, &mut new_h1);
    new_water[7] = new_h1[0] + new_ox; // hydrogen1 x
    new_water[8] = new_h1[1] + new_oy; // hydrogen1 y
    new_water[9] = new_h1[2] + new_oz; // hydrogen1 z

    let mut new_h2: Array::<f32> = Array::new(3);
    rodrigues_rotation(&h2, &normalized_axis_array, angle, &original_oxygen, &mut new_h2);
    new_water[14] = new_h2[0] + new_ox; // hydrogen2 x
    new_water[15] = new_h2[1] + new_oy; // hydrogen2 y
    new_water[16] = new_h2[2] + new_oz; // hydrogen2 z

    new_water[0] = new_ox;
    new_water[1] = new_oy;
    new_water[2] = new_oz;
}

#[cube]
fn propose_perturbation<F: Float>(
    boundaries: &Array<F>,
    old_water: &Array<F>, // Flattened water coordinates [ox, oy, oz, charge_o, epsilon_o, rmin_half_o, resnum_o, h1x, h1y, h1z, charge_h1, epsilon_h1, rmin_half_h1, resnum_h1, h2x, h2y, h2z, charge_h2, epsilon_h2, rmin_half_h2, resnum_h2]
    rng_array: &Array<F>,
    epoch: u32) -> Array<F> {

    let mut new_water = Array::new(ATOM_FEATURES * 3);
    new_water[0] = old_water[0];
    new_water[1] = old_water[1];
    new_water[2] = old_water[2];
    new_water[3] = old_water[3];
    new_water[4] = old_water[4];
    new_water[5] = old_water[5];
    new_water[6] = old_water[6];
    new_water[7] = old_water[7];
    new_water[8] = old_water[8];
    new_water[9] = old_water[9];
    new_water[10] = old_water[10];
    new_water[11] = old_water[11];
    new_water[12] = old_water[12];
    new_water[13] = old_water[13];
    new_water[14] = old_water[14];
    new_water[15] = old_water[15];
    new_water[16] = old_water[16];
    new_water[17] = old_water[17];
    new_water[18] = old_water[18];
    new_water[19] = old_water[19];
    new_water[20] = old_water[20];

    let rnd_idx = (CUBE_POS_X + epoch) * 12;
    let delta_x = rng_array[rnd_idx + TRANSLATION_X_IDX];
    let delta_y = rng_array[rnd_idx + TRANSLATION_Y_IDX];
    let delta_z = rng_array[rnd_idx + TRANSLATION_Z_IDX];
    let angle_rnd = rng_array[rnd_idx + ROTATION_IDX];
    let axis_x = rng_array[rnd_idx + ROT_AXIS_X_IDX];
    let axis_y = rng_array[rnd_idx + ROT_AXIS_Y_IDX];
    let axis_z = rng_array[rnd_idx + ROT_AXIS_Z_IDX];

    // Get boundary values
    let x_min = boundaries[0];
    let x_max = boundaries[1];
    let y_min = boundaries[2];
    let y_max = boundaries[3];
    let z_min = boundaries[4];
    let z_max = boundaries[5];
    
    // Get current oxygen position
    let ox = old_water[0];
    let oy = old_water[1];
    let oz = old_water[2];
    
    // Calculate new oxygen position
    let new_ox = ox + delta_x;
    let new_oy = oy + delta_y;
    let new_oz = oz + delta_z;
    
    // Check boundaries
    // let in_bounds = (new_ox >= x_min) && (new_ox <= x_max) &&
    //                (new_oy >= y_min) && (new_oy <= y_max) &&
    //                (new_oz >= z_min) && (new_oz <= z_max);
    
    // if in_bounds {
        // Apply translation to all atoms
        new_water[0] = new_ox; // oxygen x
        new_water[1] = new_oy; // oxygen y
        new_water[2] = new_oz; // oxygen z
        
        
        // Simple rotation around oxygen (optional)
        let angle = angle_rnd; // Small rotation
        let cos_a = F::cos(angle);
        let sin_a = F::sin(angle);
        
        // Rotate hydrogen atoms around oxygen (z-axis rotation for simplicity)
        let mut h1 = Array::new(3);
        h1[0] = new_water[7];
        h1[1] = new_water[8];
        h1[2] = new_water[9];

        let mut h2 = Array::new(3);
        h2[0] = new_water[14];
        h2[1] = new_water[15];
        h2[2] = new_water[16];

        let mut new_oxygen = Array::new(3);
        new_oxygen[0] = new_ox;
        new_oxygen[1] = new_oy;
        new_oxygen[2] = new_oz; 

        let mut axis_array = Array::new(3);
        axis_array[0] = axis_x;
        axis_array[1] = axis_y;
        axis_array[2] = axis_z; 

        axis_array = gpu_geometry::normalize(&mut axis_array);

        let mut new_h1: Array::<F> = Array::new(3);
        rodrigues_rotation(&h1, &axis_array, angle, &new_oxygen, &mut new_h1);
        new_water[7] = new_h1[0] + delta_x; // hydrogen1 x
        new_water[8] = new_h1[1] + delta_y; // hydrogen1 y
        new_water[9] = new_h1[2] + delta_z; // hydrogen1 z

        let mut new_h2: Array::<F> = Array::new(3);
        rodrigues_rotation(&h2, &axis_array, angle, &new_oxygen, &mut new_h2);
        new_water[14] = new_h2[0] + delta_x; // hydrogen2 x
        new_water[15] = new_h2[1] + delta_y; // hydrogen2 y
        new_water[16] = new_h2[2] + delta_z; // hydrogen2 z
    // } 
    new_water
}


// Energies region
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
        let mut x = F::new(0.0);
        let mut y = F::new(0.0);
        let mut z = F::new(0.0);
        let mut charge = F::new(0.0);
        let mut epsilon = F::new(0.0);
        let mut rmin_half = F::new(0.0);
        let mut resnum = F::new(0.0);
        if idx < n_receptor_atoms {
            x = receptor_atoms[base_idx];
            y = receptor_atoms[base_idx + 1];
            z = receptor_atoms[base_idx + 2];
            charge = receptor_atoms[base_idx + 3];
            epsilon = receptor_atoms[base_idx + 4];
            rmin_half = receptor_atoms[base_idx + 5];
            resnum = receptor_atoms[base_idx + 6];
        } else {
            x = water_atoms[base_idx];
            y = water_atoms[base_idx + 1];
            z = water_atoms[base_idx + 2];
            charge = water_atoms[base_idx + 3];
            epsilon = water_atoms[base_idx + 4];
            rmin_half = water_atoms[base_idx + 5];
            resnum = water_atoms[base_idx + 6];
        }
        
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
        energies[base_energy_idx + idx] = energy;
    }
}

// Modified energy calculations for compact layout
#[cube]
fn compute_interactions_for_insertion(
    receptor_atoms: &Array<f32>,
    existing_waters: &Array<f32>,
    new_water: &Array<f32>,
    energies: &mut Array<f32>,
    sim_id: u32,
    active_waters: u32,
    waters_base_idx: u32,
    n_receptor_atoms: u32
) {
    const ATOM_FEATURES: u32 = 7;
    const WATER_SIZE: u32 = 3 * ATOM_FEATURES;

    // Base offset in energies array for this simulation
    // Layout: [ receptor_atom_energies..., water_molecule_energies... ]
    let sim_energy_base = sim_id * (n_receptor_atoms + active_waters);

    // ---- RECEPTOR vs NEW WATER ----
    for receptor_idx in 0..n_receptor_atoms {
        let atom_base = receptor_idx * ATOM_FEATURES;

        let rx = receptor_atoms[atom_base];
        let ry = receptor_atoms[atom_base + 1];
        let rz = receptor_atoms[atom_base + 2];
        let rcharge = receptor_atoms[atom_base + 3];
        let repsilon = receptor_atoms[atom_base + 4];
        let rrmin_half = receptor_atoms[atom_base + 5];
        let rresnum = receptor_atoms[atom_base + 6];

        let mut energy = 0.0;

        // Interact with all atoms in the new water
        for water_atom_idx in 0..3 {
            let water_base = water_atom_idx * ATOM_FEATURES;

            let wx = new_water[water_base];
            let wy = new_water[water_base + 1];
            let wz = new_water[water_base + 2];
            let wcharge = new_water[water_base + 3];
            let wepsilon = new_water[water_base + 4];
            let wrmin_half = new_water[water_base + 5];
            let wresnum = new_water[water_base + 6];

            if rresnum != wresnum {
                let dx = rx - wx;
                let dy = ry - wy;
                let dz = rz - wz;
                let r2 = dx * dx + dy * dy + dz * dz;
                let r = f32::sqrt(r2);

                // Lennard–Jones (O–O only)
                let mut lj = 0.0;
                if water_atom_idx == 0 {
                    let rmin = rrmin_half + wrmin_half;
                    let epsilon_combined = f32::sqrt(repsilon * wepsilon);
                    let ratio = rmin / r;
                    lj = epsilon_combined * (f32::powf(ratio, 12.0) - 2.0 * f32::powf(ratio, 6.0));
                }

                // Coulomb
                let coulomb = 332.0636 * rcharge * wcharge / r;

                energy += lj + coulomb;
            }
        }

        // Store receptor–water energy per receptor atom
        energies[sim_energy_base + receptor_idx] = energy;
    }

    // ---- NEW WATER vs EXISTING WATERS ----
    for existing_idx in 0..active_waters {
        let existing_water_base = waters_base_idx + existing_idx * WATER_SIZE;

        // Load existing water molecule (3 atoms)
        let mut existing_water = Array::<f32>::new(WATER_SIZE);
        load_water_from_array(existing_waters, existing_water_base, &mut existing_water);

        let mut energy = 0.0;

        for new_atom in 0..3 {
            let new_base = new_atom * ATOM_FEATURES;

            let nx = new_water[new_base];
            let ny = new_water[new_base + 1];
            let nz = new_water[new_base + 2];
            let nq = new_water[new_base + 3];
            let neps = new_water[new_base + 4];
            let nrmin_half = new_water[new_base + 5];
            let nresnum = new_water[new_base + 6];

            for exist_atom in 0..3 {
                let exist_base = exist_atom * ATOM_FEATURES;

                let ex = existing_water[exist_base];
                let ey = existing_water[exist_base + 1];
                let ez = existing_water[exist_base + 2];
                let eq = existing_water[exist_base + 3];
                let eeps = existing_water[exist_base + 4];
                let ermin_half = existing_water[exist_base + 5];
                let eresnum = existing_water[exist_base + 6];

                if nresnum != eresnum {
                    let dx = nx - ex;
                    let dy = ny - ey;
                    let dz = nz - ez;
                    let r2 = dx * dx + dy * dy + dz * dz;
                    let r = f32::sqrt(r2);

                    // Lennard–Jones (O–O only)
                    let mut lj = 0.0;
                    if new_atom == 0 && exist_atom == 0 {
                        let rmin = nrmin_half + ermin_half;
                        let epsilon_combined = f32::sqrt(neps * eeps);
                        let ratio = rmin / r;
                        lj = epsilon_combined * (f32::powf(ratio, 12.0) - 2.0 * f32::powf(ratio, 6.0));
                    }

                    // Coulomb
                    let coulomb = 332.0636 * nq * eq / r;

                    energy += lj + coulomb;
                }
            }
        }

        // Store water–water energy per existing water molecule
        energies[sim_energy_base + n_receptor_atoms + existing_idx] = energy;
    }
}

#[cube]
fn compute_interactions_for_deletion(
    receptor_atoms: &Array<f32>,
    existing_waters: &Array<f32>,
    water_to_remove: &Array<f32>,
    energies: &mut Array<f32>,
    sim_id: u32,
    active_waters: u32,
    waters_base_idx: u32,
    n_receptor_atoms: u32,
    water_idx_to_remove: u32
) {
    let receptor_energy_base = sim_id * (n_receptor_atoms + MAX_N_WATERS as u32 * 3);
    
    // Clear all energies for this simulation first
    for i in 0..(n_receptor_atoms + active_waters * 3) {
        energies[receptor_energy_base + i] = 0.0;
    }
    
    // PART 1: Calculate receptor-water interactions for the water being removed
    for receptor_idx in 0..n_receptor_atoms {
        let mut energy = 0.0;
        let receptor_atom_base = receptor_idx * ATOM_FEATURES;
        
        // Get receptor atom properties
        let rx = receptor_atoms[receptor_atom_base];
        let ry = receptor_atoms[receptor_atom_base + 1];
        let rz = receptor_atoms[receptor_atom_base + 2];
        let rcharge = receptor_atoms[receptor_atom_base + 3];
        let repsilon = receptor_atoms[receptor_atom_base + 4];
        let rrmin_half = receptor_atoms[receptor_atom_base + 5];
        let rresnum = receptor_atoms[receptor_atom_base + 6];
        
        // Interact with all 3 atoms in the water being removed (O, H1, H2)
        for water_atom_idx in 0..3 {
            let water_atom_base = water_atom_idx * ATOM_FEATURES;
            
            let wx = water_to_remove[water_atom_base];
            let wy = water_to_remove[water_atom_base + 1];
            let wz = water_to_remove[water_atom_base + 2];
            let wcharge = water_to_remove[water_atom_base + 3];
            let wepsilon = water_to_remove[water_atom_base + 4];
            let wrmin_half = water_to_remove[water_atom_base + 5];
            let wresnum = water_to_remove[water_atom_base + 6];
            
            // Only calculate if different residues (avoid self-interaction)
            if rresnum != wresnum {
                let dx = rx - wx;
                let dy = ry - wy;
                let dz = rz - wz;
                let r2 = dx * dx + dy * dy + dz * dz;
                
                if r2 > 0.0 { // Avoid division by zero
                    let r = f32::sqrt(r2);
                    
                    // Lennard-Jones interaction (only for oxygen-oxygen interactions)
                    let mut lj = 0.0;
                    if water_atom_idx == 0 { // Water oxygen (first atom in water)
                        let rmin = rrmin_half + wrmin_half;
                        if rmin > 0.0 && r > 0.0 {
                            let rmin_over_r_6 = f32::powf(rmin / r, 6.0);
                            let rmin_over_r_12 = rmin_over_r_6 * rmin_over_r_6;
                            lj = repsilon * (rmin_over_r_12 - 2.0 * rmin_over_r_6);
                        }
                    }
                    
                    // Coulomb interaction (all atom pairs)
                    let coulomb = 332.0636 * rcharge * wcharge / r;
                    energy += lj + coulomb;
                }
            }
        }
        
        // Store the total interaction energy between this receptor atom and the water being removed
        energies[receptor_energy_base + receptor_idx] = energy;
    }
    
    // PART 2: Calculate water-water interactions between the water being removed and all other active waters
    for other_water_idx in 0..active_waters {
        // Skip self-interaction
        if other_water_idx != water_idx_to_remove {
        
            let other_water_base = waters_base_idx + other_water_idx * WATER_SIZE;
            let mut total_water_water_energy = 0.0;
            
            // Load the other water molecule
            let mut other_water = Array::<f32>::new(WATER_SIZE);
            for i in 0..WATER_SIZE {
                other_water[i] = existing_waters[other_water_base + i];
            }
            
            // Calculate interaction between all atoms in water_to_remove and all atoms in other_water
            for remove_atom_idx in 0..3 { // 3 atoms per water (O, H1, H2)
                for other_atom_idx in 0..3 {
                    let remove_atom_base = remove_atom_idx * ATOM_FEATURES;
                    let other_atom_base = other_atom_idx * ATOM_FEATURES;
                    
                    // Get coordinates and properties of atom from water being removed
                    let remove_x = water_to_remove[remove_atom_base];
                    let remove_y = water_to_remove[remove_atom_base + 1];
                    let remove_z = water_to_remove[remove_atom_base + 2];
                    let remove_charge = water_to_remove[remove_atom_base + 3];
                    let remove_epsilon = water_to_remove[remove_atom_base + 4];
                    let remove_rmin_half = water_to_remove[remove_atom_base + 5];
                    let remove_resnum = water_to_remove[remove_atom_base + 6];
                    
                    // Get coordinates and properties of atom from other water
                    let other_x = other_water[other_atom_base];
                    let other_y = other_water[other_atom_base + 1];
                    let other_z = other_water[other_atom_base + 2];
                    let other_charge = other_water[other_atom_base + 3];
                    let other_epsilon = other_water[other_atom_base + 4];
                    let other_rmin_half = other_water[other_atom_base + 5];
                    let other_resnum = other_water[other_atom_base + 6];
                    
                    // Only calculate if different residues (different water molecules)
                    if remove_resnum != other_resnum {
                        let dx = remove_x - other_x;
                        let dy = remove_y - other_y;
                        let dz = remove_z - other_z;
                        let r2 = dx * dx + dy * dy + dz * dz;
                        
                        if r2 > 0.0 { // Avoid division by zero
                            let r = f32::sqrt(r2);
                            
                            // Lennard-Jones interaction (only oxygen-oxygen)
                            let mut lj = 0.0;
                            if remove_atom_idx == 0 && other_atom_idx == 0 { // Both oxygens
                                let rmin = remove_rmin_half + other_rmin_half;
                                if rmin > 0.0 && r > 0.0 {
                                    // Use geometric mean for epsilon mixing rule
                                    let epsilon_mixed = f32::sqrt(remove_epsilon * other_epsilon);
                                    let rmin_over_r_6 = f32::powf(rmin / r, 6.0);
                                    let rmin_over_r_12 = rmin_over_r_6 * rmin_over_r_6;
                                    lj = epsilon_mixed * (rmin_over_r_12 - 2.0 * rmin_over_r_6);
                                }
                            }
                            
                            // Coulomb interaction (all atom pairs)
                            let coulomb = 332.0636 * remove_charge * other_charge / r;
                            total_water_water_energy += lj + coulomb;
                        }
                    }
                }
            }
            let water_energy_idx = receptor_energy_base + n_receptor_atoms + other_water_idx * 3;
            energies[water_energy_idx] = total_water_water_energy;
        }
    }
}

// Helper function to sum all energies for deletion move
#[cube]
fn get_deletion_energy_sum(
    energies: &Array<f32>,
    sim_id: u32,
    n_receptor_atoms: u32,
    active_waters: u32,
    water_idx_to_remove: u32
) -> f32 {
    let receptor_energy_base = sim_id * (n_receptor_atoms + MAX_N_WATERS as u32 * 3);
    let mut total_energy = 0.0;
    
    // Sum receptor-water interactions
    for i in 0..n_receptor_atoms {
        total_energy += energies[receptor_energy_base + i];
    }
    
    // Sum water-water interactions (excluding the water being removed)
    for i in 0..active_waters {
        if i != water_idx_to_remove {
            total_energy += energies[receptor_energy_base + n_receptor_atoms + i * 3];
        }
    }
    
    total_energy
}

#[cube]
pub fn energy_for_real_water_kernel(
    receptor_atoms: &Array<f32>,      // [n_receptor * 7] - x,y,z,charge,epsilon,rmin_half,resnum
    water_atoms: &Array<f32>,         // [n_waters * 7] - all waters in system
    target_water: &Array<f32>,        // [3 * 7] - the specific water (3 atoms: O, H, H)
    n_receptor: u32,
    n_waters: u32,
    sim_id: u32,
) -> f32 {
    let mut total_energy = 0.0f32;
    
    // Get target water residue number (from first atom - oxygen)
    let target_resnum: f32 = target_water[6];
    
    // Iterate through target water atoms (3 atoms: O, H, H)
    // let receptor_offset: u32 = receptor_atoms.len();
    // Process all receptor atoms
    for r_idx in 0u32..receptor_atoms.len() / ATOM_FEATURES {
        let r_base: u32 = r_idx * ATOM_FEATURES;

        let r_x: f32 = receptor_atoms[r_base];
        let r_y: f32 = receptor_atoms[r_base + 1u32];
        let r_z: f32 = receptor_atoms[r_base + 2u32];
        let r_charge: f32 = receptor_atoms[r_base + 3u32];
        let r_epsilon: f32 = receptor_atoms[r_base + 4u32];
        let r_rmin_half: f32 = receptor_atoms[r_base + 5u32];
        let r_resnum: f32 = receptor_atoms[r_base + 6u32];

        for target_atom_idx in 0u32..3u32 {
            let t_base: u32 = target_atom_idx * ATOM_FEATURES;
            let t_x: f32 = target_water[t_base];
            let t_y: f32 = target_water[t_base + 1u32];
            let t_z: f32 = target_water[t_base + 2u32];
            let t_charge: f32 = target_water[t_base + 3u32];
            let t_epsilon: f32 = target_water[t_base + 4u32];
            let t_rmin_half: f32 = target_water[t_base + 5u32];
            let t_resnum: f32 = target_water[t_base + 6u32];
            
            // Check if target atom is HW (hydrogen in water)
            let t_is_hw: bool = t_epsilon == 0.0f32;
            
            // Calculate distance
            let dx: f32 = t_x - r_x;
            let dy: f32 = t_y - r_y;
            let dz: f32 = t_z - r_z;
            let distance_sq: f32 = dx * dx + dy * dy + dz * dz;
            let distance: f32 = f32::sqrt(distance_sq);
            let r = f32::max(distance, 1e-8f32);
            
            // let r_is_hw: bool = r_epsilon == 0.0f32;
            
            // Lennard-Jones energy (skip if either atom is HW)
            let mut lj_energy = 0.0f32;
            if !t_is_hw {
                lj_energy = gpu_energy::lennard_jones_rmin_half(
                    t_epsilon,
                    r_epsilon,
                    r,
                    t_rmin_half,
                    r_rmin_half
                );
            }
            total_energy += lj_energy;
        
            // Electrostatic energy
            let electrostatics_energy: f32 = gpu_energy::coulomb_energy::<f32>(t_charge, r_charge, r);
            total_energy += electrostatics_energy;
        }
    }        

    let water_offset: u32 = sim_id * n_waters * ATOM_FEATURES;
    
    // Process all water atoms
    for w_atom_idx in 0u32..water_atoms.len() / ATOM_FEATURES {
        // Fix the indexing for water atoms
        let w_base: u32 = water_offset + w_atom_idx * ATOM_FEATURES;
        
        let w_x: f32 = water_atoms[w_base];
        let w_y: f32 = water_atoms[w_base + 1u32];
        let w_z: f32 = water_atoms[w_base + 2u32];
        let w_charge: f32 = water_atoms[w_base + 3u32];
        let w_epsilon: f32 = water_atoms[w_base + 4u32];
        let w_rmin_half: f32 = water_atoms[w_base + 5u32];
        let w_resnum: f32 = water_atoms[w_base + 6u32];

        for target_atom_idx in 0u32..3u32 {
            let t_base: u32 = target_atom_idx * ATOM_FEATURES;
            let t_x: f32 = target_water[t_base];
            let t_y: f32 = target_water[t_base + 1u32];
            let t_z: f32 = target_water[t_base + 2u32];
            let t_charge: f32 = target_water[t_base + 3u32];
            let t_epsilon: f32 = target_water[t_base + 4u32];
            let t_rmin_half: f32 = target_water[t_base + 5u32];
            let t_resnum: f32 = target_water[t_base + 6u32];
            
            // Check if target atom is HW (hydrogen in water)
            let t_is_hw: bool = t_epsilon == 0.0f32;
            
            // Skip if same residue (same water molecule)
            if t_resnum != w_resnum {
                // Calculate distance
                let dx: f32 = t_x - w_x;
                let dy: f32 = t_y - w_y;
                let dz: f32 = t_z - w_z;
                let distance_sq: f32 = dx * dx + dy * dy + dz * dz;
                let distance: f32 = f32::sqrt(distance_sq);
                let r = f32::max(distance, 1e-8f32);
                
                let w_is_hw: bool = w_epsilon == 0.0f32;
                
                // Lennard-Jones energy (skip if either atom is HW)
                let mut lj_energy = 0.0f32;
                if !t_is_hw && !w_is_hw {
                    lj_energy = gpu_energy::lennard_jones_rmin_half(
                        t_epsilon,
                        w_epsilon,
                        r,
                        t_rmin_half,
                        w_rmin_half
                    );
                }
                total_energy += lj_energy;
                
                // Electrostatic energy
                let electrostatics_energy: f32 = gpu_energy::coulomb_energy::<f32>(t_charge, w_charge, r);
                total_energy += electrostatics_energy;
            }
        }
    }
    
    total_energy
}



#[cube]
/// This is just for TIP3P for now
fn create_water_std(resnum: u32) -> Array<f32>{
    let mut new_water: Array<f32> = Array::new(WATER_SIZE);
    new_water[0] = 0.000;
    new_water[1] = 0.000;
    new_water[2] = 0.000;
    new_water[3] = -0.8340;
    new_water[4] = 0.15210325;
    new_water[5] = 1.7682;
    new_water[6] = resnum as f32;
    new_water[7] = 0.000;
    new_water[8] = 0.756;
    new_water[9] = 0.586;
    new_water[10] = 0.4170;
    new_water[11] = 0.000;
    new_water[12] = 0.000;
    new_water[13] = resnum as f32;
    new_water[14] = 0.000;
    new_water[15] = -0.761;
    new_water[16] = 0.594;
    new_water[17] = 0.4170;
    new_water[18] = 0.000;
    new_water[19] = 0.000;
    new_water[20] = resnum as f32;
    new_water
} 


#[cube]
pub fn fast_flat_combine<F: Float>(
    array_a: &Array<F>,
    array_b: &Array<F>,
    output: &mut Array<F>,
    a_len: u32,
    b_len: u32,
) -> u32 {
    let mut write_idx = 0u32;
    
    // Copy array_a
    for i in 0..a_len {
        if i < array_a.len() && write_idx < output.len() {
            output[write_idx] = array_a[i];
            write_idx += 1;
        }
    }
    
    // Copy array_b
    for i in 0..b_len {
        if i < array_b.len() && write_idx < output.len() {
            output[write_idx] = array_b[i];
            write_idx += 1;
        }
    }
    
    write_idx // Return the total number of elements written
}

#[cube]
pub fn combine_arrays_to_sequence<F: Float>(
    array_a: &Array<F>,
    array_b: &Array<F>,
) -> Sequence<F> {
    let mut result = Sequence::<F>::new();
    
    // Add all elements from array_a
    for i in 0..array_a.len() {
        result.push(array_a[i]);
    }
    
    // Add all elements from array_b
    for i in 0..array_b.len() {
        result.push(array_b[i]);
    }
    
    result
}