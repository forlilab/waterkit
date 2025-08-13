use std::sync::atomic::ATOMIC_ISIZE_INIT;

use cubecl::prelude::*;
use crate::{consts, gpu_energy, gpu_geometry::{self, rodrigues_rotation}, gpu_random};


/// FF values 
/// TIP3P
pub const WATER_OXYGEN_CHARGE: f32 = -0.8340;
pub const WATER_OXYGEN_EPSILON: f32 = 0.15210325;
pub const WATER_OXYGEN_RMIN_HALF: f32 = 1.7682;
pub const WATER_HYDROGEN_CHARGE: f32 = 0.4170;

pub const MAX_N_WATERS: u32 = 500;
pub const ATOM_FEATURES: u32 = 7;

// 0 -> oxygen, 1 -> hydrogen_1, 2 -> hydrogen_2, 3 -> resnum
pub const WATER_SIZE: u32 = 4 * 3;

pub const INSERTION_X_IDX: u32 = 0;
pub const INSERTION_Y_IDX: u32 = 1;
pub const INSERTION_Z_IDX: u32 = 2;
pub const TRANSLATION_X_IDX: u32 = 3;
pub const TRANSLATION_Y_IDX: u32 = 4;
pub const TRANSLATION_Z_IDX: u32 = 5;
pub const ROT_AXIS_X_IDX: u32 = 6;
pub const ROT_AXIS_Y_IDX: u32 = 7;
pub const ROT_AXIS_Z_IDX: u32 = 8;
pub const ROTATION_ANGLE: u32 = 9;
pub const WATER_TARGET_IDX: u32 = 10;
pub const ACCEPTANCE_IDX: u32 = 11;

// INSERTION: Add water at the end of active waters array
#[cube]
pub fn insertion_move(
    boundaries: &Array<f32>,
    receptor_atoms: &Array<f32>,
    water_atoms: &mut Array<f32>,
    random_numbers: &Array<f32>,
    sim_id: u32,
    active_waters: u32,
    n_receptor_atoms: u32,
    last_resnum: u32,
    B: f32,
) -> bool {
    let mut accepted = false;
    let waters_handle_idx = sim_id  * MAX_N_WATERS * WATER_SIZE;
    // Check if we have space for another water
    if active_waters < MAX_N_WATERS {
        // Calculate position for new water (at end of active waters)
        let new_water_idx = waters_handle_idx + active_waters * WATER_SIZE;

        // Create temporary water for testing
        let possible_resnum = last_resnum + active_waters + 1;
        let mut new_water = create_water_std(possible_resnum);

        // Generate new water configuration
        propose_insertion_compact(boundaries, random_numbers, &mut new_water, possible_resnum as f32);
        
        // // Calculate energy of new water interacting with receptor and existing waters
        let receptor_energy = energy_for_real_water_kernel(receptor_atoms,
            &new_water,
            n_receptor_atoms,
            sim_id);
        
        let mut waters_energy = 0.0;
        if active_waters > 0 {
            waters_energy = energy_for_real_water_with_waters_kernel(water_atoms, &new_water, sim_id, active_waters);
            // let new_energy = 0.0;
        }
        let new_energy = receptor_energy + waters_energy;

        // Calculate acceptance probability
        let deltaE = new_energy + consts::CHEMICAL_POTENTIAL;
        let acceptance_prob = f32::min(
            (1.0 / (active_waters + 1) as f32) * f32::exp(B) * f32::exp(-consts::BETA * deltaE),
            1.0
        );

        // // Check acceptances
        let rnd_acceptance = random_numbers[ACCEPTANCE_IDX];
        if rnd_acceptance < acceptance_prob {
            // ACCEPT: Copy new water to the active position
            copy_water_to_array(&new_water, water_atoms, new_water_idx);
            accepted = true;
        }
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
    sim_id: u32,
    active_waters: u32,
    n_receptor_atoms: u32,
    B: f32,
) -> bool {
    let mut deleted = false;
    if active_waters > 0 {
        let waters_handle_idx = sim_id  * MAX_N_WATERS * WATER_SIZE;

        // Select random water to delete (0 to active_waters-1)
        let water_to_delete = random_numbers[WATER_TARGET_IDX] as u32;

        // Calculate position of water to delete
        let delete_water_idx = waters_handle_idx + water_to_delete * WATER_SIZE;
        
        // Load water to be deleted for energy calculation
        let mut water_to_remove = Array::<f32>::new(WATER_SIZE);
        load_water_from_array(water_atoms, delete_water_idx, &mut water_to_remove);
        
        // Calculate energy of removing this water
        let receptor_energy = energy_for_real_water_kernel(receptor_atoms,
            &water_to_remove,
            n_receptor_atoms,
            sim_id);
        
        let mut waters_energy = 0.0;
        if active_waters > 0 {
            waters_energy = energy_for_real_water_with_waters_kernel(water_atoms, &water_to_remove, sim_id, active_waters);
        }

        let removed_energy = receptor_energy + waters_energy;
        
        // Calculate acceptance probability for deletion
        // When we remove a water, the energy change is -removed_energy (energy decreases)
        // We also lose the chemical potential contribution
        let deltaE = -removed_energy - consts::CHEMICAL_POTENTIAL;
        let acceptance_prob = f32::min(
            active_waters as f32 * f32::exp(-B) * f32::exp(-consts::BETA * deltaE),
            1.0
        );
        
        // Check acceptance
        let rnd_acceptance = random_numbers[ACCEPTANCE_IDX];
        
        if rnd_acceptance < acceptance_prob {
            // ACCEPT DELETION
            deleted = true;

            // If we're not deleting the last water, move the last water to the deleted position
            // This keeps the array compact
            if water_to_delete < active_waters {
                let last_water_idx = waters_handle_idx + active_waters * WATER_SIZE;
                move_water_in_array(water_atoms, last_water_idx, delete_water_idx);
            }
            
            // Clear the last position (where we moved the water from, or where deleted water was if it was last)
            let clear_idx = waters_handle_idx + active_waters * WATER_SIZE;
            clear_water_in_array(water_atoms, clear_idx);
        }
    }
    deleted
    // If rejected, do nothing - water stays in place
}


// TRANSLATION: Move existing water
#[cube]
pub fn translation_move(
    boundaries: &Array<f32>,
    receptor_atoms: &Array<f32>,
    water_atoms: &mut Array<f32>,
    random_numbers: &Array<f32>,
    sim_id: u32,
    active_waters: u32,
    n_receptor_atoms: u32,
) {
    let waters_handle_idx = sim_id  * MAX_N_WATERS * WATER_SIZE;
    // Select random active water to move
    let water_to_move = random_numbers[WATER_TARGET_IDX] as u32;
    let move_water_idx = waters_handle_idx + water_to_move * WATER_SIZE;
    
    // Load current water
    let mut old_water = Array::<f32>::new(WATER_SIZE);
    load_water_from_array(water_atoms, move_water_idx, &mut old_water);
    
    let receptor_energy_old = energy_for_real_water_kernel(receptor_atoms,
            &old_water,
            n_receptor_atoms,
            sim_id);
    let waters_energy_old = energy_for_real_water_with_waters_kernel(water_atoms, &old_water, sim_id, active_waters);
    
    let energy_old = receptor_energy_old + waters_energy_old;
    
    // Propose new position
    let new_water = propose_perturbation(boundaries, &old_water, random_numbers);
    
    let receptor_energy_new = energy_for_real_water_kernel(receptor_atoms,
            &new_water,
            n_receptor_atoms,
            sim_id);
    let waters_energy_new = energy_for_real_water_with_waters_kernel(water_atoms, &new_water, sim_id, active_waters);

    let energy_new = receptor_energy_new + waters_energy_new;

    // Accept/reject and update if accepted
    let rnd_acceptance = random_numbers[ACCEPTANCE_IDX];
    
    let deltaE = energy_new - energy_old; // Calculate actual energy difference
    let acceptance_prob = f32::min(f32::exp(-consts::BETA * deltaE), 1.0);
    
    if rnd_acceptance < acceptance_prob {
        copy_water_to_array(&new_water, water_atoms, move_water_idx);
    }
}

// Helper functions for compact array operations
#[cube]
pub fn copy_water_to_array(source: &Array<f32>, dest: &mut Array<f32>, dest_base_idx: u32) {
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
}


// Modified insertion proposal for compact layout
#[cube]
fn propose_insertion_compact(
    boundaries: &Array<f32>,
    random_numbers: &Array<f32>,
    new_water: &mut Array<f32>,
    resnum: f32,
) {
    let delta_x = random_numbers[INSERTION_X_IDX];
    let delta_y = random_numbers[INSERTION_Y_IDX];
    let delta_z = random_numbers[INSERTION_Z_IDX];
    let angle_rnd = random_numbers[ROTATION_ANGLE];
    let axis_x = random_numbers[ROT_AXIS_X_IDX];
    let axis_y = random_numbers[ROT_AXIS_Y_IDX];
    let axis_z = random_numbers[ROT_AXIS_Z_IDX];

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
    h1[0] = new_water[4];
    h1[1] = new_water[5];
    h1[2] = new_water[6];

    let mut h2 = Array::new(3);
    h2[0] = new_water[8];
    h2[1] = new_water[9];
    h2[2] = new_water[10];

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
    new_water[4] = new_h1[0] + new_ox; // hydrogen1 x
    new_water[5] = new_h1[1] + new_oy; // hydrogen1 y
    new_water[6] = new_h1[2] + new_oz; // hydrogen1 z

    let mut new_h2: Array::<f32> = Array::new(3);
    rodrigues_rotation(&h2, &normalized_axis_array, angle, &original_oxygen, &mut new_h2);
    new_water[8] = new_h2[0] + new_ox; // hydrogen2 x
    new_water[9] = new_h2[1] + new_oy; // hydrogen2 y
    new_water[10] = new_h2[2] + new_oz; // hydrogen2 z

    new_water[0] = new_ox;
    new_water[1] = new_oy;
    new_water[2] = new_oz;
}

#[cube]
fn propose_perturbation(
    boundaries: &Array<f32>,
    old_water: &Array<f32>, // Flattened water coordinates [ox, oy, oz, charge_o, epsilon_o, rmin_half_o, resnum_o, h1x, h1y, h1z, charge_h1, epsilon_h1, rmin_half_h1, resnum_h1, h2x, h2y, h2z, charge_h2, epsilon_h2, rmin_half_h2, resnum_h2]
    rng_array: &Array<f32>) -> Array<f32> {

    let resnum = old_water[3] as u32;
    let mut new_water = create_water_std(resnum);
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

    let delta_x = rng_array[TRANSLATION_X_IDX];
    let delta_y = rng_array[TRANSLATION_Y_IDX];
    let delta_z = rng_array[TRANSLATION_Z_IDX];
    let angle_rnd = rng_array[ROTATION_ANGLE];
    let axis_x = rng_array[ROT_AXIS_X_IDX];
    let axis_y = rng_array[ROT_AXIS_Y_IDX];
    let axis_z = rng_array[ROT_AXIS_Z_IDX];

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
    let in_bounds = (new_ox >= x_min) && (new_ox <= x_max) &&
                   (new_oy >= y_min) && (new_oy <= y_max) &&
                   (new_oz >= z_min) && (new_oz <= z_max);
    
    if in_bounds {
        let angle = angle_rnd; // Small rotation
        
        // Rotate hydrogen atoms around oxygen (z-axis rotation for simplicity)
        let mut h1 = Array::new(3);
        h1[0] = new_water[4];
        h1[1] = new_water[5];
        h1[2] = new_water[6];

        let mut h2 = Array::new(3);
        h2[0] = new_water[8];
        h2[1] = new_water[9];
        h2[2] = new_water[10];

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

        let mut normalized_axis_array = gpu_geometry::normalize(&mut axis_array);

        let mut new_h1: Array::<f32> = Array::new(3);
        rodrigues_rotation(&h1, &axis_array, angle, &original_oxygen, &mut new_h1);
        new_water[7] = new_h1[0] + new_ox; // hydrogen1 x
        new_water[8] = new_h1[1] + new_oy; // hydrogen1 y
        new_water[9] = new_h1[2] + new_oz; // hydrogen1 z

        let mut new_h2: Array::<f32> = Array::new(3);
        rodrigues_rotation(&h2, &axis_array, angle, &original_oxygen, &mut new_h2);
        new_water[14] = new_h2[0] + new_ox; // hydrogen2 x
        new_water[15] = new_h2[1] + new_oy; // hydrogen2 y
        new_water[16] = new_h2[2] + new_oz; // hydrogen2 z

        new_water[0] = new_ox;
        new_water[1] = new_oy;
        new_water[2] = new_oz;
    } 
    new_water
}


#[cube]
pub fn energy_for_real_water_kernel(
    receptor_atoms: &Array<f32>,      // [n_receptor * 7]
    target_water: &Array<f32>,        // [3 * 4]
    n_receptor: u32,
    sim_id: u32,
) -> f32 {
    let mut total_energy = 0.0f32;
    let water_atom_stride = 4;
    let atoms_per_water = 3;
    let receptor_stride = ATOM_FEATURES; // 7
    let target_oxygen_base = 0 * water_atom_stride;
    let target_resnum = target_water[target_oxygen_base + 3];

    // === Interact with receptor atoms ===
    let n_receptor_atoms = receptor_atoms.len() as u32 / receptor_stride;
    for r_idx in 0..n_receptor_atoms {
        let r_base = r_idx * receptor_stride;

        let r_x = receptor_atoms[r_base];
        let r_y = receptor_atoms[r_base + 1];
        let r_z = receptor_atoms[r_base + 2];
        let r_charge = receptor_atoms[r_base + 3];
        let r_epsilon = receptor_atoms[r_base + 4];
        let r_rmin_half = receptor_atoms[r_base + 5];
        let r_resnum = receptor_atoms[r_base + 6];

        for target_atom_idx in 0..atoms_per_water {
            let t_base = target_atom_idx * water_atom_stride;

            let t_x = target_water[t_base];
            let t_y = target_water[t_base + 1];
            let t_z = target_water[t_base + 2];
            let t_resnum_t = target_water[t_base + 3];

            let mut t_charge = 0.0;
            let mut t_epsilon = 0.0;
            let mut t_rmin_half = 0.0;
            if target_atom_idx == 0 {
                t_charge = WATER_OXYGEN_CHARGE;
                t_epsilon = WATER_OXYGEN_EPSILON;
                t_rmin_half = WATER_OXYGEN_RMIN_HALF;
            } else {
                t_charge = WATER_HYDROGEN_CHARGE;
            }

            let t_is_hw = t_epsilon == 0.0;

            let dx = t_x - r_x;
            let dy = t_y - r_y;
            let dz = t_z - r_z;
            let distance_sq = dx*dx + dy*dy + dz*dz;
            let distance = f32::sqrt(distance_sq);
            let r_val = f32::max(distance, 1e-8);

            let mut lj_energy = 0.0;
            if !t_is_hw {
                lj_energy = gpu_energy::lennard_jones_rmin_half(
                    t_epsilon, r_epsilon, r_val, t_rmin_half, r_rmin_half);
            }
            total_energy += lj_energy;

            let electrostatics_energy = gpu_energy::coulomb_energy::<f32>(t_charge, r_charge, r_val);
            total_energy += electrostatics_energy;
        }
    }
    total_energy
}

#[cube]
fn energy_for_real_water_with_waters_kernel(
    water_atoms: &Array<f32>, 
    target_water: &Array<f32>, 
    sim_id: u32,
    active_waters: u32
) -> f32 {
    let mut total_energy: f32 = 0.0;
    let water_atom_stride = 4;
    let atoms_per_water = 3;
    let waters_base_idx = sim_id * MAX_N_WATERS * WATER_SIZE;
    
    // Only iterate through active waters, not all possible waters
    for w_idx in 0..active_waters {
        let water_base = waters_base_idx + w_idx * WATER_SIZE;
        
        // Process each atom in this water molecule
        for w_atom_idx in 0..atoms_per_water {
            let base = water_base + w_atom_idx * water_atom_stride;
            let w_x = water_atoms[base];
            let w_y = water_atoms[base + 1];
            let w_z = water_atoms[base + 2];
            let w_resnum = water_atoms[base + 3];
            
            // Skip if this is an uninitialized water (resnum = 0)
            if w_resnum != 0.0 {
                let mut w_charge = 0.0;
                let mut w_epsilon = 0.0;
                let mut w_rmin_half = 0.0;
                
                if w_atom_idx == 0 {  // Oxygen
                    w_charge = WATER_OXYGEN_CHARGE;
                    w_epsilon = WATER_OXYGEN_EPSILON;
                    w_rmin_half = WATER_OXYGEN_RMIN_HALF;
                } else {  // Hydrogen
                    w_charge = WATER_HYDROGEN_CHARGE;
                }
                
                // Calculate interaction with target water
                for target_atom_idx in 0..atoms_per_water {
                    let t_base = target_atom_idx * water_atom_stride;
                    let t_x = target_water[t_base];
                    let t_y = target_water[t_base + 1];
                    let t_z = target_water[t_base + 2];
                    let t_resnum = target_water[t_base + 3];
                    
                    let mut t_charge = 0.0;
                    let mut t_epsilon = 0.0;
                    let mut t_rmin_half = 0.0;
                    
                    if target_atom_idx == 0 {  // Oxygen
                        t_charge = WATER_OXYGEN_CHARGE;
                        t_epsilon = WATER_OXYGEN_EPSILON;
                        t_rmin_half = WATER_OXYGEN_RMIN_HALF;
                    } else {  // Hydrogen
                        t_charge = WATER_HYDROGEN_CHARGE;
                    }
                    
                    // Skip same residue interactions
                    if t_resnum != w_resnum {
                    
                        let dx = t_x - w_x;
                        let dy = t_y - w_y;
                        let dz = t_z - w_z;
                        let distance_sq = dx * dx + dy * dy + dz * dz;
                        let distance = f32::sqrt(distance_sq);
                        let r_val = f32::max(distance, 1e-8);
                        
                        // Calculate LJ energy (only for O-O interactions)
                        let w_is_hw = w_epsilon == 0.0;
                        let t_is_hw = t_epsilon == 0.0;
                        
                        if !t_is_hw && !w_is_hw {
                            let lj_energy = gpu_energy::lennard_jones_rmin_half(
                                t_epsilon, w_epsilon, r_val, t_rmin_half, w_rmin_half
                            );
                            total_energy += lj_energy;
                        }
                        
                        // Calculate electrostatic energy
                        let electrostatics_energy = gpu_energy::coulomb_energy::<f32>(
                            t_charge, w_charge, r_val
                        );
                        total_energy += electrostatics_energy;
                    }
                }
            }
        }
    }
    
    total_energy
}


#[cube]
/// This is just for TIP3P for now
pub fn create_water_std(resnum: u32) -> Array<f32>{
    let mut new_water: Array<f32> = Array::new(WATER_SIZE);
    new_water[0] = 0.000;
    new_water[1] = 0.000;
    new_water[2] = 0.000;
    new_water[3] = resnum as f32;
    new_water[4] = 0.000;
    new_water[5] = 0.756;
    new_water[6] = 0.586;
    new_water[7] = resnum as f32;
    new_water[8] = 0.000;
    new_water[9] = -0.761;
    new_water[10] = 0.594;
    new_water[11] = resnum as f32;
    new_water
} 