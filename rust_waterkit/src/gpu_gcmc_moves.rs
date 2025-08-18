use cubecl::prelude::*;
use crate::consts;
use crate::gpu_energy;
use crate::gpu_geometry::{self, rodrigues_rotation};
use crate::gpu_random;

#[cube]
pub fn randomize_water(random_numbers: &Array<f32>) -> Array<f32> {
    let d: f32 = f32::cast_from(0f32);
    let rot_axis_x = random_numbers[consts::ROT_AXIS_X_IDX];
    let rot_axis_y = random_numbers[consts::ROT_AXIS_Y_IDX];
    let rot_axis_z = random_numbers[consts::ROT_AXIS_Z_IDX];
    let rot_angle = random_numbers[consts::ROTATION_ANGLE];

    let mut axis_arr = Array::new(3);
    axis_arr[0] = rot_axis_x;
    axis_arr[1] = rot_axis_y;
    axis_arr[2] = rot_axis_z;
    let normalized_axis = gpu_geometry::normalize(&mut axis_arr);

    let mut new_water = create_water_std(0);
    let mut pivot = Array::new(3);
    pivot[0] = new_water[0];
    pivot[1] = new_water[1];
    pivot[2] = new_water[2];

    let mut h1 = Array::new(3);
    h1[0] = new_water[4];
    h1[1] = new_water[5];
    h1[2] = new_water[6];

    let mut h2 = Array::new(3);
    h2[0] = new_water[8];
    h2[1] = new_water[9];
    h2[2] = new_water[10];

    let mut new_h1 = Array::new(3);
    gpu_geometry::rodrigues_rotation(&h1, &normalized_axis, rot_angle, &pivot, &mut new_h1);

    let mut new_h2 = Array::new(3);
    gpu_geometry::rodrigues_rotation(&h2, &normalized_axis, rot_angle, &pivot, &mut new_h2);
    
    new_water[4] = new_h1[0];
    new_water[5] = new_h1[1];
    new_water[6] = new_h1[2];
    new_water[8] = new_h2[0];
    new_water[9] = new_h2[1];
    new_water[10] = new_h2[2];

    new_water
}

// INSERTION: Add water at the end of active waters array
#[cube]
pub fn insertion_move(
    boundaries: &Array<f32>,
    receptor_atoms: &Array<f32>,
    water_atoms: &mut Array<f32>,
    random_numbers: &Array<f32>,
    sim_id: u32,
    active_waters: u32,
    res_counter: &mut Array<u32>,
    n_receptor_atoms: u32,
    last_resnum: u32,
    B: f32,
) -> bool {
    let mut accepted = false;
    let waters_handle_idx = sim_id  * consts::MAX_N_WATERS * consts::WATER_SIZE;
    // Check if we have space for another water
    // debug_print!("Insertion move - active waters: %d\n", active_waters);
    if active_waters < consts::MAX_N_WATERS {
        // Calculate position for new water (at end of active waters)
        let new_water_idx = waters_handle_idx + active_waters * consts::WATER_SIZE;

        // Create temporary water for testing
        let counter_idx = sim_id;
        let current_res = res_counter[counter_idx];
        let possible_resnum = current_res + 1;
        let mut new_water = randomize_water(random_numbers);

        // Generate new water configuration
        let insertion_within_boundaries = propose_insertion_compact(boundaries, random_numbers, &mut new_water, possible_resnum as f32);
        if insertion_within_boundaries {
            new_water[3] = possible_resnum as f32;
            new_water[7] = possible_resnum as f32;
            new_water[11] = possible_resnum as f32;

            // // Calculate energy of new water interacting with receptor and existing waters
            // Single-thread
            let receptor_energy = energy_for_real_water_kernel(receptor_atoms,
                &new_water,
                n_receptor_atoms,
                sim_id);

            let waters_energy = energy_for_real_water_with_waters_kernel(water_atoms, &new_water, sim_id, active_waters);
            // debug_print!("Insertion move - receptor's energy: %f\n", receptor_energy);
            // debug_print!("Insertion move - waters' energy: %f\n", waters_energy);
            // debug_print!("Insertion move waters energy: %f\n", waters_energy);
            let new_energy = receptor_energy + waters_energy;
            // debug_print!("Insertion move new energy: %f\n", new_energy);
            // Calculate acceptance probability
            let deltaE = new_energy + consts::CHEMICAL_POTENTIAL;
            // debug_print!("Insertion move deltaE energy: %f\n", deltaE);
            let acceptance_prob = f32::min(
                (1.0 / (active_waters + 1) as f32) * f32::exp(B) * f32::exp(-consts::BETA * deltaE),
                1.0
            );
            // debug_print!("Insertion move acceptance prob: %f\n", acceptance_prob);

            // // Check acceptances
            let rnd_acceptance = random_numbers[consts::ACCEPTANCE_IDX];
            // debug_print!("Insertion move random acceptance: %f\n", rnd_acceptance);
            if rnd_acceptance < acceptance_prob {
            // if new_energy < 0.0 {
                // ACCEPT: Copy new water to the active position
                // Wait for energy calculations to complete -> avoid race conditions
                sync_cube();
                copy_water_to_array(&new_water, water_atoms, new_water_idx);
                res_counter[counter_idx] = possible_resnum;
                accepted = true;
                // debug_print!("Insertion move - water %d inserted\n", possible_resnum);
            }
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
    // debug_print!("Deletion move - active waters: %d\n", active_waters);
    if active_waters > 0 {
        let waters_handle_idx = sim_id  * consts::MAX_N_WATERS * consts::WATER_SIZE;

        // Select random water to delete (0 to active_waters-1)
        let water_to_delete = random_numbers[consts::WATER_TARGET_IDX] as u32;

        // Calculate position of water to delete
        let delete_water_idx = waters_handle_idx + water_to_delete * consts::WATER_SIZE;
        // debug_print!("Water to delete: %d from %d active waters\n", water_to_delete, active_waters);
        // Load water to be deleted for energy calculation
        let mut water_to_remove = Array::<f32>::new(consts::WATER_SIZE);
        load_water_from_array(water_atoms, delete_water_idx, &mut water_to_remove);
        
        // Calculate energy of removing this water
        let receptor_energy = energy_for_real_water_kernel(receptor_atoms,
            &water_to_remove,
            n_receptor_atoms,
            sim_id);
        
        let waters_energy = energy_for_real_water_with_waters_kernel(water_atoms, &water_to_remove, sim_id, active_waters);
        // debug_print!("Deletion move - receptor's energy: %f\n", receptor_energy);
        // debug_print!("Deletion move - waters' energy: %f\n", waters_energy);
        let removed_energy = receptor_energy + waters_energy;
        
        // Calculate acceptance probability for deletion
        // When we remove a water, the energy change is -removed_energy (energy decreases)
        // We also lose the chemical potential contribution
        let deltaE = -removed_energy - consts::CHEMICAL_POTENTIAL;
        let acceptance_prob = f32::min(
            active_waters as f32 * f32::exp(-B) * f32::exp(-consts::BETA * deltaE),
            1.0
        );
        // debug_print!("Deletion move - deltaE: %f\n", deltaE);
        // debug_print!("Deletion move - acceptance prob: %f\n", acceptance_prob);
        // Check acceptance
        let rnd_acceptance = random_numbers[consts::ACCEPTANCE_IDX];
        
        if rnd_acceptance < acceptance_prob {
            // ACCEPT DELETION
            deleted = true;
            
            // FIXED: Only move water if we're not deleting the last water
            if water_to_delete < active_waters - 1 {
                // Move the LAST water (index active_waters - 1) to the deleted position
                let last_water_idx = waters_handle_idx + (active_waters - 1) * consts::WATER_SIZE;
                // debug_print!("Moving last water from index %d to deleted position %d\n", 
                //             last_water_idx, delete_water_idx);
                
                // Wait for energy calculations to complete -> avoid race conditions
                sync_cube();

                move_water_in_array(water_atoms, last_water_idx, delete_water_idx);
            // } else {
                // debug_print!("Deleted water was already the last one, no moving needed\n", active_waters);
            }
            
            // FIXED: Clear the last position (where the last water was before moving)
            let last_water_idx = waters_handle_idx + (active_waters - 1) * consts::WATER_SIZE;
            // debug_print!("Clearing water at last position: %d\n", last_water_idx);
            clear_water_in_array(water_atoms, last_water_idx);
        }
    }
    // debug_print!("Deletion move - accepted: %d\n", deleted);
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
    production_mc: bool
) {
    let waters_handle_idx = sim_id  * consts::MAX_N_WATERS * consts::WATER_SIZE;
    // Select random active water to move
    let water_to_move = random_numbers[consts::WATER_TARGET_IDX] as u32;
    let move_water_idx = waters_handle_idx + water_to_move * consts::WATER_SIZE;
    
    // Load current water
    let mut old_water = Array::<f32>::new(consts::WATER_SIZE);
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
    let rnd_acceptance = random_numbers[consts::ACCEPTANCE_IDX];
    
    let deltaE = energy_new - energy_old; // Calculate actual energy difference

    if !production_mc {
        let acceptance_prob = f32::min(f32::exp(-consts::BETA * deltaE), 1.0);

        if rnd_acceptance < acceptance_prob {
                // Wait for energy calculations to complete -> avoid race conditions
            sync_cube();
            copy_water_to_array(&new_water, water_atoms, move_water_idx);
        }
    } else {
        if energy_new < energy_old {
            // Wait for energy calculations to complete -> avoid
            sync_cube();
            copy_water_to_array(&new_water, water_atoms, move_water_idx);
        } else {
            let factor = f32::cast_from(consts::BOLTZMANN_K * consts::TEMPERATURE);
            let p_acc = f32::min(f32::exp(-deltaE / factor), 1.0);
            if rnd_acceptance < p_acc {
                // Wait for energy calculations to complete -> avoid race conditions
                sync_cube();
                copy_water_to_array(&new_water, water_atoms, move_water_idx);
            }
        }
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
pub fn load_water_from_array(source: &Array<f32>, source_base_idx: u32, dest: &mut Array<f32>) {
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
pub fn move_water_in_array(waters: &mut Array<f32>, from_idx: u32, to_idx: u32) {
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
pub fn clear_water_in_array(waters: &mut Array<f32>, base_idx: u32) {
    waters[base_idx] = 0.0;
    waters[base_idx+1] = 0.0;
    waters[base_idx+2] = 0.0;
    waters[base_idx+3] = 0.0;
    waters[base_idx+4] = 0.0;
    waters[base_idx+5] = 0.0;
    waters[base_idx+6] = 0.0;
    waters[base_idx+7] = 0.0;
    waters[base_idx+8] = 0.0;
    waters[base_idx+9] = 0.0;
    waters[base_idx+10] = 0.0;
    waters[base_idx+11] = 0.0;
}

#[cube]
pub fn create_water_std(resnum: u32) -> Array<f32>{
    let mut new_water: Array<f32> = Array::new(consts::WATER_SIZE);
    #[cfg(feature = "tip3p")]
    new_water[0] = 0.000f32;
    new_water[1] = 0.000f32;
    new_water[2] = 0.000f32;
    new_water[3] = resnum as f32;
    new_water[4] = 0.000f32;
    new_water[5] = 0.756f32;
    new_water[6] = 0.586f32;
    new_water[7] = resnum as f32;
    new_water[8] = 0.000f32;
    new_water[9] = -0.761f32;
    new_water[10] = 0.594f32;
    new_water[11] = resnum as f32;

    #[cfg(feature = "tip3pfp")]
    new_water[0] = 0.000f32;
    new_water[1] = 0.000f32;
    new_water[2] = -0.018f32;
    new_water[3] = resnum as f32;
    new_water[4] = 0.000f32;
    new_water[5] = 0.761f32;
    new_water[6] = 0.595f32;
    new_water[7] = resnum as f32;
    new_water[8] = 0.000f32;
    new_water[9] = -0.761f32;
    new_water[10] = 0.594f32;
    new_water[11] = resnum as f32;

    new_water

} 


// Modified insertion proposal for compact layout
#[cube]
pub fn propose_insertion_compact(
    boundaries: &Array<f32>,
    random_numbers: &Array<f32>,
    new_water: &mut Array<f32>,
    resnum: f32,
) -> bool {
    let mut good_to_go = false;
    let delta_x = random_numbers[consts::INSERTION_X_IDX];
    let delta_y = random_numbers[consts::INSERTION_Y_IDX];
    let delta_z = random_numbers[consts::INSERTION_Z_IDX];

    // Calculate new oxygen position
    let new_ox = delta_x;
    let new_oy = delta_y;
    let new_oz = delta_z;
    
    // Get boundary values
    let x_min = boundaries[0];
    let x_max = boundaries[1];
    let y_min = boundaries[2];
    let y_max = boundaries[3];
    let z_min = boundaries[4];
    let z_max = boundaries[5];
    let in_bounds = (new_ox >= x_min) && (new_ox <= x_max) &&
                   (new_oy >= y_min) && (new_oy <= y_max) &&
                   (new_oz >= z_min) && (new_oz <= z_max);
    
    if in_bounds {
        new_water[0] = new_ox;
        new_water[1] = new_oy;
        new_water[2] = new_oz;
        new_water[4] = new_water[4] + new_ox; // hydrogen1 x
        new_water[5] = new_water[5] + new_oy; // hydrogen1 y
        new_water[6] = new_water[6] + new_oz; // hydrogen1 z
        new_water[8] = new_water[8] + new_ox; // hydrogen2 x
        new_water[9] = new_water[9] + new_oy; // hydrogen2 y
        new_water[10] = new_water[10] + new_oz; // hydrogen2 z
        good_to_go = true;
    }
    good_to_go
}

#[cube]
pub fn propose_perturbation(
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

    let delta_x = rng_array[consts::TRANSLATION_X_IDX];
    let delta_y = rng_array[consts::TRANSLATION_Y_IDX];
    let delta_z = rng_array[consts::TRANSLATION_Z_IDX];
    let angle_rnd = rng_array[consts::ROTATION_ANGLE];
    let axis_x = rng_array[consts::ROT_AXIS_X_IDX];
    let axis_y = rng_array[consts::ROT_AXIS_Y_IDX];
    let axis_z = rng_array[consts::ROT_AXIS_Z_IDX];

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
        h1[0] = new_water[4] + delta_x;
        h1[1] = new_water[5] + delta_y;
        h1[2] = new_water[6] + delta_z;

        let mut h2 = Array::new(3);
        h2[0] = new_water[8] + delta_x;
        h2[1] = new_water[9] + delta_y;
        h2[2] = new_water[10] + delta_z;

        let mut new_oxygen = Array::new(3);
        new_oxygen[0] = new_ox;
        new_oxygen[1] = new_oy;
        new_oxygen[2] = new_oz; 

        let mut axis_array = Array::new(3);
        axis_array[0] = axis_x;
        axis_array[1] = axis_y;
        axis_array[2] = axis_z; 

        let mut normalized_axis_array = gpu_geometry::normalize(&mut axis_array);

        let mut new_h1: Array::<f32> = Array::new(3);
        rodrigues_rotation(&h1, &normalized_axis_array, angle, &new_oxygen, &mut new_h1);
        new_water[4] = new_h1[0]; // hydrogen1 x
        new_water[5] = new_h1[1]; // hydrogen1 y
        new_water[6] = new_h1[2]; // hydrogen1 z

        let mut new_h2: Array::<f32> = Array::new(3);
        rodrigues_rotation(&h2, &normalized_axis_array, angle, &new_oxygen, &mut new_h2);
        new_water[8] = new_h2[0]; // hydrogen2 x
        new_water[9] = new_h2[1]; // hydrogen2 y
        new_water[10] = new_h2[2]; // hydrogen2 z

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
    let receptor_stride = consts::ATOM_FEATURES; // 7
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

            #[cfg(feature = "tip3p")]
            if target_atom_idx == 0 {  // Oxygen
                t_charge = -0.8340;
                t_epsilon = 0.15210325;
                t_rmin_half = 1.7682;
            } else {  // Hydrogen
                t_charge = 0.4170;
            }

            #[cfg(feature = "tip3pfp")]
            if target_atom_idx == 0 {  // Oxygen
                t_charge = -0.8484;
                t_epsilon = 0.15586604;
                t_rmin_half = 1.7835723;
            } else {  // Hydrogen
                t_charge = 0.4242;
            }

            let t_is_hw = t_epsilon == 0.0;

            let dx = t_x - r_x;
            let dy = t_y - r_y;
            let dz = t_z - r_z;
            let distance_sq = dx*dx + dy*dy + dz*dz;
            let distance = f32::sqrt(distance_sq);
            let r_val = f32::max(distance, 1e-8);

            // if r_val < f32::cast_from(consts::ELECTROSTATICS_CUTOFF) {
                let mut lj_energy = 0.0;
                if !t_is_hw {
                    lj_energy = gpu_energy::lennard_jones_rmin_half(
                        t_epsilon, r_epsilon, r_val, t_rmin_half, r_rmin_half);
                    //     debug_print!("Receptor - Distance: %f for %d and %d\n", r_val, t_resnum_t, r_resnum);
                    // debug_print!("Receptor - LJ energy: %f\n", lj_energy);
                }
                total_energy += lj_energy;

                let electrostatics_energy = gpu_energy::coulomb_energy::<f32>(t_charge, r_charge, r_val);
                total_energy += electrostatics_energy;
            // }
        }
    }
    total_energy
}

#[cube]
pub fn energy_for_real_water_with_waters_kernel(
    water_atoms: &Array<f32>,
    target_water: &Array<f32>,
    sim_id: u32,
    active_waters: u32
) -> f32 {
    let mut total_energy: f32 = 0.0;
    let water_atom_stride = 4;
    let atoms_per_water = 3;
    
    // debug_print!("\nActive waters: %d\n", active_waters);
    
    // Calculate total number of active atoms (3 atoms per water)
    let total_active_atoms = active_waters * atoms_per_water;
    let atoms_base_idx = sim_id * consts::MAX_N_WATERS * atoms_per_water;
    // debug_print!("atoms_base_idx = %d, total_active_atoms = %d\n", atoms_base_idx, total_active_atoms);

    // Iterate through all active water atoms
    for atom_idx in 0..total_active_atoms {
        let global_atom_idx = atoms_base_idx + atom_idx;
        // FIXED: Don't multiply by water_atom_stride again!
        let atom_array_idx = global_atom_idx * water_atom_stride;
        
        // debug_print!("atom_idx: %d, global_atom_idx: %d, array_idx: %d\n", 
        //             atom_idx, global_atom_idx, atom_array_idx);
        
        let w_x = water_atoms[atom_array_idx];
        let w_y = water_atoms[atom_array_idx + 1];
        let w_z = water_atoms[atom_array_idx + 2];
        let w_resnum = water_atoms[atom_array_idx + 3];
        
        // Cast to u32 to avoid float interpretation issues
        let w_resnum_int = w_resnum as u32;
        
        // debug_print!("Water atom position: (%f, %f, %f), resnum: %d\n", 
        //             w_x, w_y, w_z, w_resnum_int);
        
        let mut w_charge = 0.0;
        let mut w_epsilon = 0.0;
        let mut w_rmin_half = 0.0;
        
        // Determine atom type based on position within the water molecule
        let atom_type_idx = atom_idx % atoms_per_water;
        // debug_print!("Atom type idx: %d\n", atom_type_idx);
        
        #[cfg(feature = "tip3p")]
        if atom_type_idx == 0 { // Oxygen
            w_charge = -0.8340;
            w_epsilon = 0.15210325;
            w_rmin_half = 1.7682;
        } else { // Hydrogen
            w_charge = 0.4170;
        }
        
        #[cfg(feature = "tip3pfp")]
        if atom_type_idx == 0 { // Oxygen
            w_charge = -0.8484;
            w_epsilon = 0.15586604;
            w_rmin_half = 1.7835723;
        } else { // Hydrogen
            w_charge = 0.4242;
        }
        
        // debug_print!("Water atom charge: %f, epsilon: %f\n", w_charge, w_epsilon);
        
        // Calculate interaction with target water
        for target_atom_idx in 0..atoms_per_water {
            let t_base = target_atom_idx * water_atom_stride;
            let t_x = target_water[t_base];
            let t_y = target_water[t_base + 1];
            let t_z = target_water[t_base + 2];
            let t_resnum = target_water[t_base + 3];
            let t_resnum_int = t_resnum as u32;

            let mut t_charge = 0.0;
            let mut t_epsilon = 0.0;
            let mut t_rmin_half = 0.0;
            
            #[cfg(feature = "tip3p")]
            if target_atom_idx == 0 { // Oxygen
                t_charge = -0.8340;
                t_epsilon = 0.15210325;
                t_rmin_half = 1.7682;
            } else { // Hydrogen
                t_charge = 0.4170;
            }
            
            #[cfg(feature = "tip3pfp")]
            if target_atom_idx == 0 { // Oxygen
                t_charge = -0.8484;
                t_epsilon = 0.15586604;
                t_rmin_half = 1.7835723;
            } else { // Hydrogen
                t_charge = 0.4242;
            }
            
            // debug_print!("t_resnum: %d and w_resnum: %d\n", t_resnum_int, w_resnum_int);
            
            // Skip same residue interactions
            if t_resnum_int != w_resnum_int {
                let dx = t_x - w_x;
                let dy = t_y - w_y;
                let dz = t_z - w_z;
                let distance_sq = dx * dx + dy * dy + dz * dz;
                let distance = f32::sqrt(distance_sq);
                let r_val = f32::max(distance, 1e-8);
                
                // debug_print!("Distance: %f for %d and %d\n", r_val, t_resnum_int, w_resnum_int);
                // if r_val < f32::cast_from(consts::ELECTROSTATICS_CUTOFF) {
                    // Calculate LJ energy (only for O-O interactions)
                    let w_is_hw = w_epsilon == 0.0;
                    let t_is_hw = t_epsilon == 0.0;
                    
                    if !t_is_hw && !w_is_hw {
                        let lj_energy = gpu_energy::lennard_jones_rmin_half(
                            t_epsilon, w_epsilon, r_val, t_rmin_half, w_rmin_half
                        );
                        // debug_print!("Water - Distance: %f for %d and %d\n", r_val, t_resnum_int, w_resnum_int);
                        // debug_print!("Water - LJ energy: %f\n", lj_energy);
                        total_energy += lj_energy;
                    }
                    
                    // Calculate electrostatic energy
                    let electrostatics_energy = gpu_energy::coulomb_energy::<f32>(
                        t_charge, w_charge, r_val
                    );
                    // debug_print!("Electrostatic energy: %f\n", electrostatics_energy);
                    total_energy += electrostatics_energy;
                // }
            }
        }
    }
    
    // debug_print!("Final total energy: %f\n", total_energy);
    total_energy
}
