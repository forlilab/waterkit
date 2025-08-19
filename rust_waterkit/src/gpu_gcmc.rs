use core::f32;

use crate::atom::Atom;
use crate::{consts, gpu_energy, gpu_gcmc_moves, gpu_geometry, gpu_multithread, gpu_random};
use crate::water::WaterMolecule;
use cubecl::std::tensor::TensorHandle;
use cubecl::{compute, prelude::*};
use cubecl_random::{random_normal, random_uniform};
use fixed::types::extra::Unsigned;
use nalgebra::Scalar;
use rand::distributions::Uniform;
use rand::prelude::Distribution;
use rand::rngs::ThreadRng;
use rand::{random, Rng};

pub const NUM_THREADS: u32 = 256;

#[cube(launch)]
pub fn run_gcmc_multithread(
    boundaries: &Array<f32>,
    receptor_atoms: &Array<f32>,
    water_atoms: &mut Array<f32>,
    seeds: &mut Array<u32>,
    num_waters: &mut Array<u32>,
    last_resnum: u32,
    B: f32,
    volume: f32,
    steps: u32,
    target_n_waters: u32,
) {
    let sim_id = CUBE_POS_X;
    let n_receptor_atoms = receptor_atoms.len() / consts::ATOM_FEATURES;

    // Persistent per-sim state
    let mut rng_state = seeds[sim_id];
    let mut active_waters = num_waters[sim_id];

    // Local scratch
    let mut random_numbers = Array::new(12u32);
    let mut trial_water = Array::new(12u32);

    // Shared buffers
    let mut sh_trial = SharedMemory::<f32>::new(consts::WATER_SIZE);
    let mut sh_trial_old = SharedMemory::<f32>::new(consts::WATER_SIZE); // NEW: for displacement "old" pose
    let mut move_type = 0u32;

    let mut shared_active_waters = SharedMemory::<u32>::new(1);
    if UNIT_POS_X == 0 { shared_active_waters[0] = active_waters; }

    for _step in 0..steps {
        // ===========================
        // SINGLE THREAD: prepare move
        // ===========================
        if UNIT_POS_X == 0 {
            shared_active_waters[0] = active_waters;

            // Randoms (you already had these)
            random_numbers[consts::INSERTION_X_IDX] = gpu_random::random_range(&mut rng_state, boundaries[0], boundaries[1]);
            random_numbers[consts::INSERTION_Y_IDX] = gpu_random::random_range(&mut rng_state, boundaries[2], boundaries[3]);
            random_numbers[consts::INSERTION_Z_IDX] = gpu_random::random_range(&mut rng_state, boundaries[4], boundaries[5]);
            random_numbers[consts::TRANSLATION_X_IDX] = gpu_random::random_range(&mut rng_state, -0.3f32, 0.3f32);
            random_numbers[consts::TRANSLATION_Y_IDX] = gpu_random::random_range(&mut rng_state, -0.3f32, 0.3f32);
            random_numbers[consts::TRANSLATION_Z_IDX] = gpu_random::random_range(&mut rng_state, -0.3f32, 0.3f32);
            random_numbers[consts::ROT_AXIS_X_IDX]   = gpu_random::random_float(&mut rng_state);
            random_numbers[consts::ROT_AXIS_Y_IDX]   = gpu_random::random_float(&mut rng_state);
            random_numbers[consts::ROT_AXIS_Z_IDX]   = gpu_random::random_float(&mut rng_state);
            random_numbers[consts::ROTATION_ANGLE]   = gpu_random::random_range(&mut rng_state, -180.0f32, 180.0f32);
            random_numbers[consts::ACCEPTANCE_IDX]   = gpu_random::random_float(&mut rng_state);

            // Pick target if needed
            if active_waters > 0 {
                random_numbers[consts::WATER_TARGET_IDX] = gpu_random::random_int_range(&mut rng_state, active_waters) as f32;
            } else {
                random_numbers[consts::WATER_TARGET_IDX] = 0.0f32;
            }

            // ---------- MOVE SELECTION ----------
            // 0 = insertion, 1 = deletion, 2 = displacement
            if active_waters == 0 {
                move_type = 0;
            } else if active_waters >= target_n_waters {
                // choose deletion or displacement
                let r = gpu_random::random_int_range(&mut rng_state, 2);
                move_type = 1 + r; // 1 or 2
            } else {
                // choose among 0,1,2
                move_type = gpu_random::random_int_range(&mut rng_state, 3);
            }

            if move_type == 0 {
                // ---------- PREPARE INSERTION ----------
                let counter_idx = sim_id;
                let current_res = num_waters[counter_idx];
                let possible_resnum = current_res + 1;
                let mut new_water = gpu_gcmc_moves::randomize_water(&random_numbers);
                gpu_gcmc_moves::propose_insertion_compact(boundaries, &random_numbers, &mut new_water, possible_resnum as f32);

                // flatten to compact [x y z res | x y z res | x y z res]
                trial_water[0]  = new_water[0];  trial_water[1]  = new_water[1];
                trial_water[2]  = new_water[2];  trial_water[3]  = possible_resnum as f32;
                trial_water[4]  = new_water[4];  trial_water[5]  = new_water[5];
                trial_water[6]  = new_water[6];  trial_water[7]  = possible_resnum as f32;
                trial_water[8]  = new_water[8];  trial_water[9]  = new_water[9];
                trial_water[10] = new_water[10]; trial_water[11] = possible_resnum as f32;

                for i in 0..consts::WATER_SIZE { sh_trial[i] = trial_water[i]; }
            } else if move_type == 1 {
                // ---------- PREPARE DELETION ----------
                let waters_handle_idx = sim_id * target_n_waters * consts::WATER_SIZE;
                let del_idx = random_numbers[consts::WATER_TARGET_IDX] as u32;
                let src = waters_handle_idx + del_idx * consts::WATER_SIZE;
                gpu_gcmc_moves::load_water_from_array(water_atoms, src, &mut trial_water);
                for i in 0..consts::WATER_SIZE { sh_trial[i] = trial_water[i]; }
            } else {
                // ---------- PREPARE DISPLACEMENT ----------
                let waters_handle_idx = sim_id * target_n_waters * consts::WATER_SIZE;
                let tgt_idx = random_numbers[consts::WATER_TARGET_IDX] as u32;
                let src = waters_handle_idx + tgt_idx * consts::WATER_SIZE;

                // Load OLD pose
                let mut old_water = Array::new(consts::WATER_SIZE as u32);
                gpu_gcmc_moves::load_water_from_array(water_atoms, src, &mut old_water);

                // Build TRIAL (translate + rotate)
                let mut new_water = gpu_gcmc_moves::propose_perturbation(boundaries, &old_water, &random_numbers);

                // flatten to compact [x y z res | x y z res | x y z res]
                trial_water[0]  = new_water[0];  trial_water[1]  = new_water[1];
                trial_water[2]  = new_water[2];  trial_water[3]  = new_water[3];
                trial_water[4]  = new_water[4];  trial_water[5]  = new_water[5];
                trial_water[6]  = new_water[6];  trial_water[7]  = new_water[7];
                trial_water[8]  = new_water[8];  trial_water[9]  = new_water[9];
                trial_water[10] = new_water[10]; trial_water[11] = new_water[11];

                // Broadcast both old and trial
                for i in 0..consts::WATER_SIZE { sh_trial_old[i] = old_water[i]; }
                for i in 0..consts::WATER_SIZE { sh_trial[i] = new_water[i]; }

                // Keep local copies for thread 0 scalar work later
                for i in 0..consts::WATER_SIZE { trial_water[i] = new_water[i]; }
            }
        }

        // ensure shared is ready
        sync_cube();
        let current_active_waters = shared_active_waters[0];

        // ==================================
        // MULTITHREAD: compute energies
        // ==================================

        // Copy shared trial into per-thread array
        let mut trial_copy = Array::new(consts::WATER_SIZE as u32);
        for i in 0..consts::WATER_SIZE { trial_copy[i] = sh_trial[i]; }

        // For displacement we’ll also need the OLD pose receptor energy in parallel
        let mut old_copy = Array::new(consts::WATER_SIZE as u32);
        for i in 0..consts::WATER_SIZE { old_copy[i] = sh_trial_old[i]; }

        // Compute receptor energies
        let e_rec_new = gpu_multithread::energy_for_real_water_kernel_parallel(
            receptor_atoms, &trial_copy, n_receptor_atoms, sim_id
        );

        // If not displacement, e_rec_old is unused; keep 0.0 to avoid UB
        let mut e_rec_old = 0.0f32;

        // For displacement: compute old receptor contribution as well
        // We can gate this with move_type without branching divergence for UNIT 0 only,
        // but a small extra cost to compute here is fine.
        let e_rec_old_tmp = gpu_multithread::energy_for_real_water_kernel_parallel(
            receptor_atoms, &old_copy, n_receptor_atoms, sim_id
        );

        // Barrier before reductions in thread 0
        sync_cube();

        // ======================================
        // SINGLE THREAD: reduce + accept/reject
        // ======================================
        if UNIT_POS_X == 0 {
            let rnd_acceptance = random_numbers[consts::ACCEPTANCE_IDX];

            if move_type == 0 {
                // ---------- INSERTION ----------
                let e_wat_new = gpu_gcmc_moves::energy_for_real_water_with_waters_kernel(
                    water_atoms, &trial_water, sim_id, active_waters, target_n_waters
                );
                let new_energy = e_rec_new + e_wat_new;

                let delta_e = new_energy + consts::CHEMICAL_POTENTIAL;
                let pref = (1.0f32 / (active_waters + 1) as f32) * f32::exp(B);
                let p_acc = f32::min(pref * f32::exp(-consts::BETA * delta_e), 1.0f32);

                if rnd_acceptance < p_acc {
                    let waters_handle_idx = sim_id * target_n_waters * consts::WATER_SIZE;
                    let dst = waters_handle_idx + active_waters * consts::WATER_SIZE;
                    gpu_gcmc_moves::copy_water_to_array(&trial_water, water_atoms, dst);
                    active_waters += 1;
                    let counter_idx = sim_id;
                    num_waters[counter_idx] = num_waters[counter_idx] + 1;
                }

            } else if move_type == 1 && active_waters > 0 {
                // ---------- DELETION ----------
                // We already computed e_rec_new above using the candidate (which is the one to delete).
                // That’s how your original code defined old_contrib = new_energy.
                let e_wat_del = gpu_gcmc_moves::energy_for_real_water_with_waters_kernel(
                    water_atoms, &trial_water, sim_id, active_waters, target_n_waters
                );
                let old_contrib = e_rec_new + e_wat_del;

                let delta_e = -old_contrib - consts::CHEMICAL_POTENTIAL;
                let pref = (active_waters as f32) * f32::exp(-B);
                let p_acc = f32::min(pref * f32::exp(-consts::BETA * delta_e), 1.0f32);

                if rnd_acceptance < p_acc {
                    let waters_handle_idx = sim_id * target_n_waters * consts::WATER_SIZE;
                    let del_idx = random_numbers[consts::WATER_TARGET_IDX] as u32;
                    let dst = waters_handle_idx + del_idx * consts::WATER_SIZE;
                    let last_idx = waters_handle_idx + (active_waters - 1) * consts::WATER_SIZE;

                    if del_idx + 1 < active_waters {
                        gpu_gcmc_moves::move_water_in_array(water_atoms, last_idx, dst);
                    }
                    gpu_gcmc_moves::clear_water_in_array(water_atoms, last_idx);
                    active_waters -= 1;
                }

            } else if move_type == 2 && active_waters > 0 {
                // ---------- DISPLACEMENT ----------
                let waters_handle_idx = sim_id * target_n_waters * consts::WATER_SIZE;
                let tgt_idx = random_numbers[consts::WATER_TARGET_IDX] as u32;
                let src = waters_handle_idx + tgt_idx * consts::WATER_SIZE;

                // OLD pose was in sh_trial_old / old_copy
                // NEW pose is in trial_water / trial_copy

                // Receptor parts
                e_rec_old = e_rec_old_tmp;
                let e_rec_new_used = e_rec_new;

                // Water-water parts (exclude self index!)
                // Replace with your "exclude index" variant if you have it:
                let e_wat_old = gpu_gcmc_moves::energy_for_real_water_with_waters_kernel(
                    water_atoms, &old_copy, sim_id, active_waters, target_n_waters
                );
                let e_wat_new = gpu_gcmc_moves::energy_for_real_water_with_waters_kernel(
                    water_atoms, &trial_water, sim_id, active_waters, target_n_waters
                );

                let old_energy = e_rec_old + e_wat_old;
                let new_energy = e_rec_new_used + e_wat_new;
                let delta_e = new_energy - old_energy;

                let p_acc = f32::min(f32::exp(-consts::BETA * delta_e), 1.0f32);

                if rnd_acceptance < p_acc {
                    // Accept: write new pose into its original slot
                    gpu_gcmc_moves::copy_water_to_array(&trial_water, water_atoms, src);
                }
            }
        }

        sync_cube();
    }

    #[cfg(feature = "mc_after_gcmc")]
    // In case we want to run MC after GCMC
    for _step in 0..75000 {
        // ===========================
        // SINGLE THREAD: prepare move
        // ===========================
        if UNIT_POS_X == 0 {
            shared_active_waters[0] = active_waters;

            // Randoms (you already had these)
            random_numbers[consts::INSERTION_X_IDX] = gpu_random::random_range(&mut rng_state, boundaries[0], boundaries[1]);
            random_numbers[consts::INSERTION_Y_IDX] = gpu_random::random_range(&mut rng_state, boundaries[2], boundaries[3]);
            random_numbers[consts::INSERTION_Z_IDX] = gpu_random::random_range(&mut rng_state, boundaries[4], boundaries[5]);
            random_numbers[consts::TRANSLATION_X_IDX] = gpu_random::random_range(&mut rng_state, -0.3f32, 0.3f32);
            random_numbers[consts::TRANSLATION_Y_IDX] = gpu_random::random_range(&mut rng_state, -0.3f32, 0.3f32);
            random_numbers[consts::TRANSLATION_Z_IDX] = gpu_random::random_range(&mut rng_state, -0.3f32, 0.3f32);
            random_numbers[consts::ROT_AXIS_X_IDX]   = gpu_random::random_float(&mut rng_state);
            random_numbers[consts::ROT_AXIS_Y_IDX]   = gpu_random::random_float(&mut rng_state);
            random_numbers[consts::ROT_AXIS_Z_IDX]   = gpu_random::random_float(&mut rng_state);
            random_numbers[consts::ROTATION_ANGLE]   = gpu_random::random_range(&mut rng_state, -180.0f32, 180.0f32);
            random_numbers[consts::ACCEPTANCE_IDX]   = gpu_random::random_float(&mut rng_state);

            // Pick target if needed
            if active_waters > 0 {
                random_numbers[consts::WATER_TARGET_IDX] = gpu_random::random_int_range(&mut rng_state, active_waters) as f32;
            } else {
                random_numbers[consts::WATER_TARGET_IDX] = 0.0f32;
            }

            // ---------- PREPARE DISPLACEMENT ----------
            let waters_handle_idx = sim_id * target_n_waters * consts::WATER_SIZE;
            let tgt_idx = random_numbers[consts::WATER_TARGET_IDX] as u32;
            let src = waters_handle_idx + tgt_idx * consts::WATER_SIZE;

            // Load OLD pose
            let mut old_water = Array::new(consts::WATER_SIZE as u32);
            gpu_gcmc_moves::load_water_from_array(water_atoms, src, &mut old_water);

            // Build TRIAL (translate + rotate)
            let mut new_water = gpu_gcmc_moves::propose_perturbation(boundaries, &old_water, &random_numbers);

            // flatten to compact [x y z res | x y z res | x y z res]
            trial_water[0]  = new_water[0];  trial_water[1]  = new_water[1];
            trial_water[2]  = new_water[2];  trial_water[3]  = new_water[3];
            trial_water[4]  = new_water[4];  trial_water[5]  = new_water[5];
            trial_water[6]  = new_water[6];  trial_water[7]  = new_water[7];
            trial_water[8]  = new_water[8];  trial_water[9]  = new_water[9];
            trial_water[10] = new_water[10]; trial_water[11] = new_water[11];

            // Broadcast both old and trial
            for i in 0..consts::WATER_SIZE { sh_trial_old[i] = old_water[i]; }
            for i in 0..consts::WATER_SIZE { sh_trial[i] = new_water[i]; }

            // Keep local copies for thread 0 scalar work later
            for i in 0..consts::WATER_SIZE { trial_water[i] = new_water[i]; }
        }

        // ensure shared is ready
        sync_cube();
        let current_active_waters = shared_active_waters[0];

        // ==================================
        // MULTITHREAD: compute energies
        // ==================================

        // Copy shared trial into per-thread array
        let mut trial_copy = Array::new(consts::WATER_SIZE as u32);
        for i in 0..consts::WATER_SIZE { trial_copy[i] = sh_trial[i]; }

        // For displacement we’ll also need the OLD pose receptor energy in parallel
        let mut old_copy = Array::new(consts::WATER_SIZE as u32);
        for i in 0..consts::WATER_SIZE { old_copy[i] = sh_trial_old[i]; }

        // Compute receptor energies
        let e_rec_new = gpu_multithread::energy_for_real_water_kernel_parallel(
            receptor_atoms, &trial_copy, n_receptor_atoms, sim_id
        );

        // If not displacement, e_rec_old is unused; keep 0.0 to avoid UB
        let mut e_rec_old = 0.0f32;

        // For displacement: compute old receptor contribution as well
        // We can gate this with move_type without branching divergence for UNIT 0 only,
        // but a small extra cost to compute here is fine.
        let e_rec_old_tmp = gpu_multithread::energy_for_real_water_kernel_parallel(
            receptor_atoms, &old_copy, n_receptor_atoms, sim_id
        );

        // Barrier before reductions in thread 0
        sync_cube();

        // ======================================
        // SINGLE THREAD: reduce + accept/reject
        // ======================================
        if UNIT_POS_X == 0 {
            let rnd_acceptance = random_numbers[consts::ACCEPTANCE_IDX];
            // ---------- DISPLACEMENT ----------
            let waters_handle_idx = sim_id * target_n_waters * consts::WATER_SIZE;
            let tgt_idx = random_numbers[consts::WATER_TARGET_IDX] as u32;
            let src = waters_handle_idx + tgt_idx * consts::WATER_SIZE;

            // OLD pose was in sh_trial_old / old_copy
            // NEW pose is in trial_water / trial_copy

            // Receptor parts
            e_rec_old = e_rec_old_tmp;
            let e_rec_new_used = e_rec_new;

            // Water-water parts (exclude self index!)
            // Replace with your "exclude index" variant if you have it:
            let e_wat_old = gpu_gcmc_moves::energy_for_real_water_with_waters_kernel(
                water_atoms, &old_copy, sim_id, active_waters, target_n_waters
            );
            let e_wat_new = gpu_gcmc_moves::energy_for_real_water_with_waters_kernel(
                water_atoms, &trial_water, sim_id, active_waters, target_n_waters
            );

            let old_energy = e_rec_old + e_wat_old;
            let new_energy = e_rec_new_used + e_wat_new;
            let delta_e = new_energy - old_energy;

            let factor = f32::cast_from(consts::BOLTZMANN_K * consts::TEMPERATURE);
            let p_acc = f32::min(f32::exp(-delta_e / factor), 1.0);

            if rnd_acceptance < p_acc {
                // Accept: write new pose into its original slot
                gpu_gcmc_moves::copy_water_to_array(&trial_water, water_atoms, src);
            }
        }

        sync_cube();
    }

    if UNIT_POS_X == 0 {
        seeds[sim_id] = rng_state;
        num_waters[sim_id] = active_waters;
    }
}

#[cube(launch)]
fn run_gcmc(
    boundaries: &Array<f32>,
    receptor_atoms: &Array<f32>,
    water_atoms: &mut Array<f32>,
    seeds: &mut Array<u32>,           // persistent RNG state
    num_waters: &mut Array<u32>,      // persistent per-sim water count
    waters_res_numbers: &mut Array<u32>, // persistent water residue numbers
    last_resnum: u32,
    B: f32,
    volume: f32,
    steps: u32,                       // steps per launch (batch)
    target_n_waters: u32, // target number of waters per simulation
) {
    let sim_id = CUBE_POS_X;
    let n_receptor_atoms = receptor_atoms.len() / consts::ATOM_FEATURES;

    // Load persistent RNG state
    let mut rng_state = seeds[sim_id];
    // Load current water count for this simulation
    let mut active_waters = 0;

    // Preallocate random number buffer
    // Random numbers should follow this order:
    // [insertion_x, insertion_y, insertion_z, translation_x, translation_y, translation_z, axis_x, axis_y, axis_z, angle, target_water, acceptance_probability]
    let mut random_numbers = Array::<f32>::new(12);

    for step in 0..steps {
        // debug_print!("\nEpoch: %d\n", step);
        let move_type = gpu_random::random_int_range(&mut rng_state, 2) as u32; // 0: insertion, 1: deletion, 2: displacement

        // Fill random numbers once per move
        random_numbers[0] = gpu_random::random_range(&mut rng_state, boundaries[0], boundaries[1]);
        random_numbers[1] = gpu_random::random_range(&mut rng_state, boundaries[2], boundaries[3]);
        random_numbers[2] = gpu_random::random_range(&mut rng_state, boundaries[4], boundaries[5]);
        random_numbers[3] = gpu_random::random_range(&mut rng_state, -0.5, 0.5);
        random_numbers[4] = gpu_random::random_range(&mut rng_state, -0.5, 0.5);
        random_numbers[5] = gpu_random::random_range(&mut rng_state, -0.5, 0.5);
        random_numbers[6] = gpu_random::random_float(&mut rng_state);
        random_numbers[7] = gpu_random::random_float(&mut rng_state);
        random_numbers[8] = gpu_random::random_float(&mut rng_state);
        random_numbers[9] = gpu_random::random_range(&mut rng_state, -180_f32, 180_f32);
        if active_waters > 0 {
        random_numbers[10] = gpu_random::random_int_range(&mut rng_state, active_waters) as f32;
        } else {
            random_numbers[10] = 0.0;
        }
        random_numbers[11] = gpu_random::random_float(&mut rng_state);

        let target_water = random_numbers[10] as u32;
        // Bounds check: don’t add water if array full
        if active_waters < target_n_waters {
            if move_type == 0 {
                // debug_print!("Insertion move - %d\n", active_waters);
                if gpu_gcmc_moves::insertion_move(
                    boundaries,
                    receptor_atoms,
                    water_atoms,
                    &random_numbers,
                    sim_id,
                    active_waters,
                    waters_res_numbers,
                    n_receptor_atoms,
                    last_resnum,
                    B,
                    target_n_waters
                ) {
                    // let accepted = true;
                    // debug_print!("Insertion accepted: %d\n", accepted);
                    active_waters += 1;
                }
            } else if move_type == 1 && active_waters > 0 {
                // debug_print!("Deletion move - %d - water %d\n", target_water);
                // Deletion move
                if gpu_gcmc_moves::deletion_move(
                    receptor_atoms,
                    water_atoms,
                    &random_numbers,
                    sim_id,
                    active_waters,
                    n_receptor_atoms,
                    B,
                    target_n_waters
                ) {
                    // let deleted = true;
                    // debug_print!("Deletion accepted: %d\n", deleted);
                    active_waters -= 1;
                }
            } else if move_type == 2 && active_waters > 0 {
                // debug_print!("Displacement move - %d - water %d\n", active_waters, target_water);
                // Displacement move
                gpu_gcmc_moves::translation_move(
                    boundaries,
                    receptor_atoms,
                    water_atoms,
                    &random_numbers,
                    sim_id,
                    active_waters,
                    n_receptor_atoms,
                    false, // production_mc
                    target_n_waters
                );
            }
            sync_cube();
        }
    }
    
    // Run normal MC
    for step in 0..100000 {
        // Fill random numbers once per move
        random_numbers[0] = gpu_random::random_range(&mut rng_state, boundaries[0], boundaries[1]);
        random_numbers[1] = gpu_random::random_range(&mut rng_state, boundaries[2], boundaries[3]);
        random_numbers[2] = gpu_random::random_range(&mut rng_state, boundaries[4], boundaries[5]);
        random_numbers[3] = gpu_random::random_range(&mut rng_state, -0.3, 0.3);
        random_numbers[4] = gpu_random::random_range(&mut rng_state, -0.3, 0.3);
        random_numbers[5] = gpu_random::random_range(&mut rng_state, -0.3, 0.3);
        random_numbers[6] = gpu_random::random_float(&mut rng_state);
        random_numbers[7] = gpu_random::random_float(&mut rng_state);
        random_numbers[8] = gpu_random::random_float(&mut rng_state);
        random_numbers[9] = gpu_random::random_range(&mut rng_state, -180_f32, 180_f32);
        random_numbers[10] = gpu_random::random_int_range(&mut rng_state, active_waters) as f32;
        random_numbers[11] = gpu_random::random_float(&mut rng_state);
        
        let target_water = random_numbers[10] as u32;
        // debug_print!("Production move - %d - water %d\n", active_waters, target_water);
        gpu_gcmc_moves::translation_move(
            boundaries,
            receptor_atoms,
            water_atoms,
            &random_numbers,
            sim_id,
            active_waters,
            n_receptor_atoms,
            true, // production_mc
            target_n_waters
        );
    }

    // Save updated state for next batch
    seeds[sim_id] = rng_state;
}

pub fn simulate<R: Runtime>(
    n_simulations: usize,
    device: &R::Device,
    receptor_atoms: Vec<Atom>,
    water_configuration: WaterMolecule,
    cutoff: f32,
    boundaries: Vec<f32>,
    volume: f32,
    num_steps: usize,
    target_n_waters: usize) -> Vec<f32> {
    let client: ComputeClient<<R as Runtime>::Server, <R as Runtime>::Channel> = R::client(device);

    // Try to use cubecl-random to create random seeds to pass to the kernels
    // These are the seeds that will be used to generate random numbers for the simulation
    // One seed per simulation per epoch -> random molecule to affect will be picked with xorshift function + the seed
    debug_buffer_calculations(n_simulations, target_n_waters);
    let seed_tensor = TensorHandle::<R, u32>::empty(&client, [n_simulations*num_steps].to_vec());
    random_uniform::<R, u32>(&client, u32::MIN, u32::MAX - 1, seed_tensor.as_ref());

    println!("# Atoms in the receptor: {}", receptor_atoms.len());

    // Move the receptor to the global memory on the GPU
    let mut receptor_atoms_buffer = Vec::with_capacity(receptor_atoms.len() * consts::ATOM_FEATURES as usize);
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

    let max_n_waters = target_n_waters as usize;
    println!("N FRAMES: {}", n_simulations);
    // The only thing that I need for the waters is the coordinates (3*3)
    let mut waters_buffer = Vec::with_capacity(n_simulations * max_n_waters * consts::WATER_SIZE as usize);

    for idx in 0..(n_simulations * max_n_waters) {
        waters_buffer.push(0.0);
        waters_buffer.push(0.0);
        waters_buffer.push(0.0);
        waters_buffer.push(0.);
        waters_buffer.push(0.0);
        waters_buffer.push(0.0);
        waters_buffer.push(0.0);
        waters_buffer.push(0.);
        waters_buffer.push(0.0);
        waters_buffer.push(0.0);
        waters_buffer.push(0.0);
        waters_buffer.push(0.);
    }

    let waters_res_numbers: Vec<u32> = vec![0; n_simulations];

    println!("N WATERS: {}", waters_buffer.len());
    let boundaries_handle = client.create(f32::as_bytes(&boundaries));
    let receptor_atoms_handle = client.create(f32::as_bytes(&receptor_atoms_buffer));
    let water_atoms_handle = client.create(f32::as_bytes(&waters_buffer));
    let wat_num_handle = client.empty(n_simulations * core::mem::size_of::<u32>());
    let waters_res_numbers_handle = client.create(u32::as_bytes(&waters_res_numbers));
    
    let mut rec_partials = vec![0.0; n_simulations * CUBE_DIM_X as usize];
    let mut wat_partials = vec![0.0; n_simulations * CUBE_DIM_X as usize];

    let rec_partials_handle = client.empty(n_simulations * CUBE_DIM_X as usize * core::mem::size_of::<f32>());
    let wat_partials_handle = client.empty(n_simulations * CUBE_DIM_X as usize * core::mem::size_of::<f32>());

    let volume_var = volume / consts::STANDARD_VOLUME;
    let B = consts::CHEMICAL_POTENTIAL * consts::BETA + volume_var.ln();
    if NUM_THREADS < 2 {
        unsafe {
            run_gcmc::launch::<R>(
                &client, 
                CubeCount::Static(n_simulations as u32, 1, 1),
                CubeDim::new(1,1, 1),
                ArrayArg::from_raw_parts::<f32>(&boundaries_handle, 6, 1), 
                ArrayArg::from_raw_parts::<f32>(&receptor_atoms_handle, receptor_atoms_buffer.len(), 1), 
                ArrayArg::from_raw_parts::<f32>(&water_atoms_handle, waters_buffer.len(), 1), 
                ArrayArg::from_raw_parts::<u32>(&seed_tensor.handle, n_simulations*num_steps, 1), 
                ArrayArg::from_raw_parts::<u32>(&wat_num_handle, n_simulations, 1),
                ArrayArg::from_raw_parts::<i32>(&waters_res_numbers_handle, n_simulations, 1),
                ScalarArg {elem: receptor_atoms_buffer[receptor_atoms_buffer.len() -1] as u32}, 
                ScalarArg {elem: B},
                ScalarArg {elem: volume}, 
                ScalarArg {elem: num_steps as u32},
                ScalarArg {elem: target_n_waters as u32},
            );
        }
    } else {
        unsafe {
            run_gcmc_multithread::launch::<R>(
                &client, 
                CubeCount::Static(n_simulations as u32, 1, 1),
                CubeDim::new(NUM_THREADS,1, 1),
                ArrayArg::from_raw_parts::<f32>(&boundaries_handle, 6, 1), 
                ArrayArg::from_raw_parts::<f32>(&receptor_atoms_handle, receptor_atoms_buffer.len(), 1), 
                ArrayArg::from_raw_parts::<f32>(&water_atoms_handle, waters_buffer.len(), 1), 
                ArrayArg::from_raw_parts::<u32>(&seed_tensor.handle, n_simulations*num_steps, 1), 
                ArrayArg::from_raw_parts::<u32>(&wat_num_handle, n_simulations, 1),
                ScalarArg {elem: receptor_atoms_buffer[receptor_atoms_buffer.len() -1] as u32}, 
                ScalarArg {elem: B},
                ScalarArg {elem: volume}, 
                ScalarArg {elem: num_steps as u32},
                ScalarArg {elem: target_n_waters as u32},
            );
        }
    }
    let bytes = client.read_one(water_atoms_handle.clone().binding());
    let output: Vec<f32> = f32::from_bytes(&bytes).to_vec();

    output
}


pub fn debug_buffer_calculations(n_simulations: usize, target_n_waters: usize) {
    println!("=== BUFFER SIZE DEBUGGING ===");
    println!("MAX_N_WATERS: {}", target_n_waters);
    println!("WATER_SIZE: {}", consts::WATER_SIZE);
    println!("n_simulations: {}", n_simulations);
    
    // Check for potential overflow in index calculations
    let waters_per_sim = target_n_waters as usize * consts::WATER_SIZE as usize;
    let total_water_elements = n_simulations * waters_per_sim;
    
    println!("Waters per simulation: {}", waters_per_sim);
    println!("Total water buffer elements: {}", total_water_elements);
    println!("Total buffer size (MB): {:.2}", (total_water_elements * 4) as f64 / 1_048_576.0);
    
    // Check if calculations would overflow u32
    let max_sim_id = n_simulations - 1;
    let max_base_idx = max_sim_id as u32 * target_n_waters as u32 * consts::WATER_SIZE;
    let max_water_idx = max_base_idx + (target_n_waters as u32 - 1) * consts::WATER_SIZE;
    
    println!("Max simulation base index: {}", max_base_idx);
    println!("Max water index: {}", max_water_idx);
    println!("Total buffer length: {}", total_water_elements);
    
    if max_water_idx as usize >= total_water_elements {
        println!("❌ OVERFLOW DETECTED! Max index {} >= buffer size {}", 
                 max_water_idx, total_water_elements);
    } else {
        println!("✅ Index calculations look safe");
    }
    
    // Check for u32 overflow in the multiplication itself
    let check_overflow = (max_sim_id as u64) * (target_n_waters as u64) * (consts::WATER_SIZE as u64);
    if check_overflow > u32::MAX as u64 {
        println!("❌ U32 OVERFLOW in index calculation! {} > {}", check_overflow, u32::MAX);
    } else {
        println!("✅ No u32 overflow in index calculations");
    }
}