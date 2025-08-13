use core::f32;

use crate::atom::Atom;
use crate::{consts, gpu_energy, gpu_gcmc_moves, gpu_geometry, gpu_random};
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

#[cube(launch)]
fn run_gcmc(
    boundaries: &Array<f32>,
    receptor_atoms: &Array<f32>,
    water_atoms: &mut Array<f32>,
    seeds: &mut Array<u32>,           // persistent RNG state
    num_waters: &mut Array<u32>,      // persistent per-sim water count
    last_resnum: u32,
    B: f32,
    volume: f32,
    steps: u32,                       // steps per launch (batch)
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

        let move_type = gpu_random::random_int_range(&mut rng_state, 3) as u32; // 0: insertion, 1: deletion, 2: displacement

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
        random_numbers[10] = gpu_random::random_int_range(&mut rng_state, active_waters) as f32;
        random_numbers[11] = gpu_random::random_float(&mut rng_state);

        // Bounds check: don’t add water if array full
        if active_waters < consts::MAX_N_WATERS {
            if move_type == 0 {
                // let base_water_idx = waters_base_idx + active_waters * gpu_gcmc_moves::WATER_SIZE;
                if gpu_gcmc_moves::insertion_move(
                    boundaries,
                    receptor_atoms,
                    water_atoms,
                    &random_numbers,
                    sim_id,
                    active_waters,
                    n_receptor_atoms,
                    last_resnum,
                    B,
                ) {
                    active_waters += 1;
                }
            } else if move_type == 1 && active_waters > 0 {
                // Deletion move
                if gpu_gcmc_moves::deletion_move(
                    receptor_atoms,
                    water_atoms,
                    &random_numbers,
                    sim_id,
                    active_waters,
                    n_receptor_atoms,
                    B
                ) {
                    active_waters -= 1;
                }
            } else if move_type == 2 && active_waters > 0 {
                // Displacement move
                gpu_gcmc_moves::translation_move(
                    boundaries,
                    receptor_atoms,
                    water_atoms,
                    &random_numbers,
                    sim_id,
                    active_waters,
                    n_receptor_atoms,
                );
            }
        }
    }
    
    // Run normal MC
    for step in 0..100000 {
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
        random_numbers[10] = gpu_random::random_int_range(&mut rng_state, active_waters) as f32;
        random_numbers[11] = gpu_random::random_float(&mut rng_state);
        
        gpu_gcmc_moves::translation_move(
                    boundaries,
                    receptor_atoms,
                    water_atoms,
                    &random_numbers,
                    sim_id,
                    active_waters,
                    n_receptor_atoms,
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
    num_steps: usize) -> Vec<f32> {
    let client: ComputeClient<<R as Runtime>::Server, <R as Runtime>::Channel> = R::client(device);

    // Try to use cubecl-random to create random seeds to pass to the kernels
    // These are the seeds that will be used to generate random numbers for the simulation
    // One seed per simulation per epoch -> random molecule to affect will be picked with xorshift function + the seed
    debug_buffer_calculations(n_simulations);
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

    let max_n_waters = consts::MAX_N_WATERS as usize;
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
    println!("N WATERS: {}", waters_buffer.len());
    let boundaries_handle = client.create(f32::as_bytes(&boundaries));
    let receptor_atoms_handle = client.create(f32::as_bytes(&receptor_atoms_buffer));
    let water_atoms_handle = client.create(f32::as_bytes(&waters_buffer));
    let wat_num_handle = client.empty(n_simulations * core::mem::size_of::<u32>());
    
    let volume_var = volume / consts::STANDARD_VOLUME;
    let B = consts::CHEMICAL_POTENTIAL * consts::BETA + volume_var.ln();

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
            ScalarArg {elem: receptor_atoms_buffer[receptor_atoms_buffer.len() -1] as u32}, 
            ScalarArg {elem: B},
            ScalarArg {elem: volume}, 
            ScalarArg {elem: num_steps as u32},
        );
    }

    let bytes = client.read_one(water_atoms_handle.clone().binding());
    let output: Vec<f32> = f32::from_bytes(&bytes).to_vec();

    output
}


pub fn debug_buffer_calculations(n_simulations: usize) {
    println!("=== BUFFER SIZE DEBUGGING ===");
    println!("MAX_N_WATERS: {}", consts::MAX_N_WATERS);
    println!("WATER_SIZE: {}", consts::WATER_SIZE);
    println!("n_simulations: {}", n_simulations);
    
    // Check for potential overflow in index calculations
    let waters_per_sim = consts::MAX_N_WATERS as usize * consts::WATER_SIZE as usize;
    let total_water_elements = n_simulations * waters_per_sim;
    
    println!("Waters per simulation: {}", waters_per_sim);
    println!("Total water buffer elements: {}", total_water_elements);
    println!("Total buffer size (MB): {:.2}", (total_water_elements * 4) as f64 / 1_048_576.0);
    
    // Check if calculations would overflow u32
    let max_sim_id = n_simulations - 1;
    let max_base_idx = max_sim_id as u32 * consts::MAX_N_WATERS * consts::WATER_SIZE;
    let max_water_idx = max_base_idx + (consts::MAX_N_WATERS - 1) * consts::WATER_SIZE;
    
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
    let check_overflow = (max_sim_id as u64) * (consts::MAX_N_WATERS as u64) * (consts::WATER_SIZE as u64);
    if check_overflow > u32::MAX as u64 {
        println!("❌ U32 OVERFLOW in index calculation! {} > {}", check_overflow, u32::MAX);
    } else {
        println!("✅ No u32 overflow in index calculations");
    }
}