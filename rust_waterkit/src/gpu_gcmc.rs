use core::f32;

use crate::atom::Atom;
use crate::gpu_gcmc_moves::WATER_SIZE;
use crate::{consts, gpu_gcmc_moves, gpu_geometry, gpu_random};
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

const MAX_N_WATERS: u32 = 500;
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
    let n_receptor_atoms = receptor_atoms.len() / ATOM_FEATURES;

    // Load persistent RNG state
    let mut rng_state = seeds[sim_id];
    // Load current water count for this simulation
    let mut active_waters = 0;

    // Preallocate random number buffer
    let mut random_numbers = Array::<f32>::new(8);

    // Base offset for water atoms of this simulation
    // let waters_base_idx = sim_id * MAX_N_WATERS * gpu_gcmc_moves::WATER_SIZE;

    for step in 0..steps {
        // Fill random numbers once per move
        random_numbers[0] = gpu_random::random_range(&mut rng_state, boundaries[0], boundaries[1]);
        random_numbers[1] = gpu_random::random_range(&mut rng_state, boundaries[2], boundaries[3]);
        random_numbers[2] = gpu_random::random_range(&mut rng_state, boundaries[4], boundaries[5]);
        random_numbers[3] = gpu_random::random_range(&mut rng_state, -180.0, 180.0);
        random_numbers[4] = gpu_random::random_range(&mut rng_state, -0.5, 0.5);
        random_numbers[5] = gpu_random::random_range(&mut rng_state, -0.5, 0.5);
        random_numbers[6] = gpu_random::random_range(&mut rng_state, -0.5, 0.5);
        random_numbers[7] = gpu_random::random_float(&mut rng_state);

        // Bounds check: don’t add water if array full
        if active_waters < MAX_N_WATERS {
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
        }
    }

    // Save updated state for next batch
    seeds[sim_id] = rng_state;
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
// ) -> Vec<f32> {
//     const EPOCHS_PER_BATCH: usize = 5000; // Adjust based on your GPU timeout limits
    
//     let client: ComputeClient<<R as Runtime>::Server, <R as Runtime>::Channel> = R::client(device);

//     debug_buffer_calculations(n_simulations);
    
//     // Calculate batching parameters
//     let num_batches = (num_steps + EPOCHS_PER_BATCH - 1) / EPOCHS_PER_BATCH; // Ceiling division
//     println!("Running {} total epochs in {} batches of up to {} epochs each", 
//              num_steps, num_batches, EPOCHS_PER_BATCH);

//     println!("# Atoms in the receptor: {}", receptor_atoms.len());

//     // Move the receptor to the global memory on the GPU (this stays the same across batches)
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

//     // Initialize waters buffer (this will accumulate results across batches)
//     let mut accumulated_waters_buffer = Vec::with_capacity(n_simulations * max_n_waters * gpu_gcmc_moves::WATER_SIZE as usize);
//     for idx in 0..(n_simulations * max_n_waters) {
//         for _ in 0..gpu_gcmc_moves::WATER_SIZE {
//             accumulated_waters_buffer.push(0.0);
//         }
//     }

//     // Create persistent GPU buffers that we'll reuse across batches
//     let boundaries_handle = client.create(f32::as_bytes(&boundaries));
//     let receptor_atoms_handle = client.create(f32::as_bytes(&receptor_atoms_buffer));
    
//     let volume_var = volume / consts::STANDARD_VOLUME;
//     let B = consts::CHEMICAL_POTENTIAL * consts::BETA + volume_var.ln();

//     // Process each batch
//     for batch_idx in 0..num_batches {
//         let batch_start = batch_idx * EPOCHS_PER_BATCH;
//         let batch_end = std::cmp::min(batch_start + EPOCHS_PER_BATCH, num_steps);
//         let current_batch_size = batch_end - batch_start;
        
//         println!("Processing batch {}/{}: epochs {}-{} ({} epochs)", 
//                  batch_idx + 1, num_batches, batch_start, batch_end - 1, current_batch_size);

//         // Create seeds for this batch only
//         let seed_tensor = TensorHandle::<R, u32>::empty(&client, [n_simulations * current_batch_size].to_vec());
//         random_uniform::<R, u32>(&client, 42 + batch_idx as u32, u32::MAX - 1, seed_tensor.as_ref());

//         // Create waters buffer for this batch (start with accumulated results)
//         let water_atoms_handle = client.create(f32::as_bytes(&accumulated_waters_buffer));
//         let wat_num_handle = client.empty(n_simulations * core::mem::size_of::<u32>());

//         unsafe {
//             run_gcmc::launch_unchecked::<R>(
//                 &client,
//                 CubeCount::Static(n_simulations as u32, 1, 1),
//                 CubeDim::new(1, 1, 1),
//                 ArrayArg::from_raw_parts::<f32>(&boundaries_handle, 6, 1),
//                 ArrayArg::from_raw_parts::<f32>(&receptor_atoms_handle, receptor_atoms_buffer.len(), 1),
//                 ArrayArg::from_raw_parts::<f32>(&water_atoms_handle, accumulated_waters_buffer.len(), 1),
//                 ArrayArg::from_raw_parts::<u32>(&seed_tensor.handle, n_simulations * current_batch_size, 1),
//                 ArrayArg::from_raw_parts::<u32>(&wat_num_handle, n_simulations, 1),
//                 ScalarArg { elem: receptor_atoms_buffer[receptor_atoms_buffer.len() - 1] as u32 },
//                 ScalarArg { elem: B },
//                 ScalarArg { elem: volume },
//                 ScalarArg { elem: current_batch_size as u32 }, // Pass current batch size instead of total
//             );
//         }

//         // Synchronize to prevent timeout and ensure batch completion
//         client.sync();

//         // Read back results from this batch and update accumulated buffer
//         let bytes = client.read_one(water_atoms_handle.clone().binding());
//         let batch_output: Vec<f32> = f32::from_bytes(&bytes).to_vec();
        
//         // Update accumulated results
//         accumulated_waters_buffer = batch_output;

//         println!("Batch {} completed", 
//                  batch_idx + 1);
//     }

//     println!("All {} batches completed successfully!", num_batches);
//     println!("Final total waters: {}", accumulated_waters_buffer.len() / 4 / 3);

//     accumulated_waters_buffer
// }

#[cube(launch_unchecked)]
fn run_stupid_random(
    seeds: &mut Array<u32>,           // persistent RNG state
    steps: u32,                       // steps per launch (batch)
) {
    let sim_id = ABSOLUTE_POS;
    // Load persistent RNG state
    let mut rng_state = seeds[sim_id];
    // Load current water count for this simulation

    // Preallocate random number buffer
    let mut random_numbers = Array::<f32>::new(8);

    for step in 0..steps {
        // Fill random numbers once per move
        random_numbers[0] = gpu_random::random_range(&mut rng_state, -80.0, 20.0);
        random_numbers[1] = gpu_random::random_range(&mut rng_state, -2.5, 34.2);
        random_numbers[2] = gpu_random::random_range(&mut rng_state, -19.32, 76.3);
        random_numbers[3] = gpu_random::random_range(&mut rng_state, -180.0, 180.0);
        random_numbers[4] = gpu_random::random_range(&mut rng_state, -0.5, 0.5);
        random_numbers[5] = gpu_random::random_range(&mut rng_state, -0.5, 0.5);
        random_numbers[6] = gpu_random::random_range(&mut rng_state, -0.5, 0.5);
        random_numbers[7] = gpu_random::random_float(&mut rng_state);
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
    // The only thing that I need for the waters is the coordinates (3*3)
    let mut waters_buffer = Vec::with_capacity(n_simulations * max_n_waters * gpu_gcmc_moves::WATER_SIZE as usize);

    for idx in 0..(n_simulations * max_n_waters) {
        let water = water_configuration.as_vec();
        let o_c = water[0].coords();
        let h1_c = water[1].coords();
        let h2_c = water[2].coords();
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
    // let random_numbers_handle = client.create(f32::as_bytes(&random_buffer));
    let wat_num_handle = client.empty(n_simulations * core::mem::size_of::<u32>());
    
    let volume_var = volume / consts::STANDARD_VOLUME;
    let B = consts::CHEMICAL_POTENTIAL * consts::BETA + volume_var.ln();

    // let threads_per_workgroup: u32 = 256;
    // let num_workgroups = (n_receptor_atoms as u32 + threads_per_workgroup - 1) / threads_per_workgroup;

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

    // unsafe {
    //     run_stupid_random::launch_unchecked::<R>(
    //         &client, 
    //         CubeCount::Static(1, 1, 1),
    //         CubeDim::new(1,1, 1),
    //         ArrayArg::from_raw_parts::<u32>(&seed_tensor.handle, n_simulations * num_steps as usize, 1), 
    //         ScalarArg {elem: num_steps as u32},
    //     );
    // }

    let bytes = client.read_one(water_atoms_handle.clone().binding());
    let output: Vec<f32> = f32::from_bytes(&bytes).to_vec();
    // println!("{}", output.len() / 4 / 3);

    // let n_waters_bytes = client.read_one(wat_num_handle.clone().binding());
    // let output_n_waters: Vec<f32> = f32::from_bytes(&n_waters_bytes).to_vec();
    // println!("# Accepted: {:?}", output_n_waters);

    output
}


pub fn debug_buffer_calculations(n_simulations: usize) {
    println!("=== BUFFER SIZE DEBUGGING ===");
    println!("MAX_N_WATERS: {}", MAX_N_WATERS);
    println!("WATER_SIZE: {}", gpu_gcmc_moves::WATER_SIZE);
    println!("n_simulations: {}", n_simulations);
    
    // Check for potential overflow in index calculations
    let waters_per_sim = MAX_N_WATERS as usize * gpu_gcmc_moves::WATER_SIZE as usize;
    let total_water_elements = n_simulations * waters_per_sim;
    
    println!("Waters per simulation: {}", waters_per_sim);
    println!("Total water buffer elements: {}", total_water_elements);
    println!("Total buffer size (MB): {:.2}", (total_water_elements * 4) as f64 / 1_048_576.0);
    
    // Check if calculations would overflow u32
    let max_sim_id = n_simulations - 1;
    let max_base_idx = max_sim_id as u32 * MAX_N_WATERS * gpu_gcmc_moves::WATER_SIZE;
    let max_water_idx = max_base_idx + (MAX_N_WATERS - 1) * gpu_gcmc_moves::WATER_SIZE;
    
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
    let check_overflow = (max_sim_id as u64) * (MAX_N_WATERS as u64) * (gpu_gcmc_moves::WATER_SIZE as u64);
    if check_overflow > u32::MAX as u64 {
        println!("❌ U32 OVERFLOW in index calculation! {} > {}", check_overflow, u32::MAX);
    } else {
        println!("✅ No u32 overflow in index calculations");
    }
}