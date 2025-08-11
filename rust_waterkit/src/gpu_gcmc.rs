use core::f32;

use crate::atom::Atom;
use crate::gpu_gcmc_moves::WATER_SIZE;
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

const MAX_N_WATERS: u32 = 200;
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

#[cube(launch_unchecked)]
fn run_gcmc(
    boundaries: &Array<f32>,
    receptor_atoms: &Array<f32>,
    water_atoms: &mut Array<f32>,
    random_numbers: &Array<f32>,
    seeds: &Array<u32>,
    num_waters: &mut Array<u32>,
    last_resnum: u32,
    B: f32,
    volume: f32,
    epochs: u32,
) {
    let sim_id = CUBE_POS_X;
    let n_receptor_atoms = receptor_atoms.len() / ATOM_FEATURES;
    let n_water_atoms = water_atoms.len() /4;
    
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
            num_waters[sim_id] += active_waters;

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
    // The only thing that I need for the waters is the coordinates (3*3)
    let mut waters_buffer = Vec::with_capacity(n_simulations * max_n_waters * gpu_gcmc_moves::WATER_SIZE as usize);

    for idx in 0..(n_simulations * max_n_waters) {
        let water = water_configuration.as_vec();
        let o_c = water[0].coords();
        let h1_c = water[1].coords();
        let h2_c = water[2].coords();
        waters_buffer.push(o_c[0] as f32);
        waters_buffer.push(o_c[1] as f32);
        waters_buffer.push(o_c[2] as f32);
        waters_buffer.push(0.);
        // waters_buffer.push(0.);
        // waters_buffer.push(0.);
        // waters_buffer.push(0.);
        waters_buffer.push(h1_c[0] as f32);
        waters_buffer.push(h1_c[1] as f32);
        waters_buffer.push(h1_c[2] as f32);
        waters_buffer.push(0.);
        // waters_buffer.push(0.);
        // waters_buffer.push(0.);
        // waters_buffer.push(0.);
        waters_buffer.push(h2_c[0] as f32);
        waters_buffer.push(h2_c[1] as f32);
        waters_buffer.push(h2_c[2] as f32);
        waters_buffer.push(0.);
        // waters_buffer.push(0.);
        // waters_buffer.push(0.);
        // waters_buffer.push(0.);
    }
    println!("N WATERS: {}", waters_buffer.len());
    let boundaries_handle = client.create(f32::as_bytes(&boundaries));
    let receptor_atoms_handle = client.create(f32::as_bytes(&receptor_atoms_buffer));
    let water_atoms_handle = client.create(f32::as_bytes(&waters_buffer));
    let random_numbers_handle = client.create(f32::as_bytes(&random_buffer));
    let wat_num_handle = client.empty(n_simulations * core::mem::size_of::<u32>());
    
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
            ArrayArg::from_raw_parts::<u32>(&wat_num_handle, n_simulations, 1),
            ScalarArg {elem: receptor_atoms_buffer[receptor_atoms_buffer.len() -1] as u32}, 
            ScalarArg {elem: B},
            ScalarArg {elem: volume}, 
            ScalarArg {elem: num_steps as u32},
        );
    }

    let bytes = client.read_one(water_atoms_handle.clone().binding());
    let output: Vec<f32> = f32::from_bytes(&bytes).to_vec();
    println!("{}", output.len() / 4 / 3);

    let wat_bytes = client.read_one(wat_num_handle.clone().binding());
    let w_out = u32::from_bytes(&wat_bytes).to_vec();
    println!("{:?}", w_out);
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