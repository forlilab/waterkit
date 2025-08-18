use cubecl::prelude::*;
use crate::consts;
use crate::gpu_gcmc;
use crate::gpu_gcmc_moves;
use crate::gpu_energy;
use crate::gpu_geometry::{self, rodrigues_rotation};
use crate::gpu_random;

// MOVES
#[cube]
pub fn create_insertion_move(
    boundaries: &Array<f32>,
    random_numbers: &Array<f32>,
    possible_resnum: u32
) -> Array<f32> {

    let mut new_water: Array<f32> = gpu_gcmc_moves::randomize_water(random_numbers);

    // Generate new water configuration
    gpu_gcmc_moves::propose_insertion_compact(boundaries, random_numbers, &mut new_water, possible_resnum as f32);
    new_water
}

#[cube]
pub fn create_deletion_move(
    water_atoms: &mut Array<f32>,
    random_numbers: &Array<f32>,
    delete_water_idx: u32
) -> Array<f32> {
    // Load water to be deleted for energy calculation
    let mut water_to_remove = Array::<f32>::new(consts::WATER_SIZE);
    gpu_gcmc_moves::load_water_from_array(water_atoms, delete_water_idx, &mut water_to_remove);
    water_to_remove
}

#[cube]
pub fn create_displacement_move(
    boundaries: &Array<f32>,
    random_numbers: &Array<f32>,
    old_water: &Array<f32>
) -> Array<f32> {
    // Propose new position
    let new_water = gpu_gcmc_moves::propose_perturbation(boundaries, old_water, random_numbers);
    new_water
}


// ENERGY
#[cube]
pub fn energy_for_real_water_kernel_parallel(
    receptor_atoms: &Array<f32>, // [n_receptor * 7]
    target_water: &Array<f32>, // [3 * 4]
    n_receptor: u32,
    sim_id: u32,
) -> f32 {
    let mut shared_partials = SharedMemory::<f32>::new(gpu_gcmc::NUM_THREADS);
    let tid = UNIT_POS_X;
    let water_atom_stride = 4;
    let atoms_per_water = 3;
    let receptor_stride = consts::ATOM_FEATURES; // 7
    
    // Initialize shared memory
    shared_partials[tid] = 0.0;
    sync_cube();
    
    let mut partial_energy = 0.0f32;
    let n_receptor_atoms = receptor_atoms.len() as u32 / receptor_stride;
    
    // Each thread processes atoms: tid, tid + CUBE_DIM_X, tid + 2*CUBE_DIM_X, ...
    let mut r_idx = tid;
    while r_idx < n_receptor_atoms {
        let r_base = r_idx * receptor_stride;
        let r_x = receptor_atoms[r_base];
        let r_y = receptor_atoms[r_base + 1];
        let r_z = receptor_atoms[r_base + 2];
        let r_charge = receptor_atoms[r_base + 3];
        let r_epsilon = receptor_atoms[r_base + 4];
        let r_rmin_half = receptor_atoms[r_base + 5];
        let r_resnum = receptor_atoms[r_base + 6];
        
        // if tid < 4 && r_idx < n_receptor_atoms { // Only first few threads and atoms
        //     debug_print!("tid=%d processing r_idx=%d: coords=(%f,%f,%f) charge=%f eps=%f\n",
        //                 tid, r_idx, r_x, r_y, r_z, r_charge, r_epsilon);
        //     }   

        // Interact with all target water atoms
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
            
            let t_is_hw = t_epsilon == 0.0;
            
            let dx = t_x - r_x;
            let dy = t_y - r_y;
            let dz = t_z - r_z;
            let distance_sq = dx*dx + dy*dy + dz*dz;
            let distance = f32::sqrt(distance_sq);
            let r_val = f32::max(distance, 1e-8);
            
            // Calculate Lennard-Jones energy (only for non-hydrogen target atoms)
            let mut lj_energy = 0.0;
            if !t_is_hw {
                lj_energy = gpu_energy::lennard_jones_rmin_half(
                    t_epsilon, r_epsilon, r_val, t_rmin_half, r_rmin_half
                );
            }
            partial_energy += lj_energy;
            
            // Calculate Coulomb energy
            let electrostatics_energy = gpu_energy::coulomb_energy::<f32>(t_charge, r_charge, r_val);
            partial_energy += electrostatics_energy;
        }
        
        // Move to next atom for this thread
        r_idx += CUBE_DIM_X;
        // r_idx += 1;
    }
    // debug_print!("tid=%d: partial_energy=%f\n", tid, partial_energy);
    // Store this thread's partial result
    shared_partials[tid] = partial_energy;
    sync_cube();
    
    // if tid == 0 {
    // debug_print!("Before reduction:%d\n", CUBE_DIM_X);
    // let min_v = f32::min(8f32, f32::cast_from(CUBE_DIM_X));
    // for i in 0..u32::cast_from(min_v) { // First 8 threads
    //     let val = shared_partials[i];
    //     debug_print!("  out_partials[%d] = %f\n", i, val);
    //     }
    // }
    // Parallel tree reduction
    let mut step = CUBE_DIM_X / 2;
    while step > 0 {
        if tid < step {
            let old_val = shared_partials[tid];
            let add_val = shared_partials[tid + step];
            shared_partials[tid] = old_val + add_val;
            
            // // Debug first few reduction steps
            // if step >= CUBE_DIM_X / 4 && tid < 4 {
            //     // let index = tid;
            //     debug_print!("Reduction step=%d: tid=%d, %f + %f\n",
            //                 step, tid, old_val, add_val);
            // }
        }
        sync_cube();
        step /= 2;
    }

    let total = shared_partials[0];
    // if tid == 0 {
    //     debug_print!("Final total: %f\n", total);
    // }
    // All threads return the total
    
    sync_cube();
    
    total
}

#[cube]
pub fn energy_for_real_water_with_waters_kernel_parallel(
    water_atoms: &Array<f32>,
    target_water: &Array<f32>,
    sim_id: u32,
    active_waters: u32,
    out_partials: &mut Array<f32>,
) {
    // Initialize output immediately
    let out_base = sim_id * CUBE_DIM_X;
    out_partials[out_base + UNIT_POS_X] = 0.0;

    // Early exit if no waters to interact with
    // if active_waters == 0 {
    //     terminate!()
    // }

    let WATER_STRIDE: u32 = u32::cast_from(4);
    let ATOMS_PER_WATER: u32 = u32::cast_from(3);
    
    let mut partial: f32 = 0.0;
    let waters_base = sim_id * consts::MAX_N_WATERS * consts::WATER_SIZE;

    // Add bounds checking
    // debug_print!("Thread %d: active_waters=%d, CUBE_DIM_X=%d\n", UNIT_POS_X, active_waters, CUBE_DIM_X);

    // strided over existing (active) waters
    let mut w = UNIT_POS_X;
    while w < active_waters {
        // debug_print!("Thread %d: processing water %d\n", UNIT_POS_X, w);
        
        let wb = waters_base + w * consts::WATER_SIZE;
        
        // Bounds check
        // if wb + consts::WATER_SIZE > water_atoms.len() {
        //     // debug_print!("Thread %d: BOUNDS ERROR - wb=%d, array_len=%d\n", UNIT_POS_X, wb, water_atoms.len());
        //     break;
        // }
        
        // loop atoms (O,H1,H2) of existing water
        let mut a = 0u32;
        while a < ATOMS_PER_WATER {
            let abase = wb + a * WATER_STRIDE;
            let wx = water_atoms[abase + 0];
            let wy = water_atoms[abase + 1];
            let wz = water_atoms[abase + 2];

            // params for existing-water atom
            let mut wq: f32 = 0.0;
            let mut w_eps: f32 = 0.0;
            let mut w_rminh: f32 = 0.0;

            #[cfg(feature = "tip3p")]
            {
                if a == 0 {
                    wq = -0.8340;
                    w_eps = 0.15210325;
                    w_rminh = 1.7682;
                } else {
                    wq = 0.4170;
                }
            }

            #[cfg(feature = "tip3pfp")]
            {
                if a == 0 {
                    wq = -0.8484;
                    w_eps = 0.15586604;
                    w_rminh = 1.7835723;
                } else {
                    wq = 0.4242;
                }
            }

            // interact with target water atoms
            let mut t = 0u32;
            while t < ATOMS_PER_WATER {
                let tbase = t * WATER_STRIDE;
                let tx = target_water[tbase + 0];
                let ty = target_water[tbase + 1];
                let tz = target_water[tbase + 2];
                
                let dx = tx - wx;
                let dy = ty - wy;
                let dz = tz - wz;
                let distance_sq = dx * dx + dy * dy + dz * dz;
                
                // Skip if atoms are essentially overlapping
                if distance_sq > 1e-16 {
                    
                    let distance = f32::sqrt(distance_sq);
                    let r = f32::max(distance, 1e-8);

                    // target atom params
                    let mut tq: f32 = 0.0;
                    let mut t_eps: f32 = 0.0;
                    let mut t_rminh: f32 = 0.0;

                    #[cfg(feature = "tip3p")]
                    {
                        if t == 0 {
                            tq = -0.8340;
                            t_eps = 0.15210325;
                            t_rminh = 1.7682;
                        } else {
                            tq = 0.4170;
                        }
                    }

                    #[cfg(feature = "tip3pfp")]
                    {
                        if t == 0 {
                            tq = -0.8484;
                            t_eps = 0.15586604;
                            t_rminh = 1.7683;
                        } else {
                            tq = 0.4242;
                        }
                    }

                    // LJ only for O–O (no LJ on hydrogens)
                    if a == 0 && t == 0 {
                        partial += gpu_energy::lennard_jones_rmin_half(
                            t_eps, w_eps, r, t_rminh, w_rminh
                        );
                    }

                    // Coulomb for all atom pairs
                    partial += gpu_energy::coulomb_energy::<f32>(tq, wq, r);
                    
                    t += 1;
                }
            }
            a += 1;
        }
        
        // debug_print!("Thread %d: completed water %d, moving to %d\n", UNIT_POS_X, w, w + CUBE_DIM_X);
        w += CUBE_DIM_X;
    }

    // debug_print!("Thread %d: writing partial=%f to index %d\n", UNIT_POS_X, partial, out_base + UNIT_POS_X);
    // write partial to per-sim/thread slot in global buffer
    out_partials[out_base + UNIT_POS_X] = partial;
}

// #[cube]
// pub fn energy_for_real_water_with_waters_kernel_parallel(
//     water_atoms: &Array<f32>, // all waters for the sim: MAX_N_WATERS * WATER_SIZE
//     target_water: &Array<f32>, // (x,y,z,resnum) × 3
//     sim_id: u32,
//     active_waters: u32,
//     // OUT: per-thread partials buffer, length must be >= n_sims * CUBE_DIM_X
//     out_partials: &mut Array<f32>,
// ) {
//     let WATER_STRIDE: u32 = u32::cast_from(4);
//     let ATOMS_PER_WATER: u32 = u32::cast_from(3);
    
//     let mut partial: f32 = 0.0;
    
//     // Initialize output slot (clear any previous value)
//     let out_base = sim_id * CUBE_DIM_X;
//     out_partials[out_base + UNIT_POS_X] = 0.0;
    
//     let waters_base = sim_id * consts::MAX_N_WATERS * consts::WATER_SIZE;

//     // strided over existing (active) waters
//     let mut w = UNIT_POS_X;
//     while w < active_waters {
//         let wb = waters_base + w * consts::WATER_SIZE;
        
//         // loop atoms (O,H1,H2) of existing water
//         let mut a = 0u32;
//         while a < ATOMS_PER_WATER {
//             let abase = wb + a * WATER_STRIDE;
//             let wx = water_atoms[abase + 0];
//             let wy = water_atoms[abase + 1];
//             let wz = water_atoms[abase + 2];
//             // let w_res = water_atoms[abase + 3]; // not used for energy

//             // params for existing-water atom
//             let mut wq: f32 = 0.0;
//             let mut w_eps: f32 = 0.0;
//             let mut w_rminh: f32 = 0.0;

//             #[cfg(feature = "tip3p")]
//             {
//                 if a == 0 {
//                     wq = -0.8340;
//                     w_eps = 0.15210325;
//                     w_rminh = 1.7682;
//                 } else {
//                     wq = 0.4170;
//                 }
//             }

//             #[cfg(feature = "tip3pfp")]
//             {
//                 if a == 0 {
//                     wq = -0.8484;
//                     w_eps = 0.15586604;
//                     w_rminh = 1.7683;
//                 } else {
//                     wq = 0.4242;
//                 }
//             }

//             // interact with target water atoms
//             let mut t = 0u32;
//             while t < ATOMS_PER_WATER {
//                 let tbase = t * WATER_STRIDE;
//                 let tx = target_water[tbase + 0];
//                 let ty = target_water[tbase + 1];
//                 let tz = target_water[tbase + 2];
                
//                 let dx = tx - wx;
//                 let dy = ty - wy;
//                 let dz = tz - wz;
//                 let distance_sq = dx * dx + dy * dy + dz * dz;
                
//                 // Skip if atoms are essentially overlapping
//                 if distance_sq > 1e-16 {
                
//                     let distance = f32::sqrt(distance_sq);
//                     let r = f32::max(distance, 1e-8);

//                     // target atom params
//                     let mut tq: f32 = 0.0;
//                     let mut t_eps: f32 = 0.0;
//                     let mut t_rminh: f32 = 0.0;

//                     #[cfg(feature = "tip3p")]
//                     {
//                         if t == 0 {
//                             tq = -0.8340;
//                             t_eps = 0.15210325;
//                             t_rminh = 1.7682;
//                         } else {
//                             tq = 0.4170;
//                         }
//                     }

//                     #[cfg(feature = "tip3pfp")]
//                     {
//                         if t == 0 {
//                             tq = -0.8484;
//                             t_eps = 0.15586604;
//                             t_rminh = 1.7683;
//                         } else {
//                             tq = 0.4242;
//                         }
//                     }

//                     // LJ only for O–O (no LJ on hydrogens)
//                     if a == 0 && t == 0 {
//                         partial += gpu_energy::lennard_jones_rmin_half(
//                             t_eps, w_eps, r, t_rminh, w_rminh
//                         );
//                     }

//                     // Coulomb for all atom pairs
//                     partial += gpu_energy::coulomb_energy::<f32>(tq, wq, r);
                    
//                     t += 1;
//                 }
//             }
//             a += 1;
//         }
//         w += CUBE_DIM_X;
//     }

//     // write partial to per-sim/thread slot in global buffer
//     out_partials[out_base + UNIT_POS_X] = partial;
// }