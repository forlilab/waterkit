use core::f64;
use std::collections::HashSet;
use std::fs::File;
use std::io::Write;
use std::time::SystemTime;
use ncollide3d::shape::FeatureId;
use pyo3::prelude::*;
use rand::seq::SliceRandom;
use rand::thread_rng;
use rand::Rng;

use rayon::prelude::*;
use kiddo::{KdTree, SquaredEuclidean};

use crate::anchor_point::AnchorPoint;
use crate::atom::Atom;
use crate::energy::energy;
use crate::energy::energy_for_real_water;
use crate::gcmc::GCMC;
use crate::geometry;
use crate::geometry::dihedral;
use crate::grid::Grid3D;
use crate::grid::GridPoint;
use crate::grid::ProbeType;
use crate::consts;
use crate::monte_carlo;
use crate::optimizer;
use crate::optimizer::optimize;
use crate::optimizer::optimize_using_grids;
use crate::optimizer::SimulatedAnnealing;
use crate::replica_exchange;
use crate::sampling::build_kd_tree;
use crate::sampling::sample_using_grids;
use crate::sampling::sample_without_layers;
use crate::sampling::sample_without_layers_without_anchor_points;
use crate::setup::setup_grid;
use crate::utils::to_pdb;
use crate::utils::plot_optimization;
use crate::water::WaterMolecule;
use crate::energy;
use crate::utils;
use crate::waterkit_system;
use crate::waterkit_system::AtomSystem;
use crate::waterkit_system::AtomType;
use crate::waterkit_system::WaterSystem;

fn run_single_waterkit_gcmc_re(receptor_points: &[Atom],
    water_configurations: &Vec<[f64; 6]>,
    mut grid: Grid3D,
    epoch: usize) -> (Vec<Atom>, Vec<Atom>) {

    let mut receptor_map = receptor_points.to_vec();
    let mut last_residue_number = receptor_points.iter().map(|n| n.residue_number).max().unwrap_or(1);
    let distance_cutoff = 10.0;
    let mut receptor_points_tree = None;
    if receptor_map.len() > 0 {
        receptor_points_tree = Some(build_kd_tree(&receptor_map.clone()));
    }
    let mut gird_points_for_placement = Vec::new();
    if receptor_points_tree.is_some() {
        gird_points_for_placement.extend(grid.all_points().into_iter().filter(|p| 
            {
                let d = distance_to_protein_grid(p, &receptor_points_tree.clone().unwrap());
                d <= distance_cutoff && d >= 1.5
        }).map(|p| p.coords));
    } else {
        gird_points_for_placement.extend(grid.all_points().into_iter().map(|p| p.coords));
    }

    let mut rng = rand::thread_rng();
    let bulk_water_density = 0.0334; // molecules/A^3
    let voxel_volume = grid.spacing * grid.spacing * grid.spacing;
    let total_volume = (voxel_volume * gird_points_for_placement.len() as f64);
    let target_n_waters = (total_volume * bulk_water_density * 0.9) as usize;
    let min_max = find_min_max(&gird_points_for_placement);
    // if min_max.is_some() {
    let (min, max) = min_max.unwrap();
    
//     let n_replicas = consts::CHEMICAL_POTENTIALS.len();
//     let mut replica_states = Vec::with_capacity(n_replicas);
//     let water_configuration = WaterMolecule::new( 
//         [0.000, 0.000, 0.000], 
//         [0.000, 0.756, 0.586], 
//         [0.000, -0.756, 0.585],
//     "A".to_string(),
// 0);
//     for i in 0..consts::CHEMICAL_POTENTIALS.len() {
//         replica_states.push(replica_exchange::ReplicaState {
//             simulator: GCMC::new(water_configuration.clone(),
//                 distance_cutoff, 
//                 min[0], max[0], 
//                 min[1], max[1],
//                  min[2], max[2],
//                   consts::CHEMICAL_POTENTIALS[i], 
//                   consts::BETA, 
//                   consts::STANDARD_VOLUME, 
//                 consts::GCMC_STEPS),
//             num_waters: 0,
//             energy: 0.0,
//             positions: vec![],
//         })
//     }

//     for step in 0..consts::RE_STEPS {
//         println!("Step: {}", step);
//         for replica_state in 0..replica_states.len() {
//             if step > 0 {
//                 let last_residue_number = replica_states[replica_state].simulator.waters.last().unwrap().get_res_number(); 
//             }
//             // GCMC for each replica
            
//                 // gcmc.set_waters(new_water_molecules);
//             let simulation = replica_states[replica_state].simulator.gcmc_simulation(&receptor_map, total_volume, last_residue_number);
//             if simulation.is_ok() {
//                 let waters = simulation.unwrap();
//                 replica_states[replica_state].num_waters = waters.len();
//                 replica_states[replica_state].energy = replica_states[replica_state].simulator.system_energy;
//                 replica_states[replica_state].positions = replica_states[replica_state].simulator.atoms.clone();
//             }
//         }
//         for i in 0..n_replicas - 1 {
//             let system_i = &replica_states[i];
//             let system_j = &replica_states[i + 1];
//             let mu_i = consts::CHEMICAL_POTENTIALS[i];
//             let mu_j = consts::CHEMICAL_POTENTIALS[i + 1];
//             let n_i = system_i.num_waters;
//             let n_j = system_j.num_waters;
//             let u_i = system_i.energy;
//             let u_j = system_j.energy;

//             let prob = replica_exchange::exchange_probability(mu_i, mu_j, n_i, n_j, u_i, u_j, consts::BETA);
//             if rng.gen::<f64>() < prob {
//                 // Swap configurations
//                 replica_states.swap(i, i + 1);
//                 println!("Swapping configurations: {} - {}", i, i+1);
//             }
//         }
//     }

    // Parallel version
    let n_replicas = consts::CHEMICAL_POTENTIALS.len();
    let mut replica_states = Vec::with_capacity(n_replicas);
    let water_configuration = WaterMolecule::new(
        [0.000, 0.000, 0.000],
        [0.000, 0.756, 0.586],
        [0.000, -0.756, 0.585],
        "A".to_string(),
        0,
    );

    // Initialize replica states (unchanged)
    for i in 0..consts::CHEMICAL_POTENTIALS.len() {
        replica_states.push(replica_exchange::ReplicaState {
            simulator: GCMC::new(
                water_configuration.clone(),
                distance_cutoff,
                min[0], max[0],
                min[1], max[1],
                min[2], max[2],
                consts::CHEMICAL_POTENTIALS[i],
                consts::BETA,
                consts::STANDARD_VOLUME,
                consts::GCMC_STEPS,
            ),
            num_waters: 0,
            energy: 0.0,
            positions: vec![],
        });
    }

    // Assume rng is defined elsewhere, e.g., let mut rng = rand::thread_rng();
    for step in 0..consts::RE_STEPS {
        println!("Step: {}", step);

        // Parallelize the GCMC simulation loop
        replica_states.par_iter_mut().enumerate().for_each(|(replica_state, state)| {
            // Get last residue number if step > 0
            let last_residue_number = if step > 0 {
                state.simulator.waters.last().unwrap().get_res_number()
            } else {
                last_residue_number
            };

            // Perform GCMC simulation
            let simulation = state.simulator.gcmc_simulation(&receptor_map, total_volume, last_residue_number);
            if simulation.is_ok() {
                let waters = simulation.unwrap();
                state.num_waters = waters.len();
                state.energy = state.simulator.system_energy;
                state.positions = state.simulator.atoms.clone();
            }
        });

        // Replica exchange loop (remains sequential)
        for i in 0..n_replicas - 1 {
            let system_i = &replica_states[i];
            let system_j = &replica_states[i + 1];
            let mu_i = consts::CHEMICAL_POTENTIALS[i];
            let mu_j = consts::CHEMICAL_POTENTIALS[i + 1];
            let n_i = system_i.num_waters;
            let n_j = system_j.num_waters;
            let u_i = system_i.energy;
            let u_j = system_j.energy;

            let prob = replica_exchange::exchange_probability(mu_i, mu_j, n_i, n_j, u_i, u_j, consts::BETA);
            if rng.gen::<f64>() < prob {
                // Swap configurations
                replica_states.swap(i, i + 1);
                println!("Swapping configurations: {} - {}", i, i + 1);
            }
        }
    }
    for (idx, replica) in replica_states.into_iter().enumerate() {
        if replica.simulator.mu == consts::CHEMICAL_POTENTIAL {
            to_pdb(&replica.positions, &format!("test/water_replica_{idx}_optimized.pdb"), None);
        }
        // waters.par_iter().enumerate()
        // .for_each(|(idx, (unoptimized_system, optimized_system))| {
        //     to_pdb(&unoptimized_system, &format!("{save_path}/water_{idx}_unoptimized.pdb"), None);
        //     to_pdb(&optimized_system, &format!("{save_path}/water_{idx}_optimized.pdb"), None)}
        // );
    }
    (Vec::new(), Vec::new())
}

fn run_single_waterkit_gcmc_sa(receptor_points: &[Atom], 
    water_configurations: &Vec<[f64; 6]>, 
    mut grid: Grid3D,
    epoch: usize) -> (Vec<Atom>, Vec<Atom>) {

    let mut receptor_map = receptor_points.to_vec();
    let mut last_residue_number = receptor_points.iter().map(|n| n.residue_number).max().unwrap_or(1);
    let distance_cutoff = 10.0;
    let mut receptor_points_tree = None;
    if receptor_map.len() > 0 {
        receptor_points_tree = Some(build_kd_tree(&receptor_map.clone()));
    }
    let mut gird_points_for_placement = Vec::new();
    if receptor_points_tree.is_some() {
        gird_points_for_placement.extend(grid.all_points().into_iter().filter(|p| 
            {
                let d = distance_to_protein_grid(p, &receptor_points_tree.clone().unwrap());
                d <= distance_cutoff && d >= 1.5
        }).map(|p| p.coords));
    } else {
        gird_points_for_placement.extend(grid.all_points().into_iter().map(|p| p.coords));
    }
    
    // GCMC
    let bulk_water_density = 0.0334; // molecules/A^3
    let voxel_volume = grid.spacing * grid.spacing * grid.spacing;
    let total_volume = (voxel_volume * gird_points_for_placement.len() as f64);
    let target_n_waters = (total_volume * bulk_water_density * 0.9) as usize;
    let min_max = find_min_max(&gird_points_for_placement);
    if min_max.is_some() {
        let (min, max) = min_max.unwrap();
        let water_configuration = WaterMolecule::new( 
            [0.000, 0.000, 0.000], 
            [0.000, 0.756, 0.586], 
            [0.000, -0.756, 0.585],
        "A".to_string(),
    0);
        let mut gcmc = GCMC::new(water_configuration, distance_cutoff, min[0], max[0], min[1], max[1], min[2], max[2], consts::CHEMICAL_POTENTIAL, consts::BETA, consts::STANDARD_VOLUME, consts::GCMC_STEPS);
        // gcmc.set_waters(new_water_molecules);
        let simulation = gcmc.gcmc_simulation(&receptor_map,  total_volume, last_residue_number);
        if simulation.is_ok() {
            let water_molecules = simulation.unwrap();
            // let mut retvalue = Vec::with_capacity(water_molecules.len() * 3);
            // for w in water_molecules {
            //     retvalue.extend(w.as_vec());
            // }
            // return (Vec::new(), vec![retvalue]);
    // GCMC

    // SA
            let mut system_waters: Vec<WaterSystem> = Vec::with_capacity(water_molecules.len());
            let mut  unoptimized_water_atoms = Vec::with_capacity(water_molecules.len() * 3);
            let mut system_atoms = Vec::new();
            
            let mut cnt = 0;
            for water in water_molecules.iter() {
                let water_vec = water.as_vec();
                for atom in water_vec {
                    let atom_type = match atom.atom_type().as_str() {
                        "HW" => AtomType::WaterH,
                        "OW" => AtomType::WaterO,
                        _ => panic!("Unknown atom type: {}", atom.atom_type()),
                    };
                    unoptimized_water_atoms.push(atom.clone());
                    system_atoms.push(AtomSystem::new(atom, atom_type));
                    // system_atoms.push(atom.clone());
                }
                system_waters.push(WaterSystem::new(cnt, cnt+1, cnt+2));
                cnt += 3;
            }
            for atom in receptor_points {
                system_atoms.push(AtomSystem::new(atom.clone(), AtomType::Protein))
            }
            let system = waterkit_system::System::new(system_atoms, system_waters);
            let mut sa = optimizer::SimulatedAnnealing::new(
                system,
                water_molecules,
                1200.0,
                300.0,
                0.995,
                12.0,
            );
            // println!("Starting to optimize!");
            let acceptance_rate = sa.run();
            let mut waters = Vec::with_capacity(sa.waters.len() * 3);
            for w in sa.waters {
                waters.extend(w.as_vec());
            }
            return (Vec::new(), waters);
    // SA
        }
    }
    return (Vec::new(), Vec::new());
}

// fn run_single_waterkit_with_grids(receptor_points: &[Atom], 
//     water_configurations: &Vec<[f64; 6]>, 
//     anchor_points: &[AnchorPoint], 
//     mut grid: Grid3D,
//     epoch: usize, 
//     num_steps: i32, 
//     optimization_steps: i32) -> (Vec<Atom>, Vec<Vec<Atom>>) {
    
//     let mut receptor_map = receptor_points.to_vec();
//     let mut last_residue_number = receptor_points.iter().map(|n| n.residue_number).max().unwrap_or(1);

//     let mut mutable_anchor_points = anchor_points.to_vec();
//     let mut new_water_molecules = Vec::new();
//     let use_layers = true;
//     let use_mc_sa = false;
//     let use_gcmc = false;
//     if !use_gcmc {
//         if use_layers {
//             for _i in 0..3 {
//                 sample_using_grids(_i, &mut grid, &mut mutable_anchor_points, &water_configurations, &mut new_water_molecules, &mut last_residue_number);
//             }
//             println!("Waters placed: {}", new_water_molecules.len());
//         } else if use_mc_sa {
//             println!("Starting to sample run {}!", epoch);
//             let distance_cutoff = 20.0;
//             let receptor_points_tree: kiddo::float::kdtree::KdTree<f64, u64, 3, 32, u32> = build_kd_tree(&receptor_map);
//             // sample_without_layers_without_anchor_points(&mut grid, &None, water_configurations, &mut new_water_molecules, &mut last_residue_number, distance_cutoff);
//             // sample_without_layers(&mut grid, &receptor_points_tree,  &mut mutable_anchor_points, water_configurations, &mut new_water_molecules, &mut last_residue_number, distance_cutoff);
//             sample_without_layers_without_anchor_points(&mut grid, &Some(receptor_points_tree), water_configurations, &mut new_water_molecules, &mut last_residue_number, distance_cutoff);
//         } 
//         println!("Done sampling run {}!", epoch);
//         // let mut system_waters: Vec<Atom> = Vec::with_capacity(new_water_molecules.len() * 3);
//         let mut system_waters: Vec<WaterSystem> = Vec::with_capacity(new_water_molecules.len());
//         let mut  unoptimized_water_atoms = Vec::with_capacity(new_water_molecules.len() * 3);
//         let mut system_atoms = Vec::new();
        
//         let mut cnt = 0;
//         for water in new_water_molecules.iter() {
//             let water_vec = water.as_vec();
//             for atom in water_vec {
//                 let atom_type = match atom.atom_type().as_str() {
//                     "HW" => AtomType::WaterH,
//                     "OW" => AtomType::WaterO,
//                     _ => panic!("Unknown atom type: {}", atom.atom_type()),
//                 };
//                 unoptimized_water_atoms.push(atom.clone());
//                 system_atoms.push(AtomSystem::new(atom, atom_type));
//                 // system_atoms.push(atom.clone());
//             }
//             system_waters.push(WaterSystem::new(cnt, cnt+1, cnt+2));
//             cnt += 3;
//         }
//         for atom in receptor_points {
//             system_atoms.push(AtomSystem::new(atom.clone(), AtomType::Protein))
//         }
//         let system = waterkit_system::System::new(system_atoms, system_waters);
//         // let system = waterkit_system::System::new(&mut system_atoms);
//         // let mut waters = Vec::with_capacity(new_water_molecules.len() * 3);
//         let mut sa = optimizer::SimulatedAnnealing::new(
//             system,
//             new_water_molecules,
//             1200.0,
//             300.0,
//             0.995,
//             // 0.95,
//             12.0,
//         );
//         // println!("Starting to optimize!");
//         let acceptance_rate = sa.run();
//         println!("Acceptance rate run {}: {}%", epoch, acceptance_rate);
//         // optimize_water_nw_with_grids_sa(&mut new_water_molecules, &receptor_map, &mut grid, num_steps, optimization_steps);
        
//         // for w in new_water_molecules.into_iter() {
//         //     waters.extend(w.as_vec().into_iter());
//         // }
//         // for w in sa.waters {
//         //     waters.extend(w.as_vec());
//         // }
//         let mut frames = Vec::with_capacity(sa.frames.len());
//         for (idx, frame) in sa.frames.into_iter().enumerate() {
//             let mut single_frame= Vec::with_capacity(frame.len() * 3);
//             for w in frame {
//                 single_frame.extend(w.as_vec());
//             }
//             frames.push(single_frame);
//         }
//         return (unoptimized_water_atoms, frames);
//     } else {
//         // Here we are testing the GCMC
//         let distance_cutoff = 12.0;
//         let mut receptor_points_tree = None;
//         if receptor_map.len() > 0 {
//             receptor_points_tree = Some(build_kd_tree(&receptor_map.clone()));
//         }
//         let mut gird_points_for_placement = Vec::new();
//         if receptor_points_tree.is_some() {
//             gird_points_for_placement.extend(grid.all_points().into_iter().filter(|p| 
//                 {
//                     let d = distance_to_protein_grid(p, &receptor_points_tree.clone().unwrap());
//                     d <= distance_cutoff && d >= 1.5
//             }).map(|p| p.coords));
//         } else {
//             gird_points_for_placement.extend(grid.all_points().into_iter().map(|p| p.coords));
//         }

//         // let mut file = File::create("grid_points_for_placement.xyz").unwrap();
//         // writeln!(file, "{}", gird_points_for_placement.len()).unwrap();
//         // // writeln!(file, "Occupied Voxels for Receptor")?;
//         // for point in gird_points_for_placement.iter() {
//         //     let coords = point;
//         //     writeln!(file, "He {:.3} {:.3} {:.3}", coords[0], coords[1], coords[2]).unwrap();
//         // }
        
//         let bulk_water_density = 0.0334; // molecules/A^3
//         let voxel_volume = grid.spacing * grid.spacing * grid.spacing;
//         let total_volume = (voxel_volume * gird_points_for_placement.len() as f64);
//         // println!("V: {}", total_volume);
//         let target_n_waters = (total_volume * bulk_water_density * 0.9) as usize;
//         // println!("Target # of waters for cutoff {}A: {}", distance_cutoff, target_n_waters);
//         let min_max = find_min_max(&gird_points_for_placement);
//         if min_max.is_some() {
//             let (min, max) = min_max.unwrap();
//             let mut gcmc = GCMC::new(distance_cutoff, min[0], max[0], min[1], max[1], min[2], max[2]);
//             let water_configuration = WaterMolecule::new( 
//                 [0.000, 0.000, 0.000], 
//                 [0.000, 0.756, 0.586], 
//                 [0.000, -0.756, 0.585],
//             "A".to_string(),
//         0);
//             let simulation = gcmc.gcmc_simulation(&receptor_map, water_configuration, total_volume, last_residue_number);
//             if simulation.is_ok() {
//                 let water_frames = simulation.unwrap();
//                 let mut frames = Vec::with_capacity(water_frames.len());
//                 for frame in water_frames {
//                     let mut atoms = Vec::with_capacity(frame.len() * 3);
//                     for mol in frame {
//                         atoms.push(mol.oxygen);
//                         atoms.push(mol.hydrogen_1);
//                         atoms.push(mol.hydrogen_2);
//                     }
//                     frames.push(atoms);
//                 }
//                 // println!("{} frames generated!", frames.len());
//                 return (Vec::new(), frames);
//             }
//         }
//         return (Vec::new(), Vec::new());
//     }
// }

fn find_min_max(coords: &[[f64; 3]]) -> Option<([f64; 3], [f64; 3])> {
    if coords.is_empty() {
        return None;
    }

    let mut min = coords[0];
    let mut max = coords[0];

    for point in coords.iter().skip(1) {
        for i in 0..3 {
            if point[i] < min[i] {
                min[i] = point[i];
            }
            if point[i] > max[i] {
                max[i] = point[i];
            }
        }
    }

    Some((min, max))
}

fn distance_to_protein_grid(point: &GridPoint, tree: &KdTree<f64, 3>) -> f64 {
    let nearest = tree.nearest_one::<SquaredEuclidean>(&point.coords);
    nearest.distance.sqrt() // Convert squared distance to Euclidean distance
}

pub fn optimize_water_nw_with_grids(new_waters: &mut Vec<WaterMolecule>, grid: &mut Grid3D, num_steps: i32, optimization_steps: i32) {
    // let mut optimized_waters = Vec::new();
    let mut rng = thread_rng();

    for step in 0..num_steps {
        if let Some(water) = new_waters.choose_mut(&mut rng) {
            optimize_using_grids(water, grid, optimization_steps, consts::TEMPERATURE);
        }
    }
}

pub fn optimize_water_nw_with_grids_sa(new_waters: &mut Vec<WaterMolecule>, receptor_atoms: &Vec<Atom>, grid: &mut Grid3D, num_steps: i32, optimization_steps: i32) {
    // Worth trying reannealing after there's no acceptance for x epochs
    
    let plot_energies = false;

    let mut rng = thread_rng();
    let cooling_rate = 0.98;
    let mut starting_temp = 1200.;
    let final_temp = 0.001;
    
    // Reannealing
    // let reanneal_threshold = 5000;  // Steps before checking for stagnation
    // let reanneal_factor = 1.5;      // Reset temp to 50% of current if stagnation occurs
    // let mut rejection_counter = 0;
    // let max_rejections = 1000;
    
    // For the plot
    // let mut waters_energies = Vec::with_capacity((num_steps/100) as usize);
    // let mut receptor_energies = Vec::with_capacity((num_steps/100) as usize);
    // let mut total_energies = Vec::with_capacity((num_steps/100) as usize);
    // let (waters_energy, receptor_energy) = energy::get_system_energy(new_waters, receptor_atoms);
    // let mut initial_energy = waters_energy + receptor_energy;
    // let mut upper_b = initial_energy.max(waters_energy).max(receptor_energy);
    // let mut lower_b = initial_energy.min(waters_energy).min(receptor_energy);
    // waters_energies.push(waters_energy);
    // receptor_energies.push(receptor_energy); 
    // total_energies.push(initial_energy);

    for step in 0..num_steps {
        if let Some(water) = new_waters.choose_mut(&mut rng) {
            // if monte_carlo::boltzmann_acceptance_rejection(&water.get_energy(), &consts::BOLTZMANN_ENERGY_CUTOFF, &starting_temp, &consts::BOLTZMANN_K){
                let accepted = optimize_using_grids(water, grid, optimization_steps, starting_temp);
            // }

            // Track rejected moves
            // if accepted {
            //     rejection_counter = 0;  // Reset if a move is accepted
            // } else {
            //     rejection_counter += 1;
            // }

            // // Check for reannealing
            // if rejection_counter >= max_rejections {
            //     starting_temp = (starting_temp * reanneal_factor);
            //     rejection_counter = 0;  // Reset counter after reannealing
            //     println!("Reannealing at step {}: Reset temp to {:.2}", step, starting_temp);
            // }

            // Cooling schedule
            if step % 10 == 0 {

                // For the plot
                // let (new_water_energy, new_receptor_energy) = energy::get_system_energy(new_waters, receptor_atoms);
                // let new_energy = new_water_energy + new_receptor_energy;
                // upper_b = upper_b.max(new_energy).max(new_water_energy).max(new_receptor_energy);
                // lower_b = lower_b.min(new_energy).min(new_water_energy).min(new_receptor_energy);
                // waters_energies.push(new_water_energy);
                // receptor_energies.push(new_receptor_energy);
                // total_energies.push(new_energy);
                // // println!("Old energy: {initial_systems_energy}, new energy: {new_energy}");
                // initial_energy = new_energy;
                
                starting_temp *= cooling_rate;
                if starting_temp < final_temp {
                    println!("Temperature reached at step {step}");
                    break;
                }
            }
            // starting_temp *= cooling_rate;
            // println!("Temp: {starting_temp}");
        }
    }
    // let plot = plot_optimization(&total_energies, &waters_energies, &receptor_energies, lower_b, upper_b, num_steps as usize, optimization_steps as usize);
}

pub fn optimize_water_network(receptor_map: &mut Vec<Atom>, new_waters: &Vec<WaterMolecule>, grid: &mut Grid3D, filename: &str) -> Vec<Atom> {
    // let start = SystemTime::now();
    let mut rng = thread_rng();
    let mut waters_res_number: Vec<usize> = new_waters.iter()
        .map(|w| w.get_res_number())
        .collect::<HashSet<_>>() // Collect unique values first
        .into_iter()
        .collect();

    waters_res_number.shuffle(&mut rng);
    for res_number in waters_res_number.iter() {
        // println!("{}", res_number);
        let mut filtered = Vec::new();

        receptor_map
            .retain(|atom| {
                if atom.residue_number == *res_number {
                    filtered.push(atom.clone());
                    false
                } else {
                    true
                }
            });

        // Here I optimize
        optimize(&mut filtered, &receptor_map, grid);

        receptor_map.extend(filtered);
    }
    
    let mut waters = Vec::new();
    receptor_map.retain(|x| {
        if x.atom_type() == "OW" || x.atom_type() == "HW" {
            waters.push(x.clone());
            false // Remove this element from receptor_map
        } else {
            true // Keep this element in receptor_map
        }
    });

    // let end = SystemTime::now();
    // let duration = end.duration_since(start).unwrap();
    // println!("MC Optimization took {} ms", duration.as_millis());
    
    waters

    // to_pdb(&receptor_map.iter().filter(|x| x.atom_type() == "OW" || x.atom_type() == "HW").cloned().collect::<Vec<Atom>>(), filename);
}


#[pyfunction]
pub fn optimize_disordered_hydrogens(mut receptor_points: Vec<Atom>,
                                    mut anchor_points: Vec<AnchorPoint>,
                                    mut grid: Grid3D) -> (Vec<Atom>, Vec<AnchorPoint>) {
    let mut anchor_points_updated = anchor_points.clone();

    for (idx, anchor_point) in anchor_points.iter().enumerate() {
        if let Some(disordered_hydrogen) = anchor_point.disordered_hydrogens() {
            let anchor_vector = anchor_point.anchor_vectors();
            let atom_i_xyz = disordered_hydrogen.atom_i_xyz();
            let atom_j_xyz = disordered_hydrogen.atom_j_xyz();
            let atom_k_xyz = disordered_hydrogen.atom_k_xyz();
            let atom_l_xyz = disordered_hydrogen.atom_l_xyz();

            let actual_angle = geometry::dihedral(&atom_i_xyz, 
                &atom_j_xyz, 
                &atom_k_xyz, 
                &atom_l_xyz);

            let (sampled_anchor_points, sampled_anchor_vectors) = disordered_hydrogen.sample_rotatable_hydrogen(*anchor_vector);
            let mut energies: Vec<f64> = Vec::with_capacity(sampled_anchor_points.len());
            for index in 0..sampled_anchor_points.len() {
                // Select the optimal based on the energy and Boltzmann sampling
                // Then look at the atom_j_xyz, if that is part of anchor points as well,
                // Rotate those anchor points of the same angle
                if let Some(energy) = grid.trilinear_interpolation(sampled_anchor_vectors[index], ProbeType::ODa) {
                    energies.push(energy);
                } else {
                    energies.push(0.0);
                }
            }
            let choice = monte_carlo::boltzmann_choices(&energies, None)[0];
            let optimized_hydrogen = sampled_anchor_points[choice];
            let optimized_anchor_vector = sampled_anchor_vectors[choice];
            let expected_angle = geometry::dihedral(&optimized_hydrogen, 
                &atom_j_xyz, 
                &atom_k_xyz, 
                &atom_l_xyz);
            let rotation_angle = geometry::caclulate_angle_to_apply(&actual_angle, &expected_angle);
            
            // println!("Before: {:?}", anchor_points_updated[idx].anchor_point());
            anchor_points_updated[idx].set_anchor_point_xyz(optimized_hydrogen);
            // println!("After: {:?}", anchor_points_updated[idx].anchor_point());
            anchor_points_updated[idx].set_anchor_vector_xyz(optimized_anchor_vector);
            let anchor_points_j: Vec<&AnchorPoint> = anchor_points.iter().filter(|x| x.anchor_point() == &atom_j_xyz).collect();
            if anchor_points_j.len() > 0 {
                for anchor_point_j in anchor_points_j {
                    let ap_j_idx = anchor_point_j.get_idx();
                    let ap_to_update = &mut anchor_points_updated[ap_j_idx];
                    let new_ap = geometry::rotate_point(ap_to_update.anchor_point(), &atom_j_xyz, &atom_k_xyz, rotation_angle);
                    let new_vector = geometry::rotate_point(ap_to_update.anchor_vectors(), &atom_j_xyz, &atom_k_xyz, rotation_angle);
                    ap_to_update.set_anchor_vector_xyz(new_vector);
                }
            }
            // anchor_point.set_anchor_point_xyz(optimized_hydrogen);
            // anchor_point.set_anchor_vector_xyz(optimized_anchor_vector);

            // Update receptor's coordinates
            receptor_points.iter_mut().filter(|x| x.coords() == atom_i_xyz).next().unwrap().set_coords(optimized_hydrogen);
            // println!("Before: {:?}", atom_to_modify.coords());
            // println!("After: {:?}", atom_to_modify.coords());
        }
    }
    (receptor_points, anchor_points_updated)
}


#[pyfunction]
pub fn run_parallel_waterkit(receptor_points: Vec<Atom>, 
    water_configurations: Vec<[f64; 6]>, 
    anchor_points: Vec<AnchorPoint>, 
    grid: Grid3D,
    epochs: usize,
    num_steps: i32,
    optimization_steps: i32,
    save_path: String) {

    let waters: (Vec<Vec<Atom>>, Vec<Vec<Atom>>) = (0..epochs).into_par_iter()
        .map(|epoch| run_single_waterkit_gcmc_sa(
                &receptor_points,
                &water_configurations,
                grid.clone(),
                epoch,
            )).collect();

    println!("Done sampling...saving results!");
    
    waters.par_iter().enumerate()
        .for_each(|(idx, (unoptimized_system, optimized_system))| {
            to_pdb(&unoptimized_system, &format!("{save_path}/water_{idx}_unoptimized.pdb"), None);
            to_pdb(&optimized_system, &format!("{save_path}/water_{idx}_optimized.pdb"), None)}
        );
}

#[pyfunction]
pub fn run_waterkit_gcmcre(receptor_points: Vec<Atom>, 
    water_configurations: Vec<[f64; 6]>,
    grid: Grid3D,
    num_frames: usize, 
    save_path: String) {
        let waters: (Vec<Vec<Atom>>, Vec<Vec<Atom>>) = (0..num_frames).into_par_iter()
        .map(|epoch| run_single_waterkit_gcmc_re(
                &receptor_points,
                &water_configurations,
                grid.clone(),
                epoch
            )).collect();

    println!("Done sampling...saving results!");
    
    waters.par_iter().enumerate()
        .for_each(|(idx, (unoptimized_system, optimized_system))| {
            to_pdb(&unoptimized_system, &format!("{save_path}/water_{idx}_unoptimized.pdb"), None);
            to_pdb(&optimized_system, &format!("{save_path}/water_{idx}_optimized.pdb"), None)}
    );

}

#[pyfunction]
pub fn get_energies_for_system(receptor_points: Vec<Atom>, 
    waters: Vec<[Atom; 3]>, center: [f64; 3], x: f64, y: f64, z: f64) {
    
    let grid_receptor = setup_grid(&receptor_points, x, y, z, 0.375, center);

    for (index, water) in waters.iter().enumerate() {
        let mut points = receptor_points.clone();
        // println!("Points before: {}", points.len());
        let water_atoms = vec![water[0].clone(), water[1].clone(), water[2].clone()];
        let waters_excluding: Vec<&[Atom; 3]> = waters.iter().filter(|x| &water != x).collect();
        let mut waters_network = Vec::new();
        for excluded_w in waters_excluding {
            for excluded_atom in excluded_w {
                points.push(excluded_atom.clone());
                waters_network.push(excluded_atom.clone());
            }
        }
        // println!("Waters network: {}", waters_network.len());
        // println!("Points after: {}", points.len());
        let use_grids = true;

        if use_grids {
            // let grid_receptor_and_w = setup_grid(&points, 21.0, 24.0, 26.0, 0.375, [71.5, 73.1, 243.4]);
            let grid_receptor_and_w = setup_grid(&points, x, y, z, 0.375, center);

            let oxygen = water_atoms[0].clone();
            println!("{:?}", oxygen.coords());
            let h1 = water_atoms[1].clone();
            let h2 = water_atoms[2].clone();
            // println!("{}", receptor_points.len());
            // let mut energy = energy_for_real_water(&points, &vec![oxygen.clone()]);
            let mut energy = grid_receptor_and_w.trilinear_interpolation(oxygen.coords(), ProbeType::OW).unwrap();
            println!("Total LJ: {}", energy);
            let e_elec = grid_receptor_and_w.trilinear_interpolation(oxygen.coords(), ProbeType::HW).unwrap() * consts::OXYGEN_W_Q;
            println!("Total Coulomb: {}", e_elec);
            energy += e_elec;
            println!("{index} {index} {energy} O (rec+wat)");

            // let mut energy_rec = energy_for_real_water(&receptor_points, &vec![oxygen.clone()]);
            let mut energy_rec = grid_receptor.trilinear_interpolation(oxygen.coords(), ProbeType::OW).unwrap();
            energy_rec += grid_receptor.trilinear_interpolation(oxygen.coords(), ProbeType::HW).unwrap() * consts::OXYGEN_W_Q;
            println!("{index} {index} {energy_rec} O (just rec)");

            // let e = energy_for_real_water(&points, &vec![h1.clone()]);
            let e = grid_receptor_and_w.trilinear_interpolation(h1.coords(), ProbeType::HW).unwrap() * consts::HYDROGEN_W_Q;
            println!("Total Coulomb: {}", e);
            println!("{index} {index} {e} H (rec+wat)");
            energy += e;

            // let er = energy_for_real_water(&receptor_points, &vec![h1.clone()]);
            let er = grid_receptor.trilinear_interpolation(h1.coords(), ProbeType::HW).unwrap() * consts::HYDROGEN_W_Q;
            println!("{index} {index} {er} H (just rec)");
            energy_rec += er;

            // let e = energy_for_real_water(&points, &vec![h2.clone()]);
            let e = grid_receptor_and_w.trilinear_interpolation(h2.coords(), ProbeType::HW).unwrap() * consts::HYDROGEN_W_Q;
            println!("Total Coulomb: {}", e);
            println!("{index} {index} {e} H (rec+wat)");
            energy += e;

            // let er = energy_for_real_water(&receptor_points, &vec![h2.clone()]);
            let er = grid_receptor.trilinear_interpolation(h2.coords(), ProbeType::HW).unwrap() * consts::HYDROGEN_W_Q;
            println!("{index} {index} {er} H (just rec)");
            energy_rec += er;

            println!("{index} {index} {energy} HOH (rec+wat)");
            println!("{index} {index} {energy_rec} HOH (just rec)");
            println!();
        } else {
            // println!("waters network: {:?}", waters_network);
            // println!("Target water: {:?}", water_atoms);

            let oxygen = water_atoms[0].clone();
            println!("{:?}", oxygen.coords());
            let h1 = water_atoms[1].clone();
            let h2 = water_atoms[2].clone();
            println!("{}", receptor_points.len());
            let mut energy = energy_for_real_water(&points, &vec![oxygen.clone()]);
            println!("{index} {index} {energy} O (rec+wat)");

            let mut energy_w = energy_for_real_water(&waters_network, &vec![oxygen.clone()]);
            println!("{index} {index} {energy_w} O (just wat)");

            let mut energy_rec = energy_for_real_water(&receptor_points, &vec![oxygen.clone()]);
            println!("{index} {index} {energy_rec} O (just rec)");

            let e = energy_for_real_water(&points, &vec![h1.clone()]);
            println!("{index} {index} {e} H (rec+wat)");
            energy += e;

            let ew = energy_for_real_water(&waters_network, &vec![h1.clone()]);
            println!("{index} {index} {ew} H (just wat)");
            energy_w += ew;

            let er = energy_for_real_water(&receptor_points, &vec![h1.clone()]);
            println!("{index} {index} {er} H (just rec)");
            energy_rec += er;

            let e = energy_for_real_water(&points, &vec![h2.clone()]);
            println!("{index} {index} {e} H (rec+wat)");
            energy += e;

            let ew = energy_for_real_water(&waters_network, &vec![h2.clone()]);
            println!("{index} {index} {ew} H (just wat)");
            energy_w += ew;

            let er = energy_for_real_water(&receptor_points, &vec![h2.clone()]);
            println!("{index} {index} {er} H (just rec)");
            energy_rec += er;

            println!("{index} {index} {energy} HOH (rec+wat)");
            println!("{index} {index} {energy_w} HOH (just wat)");
            println!("{index} {index} {energy_rec} HOH (just rec)");
            // println!();
        }
        // println!("{:?}", water);
        // println!("{:?}\n", water_atoms);
        // let energy = energy_for_real_water(&points, &water_atoms);
        // println!("Energy for water: {}", energy);
    }

}
