use core::f64;
use std::collections::HashSet;
use std::env;
use std::fs::File;
use std::io::Write;
use std::time::{Instant, SystemTime};
use cubecl::wgpu::{WebGpu, WgpuDevice};
use cubecl::{prelude::*};
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
use crate::{geometry, gpu_gcmc, gpu_gcmc_moves};
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

fn run_mc(water_molecules: Vec<WaterMolecule>, receptor_points: &Vec<Atom> ) -> (Vec<Atom>, Vec<WaterMolecule>) {
    let mut system_waters: Vec<WaterSystem> = Vec::with_capacity(water_molecules.len());
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
        299.0,
        300.0,
        0.995,
        12.0,
        200000
    );
    let acceptance_rate = sa.run();
    let mut waters = Vec::with_capacity(sa.waters.len() * 3);
    for w in sa.waters.iter() {
        waters.extend(w.as_vec());
    }
    (waters, sa.waters)
}

fn run_single_waterkit_gcmc_sa(receptor_points: &[Atom], 
    water_configurations: &Vec<[f64; 6]>, 
    mut grid: Grid3D,
    epoch: usize,
    gcmc_steps: usize,
    sa_steps: usize,) -> (Vec<Atom>, Vec<Atom>, Vec<WaterMolecule>) {
// device: WgpuDevice) -> (Vec<Atom>, Vec<Atom>, Vec<WaterMolecule>) {
    let water_params = consts::WATER_PARAMS.get(consts::WATER_FF).unwrap();
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
        let water_model = water_params.WATER_MODEL;
        let water_configuration = WaterMolecule::new( 
            water_model[0], 
            water_model[1], 
            water_model[2],
        "A".to_string(),
    0);
        let mut gcmc = GCMC::new(water_configuration, 
            distance_cutoff, 
            min[0], max[0], 
            min[1], max[1], 
            min[2], max[2], 
            consts::CHEMICAL_POTENTIAL as f64, 
            consts::BETA as f64, 
            consts::STANDARD_VOLUME as f64, 
            consts::GCMC_STEPS);
        let simulation = gcmc.gcmc_simulation(&receptor_map,  total_volume, last_residue_number, gcmc_steps);
        if simulation.is_ok() {
            let water_molecules = simulation.unwrap();
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
                sa_steps
            );
            let acceptance_rate = sa.run();
            let mut waters = Vec::with_capacity(sa.waters.len() * 3);
            for w in sa.waters.iter() {
                waters.extend(w.as_vec());
            }
            return (Vec::new(), waters, sa.waters);
    // SA
        }
    }
    return (Vec::new(), Vec::new(), Vec::new());
}

fn run_single_waterkit_gcmcmc(receptor_points: &[Atom], 
    water_configurations: &Vec<[f64; 6]>, 
    mut grid: Grid3D,
    epoch: usize,
    gcmc_steps: usize,
    sa_steps: usize,) -> (Vec<Atom>, Vec<Atom>, Vec<WaterMolecule>) {
    // device: WgpuDevice) -> (Vec<Atom>, Vec<Atom>, Vec<WaterMolecule>) {
    let water_params = consts::WATER_PARAMS.get(consts::WATER_FF).unwrap();
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
        let water_model = water_params.WATER_MODEL;
        let water_configuration = WaterMolecule::new( 
            water_model[0], 
            water_model[1], 
            water_model[2],
        "A".to_string(),
    0);
        let mut gcmc = GCMC::new(water_configuration, 
            distance_cutoff, 
            min[0], max[0], 
            min[1], max[1], 
            min[2], max[2], 
            consts::CHEMICAL_POTENTIAL as f64, 
            consts::BETA as f64, 
            consts::STANDARD_VOLUME as f64, 
            consts::GCMC_STEPS);
        let simulation = gcmc.gcmc_simulation(&receptor_map,  total_volume, last_residue_number, gcmc_steps);
        if simulation.is_ok() {
            let water_molecules = simulation.unwrap();
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
                299.0,
                300.0,
                0.995,
                12.0,
                sa_steps
            );
            let acceptance_rate = sa.run();
            let mut waters = Vec::with_capacity(sa.waters.len() * 3);
            for w in sa.waters.iter() {
                waters.extend(w.as_vec());
            }
            return (Vec::new(), waters, sa.waters);
    // SA
        }
    }
    return (Vec::new(), Vec::new(), Vec::new());
}

fn run_single_waterkit_gcmc(receptor_points: &[Atom], 
    water_configurations: &Vec<[f64; 6]>, 
    mut grid: Grid3D,
    epoch: usize,
    gcmc_steps: usize,
    sa_steps: usize,) -> (Vec<Atom>, Vec<Atom>, Vec<WaterMolecule>) {
    // device: WgpuDevice) -> (Vec<Atom>, Vec<Atom>, Vec<WaterMolecule>) {
    let water_params = consts::WATER_PARAMS.get(consts::WATER_FF).unwrap();
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
        let water_model = water_params.WATER_MODEL;
        let water_configuration = WaterMolecule::new( 
            water_model[0], 
            water_model[1], 
            water_model[2],
        "A".to_string(),
    0);
        let mut gcmc = GCMC::new(water_configuration, 
            distance_cutoff, 
            min[0], max[0], 
            min[1], max[1], 
            min[2], max[2], 
            consts::CHEMICAL_POTENTIAL as f64, 
            consts::BETA as f64, 
            consts::STANDARD_VOLUME as f64, 
            consts::GCMC_STEPS);
        let simulation = gcmc.gcmc_simulation(&receptor_map,  total_volume, last_residue_number, gcmc_steps);
        if simulation.is_ok() {
            let water_molecules = simulation.unwrap();
            let mut waters = Vec::with_capacity(water_molecules.len() * 3);
            for w in water_molecules.iter() {
                waters.extend(w.as_vec());
            }
            return (Vec::new(), waters, water_molecules);
    // SA
        }
    }
    return (Vec::new(), Vec::new(), Vec::new());
}

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
    gcmc_steps: usize,
    sa_steps: usize,
    save_path: String) {
    let waters: Vec<(Vec<Atom>, Vec<Atom>, Vec<WaterMolecule>)> = (0..epochs).into_par_iter()
        .map(|epoch| run_single_waterkit_gcmc_sa(
                &receptor_points,
                &water_configurations,
                grid.clone(),
                epoch,
                gcmc_steps,
                sa_steps,
                // device.clone()
            )).collect();

    println!("Done sampling...saving results!");
    
    waters.par_iter().enumerate()
        .for_each(|(idx, (unoptimized_system, optimized_system, water_moleucles))| {
            // to_pdb(&unoptimized_system, &format!("{save_path}/water_{idx}_unoptimized.pdb"), None);
            to_pdb(&optimized_system, &format!("{save_path}/water_{idx}_optimized.pdb"), None)}
        );
}

#[pyfunction]
pub fn run_waterkit_gcmc(receptor_points: Vec<Atom>, 
    water_configurations: Vec<[f64; 6]>,
    grid: Grid3D,
    num_frames: usize,
    gcmc_steps: usize, 
    save_path: String) {

    
    // Initialize the device (wgpu)
    // let device = WgpuDevice::DefaultDevice;

    let waters: Vec<(Vec<Atom>, Vec<Atom>, Vec<WaterMolecule>)> = (0..num_frames).into_par_iter()
        .map(|epoch| run_single_waterkit_gcmc(
            &receptor_points,
            &water_configurations,
            grid.clone(),
            epoch,
            gcmc_steps,
            0,
            // device.clone()
        )).collect();

    println!("Done sampling...saving results!");

    waters.par_iter().enumerate()
        .for_each(|(idx, (unoptimized_system, optimized_system, water_moleucles))| {
            // to_pdb(&unoptimized_system, &format!("{save_path}/water_{idx}_unoptimized.pdb"), None);
            to_pdb(&optimized_system, &format!("{save_path}/water_{idx}_optimized.pdb"), None)}
        );
}

#[pyfunction]
pub fn run_waterkit_gcmcmc(receptor_points: Vec<Atom>, 
    water_configurations: Vec<[f64; 6]>,
    grid: Grid3D,
    num_frames: usize,
    gcmc_steps: usize,
    mc_steps: usize, 
    save_path: String) {
    
    // Initialize the device (wgpu)
    // let device = WgpuDevice::DefaultDevice;
    let waters: Vec<(Vec<Atom>, Vec<Atom>, Vec<WaterMolecule>)> = (0..num_frames).into_par_iter()
        .map(|epoch| run_single_waterkit_gcmcmc(
            &receptor_points,
            &water_configurations,
            grid.clone(),
            epoch,
            gcmc_steps,
            mc_steps,
            // device.clone()
        )).collect();

    println!("Done sampling...saving results!");

    waters.par_iter().enumerate()
        .for_each(|(idx, (unoptimized_system, optimized_system, water_moleucles))| {
            // to_pdb(&unoptimized_system, &format!("{save_path}/water_{idx}_unoptimized.pdb"), None);
            to_pdb(&optimized_system, &format!("{save_path}/water_{idx}_optimized.pdb"), None)}
        );
}

#[pyfunction]
pub fn test_gpu(receptor_points: Vec<Atom>, 
    water_configurations: Vec<[f64; 6]>,
    grid: Grid3D,
    num_frames: usize,
    gcmc_steps: usize, 
    save_path: String) {
    env::set_var("RUST_BACKTRACE", "1");
    let water_params = consts::WATER_PARAMS.get(consts::WATER_FF).unwrap();
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
    println!("Total volume: {}", total_volume);
    let target_n_waters = (total_volume * bulk_water_density * 0.9) as usize;
    let min_max = find_min_max(&gird_points_for_placement);
    if min_max.is_some() {
        let (min, max) = min_max.unwrap();
        let water_model = water_params.WATER_MODEL;
        let water_configuration = WaterMolecule::new( 
            water_model[0], 
            water_model[1], 
            water_model[2],
        "A".to_string(),
    0);

    // for epochs in epochs_test {
    let start_gcmc = Instant::now();
        println!("Simulation with {} epochs", gcmc_steps);
        #[cfg(feature = "wgpu")]
        let n_waters = gpu_gcmc::simulate::<cubecl::wgpu::WgpuRuntime>(
            num_frames,
            &Default::default(), 
            receptor_points.clone(), 
            water_configuration.clone(), 
            12.0, 
            vec![min[0] as f32, max[0] as f32, 
                min[1] as f32, max[1] as f32,
                min[2] as f32, max[2] as f32], 
            total_volume as f32,
            gcmc_steps as usize);
        #[cfg(feature = "cuda")]
        let n_waters = gpu_gcmc::simulate::<cubecl::cuda::CudaRuntime>(
            num_frames,
            &Default::default(), 
            receptor_points.clone(), 
            water_configuration.clone(), 
            12.0, 
            vec![min[0] as f32, max[0] as f32, 
                min[1] as f32, max[1] as f32,
                min[2] as f32, max[2] as f32], 
            total_volume as f32,
            gcmc_steps as usize); 
            // gcmc_steps as usize);
        // for idx in 0..n_waters.len() {
            // println!("{idx} - {}", n_waters[idx]);
            // println!("C {} {} {}", n_waters[idx], n_waters[idx+1], n_waters[idx+2]);
        // }
        println!("Done sampling GCMC: {}s", start_gcmc.elapsed().as_secs());
        
        let (frames, water_molecules) = reconstruct_waters(n_waters, num_frames,  consts::MAX_N_WATERS as usize, 4, 3);
        
        // let start_mc = Instant::now();
        // let waters: Vec<(Vec<Atom>, Vec<WaterMolecule>)> = (0..num_frames).into_par_iter()
        // .map(|epoch| run_mc(
        //     water_molecules[epoch],
        //     &receptor_points,
        // )).collect();
        // println!("Done sampling MC: {}s", start_mc.elapsed().as_secs());

        // waters.par_iter().enumerate()
        // .for_each(|(idx, (optimized_system, water_moleucles))| {
        //     // to_pdb(&unoptimized_system, &format!("{save_path}/water_{idx}_unoptimized.pdb"), None);
        //     to_pdb(&optimized_system, &format!("{save_path}/water_{idx}_optimized.pdb"), None)}
        // );

        frames.par_iter().enumerate()
        .for_each(|(idx, frame)| {
            to_pdb(&frame, &format!("{save_path}/water_{idx}_optimized.pdb"), None);
        });
    }
}

fn reconstruct_waters(waters: Vec<f32>, num_frames: usize, max_n_waters: usize, atom_features: usize, atoms_per_water: usize) -> (Vec<Vec<Atom>>, Vec<Vec<WaterMolecule>>) {
    let mut frames = Vec::new();
    let mut waters_frames: Vec<Vec<WaterMolecule>> = Vec::new();
    for sim in 0..num_frames {
        // Calculate the starting index for this simulation's water data
        let base_wat_idx = sim * max_n_waters * atom_features * atoms_per_water;
        let mut water_molecules = Vec::<Atom>::new();
        let mut water_mol = Vec::new();
        // Iterate through each potential water molecule slot
        for water_idx in 0..max_n_waters {
            // Calculate the starting index for this water molecule
            let wat_index = base_wat_idx + (water_idx * atom_features * atoms_per_water);
            
            // Check if this water slot contains valid data
            // (assuming invalid waters have coordinates of 0.0, 0.0, 0.0)
            let o_x = waters[wat_index] as f64;
            let o_y = waters[wat_index + 1] as f64;
            let o_z = waters[wat_index + 2] as f64;
            
            // Skip if this water slot is empty (all coordinates are 0)
            if o_x == 0.0 && o_y == 0.0 && o_z == 0.0 {
                continue;
            }
            
            // Extract hydrogen coordinates
            let h1_x = waters[wat_index + 4] as f64;       // Start of H1
            let h1_y = waters[wat_index + 5] as f64;
            let h1_z = waters[wat_index + 6] as f64;
            
            let h2_x = waters[wat_index + 8] as f64;   // Start of H2
            let h2_y = waters[wat_index + 9] as f64;
            let h2_z = waters[wat_index + 10] as f64;
            
            // Extract residue number (assuming it's stored in the last field of oxygen)
            let resnumber = waters[wat_index + 3] as usize;  // 7th field (index 6) of oxygen
            
            // Create water molecule
            let wat_mol = WaterMolecule::new(
                [o_x, o_y, o_z],
                [h1_x, h1_y, h1_z],
                [h2_x, h2_y, h2_z], 
                "A".to_string(), 
                resnumber
            );
            
            // Add all atoms from this water molecule
            for atom in wat_mol.as_vec() {
                water_molecules.push(atom);
            }

            water_mol.push(wat_mol);
        }
    println!("Simulation {}: Found {} water molecules\n", sim, water_molecules.len() / 3);
    frames.push(water_molecules);
    waters_frames.push(water_mol);
    
    }
    (frames, waters_frames)
}