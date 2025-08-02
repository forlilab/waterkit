use core::f64;
use std::collections::HashSet;
use std::fs::File;
use std::io::Write;
use std::time::SystemTime;
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

fn run_single_waterkit_gcmc_sa(receptor_points: &[Atom], 
    water_configurations: &Vec<[f64; 6]>, 
    mut grid: Grid3D,
    epoch: usize,
    gcmc_steps: usize,
    sa_steps: usize,
device: WgpuDevice) -> (Vec<Atom>, Vec<Atom>, Vec<WaterMolecule>) {
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
            consts::CHEMICAL_POTENTIAL, 
            consts::BETA, 
            consts::STANDARD_VOLUME, 
            consts::GCMC_STEPS);
        let simulation = gcmc.gcmc_simulation(&receptor_map,  total_volume, last_residue_number, gcmc_steps, device);
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
    sa_steps: usize,
    device: WgpuDevice) -> (Vec<Atom>, Vec<Atom>, Vec<WaterMolecule>) {
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
            consts::CHEMICAL_POTENTIAL, 
            consts::BETA, 
            consts::STANDARD_VOLUME, 
            consts::GCMC_STEPS);
        let simulation = gcmc.gcmc_simulation(&receptor_map,  total_volume, last_residue_number, gcmc_steps, device);
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
    sa_steps: usize,
    device: WgpuDevice) -> (Vec<Atom>, Vec<Atom>, Vec<WaterMolecule>) {
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
            consts::CHEMICAL_POTENTIAL, 
            consts::BETA, 
            consts::STANDARD_VOLUME, 
            consts::GCMC_STEPS);
        let simulation = gcmc.gcmc_simulation(&receptor_map,  total_volume, last_residue_number, gcmc_steps, device);
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

    // Initialize the device (wgpu)
    let device = WgpuDevice::DefaultDevice;
    let waters: Vec<(Vec<Atom>, Vec<Atom>, Vec<WaterMolecule>)> = (0..epochs).into_par_iter()
        .map(|epoch| run_single_waterkit_gcmc_sa(
                &receptor_points,
                &water_configurations,
                grid.clone(),
                epoch,
                gcmc_steps,
                sa_steps,
                device.clone()
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
    let device = WgpuDevice::DefaultDevice;

    let waters: Vec<(Vec<Atom>, Vec<Atom>, Vec<WaterMolecule>)> = (0..num_frames).into_par_iter()
        .map(|epoch| run_single_waterkit_gcmc(
            &receptor_points,
            &water_configurations,
            grid.clone(),
            epoch,
            gcmc_steps,
            0,
            device.clone()
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
    let device = WgpuDevice::DefaultDevice;
    let waters: Vec<(Vec<Atom>, Vec<Atom>, Vec<WaterMolecule>)> = (0..num_frames).into_par_iter()
        .map(|epoch| run_single_waterkit_gcmcmc(
            &receptor_points,
            &water_configurations,
            grid.clone(),
            epoch,
            gcmc_steps,
            mc_steps,
            device.clone()
        )).collect();

    println!("Done sampling...saving results!");

    waters.par_iter().enumerate()
        .for_each(|(idx, (unoptimized_system, optimized_system, water_moleucles))| {
            // to_pdb(&unoptimized_system, &format!("{save_path}/water_{idx}_unoptimized.pdb"), None);
            to_pdb(&optimized_system, &format!("{save_path}/water_{idx}_optimized.pdb"), None)}
        );
}

#[pyfunction]
pub fn get_energies_for_system(receptor_points: Vec<Atom>, 
    waters: Vec<[Atom; 3]>, center: [f64; 3], x: f64, y: f64, z: f64) {
    let water_params = consts::WATER_PARAMS.get(consts::WATER_FF).unwrap();
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
            let e_elec = grid_receptor_and_w.trilinear_interpolation(oxygen.coords(), ProbeType::HW).unwrap() * water_params.OXYGEN_W_Q;
            println!("Total Coulomb: {}", e_elec);
            energy += e_elec;
            println!("{index} {index} {energy} O (rec+wat)");

            // let mut energy_rec = energy_for_real_water(&receptor_points, &vec![oxygen.clone()]);
            let mut energy_rec = grid_receptor.trilinear_interpolation(oxygen.coords(), ProbeType::OW).unwrap();
            energy_rec += grid_receptor.trilinear_interpolation(oxygen.coords(), ProbeType::HW).unwrap() * water_params.OXYGEN_W_Q;
            println!("{index} {index} {energy_rec} O (just rec)");

            // let e = energy_for_real_water(&points, &vec![h1.clone()]);
            let e = grid_receptor_and_w.trilinear_interpolation(h1.coords(), ProbeType::HW).unwrap() * water_params.HYDROGEN_W_Q;
            println!("Total Coulomb: {}", e);
            println!("{index} {index} {e} H (rec+wat)");
            energy += e;

            // let er = energy_for_real_water(&receptor_points, &vec![h1.clone()]);
            let er = grid_receptor.trilinear_interpolation(h1.coords(), ProbeType::HW).unwrap() * water_params.HYDROGEN_W_Q;
            println!("{index} {index} {er} H (just rec)");
            energy_rec += er;

            // let e = energy_for_real_water(&points, &vec![h2.clone()]);
            let e = grid_receptor_and_w.trilinear_interpolation(h2.coords(), ProbeType::HW).unwrap() * water_params.HYDROGEN_W_Q;
            println!("Total Coulomb: {}", e);
            println!("{index} {index} {e} H (rec+wat)");
            energy += e;

            // let er = energy_for_real_water(&receptor_points, &vec![h2.clone()]);
            let er = grid_receptor.trilinear_interpolation(h2.coords(), ProbeType::HW).unwrap() * water_params.HYDROGEN_W_Q;
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
