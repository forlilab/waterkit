use core::f64;
use std::collections::HashSet;
use std::time::SystemTime;
use pyo3::prelude::*;
use rand::seq::SliceRandom;
use rand::thread_rng;
use rand::Rng;
use rayon::prelude::*;

use crate::anchor_point::AnchorPoint;
use crate::atom::Atom;
use crate::energy::energy;
use crate::energy::energy_for_real_water;
use crate::geometry;
use crate::geometry::dihedral;
use crate::grid::Grid3D;
use crate::grid::ProbeType;
use crate::consts;
use crate::monte_carlo;
use crate::optimizer::optimize;
use crate::optimizer::optimize_using_grids;
use crate::sampling::sample_using_grids;
use crate::setup::setup_grid;
use crate::utils::to_pdb;
use crate::utils::plot_optimization;
use crate::water::WaterMolecule;
use crate::energy;
use crate::utils;

fn run_single_waterkit_with_grids(receptor_points: &[Atom], 
    water_configurations: &Vec<[f64; 6]>, 
    anchor_points: &[AnchorPoint], 
    mut grid: Grid3D,
    epoch: usize, 
    num_steps: i32, 
    optimization_steps: i32) -> (Vec<Atom>, Vec<Atom>) {
    
    let mut receptor_map = receptor_points.to_vec();
    let mut last_residue_number = receptor_map.iter().map(|n| n.residue_number).max().unwrap_or(1);

    let mut mutable_anchor_points = anchor_points.to_vec();
    let mut new_water_molecules = Vec::new();
    for _i in 0..4 {
        sample_using_grids(&mut grid, &mut mutable_anchor_points, &water_configurations, &mut new_water_molecules, &mut last_residue_number);
    }

    // let waters: Vec<Atom> = receptor_map.iter().filter(|x| x.atom_type() == "OW" || x.atom_type() == "HW").cloned().collect();
    let mut waters: Vec<Atom> = Vec::with_capacity(new_water_molecules.len() * 3);
    // for w in new_water_molecules.iter() {
        // waters.extend(w.as_vec().into_iter());
        // receptor_map.extend(w.as_vec().into_iter());
    // } 
    let mut  unoptimized_water_atoms = Vec::with_capacity(new_water_molecules.len() * 3);
    for water in new_water_molecules.iter() {
        for atom in water.as_vec() {
            unoptimized_water_atoms.push(atom);
        }
    }
    // energy::set_waters_energies(&mut new_water_molecules, &receptor_map);
    // utils::to_pdb(&unoptimized_water_atoms, &format!("test/water_{epoch}_unoptimized.pdb"), Some(new_water_molecules.iter().map(|x| x.get_energy()).collect::<Vec<f64>>()));
    // utils::to_pdb(&unoptimized_water_atoms, &format!("test/water_{epoch}_unoptimized.pdb"), None);

    // unoptimized_water_atoms

    // let optimized_waters = optimize_water_network(&mut receptor_map, &new_water_molecules, &mut grid, &format!("test/water_{epoch}_optimized.pdb"));
    // let optimized_waters = optimize_water_nw_with_grids(&mut new_water_molecules, &mut grid);
    // optimized_waters
    optimize_water_nw_with_grids_sa(&mut new_water_molecules, &receptor_map, &mut grid, num_steps, optimization_steps);
    
    for w in new_water_molecules.into_iter() {
        waters.extend(w.as_vec().into_iter());
    }

    // let old_energies: Vec<f64> = new_water_molecules.iter().map(|x| x.get_energy()).collect();
    // energy::set_waters_energies(&mut new_water_molecules, &receptor_map);
    // let new_energies: Vec<f64> = new_water_molecules.iter().map(|x| x.get_energy()).collect();
    // for (idx, water) in new_water_molecules.iter_mut().enumerate() {
    //     water.set_energy(new_energies[idx] - old_energies[idx]);
    // }
    // utils::to_pdb(&waters, &format!("test/water_{epoch}_optimized.pdb"), Some(new_water_molecules.iter().map(|x| x.get_energy()).collect::<Vec<f64>>()));
    (unoptimized_water_atoms, waters)
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
    let mut waters_res_number: Vec<i32> = new_waters.iter()
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
        .map(|epoch| run_single_waterkit_with_grids(
                &receptor_points,
                &water_configurations,
                &anchor_points,
                grid.clone(),
                epoch,
                num_steps,
                optimization_steps
            )).collect();
            
    println!("Done sampling...saving results!");
    
    waters.par_iter().enumerate()
        .for_each(|(idx, (unoptimized_system, optimized_system))| {
            to_pdb(&unoptimized_system, &format!("{save_path}/water_{idx}_unoptimized.pdb"), None);
            to_pdb(&optimized_system, &format!("{save_path}/water_{idx}_optimized.pdb"), None)}
        );
    // waters.par_iter().enumerate()
    //     .for_each(|(idx, system)| to_pdb(&system, &format!("/data/phd/waterkit/rust_waterkit/test/water_{idx}_unoptimized.pdb")));
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
