use core::f64;
use std::fs::File;
use std::io::Write;
use std::sync::atomic::ATOMIC_BOOL_INIT;

use kiddo::{KdTree, SquaredEuclidean};
use plotters::prelude::NestedRange;
use rand::distributions::WeightedIndex;
use rand::prelude::Distribution;
use rand::{thread_rng, Rng};

use crate::anchor_point::AnchorPoint;
use crate::atom::Atom;
use crate::{geometry, monte_carlo as mc};
use crate::grid::{Grid3D, GridPoint, ProbeType};
use crate::consts::*;
use crate::water::WaterMolecule;

fn optimize_disordered_hydrogens() {
    
}

fn optimize_placement_order_grid(grid: &Grid3D, points: &Vec<AnchorPoint>) -> Vec<AnchorPoint> {
    let mut energies = Vec::with_capacity(points.len());
    // let mut min_points = Vec::with_capacity(points.len());
    for (i, point) in points.iter().enumerate() {
        let (min, max) = if point.hb_type() == "donor" { (1.5, 2.6) } else { (2.5, 3.6) };
        // New
        let energy = grid.trilinear_interpolation(*point.anchor_vectors(), ProbeType::ODa);
        if energy.is_some() {
            energies.push(energy.unwrap());
        }
        

        // if let Some(min_point) = grid
        //     .get_neighbors_within_distance_and_angle(point.anchor_point(), point.anchor_vectors(), max, min)
        //     .into_iter()
        //     .min_by(|a, b| a.energy_oda.partial_cmp(&b.energy_oda).unwrap())
        // {
        //     let energy = if min_point.energy_oda == 0.0 { f64::INFINITY } else { min_point.energy_oda };
        //     energies.push(energy);
        //     min_points.push((i, point)); // Store index to avoid duplicates
        // }
    }

    let mut decisions = Vec::new();
    // let mut seen_indices = std::collections::HashSet::new();
    let order = mc::boltzmann_choices(&energies, Some(energies.len()));

    // for &order_idx in &order {
    //     let (original_index, point) = min_points[order_idx];
    //     if seen_indices.insert(original_index) { // Avoid duplicate insertions
    //         if mc::boltzmann_acceptance_rejection(
    //             &energies[order_idx],
    //             &BOLTZMANN_ENERGY_CUTOFF,
    //             &TEMPERATURE,
    //             &BOLTZMANN_K,
    //         ) {
    //             decisions.push(point.clone());
    //         }
    //     }
    // }

    // New
    for &order_idx in &order {
        if mc::boltzmann_acceptance_rejection(&energies[order_idx],
             &BOLTZMANN_ENERGY_CUTOFF,
              &TEMPERATURE,
               &BOLTZMANN_K) {
            decisions.push(points[order_idx].clone());
        }
    }
    decisions
}


fn optimize_poistion_grid(grid: &Grid3D, point: &AnchorPoint) -> [f64; 3] {
    let max_disp = 0.375/2.;
    let mut min = 2.5;
    let mut max = 3.6;
    if point.hb_type() == "donor" {
        min -= 1.0;
        max -= 1.0;
    }
    let neighbors = grid.get_neighbors_within_distance_and_angle(point.anchor_point(), point.anchor_vectors(), max, min);
    // let neighbors = grid.get_neighbors_within_distance(point.anchor_point(), max, min);
    let energies: Vec<f64> = neighbors.iter().map(|x| x.energy_oda).collect();
    let choice = mc::boltzmann_choices(&energies, None);
    if choice.len() > 0 {
        let idx = choice.first().unwrap().clone();
        let new_point = neighbors[idx].clone();
        // return new_point.coords;
        let mut rng = rand::thread_rng();
        let displacement = [rng.gen_range(-max_disp..max_disp), 
            rng.gen_range(-max_disp..max_disp),
            rng.gen_range(-max_disp..max_disp)];
        
        return geometry::sum_points(&new_point.coords, &displacement);
    }
    *point.anchor_point()
}

pub fn build_kd_tree(receptor_points_on_the_grid: &Vec<Atom>) -> KdTree<f64, 3> {
    let mut tree = KdTree::new();
    for point in receptor_points_on_the_grid {
        tree.add(&point.coords(), 0); // Item is irrelevant for distance queries
    }
    tree
}

fn min_distance_to_protein(anchor: &AnchorPoint, tree: &KdTree<f64, 3>) -> f64 {
    let nearest = tree.nearest_one::<SquaredEuclidean>(&anchor.anchor_point());
    nearest.distance.sqrt() // Convert squared distance to Euclidean distance
}

fn distance_to_protein_grid(point: &GridPoint, tree: &KdTree<f64, 3>) -> f64 {
    let nearest = tree.nearest_one::<SquaredEuclidean>(&point.coords);
    nearest.distance.sqrt() // Convert squared distance to Euclidean distance
}

pub fn sample_without_layers_without_anchor_points(grid: &mut Grid3D,  
        receptor_points_tree: &Option<KdTree<f64, 3>>,
        water_configurations: &Vec<[f64; 6]>,
        new_water_molecules: &mut Vec<WaterMolecule>,
        last_residue_number: &mut usize,
        distance_cutoff: f64) {

    let mut placed_oxygens = Vec::new();
    let max_iterations = 1000;
    let k_b_t: f64 = BOLTZMANN_K * TEMPERATURE;
    
    let mut gird_points_for_placement = Vec::new();
    match receptor_points_tree {
        Some(receptor_points_tree) => {
            gird_points_for_placement.extend(grid.all_points().into_iter().filter(|p| 
            {
                let d = distance_to_protein_grid(p, &receptor_points_tree);
                d <= distance_cutoff && d >= 1.5 && !placed_oxygens.contains(&p.coords)
            }));
        },
        None => {
            gird_points_for_placement.extend(grid.all_points().into_iter().filter(|p| {
                !placed_oxygens.contains(&p.coords)
                }));
            },        
    }

    // // Save occupied voxels to XYZ file
    // let mut file = File::create("grid_points_for_placement.xyz").unwrap();
    // writeln!(file, "{}", gird_points_for_placement.len()).unwrap();
    // // writeln!(file, "Occupied Voxels for Receptor")?;
    // for point in gird_points_for_placement.iter() {
    //     let coords = point.coords;
    //     writeln!(file, "He {:.3} {:.3} {:.3}", coords[0], coords[1], coords[2]).unwrap();
    // }
    
    let bulk_water_density = 33.4; // molecules/nm^3
    let voxel_volume = grid.spacing * grid.spacing * grid.spacing;
    let total_volume = (voxel_volume * gird_points_for_placement.len() as f64) / 1000.0; // nm^3
    let target_n_waters = (total_volume * bulk_water_density * 0.6) as usize;
    println!("Target # of waters for cutoff {}A: {}", distance_cutoff, target_n_waters);

    while new_water_molecules.len() < target_n_waters {
        // println!("# of waters placed so far: {}", new_water_molecules.len());
        let mut gird_points_for_placement = Vec::new();
        match receptor_points_tree {
            Some(receptor_points_tree) => {
                gird_points_for_placement.extend(grid.all_points().into_iter().filter(|p| 
                {
                    let d = distance_to_protein_grid(p, &receptor_points_tree);
                    d <= distance_cutoff && d >= 1.5 && !placed_oxygens.contains(&p.coords)
                }));
            },
            None => {
                gird_points_for_placement.extend(grid.all_points().into_iter().filter(|p| {
                    !placed_oxygens.contains(&p.coords)
                    }));
                },        
        }
        let mut energies = Vec::with_capacity(gird_points_for_placement.len());
        for p in gird_points_for_placement.iter() {
            energies.push(p.energy_oda);
        }
        // Compute Boltzmann probabilities
        let boltzmann_factors: Vec<f64> = energies.iter().map(|&e| (-e / k_b_t).exp()).collect();
        let z: f64 = boltzmann_factors.iter().sum();
        let probabilities: Vec<f64> = boltzmann_factors.iter().map(|&f| f / z).collect();

        // Create weighted distribution for sampling
        let dist = WeightedIndex::new(&probabilities).expect("Invalid probabilities");
        let mut rng = thread_rng();
        // Sample a grid point
        let idx = dist.sample(&mut rng);
        let new_point = gird_points_for_placement[idx].coords;
        let selected_energy = energies[idx];
        if mc::boltzmann_acceptance_rejection(&selected_energy, 
            &BOLTZMANN_ENERGY_CUTOFF, 
            &TEMPERATURE, 
            &BOLTZMANN_K) {
                let (placed, mut water) = sample_waters_with_grids(
                    &new_point, 
                    water_configurations, 
                    grid, 
                    last_residue_number);
                if placed {
                    let atoms_to_update = water.as_vec();
                    placed_oxygens.push(water.oxygen.coords());
                    grid.update_energies(&atoms_to_update);
                    new_water_molecules.push(water);
                }
        }
    }
    println!("HOH placed: {}", new_water_molecules.len());
}

pub fn sample_without_layers(grid: &mut Grid3D,  
        receptor_points_tree: &KdTree<f64, 3>,
        anchor_points: &mut Vec<AnchorPoint>, 
        water_configurations: &Vec<[f64; 6]>,
        new_water_molecules: &mut Vec<WaterMolecule>,
        last_residue_number: &mut usize,
        distance_cutoff: f64) {
    let mut receptor_points_on_the_grid = Vec::new();
    for point in anchor_points.into_iter() {
        if grid.in_box(point.anchor_point()){
            receptor_points_on_the_grid.push(point.clone());
        }
    }
    let early_stop = 300;
    let mut cnt = 0;
    while !receptor_points_on_the_grid.is_empty() && cnt < early_stop {
        // println!("CNT: {}", cnt);
        // println!("Anchor points: {}", receptor_points_on_the_grid.len());
        // Sample with Boltzmann the neighbors and the actual point and based on Metropolis 
        // acceptance criteria then these are the starting anchor points
        let decisions = optimize_placement_order_grid(grid, &receptor_points_on_the_grid);
        if decisions.len() > 0 {
            let decision = decisions.first().unwrap();
            receptor_points_on_the_grid.retain(|ap| ap.anchor_point() != decision.anchor_point());
            
            let new_point = optimize_poistion_grid(grid, decision);
            // let new_point= *decision.anchor_point();
            let point_energy = grid.trilinear_interpolation(new_point, ProbeType::ODa).unwrap();
            if mc::boltzmann_acceptance_rejection(&point_energy, &BOLTZMANN_ENERGY_CUTOFF, &TEMPERATURE, &BOLTZMANN_K) {
                let (placed, mut water) = sample_waters_with_grids(&new_point, water_configurations, grid, last_residue_number);
                if placed {
                    water.guess_new_hydrogen_bonds();
                    receptor_points_on_the_grid.extend(water.hydrogen_bonds().into_iter().filter(|anchor| {
                            min_distance_to_protein(anchor, &receptor_points_tree) <= distance_cutoff
                    }));
                    let atoms_to_update = water.as_vec();
                    
                    grid.update_energies(&atoms_to_update);
                    new_water_molecules.push(water);
                    cnt = 0;           
                } else {
                    cnt += 1;
                }
            } else {
                cnt += 1;
            }
        } else {
            cnt += 1;
            // let rand_point = receptor_points_on_the_grid[thread_rng().gen_range(0..receptor_points_on_the_grid.len())].anchor_point().clone();
            // receptor_points_on_the_grid.retain(|ap| ap.anchor_point() != &rand_point);
        }
    }
    println!("HOH placed: {}", new_water_molecules.len());
}

// pub fn sample_without_layers(grid: &mut Grid3D,  
//         receptor_points_tree: &KdTree<f64, 3>,
//         anchor_points: &mut Vec<AnchorPoint>, 
//         water_configurations: &Vec<[f64; 6]>,
//         new_water_molecules: &mut Vec<WaterMolecule>,
//         last_residue_number: &mut usize,
//         distance_cutoff: f64) {
//     let mut receptor_points_on_the_grid = Vec::new();
//     for point in anchor_points.into_iter() {
//         if grid.in_box(point.anchor_point()){
//             receptor_points_on_the_grid.push(point.clone());
//         }
//     }
//     let early_stop = 70;
//     let mut cnt = 0;
//     while !receptor_points_on_the_grid.is_empty() && cnt <= early_stop {
//         // println!("CNT: {}", cnt);
//         // println!("Anchor points: {}", receptor_points_on_the_grid.len());
//         // Sample with Boltzmann the neighbors and the actual point and based on Metropolis 
//         // acceptance criteria then these are the starting anchor points
//         let decisions = optimize_placement_order_grid(grid, &receptor_points_on_the_grid);
//         if decisions.len() > 0 {
//             for (_idx, decision) in decisions.iter().enumerate() {
//                 receptor_points_on_the_grid.retain(|ap| ap.anchor_point() != decision.anchor_point());
//                 // let new_point = optimize_poistion_grid(grid, decision);
//                 let new_point= decision.anchor_point().clone();
//                 let point_energy = grid.trilinear_interpolation(new_point, ProbeType::ODa).unwrap();
//                 if mc::boltzmann_acceptance_rejection(&point_energy, &BOLTZMANN_ENERGY_CUTOFF, &TEMPERATURE, &BOLTZMANN_K) {
//                     let (placed, mut water) = sample_waters_with_grids(&new_point, water_configurations, grid, last_residue_number);
//                     if placed {
//                         water.guess_new_hydrogen_bonds();
//                         receptor_points_on_the_grid.extend(water.hydrogen_bonds().into_iter().filter(|anchor| {
//                                 min_distance_to_protein(anchor, &receptor_points_tree) <= distance_cutoff
//                         }));
//                         let atoms_to_update = water.as_vec();
                        
//                         grid.update_energies(&atoms_to_update);
//                         new_water_molecules.push(water);
//                         cnt = 0;           
//                     } else {
//                         cnt += 1;
//                     }
//                 } else {
//                     cnt += 1;
//                 }
//             }
//         }
//     }
// }

pub fn sample_using_grids(layer_id: usize,
        grid: &mut Grid3D,  
        anchor_points: &mut Vec<AnchorPoint>, 
        water_configurations: &Vec<[f64; 6]>,
        new_water_molecules: &mut Vec<WaterMolecule>,
        last_residue_number: &mut usize) -> bool {
    
    let mut new_anchor_points = Vec::new();
    let placement: bool = false;
    let mut receptor_points_on_the_grid = Vec::new();
    for point in anchor_points.into_iter() {
        if grid.in_box(point.anchor_point()){
            receptor_points_on_the_grid.push(point.clone());
        }
    }
    // println!("{} anchor points available", anchor_points.len());
    // Sample with Boltzmann the neighbors and the actual point and based on Metropolis 
    // acceptance criteria then these are the starting anchor points
    let decisions = optimize_placement_order_grid(grid, &receptor_points_on_the_grid);

    for (_idx, decision) in decisions.iter().enumerate() {
        let new_point = optimize_poistion_grid(grid, decision);

        let point_energy = grid.trilinear_interpolation(new_point, ProbeType::ODa).unwrap();
        if mc::boltzmann_acceptance_rejection(&point_energy, &BOLTZMANN_ENERGY_CUTOFF, &TEMPERATURE, &BOLTZMANN_K) {
            let (placed, mut water) = sample_waters_with_grids(&new_point, water_configurations, grid, last_residue_number);
            if placed {
                water.set_layer_id(layer_id + 1);
                water.guess_new_hydrogen_bonds();
                for hb in water.hydrogen_bonds().into_iter() {
                    new_anchor_points.push(hb);
                }
                let atoms_to_update = water.as_vec();
                grid.update_energies(&atoms_to_update);
                new_water_molecules.push(water);           
            }
        }
    }
    anchor_points.clear();
    for p in new_anchor_points.into_iter() {
        anchor_points.push(p);
    }
    placement
}

pub fn sample_waters_with_grids(oxygen_atom: &[f64; 3],
    water_configurations: &Vec<[f64; 6]>,
    grid: &Grid3D,
    last_residue_number: &mut usize) -> (bool, WaterMolecule) {
    let oxygen_position = *oxygen_atom;
    let lj_oxygen = grid.trilinear_interpolation(oxygen_position, ProbeType::OW).unwrap_or(f64::INFINITY);

    let mut best_water: Option<WaterMolecule> = None;
    let mut best_energy = f64::INFINITY;

    for configuration in water_configurations.iter() {
        // Compute H1 and H2 positions
        let h1_coords = [
            configuration[0] + oxygen_position[0],
            configuration[1] + oxygen_position[1],
            configuration[2] + oxygen_position[2],
        ];
        let h2_coords = [
            configuration[3] + oxygen_position[0],
            configuration[4] + oxygen_position[1],
            configuration[5] + oxygen_position[2],
        ];

        // Compute electrostatics, skip if any is missing
        let electrostatics_h1 = grid.trilinear_interpolation(h1_coords, ProbeType::HW);
        let electrostatics_h2 = grid.trilinear_interpolation(h2_coords, ProbeType::HW);
        let electrostatics_oxygen = grid.trilinear_interpolation(oxygen_position, ProbeType::HW);

        if electrostatics_h1.is_none() || electrostatics_h2.is_none() || electrostatics_oxygen.is_none() {
            continue; // Skip invalid configurations
        }

        let energy_value = lj_oxygen
            + electrostatics_oxygen.unwrap() * OXYGEN_W_Q
            + electrostatics_h1.unwrap() * HYDROGEN_W_Q
            + electrostatics_h2.unwrap() * HYDROGEN_W_Q;

        if energy_value < best_energy {
            best_energy = energy_value;
            best_water = Some(WaterMolecule::new(
                oxygen_position,
                h1_coords,
                h2_coords,
                "".to_string(),
                *last_residue_number + 1,
            ));
        }
    }

    if let Some(water) = best_water {
        if mc::boltzmann_acceptance_rejection(
            &best_energy,
            &BOLTZMANN_ENERGY_CUTOFF,
            &TEMPERATURE,
            &BOLTZMANN_K,
        ) {
            *last_residue_number += 1;
            return (true, water);
        }
    }

    // Return failure case
    (false, WaterMolecule::new([0.; 3], [0.; 3], [0.; 3], "".to_string(), 1000000))
}