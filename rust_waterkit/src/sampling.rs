use core::f64;
use std::collections::HashMap;

use rand::Rng;

use crate::anchor_point::AnchorPoint;
use crate::atom::Atom;
use crate::{geometry, monte_carlo as mc};
use crate::grid::{Grid3D, ProbeType};
use crate::consts::*;
use crate::water::WaterMolecule;

fn optimize_disordered_hydrogens() {
    
}

fn optimize_placement_order_grid(grid: &Grid3D, points: &Vec<AnchorPoint>) -> Vec<AnchorPoint> {
    let mut energies = Vec::with_capacity(points.len());
    let mut min_points = Vec::with_capacity(points.len());

    for (i, point) in points.iter().enumerate() {
        let (min, max) = if point.hb_type() == "donor" { (1.5, 2.6) } else { (2.5, 3.6) };

        if let Some(min_point) = grid
            .get_neighbors_within_distance_and_angle(point.anchor_point(), point.anchor_vectors(), max, min)
            .into_iter()
            .min_by(|a, b| a.energy_oda.partial_cmp(&b.energy_oda).unwrap())
        {
            let energy = if min_point.energy_oda == 0.0 { f64::INFINITY } else { min_point.energy_oda };
            energies.push(energy);
            min_points.push((i, point)); // Store index to avoid duplicates
        }
    }

    let mut decisions = Vec::new();
    let mut seen_indices = std::collections::HashSet::new();
    let order = mc::boltzmann_choices(&energies, Some(energies.len()));

    for &order_idx in &order {
        let (original_index, point) = min_points[order_idx];
        if seen_indices.insert(original_index) { // Avoid duplicate insertions
            if mc::boltzmann_acceptance_rejection(
                &energies[order_idx],
                &BOLTZMANN_ENERGY_CUTOFF,
                &TEMPERATURE,
                &BOLTZMANN_K,
            ) {
                decisions.push(point.clone());
            }
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

pub fn sample_using_grids(layer_id: usize,
        grid: &mut Grid3D,  
        anchor_points: &mut Vec<AnchorPoint>, 
        water_configurations: &Vec<[f64; 6]>,
        new_water_molecules: &mut Vec<WaterMolecule>,
        // layers_map: &mut HashMap<usize, Vec<usize>>,
        last_residue_number: &mut usize,
        // receptor_map: &mut Vec<Atom>
    ) -> usize {
    let mut n_waters_added: usize = 0;
    // let mut new_anchor_points = Vec::new();
    let placement: bool = false;
    let mut receptor_points_on_the_grid = Vec::new();
    for point in anchor_points.into_iter() {
        if grid.in_box(point.anchor_point()){
            receptor_points_on_the_grid.push(point.clone());
        }
    }
    // Sample with Boltzmann the neighbors and the actual point and based on Metropolis 
    // acceptance criteria then these are the starting anchor points
    // let start = SystemTime::now();
    let decisions = optimize_placement_order_grid(grid, &receptor_points_on_the_grid);
    // let end = SystemTime::now();
    // let duration = end.duration_since(start).unwrap();
    // println!("Placement order took {} ms", duration.as_millis());

    for (_idx, decision) in decisions.iter().enumerate() {
        
        // let start = SystemTime::now();
        let new_point = optimize_poistion_grid(grid, decision);
        // let end = SystemTime::now();
        // let duration = end.duration_since(start).unwrap();
        // println!("Position's optimization took {} ms", duration.as_millis());

        let point_energy = grid.trilinear_interpolation(new_point, ProbeType::ODa).unwrap();
        if mc::boltzmann_acceptance_rejection(&point_energy, &BOLTZMANN_ENERGY_CUTOFF, &TEMPERATURE, &BOLTZMANN_K) {
            // Now we build explicit water
            // let start = SystemTime::now();
            let (placed, mut water) = sample_waters_with_grids(&new_point, water_configurations, grid, last_residue_number);
            // let end = SystemTime::now();
            // let duration = end.duration_since(start).unwrap();
            // println!("Water's orientation took {} ms", duration.as_millis());

            if placed {
                // let start = SystemTime::now();
                water.set_layer_id(layer_id + 1);
                // water.guess_new_hydrogen_bonds();
                // for hb in water.hydrogen_bonds().into_iter() {
                //     new_anchor_points.push(hb);
                // }
                // let end = SystemTime::now();
                // let duration = end.duration_since(start).unwrap();
                // println!("New Hydrogen's bond guessing took {} ms", duration.as_millis());
                let atoms_to_update = water.as_vec();
                
                // let start = SystemTime::now();
                grid.update_energies(&atoms_to_update);
                // let end = SystemTime::now();
                // let duration = end.duration_since(start).unwrap();
                // println!("Grid's update took {} ms", duration.as_millis());
                new_water_molecules.push(water);
                n_waters_added += 1;
                // layers_map.entry(layer_id + 1)
                //     .or_insert(Vec::new())
                //     .push(new_water_molecules.len());
                // for atom in atoms_to_update {
                //     receptor_map.push(atom);
                // }           
            }
        }
    }
    // anchor_points.clear();
    // for p in new_anchor_points.into_iter() {
    //     anchor_points.push(p);
    // }
    n_waters_added
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
            + electrostatics_oxygen.unwrap() * OXYGEN_W_Q_TIP3PFB
            + electrostatics_h1.unwrap() * HYDROGEN_W_Q_TIP3PFB
            + electrostatics_h2.unwrap() * HYDROGEN_W_Q_TIP3PFB;

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