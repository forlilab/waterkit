use rayon::prelude::*;

use crate::anchor_point::AnchorPoint;
use crate::monte_carlo as mc;
use crate::grid::{Grid3D, GridPoint};
use crate::consts::*;
use crate::atom::Atom;
use crate::water::WaterMolecule;
use crate::energy::energy_for_real_water;

fn optimize_placement_order_grid(grid: &mut Grid3D, points: &Vec<AnchorPoint>) -> Vec<AnchorPoint> {
    let mut energies = Vec::new();
    let mut min_points = Vec::new();
    let mut decisions = Vec::new();

    for point in points.iter() {
        let mut min = 2.5;
        let mut max = 3.6;
        if point.hb_type() == "donor" {
            min -= 1.0;
            max -= 1.0;
        }
        let neighbors = grid.get_neighbors_within_distance_and_angle(point.anchor_point(), point.anchor_vectors(), max, min);
        if neighbors.len() > 0 {
            let mut min_point = neighbors.first().unwrap().clone();
            for neigh in neighbors.into_iter() {
                if neigh.energy < min_point.energy {
                    min_point = neigh;
                }
            }
            let energy;
            if min_point.energy == 0. { energy = f64::INFINITY; }
            else { energy = min_point.energy;}
            energies.push(energy);
            min_points.push(point);
        }
        
    }

    let order = mc::boltzmann_choices(&energies, Some(energies.len()));      
    if order.len() > 0 {
        for order_idx in order.into_iter() {
            if mc::boltzmann_acceptance_rejection(&energies[order_idx], 
                &BOLTZMANN_ENERGY_CUTOFF, 
                &TEMPERATURE, 
            &BOLTZMANN_K) {
                if !decisions.contains(min_points[order_idx]) {
                    decisions.push(min_points[order_idx].clone());
                }
            }
        }
    }
    decisions
}

fn optimize_poistion_grid(grid: &mut Grid3D, point: &AnchorPoint, add_noise: bool) -> GridPoint {
    let mut min = 2.5;
    let mut max = 3.6;
    if point.hb_type() == "donor" {
        min -= 1.0;
        max -= 1.0;
    }
    let neighbors = grid.get_neighbors_within_distance_and_angle(point.anchor_point(), point.anchor_vectors(), max, min);
    let energies: Vec<f64> = neighbors.iter().map(|x| x.energy).collect();
    let choice = mc::boltzmann_choices(&energies, None);
    if choice.len() > 0 {
        let idx = choice.first().unwrap().clone();
        let new_point = neighbors[idx].clone();
        
        return new_point;
    }
    grid.get_nearest_neighbor(point.anchor_point()).unwrap().clone()
}

/// General approach for the search
/// 1 -  Get all the energies for the anchor points
/// 2 -  MC sampling of the points to pick the starting one
/// 3 -  Get the neighbors of the selected point
/// 4 -  MC sampling of the neighbors to pick the most favorable
/// 5 -  Metropolis criteria -> if it fails use the real anchor point
///      previously selected
/// 6 -  Sample real waters configurations
/// 7 -  MC sampling of the energies for the real waters
/// 8 -  Metropolis criteria
/// 9 -  Update grid's energies
/// 10 - Update points in the receptor's map
/// 11 - Update the new anchor points
pub fn sample(grid: &mut Grid3D, receptor_points: &mut Vec<Atom>, anchor_points: &mut Vec<AnchorPoint>, water_configurations: &Vec<[f64; 6]>) -> bool {
    let mut new_anchor_points = Vec::new();
    let placement: bool = false;
    let mut receptor_points_on_the_grid = Vec::new();
    for point in anchor_points.iter() {
        if grid.in_box(point.anchor_point()){
            receptor_points_on_the_grid.push(point.clone());
        }
    }
    // Sample with Boltzmann the neighbors and the actual point and based on Metropolis 
    // acceptance criteria then these are the starting anchor points
    // let start_time = Instant::now();
    let decisions = optimize_placement_order_grid(grid, &receptor_points_on_the_grid);
    // println!("Time taken to optimize placement: {:?}", start_time.elapsed());

    for (idx, decision) in decisions.iter().enumerate() {
        
        // let start_time = Instant::now();
        let new_point = optimize_poistion_grid(grid, decision, false);
        // println!("Time taken to optimize position_grid: {:?}", start_time.elapsed());
        
        if mc::boltzmann_acceptance_rejection(&new_point.energy, &BOLTZMANN_ENERGY_CUTOFF, &TEMPERATURE, &BOLTZMANN_K) {
            // Now we build explicit water
            // let start_time = Instant::now();
            let (placed, mut water) = sample_real_waters(&new_point, water_configurations, receptor_points);
            // println!("Time taken to sample real waters: {:?}", start_time.elapsed());

            if placed {
                water.guess_new_hydrogen_bonds();
                for hb in water.hydrogen_bonds().into_iter() {
                    new_anchor_points.push(hb);
                }
                grid.update_energies(&water.as_vec());
            }
        }
        // println!("\n");
    }

    // println!("New Anchor points: {}", new_anchor_points.len());
    anchor_points.clear();
    for p in new_anchor_points.into_iter() {
        anchor_points.push(p);
    }
    placement
}

pub fn save_shell_and_energies(mut grid: Grid3D, mut receptor_points: Vec<Atom>, anchor_points: Vec<[f64; 3]>, water_configurations: &Vec<[f64; 6]>) -> (Vec<f64>, Vec<[f64; 3]>) {
    let anchor_points_to_iter = anchor_points.clone();
    let mut energies = Vec::new();
    let mut points = Vec::new();
    for anchor_point in anchor_points_to_iter {
        let shell_points = grid.get_neighbors_within_distance(&anchor_point, MAX_SHELL_DISTANCE, MIN_SHELL_DISTANCE);
        for p in shell_points {
            energies.push(p.energy);
            points.push(p.coords);
        }
    }
    (energies, points)
}



pub fn sample_real_waters(oxygen_atom: &GridPoint,
    water_configurations: &Vec<[f64; 6]>,
    receptor_points: &mut Vec<Atom>) -> (bool, WaterMolecule) {    
    // Let's parallelize
    let oxygen_position = oxygen_atom.coords;
    let possible_results: Vec<(WaterMolecule, f64)> = water_configurations
        .par_iter()
        .map(|configuration| {
            // H1 in position
            let h1_coords: [f64; 3] = [
                configuration[0] + oxygen_position[0],
                configuration[1] + oxygen_position[1],
                configuration[2] + oxygen_position[2],
            ];
            // H2 in position
            let h2_coords: [f64; 3] = [
                configuration[3] + oxygen_position[0],
                configuration[4] + oxygen_position[1],
                configuration[5] + oxygen_position[2],
            ];

            // Create water molecule
            let water = WaterMolecule::new(oxygen_position.clone(), h1_coords, h2_coords);

            // Compute energy
            let energy_value = energy_for_real_water(&receptor_points, &water.as_vec());
            (water, energy_value)
        })
        .collect();

    let mut possible_waters_energies: Vec<f64> = possible_results.iter().map(|(_, e)| *e).collect();

    let choice = mc::boltzmann_sampling(&possible_waters_energies);
    if choice.is_some() {
        let value = choice.unwrap();
        if mc::boltzmann_acceptance_rejection(&possible_waters_energies[value],
            &BOLTZMANN_ENERGY_CUTOFF,
            &TEMPERATURE,
            &BOLTZMANN_K) {
            for atom in possible_results[value].0.as_vec() {
                receptor_points.push(atom.clone());
            }
            return (true, possible_results[value].0.clone());

        }
        else {
            return (false, WaterMolecule::new([0., 0., 0.], [0., 0., 0.], [0., 0., 0.]));
        }
    }
    else {
        return (false, WaterMolecule::new([0., 0., 0.], [0., 0., 0.], [0., 0., 0.]));
    }
}

pub fn sample_waters_with_grids() {

}