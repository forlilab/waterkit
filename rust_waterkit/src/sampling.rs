use itertools::izip;
use rand::seq::SliceRandom;
use rayon::prelude::*;

use crate::anchor_point::AnchorPoint;
use crate::monte_carlo as mc;
use crate::grid::{Grid3D, GridPoint};
use crate::consts::*;
use crate::atom::Atom;
use crate::vina_ff::vina_energy;
use crate::water::WaterMolecule;
use crate::energy::{energy_for_real_water};

pub fn test_anchor_points(grid: &mut Grid3D, receptor_points: &mut Vec<Atom>, anchor_points: &mut Vec<[f64; 3]>, water_configurations: &Vec<[f64; 6]>) -> Vec<[f64; 3]> {
    let mut ap_energies = Vec::new();
    let mut ap_on_the_grid = Vec::new();
    let mut new_anchor_points = Vec::new();
    let mut placement = false;
    let mut new_water = WaterMolecule::new([0., 0., 0.], [0., 0., 0.], [0., 0., 0.]);
    let mut new_receptor_points = Vec::new();

    for ap in anchor_points.iter() {
        let p = grid.get_nearest_neighbor(&ap);
        if p.is_some() {
            ap_energies.push(p.unwrap().energy);
            ap_on_the_grid.push(p.unwrap().clone());
        }
    }

    let mc_index = mc::boltzmann_sampling(&ap_energies);
    if mc_index.is_some() {
        let mut neighbors_energies = Vec::new();
        let neighbors = grid.get_neighbors_within_distance(&anchor_points[mc_index.unwrap()], MAX_SHELL_DISTANCE, MIN_SHELL_DISTANCE);
        for neighbor in neighbors.iter() {
            neighbors_energies.push(neighbor.energy);
        }
        let neigh_mc_index = mc::boltzmann_sampling(&neighbors_energies);
        if neigh_mc_index.is_some() {
            let point_to_sample = neighbors[neigh_mc_index.unwrap()].clone();
            if mc::boltzmann_acceptance_rejection(&point_to_sample.energy, &BOLTZMANN_ENERGY_CUTOFF, &TEMPERATURE, &BOLTZMANN_K) {
                (placement, new_water) = sample_real_waters(&point_to_sample, water_configurations, receptor_points, &mut new_receptor_points);
                if placement {
                    // anchor_points.remove(mc_index.unwrap());
                    // anchor_points.push(point_to_sample.coords);
                    let oxygen_coords = new_water.as_vec()[0].coords();
                    grid.update_grid_energies(receptor_points);
                    let new_aps = grid.guess_new_hydrogen_bonds(&new_water);
                    for anchor in new_aps {
                        new_anchor_points.push(anchor);
                    }
                    let water_v = new_water.as_vec();
                    new_anchor_points.push(water_v[1].coords());
                    new_anchor_points.push(water_v[2].coords());
                    new_anchor_points.push(water_v[0].coords());
                }
                else {
                    println!("Weird stuff while placing real waters!");
                }
                // anchor_points.remove(mc_index.unwrap());
            }
            else {
                println!("Rejected during Metropolis before sampling real water with energy: {}", &point_to_sample.energy);
                // anchor_points.remove(mc_index.unwrap());
            }
        }
        else {
            let point_to_sample = &ap_on_the_grid[mc_index.unwrap()];
            if mc::boltzmann_acceptance_rejection(&point_to_sample.energy, &BOLTZMANN_ENERGY_CUTOFF, &TEMPERATURE, &BOLTZMANN_K) {
                println!("No favorable neighbor found, using the original anchor point!");
                (placement, new_water) = sample_real_waters(&point_to_sample, water_configurations, receptor_points, &mut new_receptor_points);
                if placement {
                    // anchor_points.remove(mc_index.unwrap());
                    // anchor_points.push(ap_on_the_grid[mc_index.unwrap()].coords);
                    let oxygen_coords = new_water.as_vec()[0].coords();
                    grid.update_grid_energies(receptor_points);
                    let new_aps = grid.guess_new_hydrogen_bonds(&new_water);
                    for anchor in new_aps {
                        new_anchor_points.push(anchor);
                    }
                    let water_v = new_water.as_vec();
                    new_anchor_points.push(water_v[1].coords());
                    new_anchor_points.push(water_v[2].coords());
                    new_anchor_points.push(water_v[0].coords());
                }
                else {
                    println!("Weird stuff while placing real waters after using original point!");
                }
                // anchor_points.remove(mc_index.unwrap());
            }
            else {
                println!("Rejected during Metropolis before sampling real water when using real anchor point with energy: {}", &point_to_sample.energy);
                // anchor_points.remove(mc_index.unwrap());
            }
        }

    }
    new_anchor_points
}


fn optimize_placement_order_grid(grid: &mut Grid3D, points: &Vec<AnchorPoint>) -> Vec<GridPoint> {
    let mut energies = Vec::new();
    let mut min_points = Vec::new();
    let mut decisions = Vec::new();

    for point in points.iter() {
        for v in point.anchor_vectors() {
            let mut min = 2.5;
            let mut max = 3.6;
            if point.hb_type() == "donor" {
                min -= 1.0;
                max -= 1.0;
            }
            let neighbors = grid.get_neighbors_within_distance(point.anchor_point(), max, min);
            // println!("He {} {} {}", point_coords[0], point_coords[1], point_coords[2]);
            // for neighbor in neighbors.iter() {
            //     println!("H {} {} {}", neighbor.coords[0], neighbor.coords[1], neighbor.coords[2]);
            // }
            // println!("\n");
            let min_point = neighbors.into_iter().min_by(|a, b| a.energy.partial_cmp(&b.energy).unwrap()).unwrap();
            energies.push(min_point.energy.clone());
            min_points.push(min_point.clone());
        }
        
    }
    // for (energy, point) in izip!(&energies, &min_points) {
    //     println!("{} {}, {}, {}", energy, point.coords[0], point.coords[1], point.coords[2]);
    // }

    let order = mc::boltzmann_choices(&energies, Some(energies.len()));      
    if order.len() > 0 {
        for order_idx in order.into_iter() {
            if mc::boltzmann_acceptance_rejection(&energies[order_idx], 
                &BOLTZMANN_ENERGY_CUTOFF, 
                &TEMPERATURE, 
            &BOLTZMANN_K) {
                if !decisions.contains(&min_points[order_idx]) {
                    decisions.push(min_points[order_idx].clone());
                }
            }
        }
    }
    decisions
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
    let placement: bool = false;
    let mut receptor_points_on_the_grid = Vec::new();
    for point in anchor_points.iter() {
        if grid.in_box(point.anchor_point()){
            receptor_points_on_the_grid.push(point.clone());
        }
    }
    // println!("# of points in grid: {}", receptor_points_on_the_grid.len());
    // Sample with Boltzmann the neighbors and the actual point and based on Metropolis 
    // acceptance criteria then these are the starting anchor points
    let decisions = optimize_placement_order_grid(grid, &receptor_points_on_the_grid);
    // println!("{}", decisions.len());
    for (idx, decision) in decisions.iter().enumerate() {
        // println!("H {} {} {}", decision.coords[0], decision.coords[1], decision.coords[2]);
    }

    anchor_points.clear();

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

pub fn roll_sphere_and_compute_energies_grid(
                                        receptor_points: &Vec<Atom>,
                                        x_size: f64,
                                        y_size: f64,
                                        z_size: f64,
                                        spacing: f64,
                                        center: [f64; 3]) -> Grid3D {

    let mut grid = Grid3D::new((x_size, y_size, z_size), spacing, center);
    grid.all_points_mut()
        .par_iter_mut()
        .for_each(|point| {
            let energy = vina_energy(receptor_points, &point.coords);
            point.energy =  energy;
            point.updated = true;
        });
    grid.build_kdtree();
    // println!("Tree size: {}", grid.kdtree.size());
    // println!("Finished setting the possible points");
    grid
}



pub fn sample_real_waters(oxygen_atom: &GridPoint,
    water_configurations: &Vec<[f64; 6]>,
    receptor_points: &mut Vec<Atom>,
    new_receptor_points: &mut Vec<Atom>) -> (bool, WaterMolecule) {
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

    // let mut possible_results: Vec<(WaterMolecule, f64)> = Vec::new();
    //  for configuration in water_configurations {
    //         // H1 in position
    //         let h1_coords: [f64; 3] = [
    //             configuration[0] + oxygen_position[0],
    //             configuration[1] + oxygen_position[1],
    //             configuration[2] + oxygen_position[2],
    //         ];
    //         // H2 in position
    //         let h2_coords: [f64; 3] = [
    //             configuration[3] + oxygen_position[0],
    //             configuration[4] + oxygen_position[1],
    //             configuration[5] + oxygen_position[2],
    //         ];

    //         // Create water molecule
    //         let water = WaterMolecule::new(oxygen_position.clone(), h1_coords, h2_coords);

    //         // Compute energy
    //         let energy_value = energy_for_real_water(&receptor_points, &water.as_vec());
    //         possible_results.push((water, energy_value));
    //     }

    // print pdbs for waters
    // for (idx, result) in possible_results.iter().enumerate() {
    //     let w = &result.0;
    //     let energy = result.1;
    //     for atom in w.as_vec() {
    //         let mut atom_type = String::new();
    //         if atom.atom_type() == &"OW" {
    //             atom_type = "O".to_string();
    //         }
    //         else {
    //             atom_type = "H".to_string();
    //         }
    //         to_pdb_line(&atom_type, atom.coords(), energy, idx);
    //     }
    // }

    // Let's add Jerome's water and see if it's more favorable
    let water_j = WaterMolecule::new([-3.989, 8.48, 35.816], [-4.359, 8.63, 34.356], [-4.742, 8.555, 35.23]);
    let water_j_energy = energy_for_real_water(&receptor_points, &water_j.as_vec());
    // println!("Energy for Jerome's water: {}", water_j_energy);
    let mut possible_waters_energies: Vec<f64> = possible_results.iter().map(|(_, e)| *e).collect();
    // possible_waters_energies.push(energy_for_real_water(&receptor_points, &water_j.as_vec()));

    let choice = mc::boltzmann_sampling(&possible_waters_energies);
    let energies_good: Vec<f64> = possible_waters_energies.clone().into_iter().filter(|x| x < &0.0).collect();
    // println!("Chosen oxygen's energy: {}", oxygen_atom.energy);
    // println!("Good water configurations: {}", energies_good.len());
    if choice.is_some() {
        let value = choice.unwrap();
        // println!("Chosen index: {}", value);
        // println!("Energy for chosen water: {}", possible_waters_energies[value]);
        // // println!("Chosen point: {:?}", possible_results[value].0);
        if mc::boltzmann_acceptance_rejection(&possible_waters_energies[value],
            &BOLTZMANN_ENERGY_CUTOFF,
            &TEMPERATURE,
            &BOLTZMANN_K) {
            for atom in possible_results[value].0.as_vec() {
                // println!("New points: {:?}", atom);
                receptor_points.push(atom.clone());
            }
            return (true, possible_results[value].0.clone());

        }
        else {
            // println!("Rejected during real water sampling");
            return (false, WaterMolecule::new([0., 0., 0.], [0., 0., 0.], [0., 0., 0.]));
        }
    }
    else {
        // println!("Rejected during real water sampling before acceptance");
        return (false, WaterMolecule::new([0., 0., 0.], [0., 0., 0.], [0., 0., 0.]));
    }
}


pub fn sampling_order_for_anchor_points(anchor_points: &Vec<[f64; 3]>, grid: &mut Grid3D)-> (Vec<usize>, Vec<GridPoint>) {
    let mut aps_from_grid = Vec::new();

    for ap in anchor_points {
        let p = grid.get_nearest_neighbor(ap);
        if p.is_some() {
            aps_from_grid.push(p.unwrap().clone());
        }
    }
    let energies: &Vec<f64> = &aps_from_grid.iter().map(|x| x.energy).collect();
    let order = mc::boltzmann_choices(energies, Some(energies.len()));
    (order, aps_from_grid)
}

fn to_pdb_line(atom: &String, coords: [f64; 3], energy: f64, idx: usize) {
    let c = coords;
    let e = energy;
    let line = format!(
        "{:<6}{:>5} {:^4} {:>3} {:1}{:>4}    {:>8.3}{:>8.3}{:>8.3}{:>6.2}{:>6.2}          {:>2}",
        "ATOM",
        idx,
        atom,
        "HOH",
        "A",
        1,
        c[0],
        c[1],
        c[2],
        0.0,
        e,
        atom
    );
    println!("{}", line);
    // println!("H {} {} {} {}", c[0], c[1], c[2], e);
}
