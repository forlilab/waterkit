use pyo3::ffi::Py_Initialize;
use pyo3::prelude::*;
use rayon::prelude::*;

use crate::atom::Atom;
use crate::grid::{Grid3D, GridPoint};
use crate::sampling::{boltzmann_acceptance_rejection, boltzmann_sampling, order_boltzmann_sampling, roll_sphere_and_compute_energies_grid, sample, sample_basic, sample_real_waters, sample_with_order, save_shell_and_energies};
use crate::utils::{BOLTZMANN_ENERGY_CUTOFF, BOLTZMANN_K, TEMPERATURE};


// fn stop_hydration(map: &Vec<Atom>, water_map: &Vec<Atom>) -> bool {

// } 

#[pyfunction]
pub fn get_map(receptor_points: Vec<Atom>, water_configurations: Vec<[f64; 6]>, x_size: f64, y_size: f64, z_size: f64, spacing: f64, center: [f64; 3]) -> (Vec<f64>, Vec<[f64; 3]>) {
    let mut receptor_map = receptor_points.clone();
    let grid = roll_sphere_and_compute_energies_grid(&receptor_map, x_size, y_size, z_size, spacing, center);
    let points = grid.all_points();
    
    let mut energies = Vec::new();
    let mut trajectories = Vec::new();
    for point in points {
        energies.push(point.energy);
        trajectories.push(point.coords);
    }
    (energies, trajectories)
}

#[pyfunction]
pub fn test_allowed_points(receptor_points: Vec<Atom>, x_size: f64, y_size: f64, z_size: f64, spacing: f64, center: [f64; 3]) -> (Vec<f64>, Vec<[f64; 3]>) {
    let mut receptor_map = receptor_points.clone();
    let grid = roll_sphere_and_compute_energies_grid(&receptor_map, x_size, y_size, z_size, spacing, center);
    // let points = grid.get_neighbors(&receptor_points);
    
    let mut energies = Vec::new();
    let mut trajectories = Vec::new();
    // for point in points {
    //     let coords = point.coords; 
    //     let p = grid.get(coords[0], coords[1], coords[1]);
    //     if p.is_some() {
    //         energies.push(p.unwrap().energy);
    //         trajectories.push(p.unwrap().coords);
    //     }
    // }
    (energies, trajectories)
}


/// Step 1: Map surface of the protein.
/// Step 2: Optimize OH positions for receptor
/// Step 3: Extract energies with spherical water
///         and choose according to Boltzmann the 
///         first placement. 
/// Step 3: Optimize water position by using all the 
///         possible configurations for the water and 
///         pick the most favorable according to Boltzmann
///         and the Metropolis acceptance/rejection criteria.
/// Step 4: Update the surface with the new points and keep repeat.
#[pyfunction]
pub fn run_waterkit(receptor_points: Vec<Atom>, water_configurations: Vec<[f64; 6]>, x_size: f64, y_size: f64, z_size: f64, spacing: f64, center: [f64; 3]) -> Vec<Atom> {
    // let mut placements = 0;

    let mut receptor_map = receptor_points.clone();
    // let receptor_length = receptor_map.len();
    // let mut grid = roll_sphere_and_compute_energies_grid(&receptor_points, x_size, y_size, z_size, spacing, center);
    // // grid.set_allowed_points();

    // // this need to change -> Probably the boltzmann needs to see the grid?
    // let (initial_energies, initial_trajectories) = grid.extract_energies_and_coordinates_for_allowed();
    
    // // Pick one with Monte Carlo
    // let mut initial_placement_index = boltzmann_sampling(&initial_energies);
    // if initial_placement_index.is_some() {
        
    //     placements += 1;

    //     let oxygen_position = initial_trajectories[initial_placement_index.unwrap()];
    //     // println!("Initial energies: \n{:?}", grid.get(oxygen_position[0], oxygen_position[0], oxygen_position[0]));
    //     // Now need to sample all the possible configurations. Need to translate the 
    //     // hydrogens in place and then compute the energy
    //     // println!("Before upgrading map: {:?}", &map);
    //     (receptor_map, grid) = sample_real_waters(&oxygen_position, &water_configurations, grid, receptor_map);
    //     // grid.update_allowed_points(&oxygen_position);
        
    //     // println!("After upgrading map: {:?}", &map);
        
    //     let mut tries = 0; 
    //     let mut previous_length = 0;
        
    //     let mut picked = false;

    //     // Now the energies changed after the first placement? maybe unnecessary.
    //     let (mut energies, mut trajectories) = grid.extract_energies_and_coordinates_for_allowed();
    //     // println!("Energies after first placement: \n{:?}", grid.get(oxygen_position[0], oxygen_position[0], oxygen_position[0]));
    //     while tries < 500 {
    //         placements += 1;

    //         // println!("Previous length: {}", previous_length);
    //         // println!("Tries: {}", tries);
    //         // println!("{}", (receptor_map.len() - receptor_length) / 3);
    //         // If not picked in the previous round try to draw again from the same energies
    //         // Pick one with Monte Carlo
    //         if picked {
    //             (energies, trajectories) = grid.extract_energies_and_coordinates_for_allowed();
    //         }

    //         initial_placement_index = boltzmann_sampling(&energies);
    //         if initial_placement_index.is_some() {
    //             let oxygen_position = trajectories[initial_placement_index.unwrap()];
    //             // Now need to sample all the possible configurations. Need to translate the 
    //             // hydrogens in place and then compute the energy
    //             (receptor_map, grid) = sample_real_waters(&oxygen_position, &water_configurations, grid, receptor_map);
                
    //             if (receptor_map.len() - receptor_length) == previous_length {
    //                 // println!("Real sample rejected!");
    //                 picked = false;
    //                 tries += 1; // Increment stagnant count
        
    //             } else {
    //                 picked = true;
    //                 tries = 0; // Reset count if length changes
    //                 // grid.update_allowed_points(&oxygen_position);
    //                 previous_length = receptor_map.len() - receptor_length;

    //             }
    //         }
    //         else {
    //             picked = false;
    //             tries += 1;
    //         }
            
    //     }
    // }
    // println!("Total number of placements: {}", placements);
    receptor_map
}

#[pyfunction]
pub fn get_shells(receptor_points: Vec<Atom>, 
    water_configurations: Vec<[f64; 6]>, 
    anchor_points: Vec<[f64; 3]>, 
    x_size: f64, 
    y_size: f64, 
    z_size: f64, 
    spacing: f64, 
    center: [f64; 3],
    shells: usize) -> (Vec<[f64; 3]>, Vec<f64>) {

    let mut receptor_map = receptor_points.clone();
    let mut grid = roll_sphere_and_compute_energies_grid(&receptor_points, x_size, y_size, z_size, spacing, center);
    let mut receptor_map_in_grid = grid.get_anchor_points_in_grid(&anchor_points);
    grid.set_possible_points();
    let mut shell = 0;
    
    // while shell < shells {
    //     (receptor_map, grid, receptor_map_in_grid) = sample(grid, receptor_map, receptor_map_in_grid, &water_configurations);
    //     shell += 1;
    // }

    // println!("Total number of placements: {}", placements);
    let mut allowed_grid_as_atoms = Vec::new();
    let mut e = Vec::new();
    for point in grid.get_neighbors(&receptor_map_in_grid) {
        let coords = point.coords;
        // allowed_grid_as_atoms.push(
        //     Atom::new("HW".to_string(), "1".to_string(), coords, 0.0, 0.0, 0.0)
        // );
        allowed_grid_as_atoms.push(
            coords
        );
        e.push(point.energy);
    }
    (allowed_grid_as_atoms, e)
}

#[pyfunction]
pub fn run_waterkit_simple(receptor_points: Vec<Atom>, 
    water_configurations: Vec<[f64; 6]>, 
    anchor_points: Vec<[f64; 3]>, 
    x_size: f64, 
    y_size: f64, 
    z_size: f64, 
    spacing: f64, 
    center: [f64; 3],
    shells: usize) -> Vec<Atom> {

    let mut receptor_map = receptor_points.clone();
    let mut grid = roll_sphere_and_compute_energies_grid(&receptor_points, x_size, y_size, z_size, spacing, center);
    let mut receptor_map_in_grid = grid.get_anchor_points_in_grid(&anchor_points);
    grid.set_possible_points();
    let mut shell = 0;
    let mut new_aps = Vec::new();

    while shell < shells {
        if shell < 1 {
            new_aps = sample(&mut grid, &mut receptor_map, &mut receptor_map_in_grid, &water_configurations);
        }
        else {
            new_aps = sample(&mut grid, &mut receptor_map, &mut new_aps, &water_configurations);
        }
        shell += 1;
    }

    // println!("Total number of placements: {}", placements);
    // for point in grid.possible_points() {
    //     let coords = point.coords;
    //     // allowed_grid_as_atoms.push(
    //     //     Atom::new("HW".to_string(), "1".to_string(), coords, 0.0, 0.0, 0.0)
    //     // );
    //     allowed_grid_as_atoms.push(
    //         coords
    //     );
    //     e.push(point.energy);
    // }
    receptor_map
}

#[pyfunction]
pub fn save_shell_points_with_energies(receptor_points: Vec<Atom>, 
    water_configurations: Vec<[f64; 6]>, 
    anchor_points: Vec<[f64; 3]>, 
    x_size: f64, 
    y_size: f64, 
    z_size: f64, 
    spacing: f64, 
    center: [f64; 3],
    shells: usize) -> (Vec<f64>, Vec<[f64; 3]>) {
        let mut receptor_map = receptor_points.clone();
        let mut grid = roll_sphere_and_compute_energies_grid(&receptor_points, x_size, y_size, z_size, spacing, center);
        let mut receptor_map_in_grid = grid.get_anchor_points_in_grid(&anchor_points);
        grid.set_possible_points();
        let mut shell = 0;
        
        // while shell < shells {
        let (energies, positions) = save_shell_and_energies(grid, receptor_map, receptor_map_in_grid, &water_configurations);
        (energies, positions)
    }