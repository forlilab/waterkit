use pyo3::prelude::*;
use rayon::prelude::*;

use crate::atom::Atom;
use crate::grid::Grid3D;
use crate::sampling::{boltzmann_sampling, roll_sphere_and_compute_energies, roll_sphere_and_compute_energies_grid, sample_real_waters};


// fn stop_hydration(map: &Vec<Atom>, water_map: &Vec<Atom>) -> bool {

// } 

#[pyfunction]
pub fn get_map(receptor_points: Vec<Atom>, water_configurations: Vec<[f64; 6]>, x_size: usize, y_size: usize, z_size: usize, spacing: f64, center: [f64; 3]) -> (Vec<f64>, Vec<[f64; 3]>) {
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
    let mut receptor_map = receptor_points.clone();
    let mut grid = roll_sphere_and_compute_energies_grid(&receptor_points, x_size, y_size, z_size, spacing, center);
    
    // this need to change -> Probably the boltzmann needs to see the grid?
    let (energies, trajectories) = grid.extract_energies_and_coordinates_parallel();
    
    // Pick one with Monte Carlo
    let initial_placement_index = boltzmann_sampling(&energies);
    let oxygen_position = trajectories[initial_placement_index];
    
    // Now need to sample all the possible configurations. Need to translate the 
    // hydrogens in place and then compute the energy
    // println!("Before upgrading map: {:?}", &map);
    (receptor_map, grid) = sample_real_waters(&oxygen_position, &water_configurations, grid, receptor_map);
    // println!("After upgrading map: {:?}", &map);
    
    let mut tries = 0; 
    let mut previous_length = waters_map.len();
    
    let mut picked = true;
    let mut energies = Vec::new();
    let mut trajectories = Vec::new();

    while tries < 500 {
        println!("{}", waters_map.len()/3);
        // If not picked in the previous round try to draw again from the same energies
        // Pick one with Monte Carlo
        let initial_placement_index = boltzmann_sampling(&energies);
        let oxygen_position = trajectories[initial_placement_index];
        
        // Now need to sample all the possible configurations. Need to translate the 
        // hydrogens in place and then compute the energy
        (receptor_map, grid) = sample_real_waters(&oxygen_position, &water_configurations, grid, receptor_map);

        if waters_map.len() == previous_length {
            picked = false;
            tries += 1; // Increment stagnant count
 
        } else {
            picked = true;
            tries = 0; // Reset count if length changes
        }
        previous_length = waters_map.len();
    }
    waters_map
}