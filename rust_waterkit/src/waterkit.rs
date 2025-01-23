use pyo3::prelude::*;
use rayon::str::MatchIndices;

use crate::atom::Atom;
use crate::sampling::{boltzmann_sampling, roll_sphere_and_compute_energies, sample_real_waters};


// fn stop_hydration(map: &Vec<Atom>, water_map: &Vec<Atom>) -> bool {

// } 

#[pyfunction]
pub fn get_map(receptor_points: Vec<Atom>, water_configurations: Vec<[f64; 6]>, step_size: f64, min_box: [f64; 3], max_box: [f64; 3]) -> (Vec<f64>, Vec<[f64; 3]>) {
    let surface_points: Vec<Atom> = receptor_points.iter()
            .filter(|&point| {
                let coords = point.coords();
                coords[0] > min_box[0] && coords[0] < max_box[0] &&
                coords[1] > min_box[1] && coords[1] < max_box[1] &&
                coords[2] > min_box[2] && coords[2] < max_box[2]
            })
            .cloned() // Clone each matching point to collect into a new Vec
        .collect();

    let mut map = surface_points.clone();
    let mut waters_map: Vec<Atom> = Vec::new();
    let mut receptor_map = receptor_points.clone();
    let (energies, trajectories) = roll_sphere_and_compute_energies(&receptor_points, &surface_points, step_size, &min_box, &max_box);
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
pub fn run_waterkit(receptor_points: Vec<Atom>, water_configurations: Vec<[f64; 6]>, step_size: f64, min_box: [f64; 3], max_box: [f64; 3]) -> Vec<Atom> {
    // let mut frame = Vec::new();
    let surface_points: Vec<Atom> = receptor_points.iter()
            .filter(|&point| {
                let coords = point.coords();
                coords[0] > min_box[0] && coords[0] < max_box[0] &&
                coords[1] > min_box[1] && coords[1] < max_box[1] &&
                coords[2] > min_box[2] && coords[2] < max_box[2]
            })
            .cloned() // Clone each matching point to collect into a new Vec
        .collect();

    let mut map = surface_points.clone();
    let mut waters_map: Vec<Atom> = Vec::new();
    let mut receptor_map = receptor_points.clone();
    let (energies, trajectories) = roll_sphere_and_compute_energies(&receptor_points, &surface_points, step_size, &min_box, &max_box);
    
    // Pick one with Monte Carlo
    let initial_placement_index = boltzmann_sampling(&energies);
    let oxygen_position = trajectories[initial_placement_index];
    
    // Now need to sample all the possible configurations. Need to translate the 
    // hydrogens in place and then compute the energy
    // println!("Before upgrading map: {:?}", &map);
    (map, waters_map, receptor_map) = sample_real_waters(&oxygen_position, &water_configurations, map, waters_map, receptor_map);
    // println!("After upgrading map: {:?}", &map);
    
    let mut tries = 0; 
    let mut previous_length = waters_map.len();
    
    let mut picked = true;
    let mut energies = Vec::new();
    let mut trajectories = Vec::new();

    while tries < 500 {
        println!("{}", waters_map.len()/3);
        // If not picked in the previous round try to draw again from the same energies
        if picked {
            (energies, trajectories) = roll_sphere_and_compute_energies(&receptor_map, &map, step_size, &min_box, &max_box);
        }
        // Pick one with Monte Carlo
        let initial_placement_index = boltzmann_sampling(&energies);
        let oxygen_position = trajectories[initial_placement_index];
        
        // Now need to sample all the possible configurations. Need to translate the 
        // hydrogens in place and then compute the energy
        (map, waters_map, receptor_map) = sample_real_waters(&oxygen_position, &water_configurations, map, waters_map, receptor_map);

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