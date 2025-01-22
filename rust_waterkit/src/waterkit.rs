use pyo3::prelude::*;

use crate::atom::Atom;
use crate::energy::energy;
use crate::geometry::roll_sphere_and_compute_energies;
use crate::sampling::{boltzmann_sampling, sample_real_waters};
use crate::water::WaterMolecule;



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
pub fn run_waterkit(surface_positions: Vec<Atom>, water_configurations: Vec<[f64; 6]>, step_size: f64) -> Vec<Atom> {
    // let mut frame = Vec::new();
    let mut map = surface_positions.clone();
    let mut waters_map: Vec<Atom> = Vec::new();
    let (energies, trajectories) = roll_sphere_and_compute_energies(&surface_positions, step_size);
    
    // Pick one with Monte Carlo
    let initial_placement_index = boltzmann_sampling(&energies);
    let oxygen_position = trajectories[initial_placement_index];
    
    // Now need to sample all the possible configurations. Need to translate the 
    // hydrogens in place and then compute the energy
    // println!("Before upgrading map: {:?}", &map);
    (map, waters_map) = sample_real_waters(&oxygen_position, &water_configurations, map, waters_map);
    // println!("After upgrading map: {:?}", &map);
    

    while waters_map.len() < 60 {
        println!("{}", waters_map.len());
        let (energies, trajectories) = roll_sphere_and_compute_energies(&map, step_size);
    
        // Pick one with Monte Carlo
        let initial_placement_index = boltzmann_sampling(&energies);
        let oxygen_position = trajectories[initial_placement_index];
        
        // Now need to sample all the possible configurations. Need to translate the 
        // hydrogens in place and then compute the energy
        (map, waters_map) = sample_real_waters(&oxygen_position, &water_configurations, map, waters_map);
    }
    waters_map
}