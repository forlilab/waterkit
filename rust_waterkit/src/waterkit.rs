use pyo3::prelude::*;

use crate::atom::Atom;
use crate::energy::energy;
use crate::geometry::roll_sphere_and_compute_energies;
use crate::sampling::boltzmann_sampling;
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
pub fn run_waterkit(surface_positions: Vec<Atom>, water_configurations: Vec<[f64; 6]>, step_size: f64) -> (Vec<[f64; 3]>, Vec<f64>) {
    // let mut frame = Vec::new();
    let mut map = surface_positions.clone();
    let (energies, trajectories) = roll_sphere_and_compute_energies(surface_positions, step_size);
    
    // Pick one with Monte Carlo
    let initial_placement_index = boltzmann_sampling(&energies);
    let oxygen_position = trajectories[initial_placement_index];
    // Now need to sample all the possible configurations. Need to translate the 
    // hydrogens in place and then compute the energy
    let mut possible_waters: Vec<WaterMolecule> = Vec::new();
    let mut possible_waters_energies: Vec<f64> = Vec::new();
    let mut possible_waters_coords: Vec<[f64; 3]> = Vec::new();

    let mut counter = 0;
    println!("Sampling the real waters for the oxygen in position: {:?} - {}", oxygen_position, counter);
    for configuration in water_configurations {
        let h1_coords = [configuration[0] + oxygen_position[0], 
            configuration[1] + oxygen_position[1], 
            configuration[2] + oxygen_position[2]];
        let h2_coords: [f64; 3] = [configuration[3] + oxygen_position[0], 
            configuration[4] + oxygen_position[1], 
            configuration[5] + oxygen_position[2]];
        let water = WaterMolecule::new(h1_coords, h2_coords, oxygen_position);
        let h = water.as_vec()[1].coords();
        possible_waters.push(water);
        possible_waters_coords.push(h);
        possible_waters_energies.push(energy(&map, &possible_waters.last().unwrap().as_vec()));
        counter += 1;
    }
    println!("Done sampling real waters! {}", counter);
    // frame
    (possible_waters_coords, possible_waters_energies)
}