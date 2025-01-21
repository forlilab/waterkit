use crate::atom::Atom;
use crate::geometry::roll_sphere_and_compute_energies;
use crate::sampling::boltzmann_sampling;



/// Step 1: Map surface of the protein.
/// Step 2: Extract energies with spherical water
///         and choose according to Boltzmann the 
///         first placement. 
/// Step 3: Optimize water position by using all the 
///         possible configurations for the water and 
///         pick the most favorable according to Boltzmann
///         and the Metropolis acceptance/rejection criteria.
/// Step 4: Update the surface with the new points and keep repeat.
pub fn run_waterkit(surface_positions: Vec<Atom>, water_configurations: Vec<Atom>, step_size: f64) -> Vec<Atom> {
    let mut frame = Vec::new();
    let mut map = surface_positions.clone();
    let (energies, trajectories) = roll_sphere_and_compute_energies(surface_positions, step_size);
    
    // Pick one with Monte Carlo
    let initial_placement_index = boltzmann_sampling(&energies);
    let oxygen_position = trajectories[initial_placement_index];
    // Now need to sample all the possible configurations. Need to translate the 
    // hydrogens in place and then compute the energy 
    frame
}