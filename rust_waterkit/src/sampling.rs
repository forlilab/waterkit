use crate::geometry::{euclidean_distance, sum_points};
use crate::grid::Grid3D;
use crate::utils::*;
use crate::atom::Atom;
use crate::water::WaterMolecule;
use crate::energy::{energy, energy_for_real_water, spheric_energy};
use std::fmt::Debug;
use std::sync::Mutex;
use ndarray::Array1;
use rayon::prelude::*;
use rand::prelude::*;
use rand::distributions::WeightedIndex;


/// Check if a probe sphere is accessible at a given position.
/// A position is accessible if the sphere does not overlap with any surface atom.
trait CheckAccessibility {
    fn is_accessible(self, radius: &f64, sphere_coords: &[f64; 3]) -> bool;
}

// Implement the trait for &Vec<[f64; 3]>
impl CheckAccessibility for &Vec<[f64; 3]> {
    fn is_accessible(self, radius: &f64, sphere_coords: &[f64; 3]) -> bool {
        for point in self {
            println!("Calling is_accessible() for point");
            if &euclidean_distance(&sphere_coords, &point) < radius {
                return false;
            }
        }
        true
    }
}

impl CheckAccessibility for &Vec<Atom> {
    fn is_accessible(self, radius: &f64, sphere_coords: &[f64; 3]) -> bool {
        for point in self {
            if point.is_heavy_atom() && &euclidean_distance(&sphere_coords, &point.coords()) < radius {
                return false;
            }
        }
        true
    }
}

// A generic function to accept any type that implements PrintValue
fn is_accessible<T: CheckAccessibility + Debug>(value: T, radius: &f64, sphere_coords: &[f64; 3]) -> bool {
    value.is_accessible(radius, sphere_coords)
}

fn in_box(min_boundaries: &[f64; 3], max_boundaries: &[f64; 3], positions: &[f64; 3]) -> bool {
    let x_in = positions[0] < max_boundaries[0] && positions[0] > min_boundaries[0];
    let y_in = positions[1] < max_boundaries[1] && positions[1] > min_boundaries[1];
    let z_in = positions[2] < max_boundaries[2] && positions[2] > min_boundaries[2];
    x_in && y_in && z_in
}

pub fn roll_sphere_and_compute_energies_grid(
                                        receptor_points: &Vec<Atom>,
                                        x_size: f64,
                                        y_size: f64,
                                        z_size: f64,
                                        spacing: f64,
                                        center: [f64; 3]) -> Grid3D {

    let mut grid = Grid3D::new(x_size, y_size, z_size, spacing, center);
    
    grid.all_points_mut()
        .par_iter_mut()
        .for_each(|point| {
            let energy = spheric_energy(receptor_points, &point.coords);
            point.energy =  energy;
        }); 
    grid
}


/// Parallel version of rolling the sphere
pub fn roll_sphere_and_compute_energies(
    receptor_points: &Vec<Atom>,
    surface_points: &Vec<Atom>,
    step_size: f64,
    min_boundaries: &[f64; 3],
    max_boundaries: &[f64; 3],
) -> (Vec<f64>, Vec<[f64; 3]>) {
    // Shared results with thread-safe protection
    let energies = Mutex::new(Vec::new());
    let trajectories = Mutex::new(Vec::new());
    let surface_points_cloned = surface_points.clone();

    // Parallelize the loop over surface points
    surface_points.par_iter().for_each(|point| {
        let point_coords = point.coords();

        // Temporary local storage for this thread's contributions
        let mut local_energies = Vec::new();
        let mut local_trajectories = Vec::new();

        for d_radius in FloatRange::new(0.0, RADIUS_WATER * 2.0, 0.5) {
            let probe_center = [
                point_coords[0] + d_radius,
                point_coords[1] + d_radius,
                point_coords[2] + d_radius,
            ];

            if !in_box(min_boundaries, max_boundaries, &probe_center) {continue};

            // Check accessibility at the initial position
            if is_accessible(&surface_points_cloned, &d_radius, &probe_center){
                // Compute energy
                local_energies.push(spheric_energy(&receptor_points, &probe_center));
                local_trajectories.push(probe_center);
            }

            // Roll the sphere in a grid-like manner
            for dx in FloatRange::new(-step_size, step_size, 0.5) {
                for dy in FloatRange::new(-step_size, step_size, 0.5) {
                    for dz in FloatRange::new(-step_size, step_size, 0.5) {
                        if dx == 0.0 && dy == 0.0 && dz == 0.0 {
                            continue;
                        }

                        let new_point = [dx, dy, dz];
                        let candidate_position = sum_points(&probe_center, &new_point);

                        if !in_box(min_boundaries, max_boundaries, &candidate_position) {continue};

                        if is_accessible(&surface_points_cloned, &d_radius, &candidate_position){
                            // Compute energy
                            local_energies.push(spheric_energy(&receptor_points, &candidate_position));
                            local_trajectories.push(candidate_position);
                        }
                    }
                }
            }
        }

        // Merge results into shared collections
        let mut energies_lock = energies.lock().unwrap();
        let mut trajectories_lock = trajectories.lock().unwrap();
        energies_lock.extend(local_energies);
        trajectories_lock.extend(local_trajectories);
    });

    // Collect final results
    let final_energies = Mutex::into_inner(energies).unwrap();
    let final_trajectories = Mutex::into_inner(trajectories).unwrap();
    (final_energies, final_trajectories)
}


/// Single thread version of the rolling sphere
pub fn roll_sphere_and_compute_energies_single_th(surface_points: &Vec<Atom>,
                                        step_size: f64) -> (Vec<f64>, Vec<[f64; 3]>) {

    let mut energies = Vec::new();
    let mut trajectories = Vec::new();

    let surface_points_cloned = surface_points.clone();
    
    for point in surface_points.iter() {
        // Start the probe at the surface point
        let point_coords = point.coords();
        for d_radius in FloatRange::new(0.0, RADIUS_WATER*2.0, 0.7) {
            let probe_center = [point_coords[0] + d_radius, 
                point_coords[1] + d_radius, 
                point_coords[2] + d_radius];
            // Check if the probe is accessible at the initial position
            if is_accessible(&surface_points_cloned, &d_radius, &probe_center) {
                // Compute the energy
                energies.push(spheric_energy(&surface_points_cloned, &probe_center));
                trajectories.push(probe_center);
            }

            // Roll the sphere by moving it in a grid-like manner around the initial point
            for dx in FloatRange::new(-step_size, step_size, 0.5) {
                for dy in FloatRange::new(-step_size, step_size, 0.5) {
                    for dz in FloatRange::new(-step_size, step_size, 0.5) {
                        if dx == 0.0 && dy == 0.0 && dz == 0.0 {
                            continue;
                        }
                        let new_point = [dx, dy, dz];

                        let candidate_position = sum_points(&probe_center, &new_point);

                        if is_accessible(&surface_points_cloned, &d_radius, &probe_center) {
                            // Compute the energy
                            energies.push(spheric_energy(&surface_points_cloned, &candidate_position));
                            trajectories.push(candidate_position);
                        }
                    }
                }
            }

        }
    }
    (energies, trajectories) 
}

pub fn sample_real_waters(oxygen_position: &[f64; 3], 
    water_configurations: &Vec<[f64; 6]>,
    mut grid: Grid3D,
    mut receptor_points: Vec<Atom>) -> (Vec<Atom>, Grid3D) {
    // Want to update the map when selected the new water
    // let mut possible_waters: Vec<WaterMolecule> = Vec::new();
    // let mut possible_waters_energies: Vec<f64> = Vec::new();
    // let mut possible_waters_coords: Vec<[f64; 3]> = Vec::new();
    
    // for configuration in water_configurations {
    //     // H1 in position
    //     let h1_coords: [f64; 3] = [configuration[0] + oxygen_position[0], 
    //         configuration[1] + oxygen_position[1], 
    //         configuration[2] + oxygen_position[2]];
    //     // H2 in position
    //     let h2_coords: [f64; 3] = [configuration[3] + oxygen_position[0], 
    //         configuration[4] + oxygen_position[1], 
    //         configuration[5] + oxygen_position[2]];
    //     let water: WaterMolecule = WaterMolecule::new(h1_coords, h2_coords, oxygen_position.clone());
    //     // let h: [f64; 3] = water.as_vec()[1].coords();
    //     possible_waters.push(water);
    //     // possible_waters_coords.push(h);
    //     possible_waters_energies.push(energy(&map, &possible_waters.last().unwrap().as_vec()));
    // }
    // Let's parallelize
    let possible_results: Vec<(WaterMolecule, f64, f64)> = water_configurations
        .into_par_iter()
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
            let water = WaterMolecule::new(h1_coords, h2_coords, oxygen_position.clone());

            // Compute energy
            let (energy_value, oxygen_energy) = energy_for_real_water(&receptor_points, &water.as_vec());

            (water, energy_value, oxygen_energy)
        })
        .collect();
    
    let possible_waters: Vec<WaterMolecule> = possible_results.iter().map(|(w, _, _)| w.clone()).collect();
    let possible_waters_energies: Vec<f64> = possible_results.iter().map(|(_, e, _)| *e).collect();
    let oxygen_energies: Vec<f64> = possible_results.iter().map(|(_, _, o)| *o).collect();
    let choice = boltzmann_sampling(&possible_waters_energies);
    if boltzmann_acceptance_rejection(&possible_waters_energies[choice], 
        &BOLTZMANN_ENERGY_CUTOFF, 
        &TEMPERATURE, 
        &BOLTZMANN_K) {
        for atom in possible_waters[choice].as_vec() {
            if atom.atom_id() == "OW" {
                let oxygen_coords = atom.coords();
                grid.set(oxygen_coords[0], oxygen_coords[1], oxygen_coords[2], oxygen_energies[choice]);
            }
            receptor_points.push(atom.clone());
        }

    }
    (receptor_points, grid)
}



fn boltzmann_probabilities(energies: &Vec<f64>)  -> Vec<f64> {
    let energies_array = Array1::from(energies.clone());
    let factor = BOLTZMANN_K * TEMPERATURE;
    let distribution = energies_array.mapv(|e| (-e/factor).exp());
    let distribution_sum = distribution.sum();
    
    if distribution_sum > 0.0 {
        let p = distribution.mapv(|e| e/distribution_sum);
        return p.to_vec();
    }

    // Too high energies
    vec![0.0; energies.len()]
}


/// This function returns the index of a randomly sampled energy based
/// on the Boltzmann probability distribution
pub fn boltzmann_sampling(energies: &Vec<f64>) -> usize {
    let probability_distribution = boltzmann_probabilities(energies);
    let mut rng = thread_rng();
    let dist = WeightedIndex::new(&probability_distribution).unwrap();
    dist.sample(&mut rng)
}

pub fn boltzmann_acceptance_rejection(
    new_energies: &f64,
    old_energies: &f64,
    temperature: &f64,
    boltzmann_constant: &f64,
) -> bool {
    // Element-wise comparison to create a boolean array
    let decisions = new_energies < old_energies;

    // If all transitions are favorable, return early
    if decisions {
        return decisions;
    }

    // Compute Delta E for unfavorable transitions
    let delta_e: f64 = new_energies - old_energies;

    // Compute acceptance probabilities
    let factor = boltzmann_constant * temperature;
    let p_acc: f64 = (-delta_e / factor).exp().min(1.0);

    // Perform acceptance-rejection
    let mut rng = thread_rng();
    let random_values: f64 = rng.gen();

    // Update decisions based on probabilities
    if random_values <= p_acc {
        return true;
    }

    false
}

// pub fn boltzmann_acceptance_rejection(
//     new_energies: &Array1<f64>,
//     old_energies: &Array1<f64>,
//     temperature: f64,
//     boltzmann_constant: f64,
// ) -> Array1<bool> {
//     // Element-wise comparison to create a boolean array
//     let decisions = new_energies.iter().zip(old_energies.iter())
//         .map(|(&new, &old)| new < old)
//         .collect::<Vec<bool>>(); // Collect the results as a Vec<bool>

//     let mut decisions = Array1::from(decisions); // Convert Vec<bool> to Array1<bool>

//     // If all transitions are favorable, return early
//     if decisions.iter().all(|&d| d) {
//         return decisions;
//     }

//     // Find indices of unfavorable transitions
//     let unfavorable_indices: Vec<usize> = decisions
//         .iter()
//         .enumerate()
//         .filter(|&(_, &d)| !d)
//         .map(|(i, _)| i)
//         .collect();

//     // Compute Delta E for unfavorable transitions
//     let delta_e: Vec<f64> = unfavorable_indices
//         .iter()
//         .map(|&i| new_energies[i] - old_energies[i])
//         .collect();

//     // Compute acceptance probabilities
//     let factor = boltzmann_constant * temperature;
//     let p_acc: Vec<f64> = delta_e.iter().map(|&de| (-de / factor).exp().min(1.0)).collect();

//     // Perform acceptance-rejection
//     let mut rng = thread_rng();
//     let random_values: Vec<f64> = (0..p_acc.len()).map(|_| rng.gen()).collect();

//     // Update decisions based on probabilities
//     for (i, &p) in p_acc.iter().enumerate() {
//         let idx = unfavorable_indices[i];
//         if random_values[i] <= p {
//             decisions[idx] = true;
//         }
//     }

//     decisions
// }
