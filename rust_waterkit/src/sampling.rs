use crate::grid::Grid3D;
use crate::utils::*;
use crate::atom::Atom;
use crate::water::WaterMolecule;
use crate::energy::{energy_for_real_water, spheric_energy};
use ndarray::Array1;
use rayon::prelude::*;
use rand::prelude::*;
use rand::distributions::WeightedIndex;


/// This is the main sampling engine.
/// The idea is to find all neighboring points
/// for each AP and run Metropolis MC on those points
/// to determine a good one. Sample real waters for that point
/// then move on updating the anchor points list with the new atoms
/// and repeat until no more atoms available within the 
/// distance threshold of 12. Angstrom.
pub fn sample(mut grid: Grid3D, mut receptor_points: Vec<Atom>, mut anchor_points: Vec<[f64; 3]>, water_configurations: &Vec<[f64; 6]> ) -> (Vec<Atom>, Grid3D, Vec<[f64; 3]>){
    let anchor_points_to_iter = anchor_points.clone();
    let mut new_aps = Vec::new();
    for anchor_point in anchor_points_to_iter {
        let shell_points = grid.get_neighbor_for_point(&anchor_point);
        let mut energies = Vec::new();
        let mut trajectories = Vec::new();
        // println!("# of Shell points for {:?}: {}", anchor_point, shell_points.len());
        for point in shell_points {
            energies.push(point.energy);
            trajectories.push(point.coords);
        }
        let mc_index = boltzmann_sampling(&energies);
        if mc_index.is_some()
        {
            let mut new_ap = [0.0; 3];
            let index = mc_index.unwrap();
            // println!("Energy of the selected point: {}", &energies[index]);
            if boltzmann_acceptance_rejection(&energies[index], &BOLTZMANN_ENERGY_CUTOFF, &TEMPERATURE, &BOLTZMANN_K) {
                (receptor_points, grid, new_ap) = sample_real_waters(&trajectories[index], water_configurations, grid, receptor_points);
                if new_ap != [0.0, 0.0, 0.0] {
                    println!("After placing: {:?}", grid.get(new_ap[0], new_ap[1], new_ap[2]));
                    new_aps.push(new_ap);
                }
            }
        }
        else {
            println!("Something went wrong in the Boltzmann sampling!");
        }
    }
    println!("Waters placed: {}", new_aps.len());
    (receptor_points, grid, new_aps)
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
            let energy = spheric_energy(receptor_points, &point.coords);
            point.energy =  energy;
        }); 
    println!("Finished setting the possible points");
    grid
}

pub fn sample_real_waters(oxygen_position: &[f64; 3], 
    water_configurations: &Vec<[f64; 6]>,
    mut grid: Grid3D,
    mut receptor_points: Vec<Atom>) -> (Vec<Atom>, Grid3D, [f64; 3]) {

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

    let mut new_ap: [f64; 3] = [0.0; 3];
    // let (possible_waters, possible_energies, oxygen_energies): (Vec<WaterMolecule>, Vec<f64>, Vec<f64>) = possible_results.iter().map(|(w, e, o)| (w.clone(), *e, *o)).collect();
    let possible_waters_energies: Vec<f64> = possible_results.iter().map(|(_, e, _)| *e).collect();
    // let oxygen_energies: Vec<f64> = possible_results.iter().map(|(_, _, o)| *o).collect();
    let choice = boltzmann_sampling(&possible_waters_energies);
    if choice.is_some() {
        let value = choice.unwrap();
        println!("Water's energy: {}", possible_waters_energies[value]);
        if boltzmann_acceptance_rejection(&possible_waters_energies[value], 
            &BOLTZMANN_ENERGY_CUTOFF, 
            &TEMPERATURE, 
            &BOLTZMANN_K) {
            for atom in possible_results[value].0.as_vec() {
                if atom.atom_type() == "OW" {
                    let oxygen_coords = atom.coords();
                    let a = atom.coords();
                    new_ap = a;
                    // println!("Before setting new water's energy: {:?}", grid.get(oxygen_coords[0], oxygen_coords[1], oxygen_coords[2]));
                    grid.set(oxygen_coords[0], oxygen_coords[1], oxygen_coords[2], possible_results[value].2);
                    // println!("After setting new water's energy: {:?}", grid.get(oxygen_coords[0], oxygen_coords[1], oxygen_coords[2]));
                }
                receptor_points.push(atom.clone());
            }

        }

    }
    // else {
    //     println!("Problem in sampling real water Boltzmann sampling");
    // }
    (receptor_points, grid, new_ap)
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
    // println!("Distribution: {}", distribution);
    // Too high energies
    vec![0.0; energies.len()]
}


/// This function returns the index of a randomly sampled energy based
/// on the Boltzmann probability distribution
pub fn boltzmann_sampling(energies: &Vec<f64>) -> Option<usize> {
    // println!("{:?}",energies);
    let probability_distribution = boltzmann_probabilities(energies);
    let sum: f64 = probability_distribution.iter().sum();
    if sum == 0. {
        return None;
    }
    let mut rng = thread_rng();
    let dist = WeightedIndex::new(&probability_distribution).unwrap();
    Some(dist.sample(&mut rng))
}

pub fn order_boltzmann_sampling(energies: &Vec<f64>) -> Vec<usize> {
    let probability_distribution = boltzmann_probabilities(energies);
    let sum: f64 = probability_distribution.iter().sum();
    let mut rng = thread_rng();
    let mut indices: Vec<usize> = (0..energies.len()).collect();
    let mut weights: Vec<f64> = probability_distribution.to_vec();
    let mut selected = Vec::new();

    for _ in 0..energies.len() {
        // Create the weighted index based on current weights
        let dist = WeightedIndex::new(&weights).expect("Probabilities must sum to a positive value");
        let idx = dist.sample(&mut rng);

        // Add the selected index to the result
        selected.push(indices[idx]);

        // Remove the chosen index and its weight
        indices.remove(idx);
        weights.remove(idx);
    }
    

    selected
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
