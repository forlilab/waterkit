use crate::utils::*;
use crate::atom::Atom;
use crate::water::WaterMolecule;
use crate::energy::energy;
use ndarray::Array1;
use rand::prelude::*;
use rand::distributions::WeightedIndex;

pub fn sample_real_waters(oxygen_position: &[f64; 3], 
    water_configurations: &Vec<[f64; 6]>,
    mut map: Vec<Atom>,
    mut waters_map: Vec<Atom>) -> (Vec<Atom>, Vec<Atom>) {
    // Want to update the map when selected the new water
    let mut possible_waters: Vec<WaterMolecule> = Vec::new();
    let mut possible_waters_energies: Vec<f64> = Vec::new();
    let mut possible_waters_coords: Vec<[f64; 3]> = Vec::new();
    
    for configuration in water_configurations {
        let h1_coords: [f64; 3] = [configuration[0] + oxygen_position[0], 
            configuration[1] + oxygen_position[1], 
            configuration[2] + oxygen_position[2]];
        let h2_coords: [f64; 3] = [configuration[3] + oxygen_position[0], 
            configuration[4] + oxygen_position[1], 
            configuration[5] + oxygen_position[2]];
        let water: WaterMolecule = WaterMolecule::new(h1_coords, h2_coords, oxygen_position.clone());
        let h: [f64; 3] = water.as_vec()[1].coords();
        possible_waters.push(water);
        possible_waters_coords.push(h);
        possible_waters_energies.push(energy(&map, &possible_waters.last().unwrap().as_vec()));
    }
    let choice = boltzmann_sampling(&possible_waters_energies);
    if boltzmann_acceptance_rejection(&possible_waters_energies[choice], 
        &BOLTZMANN_ENERGY_CUTOFF, 
        &300.0, 
        &BOLTZMANN_K) {
        for atom in possible_waters[choice].as_vec() {
            waters_map.push(atom.clone());
            map.push(atom.clone());
        }
    }
    (map, waters_map)
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
