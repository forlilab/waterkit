use rand::prelude::*;
use rand::distributions::WeightedIndex;
use crate::consts::{BOLTZMANN_K, TEMPERATURE};

pub fn boltzmann_probabilities(energies: &[f64]) -> Vec<f64> {
    // Compute the Boltzmann factor for each energy
    let boltzmann_factors: Vec<f64> = energies.iter()
        .map(|&energy| (-energy / (BOLTZMANN_K * TEMPERATURE)).exp())
        .collect();

    let sum: f64 = boltzmann_factors.iter().sum();
    if sum == 0.0 {
        return vec![0.0; energies.len()]; // Return uniform distribution if sum is zero
    }
    boltzmann_factors.into_iter()
        .map(|factor| factor / sum)
        .collect()
}


pub fn boltzmann_choices(energies: &[f64], num_samples: Option<usize>) -> Vec<usize> {
    let mut rng = thread_rng();
    
    if num_samples.is_some() {
        let size = num_samples.unwrap();
        let mut sampled_indices = Vec::with_capacity(size);
        let mut remaining_indices: Vec<usize> = (0..energies.len()).collect();
        let mut remaining_energies: Vec<f64> = energies.to_vec();
        for _ in 0..size {
            // Compute the Boltzmann probabilities for the remaining energies
            let probabilities = boltzmann_probabilities(&remaining_energies);
        
            // Create a weighted distribution using the probabilities
            let dist = WeightedIndex::new(&probabilities);
            if dist.is_ok() {
        
                // Sample an index from the remaining indices
                let sampled_index = dist.unwrap().sample(&mut rng);
            
                // Add the corresponding original index to the result
                sampled_indices.push(remaining_indices[sampled_index]);
            
                // Remove the sampled energy and index from the remaining lists
                remaining_indices.remove(sampled_index);
                remaining_energies.remove(sampled_index);
            } else {
                break;
            }
        }
        
        sampled_indices
    } else {
        // Compute the Boltzmann probabilities for the remaining energies
        let probabilities = boltzmann_probabilities(&energies);
    
        // Create a weighted distribution using the probabilities
        let dist = WeightedIndex::new(&probabilities);

        if dist.is_ok() {
            // Sample an index from the remaining indices
            let sampled_index = dist.unwrap().sample(&mut rng);
            vec![sampled_index]
        } else {
            return vec![];
        }
    }
}

fn inverted_boltzmann_probabilities(energies: &[f64], e_opt: f64, sigma: f64) -> Vec<f64> {
    energies.iter()
        .map(|&e| (-((e_opt - e).powi(2)) / sigma.powi(2)).exp()) // Compute probability
        .collect()
}


pub fn inverted_boltzmann_choices(energies: &[f64], num_samples: Option<usize>) -> Vec<usize> {
    let mut rng = thread_rng();
    
    if num_samples.is_some() {
        let size = num_samples.unwrap();
        let mut sampled_indices = Vec::with_capacity(size);
        let mut remaining_indices: Vec<usize> = (0..energies.len()).collect();
        let mut remaining_energies: Vec<f64> = energies.to_vec();
        for _ in 0..size {
            let probabilities = inverted_boltzmann_probabilities(&remaining_energies, -1., 0.5);
        
            // Create a weighted distribution using the probabilities
            let dist = WeightedIndex::new(&probabilities);
            if dist.is_ok() {
        
                // Sample an index from the remaining indices
                let sampled_index = dist.unwrap().sample(&mut rng);
            
                // Add the corresponding original index to the result
                sampled_indices.push(remaining_indices[sampled_index]);
            
                // Remove the sampled energy and index from the remaining lists
                remaining_indices.remove(sampled_index);
                remaining_energies.remove(sampled_index);
            } else {
                break;
            }
        }
        
        sampled_indices
    } else {
        // Compute the Boltzmann probabilities for the remaining energies
        let probabilities = boltzmann_probabilities(&energies);
    
        // Create a weighted distribution using the probabilities
        let dist = WeightedIndex::new(&probabilities);

        if dist.is_ok() {
            // Sample an index from the remaining indices
            let sampled_index = dist.unwrap().sample(&mut rng);
            vec![sampled_index]
        } else {
            return vec![];
        }
    }
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


// pub fn monte_carlo_sampling(energies: &Vec<f64>) -> usize {
//     let mut rng = rand::thread_rng();

//     // Step 1: Calculate Boltzmann weights
//     let boltzmann_weights: Vec<f64> = energies
//         .iter()
//         .map(|&e| (-e / (BOLTZMANN_K * TEMPERATURE)).exp())
//         .collect();

//     // Step 2: Normalize weights to probabilities
//     let weight_sum: f64 = boltzmann_weights.iter().sum();
//     let probabilities: Vec<f64> = boltzmann_weights.iter().map(|&w| w / weight_sum).collect();

//     // Step 3: Build cumulative distribution
//     let mut cdf: Vec<f64> = Vec::with_capacity(probabilities.len());
//     let mut cumulative = 0.0;
//     for &p in &probabilities {
//         cumulative += p;
//         cdf.push(cumulative);
//     }

//     // // Step 4: Sample states based on CDF
//     // let mut samples = Vec::with_capacity(num_samples);
//     // for _ in 0..num_samples {
//     let mut chosen_index = 0;
//     let random_value = rng.gen::<f64>(); // Random value between 0 and 1
//     if let Some((index, _)) = cdf.iter().enumerate().find(|&(_, &v)| v >= random_value) {
//         // samples.push(index);
//         chosen_index = index;
//     }
//     chosen_index
// }


// fn monte_carlo_sampling_without_replacement(
//     energies: &[f64],
//     temperature: f64,
//     num_samples: usize,
// ) -> Vec<usize> {
//     let k_boltzmann = 1.0; // Set Boltzmann constant to 1.0 for simplicity (adjust as needed)
//     let mut rng = rand::thread_rng();

//     // Step 1: Calculate Boltzmann weights
//     let mut boltzmann_weights: Vec<f64> = energies
//         .iter()
//         .map(|&e| (-e / (k_boltzmann * temperature)).exp())
//         .collect();

//     // Ensure we don't request more samples than available states
//     let total_states = boltzmann_weights.len();
//     let num_samples = num_samples.min(total_states);

//     let mut samples = Vec::with_capacity(num_samples);

//     for _ in 0..num_samples {
//         // Step 2: Normalize weights to probabilities
//         let weight_sum: f64 = boltzmann_weights.iter().sum();
//         let probabilities: Vec<f64> = boltzmann_weights.iter().map(|&w| w / weight_sum).collect();

//         // Step 3: Build cumulative distribution
//         let mut cdf: Vec<f64> = Vec::with_capacity(probabilities.len());
//         let mut cumulative = 0.0;
//         for &p in &probabilities {
//             cumulative += p;
//             cdf.push(cumulative);
//         }

//         // Step 4: Sample a state based on the CDF
//         let random_value = rng.gen::<f64>(); // Random value between 0 and 1
//         if let Some((index, _)) = cdf.iter().enumerate().find(|&(_, &v)| v >= random_value) {
//             samples.push(index);

//             // Step 5: Remove the selected state
//             boltzmann_weights[index] = 0.0; // Set weight to 0 to exclude from future selections
//         }
//     }

//     samples
// }