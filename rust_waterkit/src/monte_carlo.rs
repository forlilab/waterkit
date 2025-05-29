use rand::prelude::*;
use rand::distributions::WeightedIndex;
use crate::{consts::{BOLTZMANN_K, TEMPERATURE}, energy};

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

pub fn sa_acceptance_rejection(
    new_energies: &f64,
    old_energies: &f64,
    temperature: &f64
) -> bool {
    if new_energies <= old_energies {
        // Accept if new solution is better (lower cost)
        true
    } else {
        // Calculate acceptance probability for worse solution
        let delta_cost = new_energies - old_energies;
        let acceptance_probability = (-delta_cost / temperature).exp();
        // Accept with probability based on temperature and cost difference
        random::<f64>() < acceptance_probability
    }
}
