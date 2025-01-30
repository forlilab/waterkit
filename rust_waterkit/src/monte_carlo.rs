use ndarray::Array1;
use rand::prelude::*;
use rand::distributions::WeightedIndex;
use crate::utils::{BOLTZMANN_K, TEMPERATURE};

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
    let mut selected: Vec<usize> = (0..energies.len()).collect();
    let dist = WeightedIndex::new(&probability_distribution).expect("Probabilities must sum to a positive value");

    for _ in 0..energies.len() {
        // Create the weighted index based on current weights
        let idx = dist.sample(&mut rng);
        // Add the selected index to the result
        if !selected.contains(&idx) {
            selected[idx] = idx;
        }
        else {
            selected[idx] = usize::MAX;
        }

    }
    
    selected
        .into_iter()
        .filter(|x| *x != usize::MAX)
        .collect()
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
    // println!("Random value: {} - p_acc: {}", random_values, p_acc);
    if random_values <= p_acc {
        // println!("Accepted even if not good!");
        return true;
    }

    false
}


pub fn monte_carlo_sampling(energies: &Vec<f64>) -> usize {
    let mut rng = rand::thread_rng();

    // Step 1: Calculate Boltzmann weights
    let boltzmann_weights: Vec<f64> = energies
        .iter()
        .map(|&e| (-e / (BOLTZMANN_K * TEMPERATURE)).exp())
        .collect();

    // Step 2: Normalize weights to probabilities
    let weight_sum: f64 = boltzmann_weights.iter().sum();
    let probabilities: Vec<f64> = boltzmann_weights.iter().map(|&w| w / weight_sum).collect();

    // Step 3: Build cumulative distribution
    let mut cdf: Vec<f64> = Vec::with_capacity(probabilities.len());
    let mut cumulative = 0.0;
    for &p in &probabilities {
        cumulative += p;
        cdf.push(cumulative);
    }

    // // Step 4: Sample states based on CDF
    // let mut samples = Vec::with_capacity(num_samples);
    // for _ in 0..num_samples {
    let mut chosen_index = 0;
    let random_value = rng.gen::<f64>(); // Random value between 0 and 1
    if let Some((index, _)) = cdf.iter().enumerate().find(|&(_, &v)| v >= random_value) {
        // samples.push(index);
        chosen_index = index;
    }
    chosen_index
}


fn monte_carlo_sampling_without_replacement(
    energies: &[f64],
    temperature: f64,
    num_samples: usize,
) -> Vec<usize> {
    let k_boltzmann = 1.0; // Set Boltzmann constant to 1.0 for simplicity (adjust as needed)
    let mut rng = rand::thread_rng();

    // Step 1: Calculate Boltzmann weights
    let mut boltzmann_weights: Vec<f64> = energies
        .iter()
        .map(|&e| (-e / (k_boltzmann * temperature)).exp())
        .collect();

    // Ensure we don't request more samples than available states
    let total_states = boltzmann_weights.len();
    let num_samples = num_samples.min(total_states);

    let mut samples = Vec::with_capacity(num_samples);

    for _ in 0..num_samples {
        // Step 2: Normalize weights to probabilities
        let weight_sum: f64 = boltzmann_weights.iter().sum();
        let probabilities: Vec<f64> = boltzmann_weights.iter().map(|&w| w / weight_sum).collect();

        // Step 3: Build cumulative distribution
        let mut cdf: Vec<f64> = Vec::with_capacity(probabilities.len());
        let mut cumulative = 0.0;
        for &p in &probabilities {
            cumulative += p;
            cdf.push(cumulative);
        }

        // Step 4: Sample a state based on the CDF
        let random_value = rng.gen::<f64>(); // Random value between 0 and 1
        if let Some((index, _)) = cdf.iter().enumerate().find(|&(_, &v)| v >= random_value) {
            samples.push(index);

            // Step 5: Remove the selected state
            boltzmann_weights[index] = 0.0; // Set weight to 0 to exclude from future selections
        }
    }

    samples
}