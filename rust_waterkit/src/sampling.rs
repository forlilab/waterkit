use crate::utils::*;
use ndarray::Array1;
use rand::prelude::*;
use rand::distributions::WeightedIndex;

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
