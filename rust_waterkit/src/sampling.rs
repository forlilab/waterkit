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

pub fn accept_reject_new_energies() {

}
