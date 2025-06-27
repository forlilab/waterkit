use crate::{atom::Atom, gcmc::GCMC};

#[derive(Clone)]
pub struct ReplicaState {
    pub simulator: GCMC,
    pub num_waters: usize,  // Number of water molecules (N)
    pub energy: f64,        // Potential energy (U)
    pub positions: Vec<Atom>,  // Water molecule positions (for exchange)
}

// Compute exchange acceptance probability
pub fn exchange_probability(
    mu_i: f64, mu_j: f64,
    n_i: usize, n_j: usize,
    u_i: f64, u_j: f64,
    beta: f64,
) -> f64 {
    let delta = beta * ((mu_j - mu_i) * (n_j as f64 - n_i as f64) + (u_i - u_j));
    delta.exp().min(1.0)
}