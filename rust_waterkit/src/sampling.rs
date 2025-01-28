use crate::grid::{Grid3D, GridPoint};
use crate::utils::*;
use crate::atom::Atom;
use crate::water::WaterMolecule;
use crate::energy::{energy_for_real_water, spheric_energy};
use ndarray::Array1;
use rayon::prelude::*;
use rand::prelude::*;
use rand::distributions::WeightedIndex;

// Probably need to move everything to f32

pub fn _optimize_placement_order_grid(grid: &Grid3D, anchor_points: &Vec<[f64; 3]>) -> Vec<usize> {
    let mut energies: Vec<f64> = Vec::new();
    for ap in anchor_points {
        let neighbors = grid.get_neighbor_for_point(ap);
        let min_energy = neighbors
            .into_iter()
            .map(|x| x.energy)
            .collect::<Vec<f64>>()
            .into_iter()
            .min_by(|a, b| a.partial_cmp(b).unwrap()).unwrap();
        energies.push(min_energy);
    }
    let order = order_boltzmann_sampling(&energies);
    let mut decisions = Vec::new();
    if order.len() > 0 {
        for o in order {
            if boltzmann_acceptance_rejection(&energies[o], &BOLTZMANN_ENERGY_CUTOFF, &TEMPERATURE, &BOLTZMANN_K) {
                decisions.push(o);
            }
        }
        return decisions;
    }

    Vec::new()
}

pub fn sample_with_order(mut grid: Grid3D, mut receptor_points: Vec<Atom>, anchor_points: Vec<[f64; 3]>, water_configurations: &Vec<[f64; 6]> ) -> (Vec<Atom>, Grid3D, Vec<[f64; 3]>){
    let mut new_aps = Vec::new();
    let ap_order = _optimize_placement_order_grid(&grid, &anchor_points); 
    println!("\nRounds to do: {}\n", ap_order.len());
    for order in ap_order {

        let anchor_point = anchor_points[order];

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
                println!("Accepted oxygen's energy: {}", energies[index]);
                (receptor_points, new_ap) = sample_real_waters(&trajectories[index], water_configurations, receptor_points);
                if new_ap != [0.0, 0.0, 0.0] {
                    // println!("After placing: {:?}", grid.get(new_ap[0], new_ap[1], new_ap[2]));
                    new_aps.push(new_ap);
                    grid = update_grid_energies(&receptor_points, grid)
                }
            }
            else {
                println!("Something went wrong with Boltzmann sampling 2!");
            }
        }
        else {
            println!("Something went wrong in the Boltzmann sampling!");
        }
    }
    // grid = update_grid_energies(&receptor_points, grid, &new_aps);
    println!("Waters placed: {}", new_aps.len());
    (receptor_points, grid, new_aps)
}


/// This is the main sampling engine.
/// The idea is to find all neighboring points
/// for each AP and run Metropolis MC on those points
/// to determine a good one. Sample real waters for that point
/// then move on updating the anchor points list with the new atoms
/// and repeat until no more atoms available within the 
/// distance threshold of 12. Angstrom.
pub fn sample(mut grid: Grid3D, mut receptor_points: Vec<Atom>, anchor_points: Vec<[f64; 3]>, water_configurations: &Vec<[f64; 6]> ) -> (Vec<Atom>, Grid3D, Vec<[f64; 3]>){
    let anchor_points_to_iter = anchor_points.clone();
    let mut new_aps = Vec::new();
    for anchor_point in anchor_points_to_iter {
        let shell_points = grid.get_neighbor_for_point(&anchor_point);
        // println!("Possible points: {}", shell_points.len());
        let mut energies = Vec::new();
        let mut trajectories = Vec::new();
        // println!("# of Shell points for {:?}: {}", anchor_point, shell_points.len());
        for point in shell_points {
            energies.push(point.energy);
            trajectories.push(point.coords);
        }
        
        let mc_index = monte_carlo_sampling(&energies);
        let mut new_ap = [0.0; 3];
        // let index = mc_index.unwrap();
        let index = mc_index;
        // println!("Energy of the selected point: {}", &energies[index]);
        if boltzmann_acceptance_rejection(&energies[index], &BOLTZMANN_ENERGY_CUTOFF, &TEMPERATURE, &BOLTZMANN_K) {
            println!("\nAccepted oxygen's energy: {}", energies[index]);
            (receptor_points, new_ap) = sample_real_waters(&trajectories[index], water_configurations, receptor_points);
            if new_ap != [0.0, 0.0, 0.0] {
                // println!("After placing: {:?}", grid.get(new_ap[0], new_ap[1], new_ap[2]));
                new_aps.push(new_ap);
                grid = update_grid_energies(&receptor_points, grid);
            }
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

pub fn update_grid_energies(
    receptor_points: &Vec<Atom>,
    mut grid: Grid3D,) -> Grid3D {
    
    grid.all_points_mut()
        .par_iter_mut()
        // .filter(|point| !to_exclude.contains(&point.coords))
        .for_each(|point| {
            let energy = spheric_energy(receptor_points, &point.coords);
            point.energy = energy
        });
    grid
}

pub fn sample_real_waters(oxygen_position: &[f64; 3], 
    water_configurations: &Vec<[f64; 6]>,
    mut receptor_points: Vec<Atom>) -> (Vec<Atom>, [f64; 3]) {
    // Let's parallelize
    // let possible_results: Vec<(WaterMolecule, f64)> = water_configurations
    //     .into_par_iter()
    //     .map(|configuration| {
    //         // H1 in position
    //         let h1_coords: [f64; 3] = [
    //             configuration[0] + oxygen_position[0],
    //             configuration[1] + oxygen_position[1],
    //             configuration[2] + oxygen_position[2],
    //         ];
    //         // H2 in position
    //         let h2_coords: [f64; 3] = [
    //             configuration[3] + oxygen_position[0],
    //             configuration[4] + oxygen_position[1],
    //             configuration[5] + oxygen_position[2],
    //         ];

    //         // Create water molecule
    //         let water = WaterMolecule::new(h1_coords, h2_coords, oxygen_position.clone());

    //         // Compute energy
    //         let energy_value = energy_for_real_water(&receptor_points, &water.as_vec());
    //         (water, energy_value)
    //     })
    //     .collect();

    let mut possible_waters = Vec::new();
    let mut possible_waters_energies = Vec::new();
    for configuration in water_configurations {
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
            let energy_value = energy_for_real_water(&receptor_points, &water.as_vec());
            possible_waters.push(water);
            possible_waters_energies.push(energy_value);
    }
    let mut new_ap: [f64; 3] = [0.0; 3];
    // let possible_waters_energies: Vec<f64> = possible_results.iter().map(|(_, e)| *e).collect();
    let energies_g_0 = possible_waters_energies.iter().filter(|w| w < &&0.0).collect::<Vec<&f64>>(); 
    println!("{:?}", energies_g_0.len());
    let choice = monte_carlo_sampling(&possible_waters_energies);

    let value = choice;
    // println!("Water's energy: {}", possible_waters_energies[value]);
    if boltzmann_acceptance_rejection(&possible_waters_energies[value], 
        &BOLTZMANN_ENERGY_CUTOFF, 
        &TEMPERATURE, 
        &BOLTZMANN_K) {
        for atom in possible_waters[value].as_vec() {
            if atom.atom_type() == "OW" {
                let oxygen_coords = atom.coords();
                new_ap = oxygen_coords;
            }
            receptor_points.push(atom.clone());
        }

    }
    (receptor_points, new_ap)
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