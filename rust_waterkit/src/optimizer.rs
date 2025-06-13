use crate::consts;
use crate::consts::BOLTZMANN_K;
use crate::consts::TEMPERATURE;
use crate::energy;
use crate::geometry;
use crate::atom::Atom;
use crate::grid::Grid3D;
use crate::grid::ProbeType;
use crate::monte_carlo;
use crate::energy::energy_for_real_water;
use crate::utils;
use crate::water::WaterMolecule;
use crate::waterkit_system;
use crate::waterkit_system::System;

use core::f64;
use std::collections::HashMap;
use std::f64::consts::PI;
use std::thread::current;
use itertools::max;
use rand::prelude::*;

/// The main idea behind the optimizer is that we want to 
/// find the waters for which a position optimization is necessary.
/// The positions to optimize are chosen using Boltzmann wheighted
/// and then for each water a Monte Carlo sampling where water's
/// are sampled by rotating around the oxygen in every direction.
/// The expected result is that the water would maximize the  
/// # of H bonds between both upper and lower layers.

fn translate_oxygen(oxygen_coords: [f64; 3], h1_coords: [f64; 3], h2_coords: [f64; 3], max_disp: f64) -> [[f64; 3]; 3] {
    let mut rng = rand::thread_rng();
    let displacement = [rng.gen_range(-max_disp..max_disp), 
        rng.gen_range(-max_disp..max_disp),
        rng.gen_range(-max_disp..max_disp)];

    let new_oxygen = geometry::sum_points(&oxygen_coords, &displacement);
    let new_h1 = geometry::sum_points(&h1_coords, &displacement);
    let new_h2 = geometry::sum_points(&h2_coords, &displacement);
    [new_oxygen, new_h1, new_h2]
}

fn rotate_hydrogens(oxygen_coords: [f64; 3], h1_coords: [f64; 3], h2_coords: [f64; 3]) -> [[f64; 3]; 2] {
    let mut rng = rand::thread_rng();
    let axis = geometry::normalize(&[rng.gen(), rng.gen(), rng.gen()]);
    // let angle = rng.gen_range(-max_angle..max_angle);
    let angle = rng.gen_range(0.0..PI); // Full rotational sampling [0, 2π]
    let cos_theta = angle.cos();
    let sin_theta = angle.sin();

    let k_cross = |v: [f64; 3]| geometry::cross(&axis, &v);
    let k_dot_v = |v: [f64; 3]| geometry::scale_point(&axis, &geometry::dot(&axis, &v));

    let rotate = |v: [f64; 3]| {
        let v_rel = geometry::subtract_points(&v, &oxygen_coords);
        let term1 = geometry::scale_point(&v_rel, &cos_theta);
        let term2 = geometry::scale_point(&k_cross(v_rel), &sin_theta);
        let term3 = geometry::scale_point(&k_dot_v(v_rel), &(1.0 - cos_theta));
        geometry::sum_points(&&geometry::sum_points(&&geometry::sum_points(&term1, &term2), &term3), &oxygen_coords)
    };


    [rotate(h1_coords), rotate(h2_coords)]
}

fn get_energy(oxygen_pos: [f64; 3], h1_pos: [f64; 3], h2_pos: [f64; 3], grid: &Grid3D) -> f64 {
    // Need to interpolate the oxygen too since we are sampling small movements for this atom too 
    let lj_oxygen = grid.trilinear_interpolation(oxygen_pos, ProbeType::OW).unwrap_or(f64::INFINITY);

    let electrostatics_h1 = grid.trilinear_interpolation(h1_pos, ProbeType::HW);
    let electrostatics_h2 = grid.trilinear_interpolation(h2_pos, ProbeType::HW);
    let electrostatics_oxygen = grid.trilinear_interpolation(oxygen_pos, ProbeType::HW);
    if electrostatics_h1.is_some() && electrostatics_h2.is_some() && electrostatics_oxygen.is_some() {
        let energy_value = lj_oxygen
            + electrostatics_oxygen.unwrap() * consts::OXYGEN_W_Q_TIP3PFB
            + electrostatics_h1.unwrap() * consts::HYDROGEN_W_Q_TIP3PFB
            + electrostatics_h2.unwrap() * consts::HYDROGEN_W_Q_TIP3PFB;
        return energy_value;
    }
    f64::INFINITY
}

pub fn optimize(water: &mut Vec<Atom>, system_atoms: &Vec<Atom>, grid: &mut Grid3D) -> WaterMolecule {   
    let res_number = water[0].residue_number;
    let mut update_grids = false;
    
    // Extract initial coordinates
    let mut original_oxygen_coords = water[0].coords();
    let mut original_h1_coords = water[1].coords();
    let mut original_h2_coords = water[2].coords();

    // Monte Carlo parameters
    let num_steps = 1000;
    let mut max_disp = 0.2;
    let mut accepted_moves = 0;
    
    // Early stopping parameters
    let early_stop_threshold = 1e-4; // Minimum energy improvement required
    let max_no_improve_steps = 50; // If no improvement in X steps, stop early
    let mut no_improve_counter = 0;
    
    // Compute initial energy once
    let old_atoms = WaterMolecule::new(original_oxygen_coords, original_h1_coords, original_h2_coords, "".to_string(), res_number);
    let mut old_energy = energy_for_real_water(system_atoms, &old_atoms.as_vec());
    // let mut old_energy = get_energy(original_oxygen_coords, original_h1_coords, original_h2_coords, &grid);
    // grid.remove_points(&water);

    for step in 0..num_steps {
        // Propose new move
        let translation = translate_oxygen(original_oxygen_coords, original_h1_coords, original_h2_coords, max_disp);
        let rotation = rotate_hydrogens(translation[0], translation[1], translation[2]);

        // Create new water molecule
        let new_atoms = WaterMolecule::new(translation[0], rotation[0], rotation[1], "".to_string(), res_number);

        // Compute new energy
        let new_energy = energy_for_real_water(system_atoms, &new_atoms.as_vec());
        // let new_energy = get_energy(translation[0], rotation[0], rotation[1], &grid);

        // Compute energy difference (avoids redundant computation)
        let delta_energy = new_energy - old_energy;

        // Metropolis acceptance criterion
        if monte_carlo::boltzmann_acceptance_rejection(&new_energy, &old_energy, &TEMPERATURE, &BOLTZMANN_K) {
            // Accept the move
            update_grids = true;
            original_oxygen_coords = translation[0];
            original_h1_coords = rotation[0];
            original_h2_coords = rotation[1];
            old_energy = new_energy;  // Update energy
            accepted_moves += 1;
            no_improve_counter = 0; // Reset counter since we improved
        } else {
            no_improve_counter += 1;
        }

        // **Early stopping:** If no improvement in `max_no_improve_steps`, stop
        if no_improve_counter >= max_no_improve_steps {
            break;
        }

        // **Early stopping:** If energy change is below the threshold, stop
        if delta_energy.abs() < early_stop_threshold {
            break;
        }

        // **Adaptive step size adjustment every 10 steps**
        if step % 10 == 0 {
            let acceptance_rate = accepted_moves as f64 / (step + 1) as f64;
            if acceptance_rate < 0.3 {
                max_disp *= 0.9;  // Reduce step size
            } else if acceptance_rate > 0.7 {
                max_disp *= 1.1;  // Increase step size
            }
        }
    }

    let optimized_water = WaterMolecule::new(original_oxygen_coords, original_h1_coords, original_h2_coords, "".to_string(), res_number);

    // Update grid if needed
    if update_grids {
        grid.remove_points(&water);
        water[0].set_coords(original_oxygen_coords);
        water[1].set_coords(original_h1_coords);
        water[2].set_coords(original_h2_coords);
        grid.update_energies(&optimized_water.as_vec());    
    }
    // grid.update_energies(&optimized_water.as_vec());

    optimized_water
}

pub fn optimize_using_grids(water_molecule: &mut WaterMolecule, grid: &mut Grid3D, num_steps: i32, temp: f64) -> bool {   
    let mut accepted = false;
    let water = water_molecule.as_vec();
    // let mut update_grids = false;
    
    // Extract initial coordinates
    let mut original_oxygen_coords = water[0].coords();
    let mut original_h1_coords = water[1].coords();
    let mut original_h2_coords = water[2].coords();

    // Monte Carlo parameters
    // let num_steps = 1000;
    let mut max_disp = 0.2;
    let mut accepted_moves = 0;
    
    // Early stopping parameters
    let early_stop_threshold = 1e-4; // Minimum energy improvement required
    let max_no_improve_steps = 50; // If no improvement in X steps, stop early
    let mut no_improve_counter = 0;
    
    // Compute initial energy once
    grid.remove_points(&water_molecule.as_vec());
    let mut old_energy = get_energy(original_oxygen_coords, original_h1_coords, original_h2_coords, &grid);
    // println!("Old {old_energy}");

    for step in 0..num_steps {
        // Propose new move
        let translation = translate_oxygen(original_oxygen_coords, original_h1_coords, original_h2_coords, max_disp);
        let rotation = rotate_hydrogens(translation[0], translation[1], translation[2]);

        // Compute new energy
        let new_energy = get_energy(translation[0], rotation[0], rotation[1], &grid);

        // Compute energy difference (avoids redundant computation)
        let delta_energy = new_energy - old_energy;

        // Metropolis acceptance criterion
        if monte_carlo::boltzmann_acceptance_rejection(&new_energy, &old_energy, &temp, &BOLTZMANN_K) {
            accepted = true;
            // Accept the move
            original_oxygen_coords = translation[0];
            original_h1_coords = rotation[0];
            original_h2_coords = rotation[1];

            old_energy = new_energy;  // Update energy
            // println!("New {old_energy}");
            accepted_moves += 1;
            no_improve_counter = 0; // Reset counter since we improved
        // }
        } else {
            accepted = false;
            no_improve_counter += 1;
        }

        // **Early stopping:** If no improvement in `max_no_improve_steps`, stop
        if no_improve_counter >= max_no_improve_steps {
            break;
        }

        // **Early stopping:** If energy change is below the threshold, stop
        if delta_energy.abs() < early_stop_threshold {
            break;
        }

        // **Adaptive step size adjustment every 10 steps**
        if step % 10 == 0 {
            let acceptance_rate = accepted_moves as f64 / (step + 1) as f64;
            if acceptance_rate < 0.3 {
                max_disp *= 0.9;  // Reduce step size
            } else if acceptance_rate > 0.7 {
                max_disp *= 1.1;  // Increase step size
            }
        }
    }

    // let optimized_water = WaterMolecule::new(original_oxygen_coords, original_h1_coords, original_h2_coords, "".to_string(), res_number);
    water_molecule.update_coords(original_oxygen_coords, original_h1_coords, original_h2_coords);
    grid.update_energies(&water_molecule.as_vec());

    accepted
}



////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// // Simulated annealing
pub struct SimulatedAnnealing {
    pub system: System,
    pub waters: Vec<WaterMolecule>,
    pub waters_by_layer: HashMap<usize, Vec<WaterMolecule>>,
    pub frames: Vec<Vec<WaterMolecule>>,
    temperature: f64,
    temp_min: f64,
    cooling_rate: f64,
    cutoff: f64, // Distance cutoff for neighbor interactions
    final_temp_max_iter: usize,
}

impl<'a> SimulatedAnnealing {
    pub fn new(
        system: System,
        waters: Vec<WaterMolecule>,
        initial_temp: f64,
        temp_min: f64,
        cooling_rate: f64,
        cutoff: f64,
    ) -> Self {
        let mut waters_by_layer: HashMap<usize, Vec<WaterMolecule>> = HashMap::new();
        let final_temp_max_iter = 1000;
        let frames = Vec::with_capacity(final_temp_max_iter/1000);
        Self {
            system,
            waters,
            waters_by_layer,
            frames,
            temperature: initial_temp,
            temp_min,
            cooling_rate,
            cutoff,
            final_temp_max_iter,
        }
    }

    fn calculate_energy(&self, water: &WaterMolecule, neighbors: &[usize]) -> f64 {
        let water_atoms = water.as_vec();
        let mut e_lj = 0.0;
        let mut e_elec = 0.0;
        for &idx in neighbors {
            let neighbor_atom = &self.system.atoms[idx];
            for water_atom in &water_atoms {
                if neighbor_atom.residue_number() == water_atom.residue_number {
                    continue
                }
                // Calculate distance avoiding division by 0
                let distance = f64::max(geometry::euclidean_distance(&neighbor_atom.coords(), &water_atom.coords()), 1e-8_f64);
                if neighbor_atom.atom.atom_type() != &"HW".to_string() && water_atom.atom_type() != &"HW".to_string() {
                    let lj_energy = energy::lennard_jones_rmin_half(neighbor_atom.atom.epsilon(),
                        water_atom.epsilon(), 
                        distance,
                        neighbor_atom.atom.rmin_half(),
                        water_atom.rmin_half());
                    e_lj += lj_energy;
                    // println!("LJ: {lj_energy}");
                }
                if consts::USE_DIELECTRIC {
                    let electrostatics_energy = energy::dielectric(neighbor_atom.atom.charge(), water_atom.charge(), distance);
                    e_elec += electrostatics_energy;
                } else {
                    let electrostatics_energy = energy::coulomb_energy(neighbor_atom.atom.charge(), water_atom.charge(), distance);
                    e_elec += electrostatics_energy;
                }
                // println!("Q: {electrostatics_energy}");
            }
        }
        e_elec + e_lj
    }

    // Perturb a water molecule (translate and rotate)
    fn perturb_water(&self, water: &WaterMolecule, delta: f64, rotation_delta: f64) -> WaterMolecule {
        let mut rng = rand::thread_rng();
        let axis = geometry::normalize(&[rng.gen(), rng.gen(), rng.gen()]);
        let mut new_water = water.clone();
        

        // Small random translation (e.g., max 0.2 Å in each direction)
        // let delta = 0.5;
        let trans = [
            rng.gen_range(-delta..delta),
            rng.gen_range(-delta..delta),
            rng.gen_range(-delta..delta),
        ];

        let mut new_oxygen_coords = new_water.oxygen.coords();
        let mut new_h1_coords = new_water.hydrogen_1.coords();
        let mut new_h2_coords = new_water.hydrogen_2.coords();

        new_oxygen_coords = geometry::sum_points(&new_oxygen_coords, &trans);
        new_h1_coords = geometry::sum_points(&new_h1_coords, &trans);
        new_h2_coords = geometry::sum_points(&new_h2_coords, &trans);

        // let angle = rng.gen_range(0.0..rotation_delta);
        let angle = rng.gen_range(-rotation_delta..rotation_delta).to_radians();
        let cos_theta = angle.cos();
        let sin_theta = angle.sin();
    
        let k_cross = |v: [f64; 3]| geometry::cross(&axis, &v);
        let k_dot_v = |v: [f64; 3]| geometry::scale_point(&axis, &geometry::dot(&axis, &v));
    
        let rotate = |v: [f64; 3]| {
            let v_rel = geometry::subtract_points(&v, &new_oxygen_coords);
            let term1 = geometry::scale_point(&v_rel, &cos_theta);
            let term2 = geometry::scale_point(&k_cross(v_rel), &sin_theta);
            let term3 = geometry::scale_point(&k_dot_v(v_rel), &(1.0 - cos_theta));
            geometry::sum_points(&&geometry::sum_points(&&geometry::sum_points(&term1, &term2), &term3), &new_oxygen_coords)
        };

        new_h1_coords = rotate(new_h1_coords);
        new_h2_coords = rotate(new_h2_coords);

        new_water.update_coords(new_oxygen_coords, new_h1_coords, new_h2_coords);
        new_water
    }

    // Update the system with new water coordinates
    fn update_system(&mut self, water_idx: usize, new_water: WaterMolecule) {
        let water_atoms = new_water.as_vec();
        let water_coords: [[f64; 3]; 3] = [water_atoms[0].coords(), water_atoms[1].coords(), water_atoms[2].coords()];
        let original_atoms = self.waters[water_idx].as_vec();

        // Assuming water atoms in self.system.atoms are contiguous and in order: O, H1, H2
        self.system.update_water_position(water_idx, water_coords);

        // Update the water in the waters vector
        self.waters[water_idx] = new_water;
    }

    pub fn run(&mut self) -> f64 { 
        let mut acceptance_rate = 0;
        let mut rng = rand::thread_rng();
        let mut cnt = 0;
        let mut final_temp_cnt = 0;

        let equilibrate = false;
        let production_temp = 300.0;
        let mut start_equilibration = false;
        let mut equilibration_done = false;
        let mut accepted_production = 0;
        let mut max_displacement = 0.5;
        let mut rotational_displacement = 130.0;
        let mut acceptance_vec = Vec::new();
        let mut end_sa = 0;

        // while self.temperature > self.temp_min || final_temp_cnt < self.final_temp_max_iter {
        while final_temp_cnt < self.final_temp_max_iter {
            // Randomly select a water molecule
            let water_idx = rng.gen_range(0..self.waters.len());
            let current_water = &self.waters[water_idx];
            let base_idx = water_idx * 3;
            let exclude = [base_idx, base_idx + 1, base_idx + 2];

            // Get current neighbors and energy
            let water_positions = [
                current_water.oxygen.coords(),
                current_water.hydrogen_1.coords(),
                current_water.hydrogen_2.coords(),
            ];
            let neighbors = self.system.get_neighbors(water_positions[0], self.cutoff, &exclude);
            let current_energy = self.calculate_energy(current_water, &neighbors);

            // Remove the waters from the tree to be able to avoid clashes

            // Perturb the water molecule
            let new_water = self.perturb_water(current_water, max_displacement, rotational_displacement);
            let new_positions = [
                new_water.oxygen.coords(),
                new_water.hydrogen_1.coords(),
                new_water.hydrogen_2.coords(),
            ];


            let new_neighbors = self.system.get_neighbors(new_positions[0], self.cutoff, &exclude);
            let new_energy = self.calculate_energy(&new_water, &new_neighbors);

            // Acceptance criterion
            // println!("Old energy: {}", current_energy);
            // println!("New energy: {}\n", new_energy);
            let delta_energy = new_energy - current_energy;
            // if delta_energy < 0.0 {
            // println!("Displacement: old_energy = {}, new_energy = {}, delta_u = {}", current_energy, new_energy, delta_energy);

            // We are still in the Simulated Annealing, the acceptance is different
            if !equilibration_done {
                if delta_energy < 0.0 {
                // if monte_carlo::sa_acceptance_rejection(&new_energy, &current_energy, &self.temperature) {
                    // if current_energy > 0. {
                    //     println!("Old energy: {current_energy}\nNew energy: {new_energy}\nAccepted!\n\n");
                    // }
                    if equilibration_done && self.temperature >= production_temp {
                        accepted_production += 1;
                    }

                    acceptance_rate += 1;
                    self.update_system(water_idx, new_water);
                }
            } else {
                // We are in the production step -> normal Monte Carlo
                if monte_carlo::boltzmann_acceptance_rejection(&new_energy, 
                    &current_energy, 
                    &self.temperature, 
                    &consts::BOLTZMANN_K) {
                        if equilibration_done && self.temperature >= production_temp {
                        accepted_production += 1;
                    }

                    acceptance_rate += 1;
                    self.update_system(water_idx, new_water);
                }
            }

            // Cool down
            if cnt % 100 == 0 && !equilibration_done {
                if self.temperature > self.temp_min {
                    self.temperature *= self.cooling_rate;
                }
            }

            acceptance_vec.push((cnt, (acceptance_rate as f64/cnt as f64) * 100.0));
            cnt += 1;

            if self.temperature < self.temp_min {
                if equilibrate {
                    // Heat up
                    if !start_equilibration{
                        println!("Reached min temp (T={}). Now slowly heating up.", self.temperature);
                        start_equilibration = true;
                    }

                    if self.temperature < production_temp && start_equilibration {
                        // println!("Heating up to T: {}", self.temperature);
                        // Slowly heat up
                        self.temperature += 5.0;
                        if self.temperature >= production_temp {
                            equilibration_done = true;
                        }
                    }
                } else {
                    end_sa = cnt;
                    equilibration_done = true;
                    self.temperature = production_temp;
                }
            }

            // Production
            if self.temperature >= production_temp && equilibration_done {
                // println!("Equilibration done, starting production at T {} and step {}", self.temperature, final_temp_cnt);
                if final_temp_cnt % 100000 == 0 {
                    println!("Production steps: {}", final_temp_cnt);
                    self.frames.push(self.waters.clone());
                }
                final_temp_cnt += 1;
                
                // **Adaptive step size adjustment every 10 steps**
                if final_temp_cnt % 100 == 0 {
                    let acceptance_rate_production = accepted_production as f64 / (final_temp_cnt) as f64;
                    println!("Acceptance rate production: {}", acceptance_rate_production);
                    println!("Step size: {}", max_displacement);
                    if acceptance_rate_production < 0.02 {
                        max_displacement *= 0.9;  // Reduce step size
                        rotational_displacement *= 0.9;
                    } else if acceptance_rate_production > 0.08 {
                        max_displacement *= 1.1;  // Increase step size
                        rotational_displacement *= 1.1;
                    }
                }
            } 
            // println!("Steps: {cnt}");
        }
        utils::plot_acceptance_rate(acceptance_vec, cnt, end_sa);
        // self.frames.push(self.waters.clone());
        acceptance_rate as f64/cnt as f64
        // println!("Acceptance rate: {}%", acceptance_rate as f64/100.0);
    }

}

