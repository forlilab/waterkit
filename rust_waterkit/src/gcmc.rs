use core::panic;
use std::{collections::{HashMap, VecDeque}, fs::File, thread::current};

use kiddo::float::kdtree::KdTree;
use rand::{distributions::Uniform, distributions::WeightedIndex, prelude::Distribution, Rng};

use crate::{consts, energy, geometry, water::{self, WaterMolecule}, waterkit_system::System};

// Constants
const KB: f64 = 0.0019872041; // Boltzmann constant in kcal/mol/K
const TEMPERATURE: f64 = 298.15; // Temperature in K
const KT: f64 = KB * TEMPERATURE;
const STANDARD_VOLUME: f64 = 30.345; // Volume per water molecule in bulk
const BETA: f64 = 1.0 / (KB * TEMPERATURE);
const CHEMICAL_POTENTIAL: f64 = -6.09; // Chemical potential in kcal/mol
const LAMBDA: f64 = 0.145; // Thermal de Broglie wavelength in Å (simplified)
const NUM_STEPS: usize = 50000; // Total Monte Carlo steps
const WINDOW_SIZE: usize = 100; // Window size for rolling acceptance rate
const NUM_FRAMES: usize = 100; // Number of frames to save
const FRAME_INTERVAL: usize = NUM_STEPS / NUM_FRAMES; // Save a frame every frame_interval steps

// Vector operations for [f64; 3]
fn norm(v: &[f64; 3]) -> f64 {
    (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt()
}

fn subtract(v1: &[f64; 3], v2: &[f64; 3]) -> [f64; 3] {
    [v1[0] - v2[0], v1[1] - v2[1], v1[2] - v2[2]]
}

fn add(v1: &[f64; 3], v2: &[f64; 3]) -> [f64; 3] {
    [v1[0] + v2[0], v1[1] + v2[1], v1[2] + v2[2]]
}

// Check if a position is within 10 Å of any receptor atom
fn is_in_hydration_shell(pos: &[f64; 3], x_min: f64, x_max: f64, y_min: f64, y_max: f64, z_min: f64, z_max: f64) -> bool {
    (pos[0] >= x_min && pos[0] <= x_max) && (pos[1] >= y_min && pos[1] <= y_max) && (pos[2] >= z_min && pos[2] <= z_max) 
}

pub struct GCMC {
    pub system: System,
    pub waters: Vec<WaterMolecule>,
    pub frames: Vec<Vec<WaterMolecule>>,
    cutoff: f64, // Distance cutoff for neighbor interactions
    x_min: f64,
    x_max: f64,
    y_min: f64,
    y_max: f64,
    z_min: f64,
    z_max: f64,
}


impl GCMC {
    pub fn new(system: System, 
        cutoff: f64,
        x_min: f64, 
        x_max: f64, 
        y_min: f64, 
        y_max: f64, 
        z_min: f64, 
        z_max: f64) -> Self {
            let mut waters = Vec::new();
            let mut frames = Vec::new();

            Self {
                system,
                waters,
                frames,
                cutoff,
                x_min,
                x_max,
                y_min,
                y_max,
                z_min,
                z_max,
            }
        }

    fn randomize_water(&self,
        water: &WaterMolecule, 
        rng: &mut rand::rngs::ThreadRng) -> WaterMolecule {
        let mut new_water = water.clone();
        let oxygen_pos = new_water.oxygen.coords();
        let mut candidate_hydrogen_1_pos = water.hydrogen_1.coords();
        let mut candidate_hydrogen_2_pos = water.hydrogen_2.coords();
        
        let angle= rng.gen_range(-180_f64..180_f64).to_radians();
        let cos_theta = angle.cos();
        let sin_theta = angle.sin();
        let axis = geometry::normalize(&[rng.gen(), rng.gen(), rng.gen()]);
        let k_cross = |v: [f64; 3]| geometry::cross(&axis, &v);
        let k_dot_v = |v: [f64; 3]| geometry::scale_point(&axis, &geometry::dot(&axis, &v));
    
        let rotate = |v: [f64; 3]| {
            let v_rel = geometry::subtract_points(&v, &oxygen_pos);
            let term1 = geometry::scale_point(&v_rel, &cos_theta);
            let term2 = geometry::scale_point(&k_cross(v_rel), &sin_theta);
            let term3 = geometry::scale_point(&k_dot_v(v_rel), &(1.0 - cos_theta));
            geometry::sum_points(&&geometry::sum_points(&&geometry::sum_points(&term1, &term2), &term3), &oxygen_pos)
        };

        candidate_hydrogen_1_pos = rotate(candidate_hydrogen_1_pos);
        candidate_hydrogen_2_pos = rotate(candidate_hydrogen_2_pos);
        new_water.update_coords(oxygen_pos, candidate_hydrogen_1_pos, candidate_hydrogen_2_pos);
        new_water
    }
    // Propose a translation move, ensuring new position is in hydration shell
    // Returns a Vec<[f64; 3]> that represents oxygen, h1 and h2 positions
    fn propose_perturbation(
        &self,
        water: &WaterMolecule, 
        rng: &mut rand::rngs::ThreadRng) -> Option<WaterMolecule> {
        let mut new_water = water.clone();
        let delta = [
            Uniform::from(-0.5..0.5).sample(rng),
            Uniform::from(-0.5..0.5).sample(rng),
            Uniform::from(-0.5..0.5).sample(rng),
        ];
        let candidate_oxygen_pos = add(&water.oxygen.coords(), &delta);
        if is_in_hydration_shell(&candidate_oxygen_pos, 
            self.x_min, self.x_max, 
            self.y_min, self.y_max, 
            self.z_min, self.z_max) {
            let mut candidate_hydrogen_1_pos = add(&water.hydrogen_1.coords(), &delta);
            let mut candidate_hydrogen_2_pos = add(&water.hydrogen_2.coords(), &delta);
            
            let angle= rng.gen_range(-180_f64..180_f64).to_radians();
            let cos_theta = angle.cos();
            let sin_theta = angle.sin();
            let axis = geometry::normalize(&[rng.gen(), rng.gen(), rng.gen()]);
            let k_cross = |v: [f64; 3]| geometry::cross(&axis, &v);
            let k_dot_v = |v: [f64; 3]| geometry::scale_point(&axis, &geometry::dot(&axis, &v));
        
            let rotate = |v: [f64; 3]| {
                let v_rel = geometry::subtract_points(&v, &candidate_oxygen_pos);
                let term1 = geometry::scale_point(&v_rel, &cos_theta);
                let term2 = geometry::scale_point(&k_cross(v_rel), &sin_theta);
                let term3 = geometry::scale_point(&k_dot_v(v_rel), &(1.0 - cos_theta));
                geometry::sum_points(&&geometry::sum_points(&&geometry::sum_points(&term1, &term2), &term3), &candidate_oxygen_pos)
            };

            candidate_hydrogen_1_pos = rotate(candidate_hydrogen_1_pos);
            candidate_hydrogen_2_pos = rotate(candidate_hydrogen_2_pos);
            new_water.update_coords(candidate_oxygen_pos, candidate_hydrogen_1_pos, candidate_hydrogen_2_pos);
            Some(new_water)
        } else {
            None // Reject if outside hydration shell
        }
    }

    // Propose an insertion move, generating a position within user-specified bounds
    fn propose_insertion(
        &self,
        water: &WaterMolecule,
        rng: &mut rand::rngs::ThreadRng,
    ) -> Option<WaterMolecule> {
        let mut new_water = water.clone();
        let max_attempts = 1000; // Limit attempts to find valid position
        for _ in 0..max_attempts {
            let new_oxygen_coord = [
                Uniform::from(self.x_min..self.x_max).sample(rng),
                Uniform::from(self.y_min..self.y_max).sample(rng),
                Uniform::from(self.z_min..self.z_max).sample(rng),
            ];
            if is_in_hydration_shell(&new_oxygen_coord, 
                self.x_min, self.x_max, 
                self.y_min, self.y_max, 
                self.z_min, self.z_max) {
                let new_hydrogen_1_coord = add(&water.hydrogen_1.coords(), &new_oxygen_coord);
                let new_hydrogen_2_coord = add(&water.hydrogen_2.coords(), &new_oxygen_coord);
                new_water.update_coords(new_oxygen_coord, new_hydrogen_1_coord, new_hydrogen_2_coord);
                // println!("FOund water to inert");
                return Some(new_water);
            }
        }
        println!("No waters to insert found!");
        None // Failed to find valid position
    }

    // Propose a deletion move
    fn propose_deletion(&self, waters: &[WaterMolecule], rng: &mut rand::rngs::ThreadRng) -> (Vec<WaterMolecule>, usize) {
        let mut new_waters = waters.to_vec();
        let i = Uniform::from(0..waters.len()).sample(rng);
        new_waters.remove(i);
        (new_waters, i)
    }

        // Grand Canonical Monte Carlo simulation
    pub fn gcmc_simulation(&mut self, water_molecule: WaterMolecule, max_waters: usize, volume: f64) -> std::io::Result<Vec<Vec<WaterMolecule>>> {
        // ADAMS parameter
        // let B = CHEMICAL_POTENTIAL * BETA + (volume / LAMBDA.powi(3)).ln();
        let B = CHEMICAL_POTENTIAL * BETA + (volume / STANDARD_VOLUME).ln();
        // println!("BETA: {}", BETA);
        // println!("B: {}\nB_equil: {}", B, B_equil);
        let receptor_atoms: Vec<crate::atom::Atom> = self.system.atoms.clone().into_iter().map(|a| a.atom).collect();
        
        // New approach to keep track of the atoms
        let mut system_atoms = receptor_atoms.clone();
        // let mut water_atoms_map = HashMap::new();

        println!("# Atoms: {}", receptor_atoms.len());
        let mut rng = rand::thread_rng();
        // let mut energy = 0.0; // Initial energy is zero
        let mut trans_history = VecDeque::with_capacity(WINDOW_SIZE);
        let mut insert_history = VecDeque::with_capacity(WINDOW_SIZE);
        let mut delete_history = VecDeque::with_capacity(WINDOW_SIZE);
        let mut trans_attempts = 0;
        let mut insert_attempts = 0;
        let mut delete_attempts = 0;
        let mut trans_accepts = 0;
        let mut insert_accepts = 0;
        let mut delete_accepts = 0;
        let mut frames = Vec::with_capacity(NUM_FRAMES);

        let mut cnt = 0;
        let mut step_cnt = 0;
        for step in 0..NUM_STEPS {
            // let step_start = std::time::Instant::now();
            // println!("# of atoms in the system at step {}: {}", step_cnt, system_atoms.len());
            step_cnt += 1;
            // Choose move type: 1/3 translation, 1/3 insertion, 1/3 deletion
            let mut water_molecule_copy = self.randomize_water(&water_molecule, &mut rng);
            let mut rng = rand::thread_rng();
            let move_type = rng.gen_range(0..=2);
            if move_type == 0 && !self.waters.is_empty() {
                // Translation move
                trans_attempts += 1;
                let water_idx = rng.gen_range(0..self.waters.len());
                let mut current_water = self.waters[water_idx].clone();
                let base_idx = water_idx * 3;
                let exclude = [base_idx, base_idx + 1, base_idx + 2];

                // Using single energy
                let old_energy = energy::energy_for_real_water(&current_water.as_vec(), &system_atoms);

                if let Some(new_water) = self.propose_perturbation(
                    &current_water, 
                    &mut rng) {
                    self.waters[water_idx] = new_water;
                    
                    // Using system's energy
                    // let system_energy = energy::get_system_energy(&self.waters, &receptor_atoms);
                    // let new_energy = system_energy.0 + system_energy.1;

                    // Using single energy
                    let new_energy = energy::energy_for_real_water(&self.waters[water_idx].as_vec(), &system_atoms);
                    let delta_e = new_energy - old_energy;
                    let acceptance_prob = (-BETA * delta_e).exp().min(1.0);

                    if Uniform::from(0.0..1.0).sample(&mut rng) < acceptance_prob {   
                        // println!("Displ accepted!");
                        let new_water_atoms = self.waters[water_idx].as_vec();
                        // let mut oxygen_changed = false;
                        // let mut h1_changed = false;
                        // let mut h2_changed = false;
                        system_atoms.iter_mut().for_each(|atom| {
                            // println!("Atom: {}, waters_atom: {}", atom.atom_id(), new_water_atoms[0].atom_id());
                            if atom.atom_id() == new_water_atoms[0].atom_id() {
                                atom.set_coords(new_water_atoms[0].coords());
                            } else if atom.atom_id() == new_water_atoms[1].atom_id() {
                                atom.set_coords(new_water_atoms[1].coords());
                            } else if atom.atom_id() == new_water_atoms[2].atom_id() {
                                atom.set_coords(new_water_atoms[2].coords());
                            }
                        });
                        // energy = new_energy;
                        trans_history.push_back(1);
                        trans_accepts += 1;
                    } else {
                        self.waters[water_idx] = current_water;
                        trans_history.push_back(0);
                    }
                } else {
                    trans_history.push_back(0); // Rejected due to leaving hydration shell
                }
            } else if move_type == 1 {
            // if move_type == 0 {
                // Insertion move
                insert_attempts += 1;
                if let Some(mut new_water) = self.propose_insertion(
                    &water_molecule_copy,
                    &mut rng,
                ) {
                    new_water.set_res_number(cnt);
                    let res_n = new_water.get_res_number();
                    let n = self.waters.len() as f64;
                    self.waters.push(new_water);
                    let water_idx = self.waters.len() -1;

                    // Using system's energy
                    // let system_energy = energy::get_system_energy(&self.waters, &receptor_atoms);
                    // let new_energy = system_energy.0 + system_energy.1;
                    // let delta_e = new_energy - energy + CHEMICAL_POTENTIAL;

                    // Using single energy
                    let new_energy = energy::energy_for_real_water(&system_atoms, &self.waters[water_idx].as_vec());
                    let delta_e = new_energy + CHEMICAL_POTENTIAL;

                    let acceptance_prob = ((1.0/(n+1.0)) * B.exp() * (-BETA * delta_e).exp()).min(1.0);

                    if Uniform::from(0.0..1.0).sample(&mut rng) < acceptance_prob {
                        // println!("Inserting water with res number: {}. # of waters in the system: {}", res_n, n);
                        // println!("New energy when inserting water {}: {} (dE: {})", res_n, new_energy, delta_e);
                        let base_idx = cnt * 3;
                        system_atoms.extend(self.waters[water_idx].as_vec());
                        // water_atoms_map.insert(res_n, [base_idx, base_idx+1, base_idx+2]);
                        // self.waters.push(new_water);

                        cnt += 1;
                        // energy = new_energy;
                        insert_history.push_back(1);
                        insert_accepts += 1;
                    } else {
                        self.waters.pop();
                        insert_history.push_back(0);
                    }
                } else {
                    insert_history.push_back(0); // Failed to find valid position
                }
            } else if move_type == 2 && !self.waters.is_empty() {
                // Deletion move
                delete_attempts += 1;
                let (new_waters, removed_water_idx) = self.propose_deletion(&self.waters, &mut rng);
                
                // Using system's energy
                // let system_energy = energy::get_system_energy(&new_waters, &receptor_atoms);
                // let new_energy = system_energy.0 + system_energy.1;

                // Using single energy
                let removed_water = self.waters[removed_water_idx].clone();
                let removed_water_energy = energy::energy_for_real_water(&system_atoms, &removed_water.as_vec());
                // Subtract the removed water energy to the previous energy
                let new_energy = -removed_water_energy;

                // let delta_e = new_energy - energy - CHEMICAL_POTENTIAL;
                let delta_e = new_energy - CHEMICAL_POTENTIAL;
                let n = self.waters.len() as f64;
                let acceptance_prob = (n * (-B).exp() * (-BETA * delta_e).exp()).min(1.0);

                if Uniform::from(0.0..1.0).sample(&mut rng) < acceptance_prob {
                    let removed_resnumber = removed_water.get_res_number();
                    self.waters = new_waters;
                    // water_atoms_map.remove(&removed_resnumber);
                    system_atoms.retain(|x| x.residue_number != removed_resnumber);

                    // energy = new_energy;
                    delete_history.push_back(1);
                    delete_accepts += 1;
                } else {
                    delete_history.push_back(0);
                }
            }
        }

        let system_energy = energy::get_system_energy(&self.waters, &receptor_atoms);
        println!("\n# of waters inserted: {}", self.waters.len());
        if trans_attempts > 0 {
            let trans_rate = trans_history.iter().sum::<u8>() as f64 / trans_history.len() as f64;
            println!("# of Translations accepted: {} out of {} attempts.", trans_accepts, trans_attempts);
            println!("Translation Acceptance Rate = {:.3}", trans_rate);
        }
        if insert_attempts > 0 {
            let insert_rate = insert_history.iter().sum::<u8>() as f64 / insert_history.len() as f64;
            println!("# of Insertions accepted: {} out of {} attempts.", insert_accepts, insert_attempts);
            println!("Insertion Acceptance Rate = {:.3}", insert_rate);
        }
        if delete_attempts > 0 {
            let delete_rate = delete_history.iter().sum::<u8>() as f64 / delete_history.len() as f64;
            println!("# of Deletions accepted: {} out of {} attempts.", delete_accepts, delete_attempts);
            println!("Deletion Acceptance Rate = {:.3}", delete_rate);
        }        
        println!("Insertions proposed in {} steps: {}", step_cnt, insert_attempts);
        println!("Deletions proposed in {} steps: {}", step_cnt, delete_attempts);
        println!("System Energy: {}", system_energy.0 + system_energy.1);
        frames.push(self.waters.clone());
        Ok(frames)
    }

    // // Grand Canonical Monte Carlo simulation
    // pub fn gcmc_simulation(&mut self, water_molecule: WaterMolecule, max_waters: usize, volume: f64) -> std::io::Result<Vec<Vec<WaterMolecule>>> {
    //     // ADAMS parameter
    //     // let B = CHEMICAL_POTENTIAL * BETA + (volume / LAMBDA.powi(3)).ln();
    //     let B = CHEMICAL_POTENTIAL * BETA + (volume / STANDARD_VOLUME).ln();
    //     // println!("BETA: {}", BETA);
    //     // println!("B: {}\nB_equil: {}", B, B_equil);
    //     let receptor_atoms: Vec<crate::atom::Atom> = self.system.atoms.clone().into_iter().map(|a| a.atom).collect();
        
    //     // New approach to keep track of the atoms
    //     let mut system_atoms = receptor_atoms.clone();
    //     // let mut water_atoms_map = HashMap::new();

    //     println!("# Atoms: {}", receptor_atoms.len());
    //     let mut rng = rand::thread_rng();
    //     let mut energy = 0.0; // Initial energy is zero
    //     let mut trans_history = VecDeque::with_capacity(WINDOW_SIZE);
    //     let mut insert_history = VecDeque::with_capacity(WINDOW_SIZE);
    //     let mut delete_history = VecDeque::with_capacity(WINDOW_SIZE);
    //     let mut trans_attempts = 0;
    //     let mut insert_attempts = 0;
    //     let mut delete_attempts = 0;
    //     let mut trans_accepts = 0;
    //     let mut insert_accepts = 0;
    //     let mut delete_accepts = 0;
    //     let mut frames = Vec::with_capacity(NUM_FRAMES);

    //     let mut cnt = 0;
    //     let mut step_cnt = 0;
    //     for step in 0..NUM_STEPS {
    //         // let step_start = std::time::Instant::now();
    //         // println!("# of atoms in the system at step {}: {}", step_cnt, system_atoms.len());
    //         step_cnt += 1;
    //         // Choose move type: 1/3 translation, 1/3 insertion, 1/3 deletion
    //         let mut water_molecule_copy = self.randomize_water(&water_molecule, &mut rng);
    //         let mut rng = rand::thread_rng();
    //         let move_type = rng.gen_range(0..=2);
    //         if move_type == 0 && !self.waters.is_empty() {
    //             // Translation move
    //             trans_attempts += 1;
    //             let water_idx = rng.gen_range(0..self.waters.len());
    //             let mut current_water = self.waters[water_idx].clone();
    //             let base_idx = water_idx * 3;
    //             let exclude = [base_idx, base_idx + 1, base_idx + 2];

    //             if let Some(new_water) = self.propose_perturbation(
    //                 &current_water, 
    //                 &mut rng) {
    //                 self.waters[water_idx] = new_water;
                    
    //                 // Using system's energy
    //                 let system_energy = energy::get_system_energy(&self.waters, &receptor_atoms);
    //                 let new_energy = system_energy.0 + system_energy.1;


    //                 // let new_energy = energy::energy_for_real_water(&new_water.as_vec(), &system_atoms);
    //                 let delta_e = new_energy - energy;
    //                 let acceptance_prob = (-BETA * delta_e).exp().min(1.0);

    //                 if Uniform::from(0.0..1.0).sample(&mut rng) < acceptance_prob {    
    //                     // let new_water_atoms = new_water.as_vec();
    //                     // let atoms_indices: &[usize; 3] = water_atoms_map.get(&new_water.get_res_number()).unwrap();
    //                     // for (enum_idx, idx) in atoms_indices.iter().enumerate() {
    //                     //     // println!("Before updating system_atoms: {:?}", system_atoms[*idx].coords());
    //                     //     system_atoms[*idx].set_coords(new_water_atoms[enum_idx].coords());
    //                     //     // println!("After updating system_atoms: {:?}", system_atoms[*idx].coords());
    //                     //     // println!("Expected coords: {:?}", new_water_atoms[enum_idx].coords());
    //                     // }
                        
    //                     energy = new_energy;
    //                     trans_history.push_back(1);
    //                     trans_accepts += 1;
    //                 } else {
    //                     self.waters[water_idx] = current_water;
    //                     trans_history.push_back(0);
    //                 }
    //             } else {
    //                 trans_history.push_back(0); // Rejected due to leaving hydration shell
    //             }
    //         } else if move_type == 1 {
    //         // if move_type == 0 {
    //             // Insertion move
    //             insert_attempts += 1;
    //             if let Some(mut new_water) = self.propose_insertion(
    //                 &water_molecule_copy,
    //                 &mut rng,
    //             ) {
    //                 new_water.set_res_number(cnt);
    //                 let res_n = new_water.get_res_number();
    //                 let n = self.waters.len() as f64;
    //                 self.waters.push(new_water);

    //                 // Using system's energy
    //                 let system_energy = energy::get_system_energy(&self.waters, &receptor_atoms);
    //                 let new_energy = system_energy.0 + system_energy.1;

    //                 // let new_energy = energy::energy_for_real_water(&system_atoms, &new_water.as_vec());
    //                 let delta_e = new_energy - energy + CHEMICAL_POTENTIAL;

    //                 // let acceptance_prob = (volume / ((n + 1.0) * LAMBDA.powi(3)))
    //                 //     * (-BETA * delta_e + BETA * CHEMICAL_POTENTIAL).exp()
    //                 //     .min(1.0);
    //                 let acceptance_prob = ((1.0/(n+1.0)) * B.exp() * (-BETA * delta_e).exp()).min(1.0);

    //                 if Uniform::from(0.0..1.0).sample(&mut rng) < acceptance_prob {
    //                     // println!("Inserting water with res number: {}. # of waters in the system: {}", res_n, n);
    //                     // println!("New energy when inserting water {}: {} (dE: {})", res_n, new_energy, delta_e);
    //                     // let idx = system_atoms.len();
    //                     // system_atoms.extend(new_water.as_vec());
    //                     // water_atoms_map.insert(new_water.get_res_number(), [idx, idx+1, idx+2]);
    //                     // self.waters.push(new_water);

    //                     cnt += 1;
    //                     energy = new_energy;
    //                     insert_history.push_back(1);
    //                     insert_accepts += 1;
    //                 } else {
    //                     self.waters.pop();
    //                     insert_history.push_back(0);
    //                 }
    //             } else {
    //                 insert_history.push_back(0); // Failed to find valid position
    //             }
    //         } else if move_type == 2 && !self.waters.is_empty() {
    //             // Deletion move
    //             delete_attempts += 1;
    //             let (new_waters, removed_water_idx) = self.propose_deletion(&self.waters, &mut rng);
    //             // let removed_water = self.waters[removed_water_idx].clone();
    //             // let removed_water_energy = energy::energy_for_real_water(&system_atoms, &removed_water.as_vec());
    //             // // Subtract the removed water energy to the previous energy
    //             // let new_energy = -removed_water_energy;
                
    //             // Using system's energy
    //             let system_energy = energy::get_system_energy(&new_waters, &receptor_atoms);
    //             let new_energy = system_energy.0 + system_energy.1;

    //             let delta_e = new_energy - energy - CHEMICAL_POTENTIAL;
    //             let n = self.waters.len() as f64;
    //             // let acceptance_prob = (n * LAMBDA.powi(3) / volume)
    //             //     * (-BETA * delta_e - BETA * CHEMICAL_POTENTIAL).exp()
    //             //     .min(1.0);
    //             let acceptance_prob = (n * (-B).exp() * (-BETA * delta_e).exp()).min(1.0);

    //             if Uniform::from(0.0..1.0).sample(&mut rng) < acceptance_prob {
    //                 // let removed_resnumber = removed_water.get_res_number();
    //                 self.waters = new_waters;
    //                 // let atoms_indices =  water_atoms_map.get(&removed_resnumber).unwrap();
    //                 // for idx in atoms_indices {
    //                 //     system_atoms.remove(*idx);
    //                 // }
    //                 // water_atoms_map.remove(&removed_resnumber);

    //                 energy = new_energy;
    //                 delete_history.push_back(1);
    //                 delete_accepts += 1;
    //             } else {
    //                 delete_history.push_back(0);
    //             }
    //         }

    //         // Compute rolling acceptance rates
    //         // if step % WINDOW_SIZE == 0 && step > 0 {
    //         //     if trans_attempts > 0 {
    //         //         let trans_rate = trans_history.iter().sum::<u8>() as f64 / trans_history.len() as f64;
    //         //         println!("Step {}: Translation Acceptance Rate = {:.3}", step + 1, trans_rate);
    //         //     }
    //         //     if insert_attempts > 0 {
    //         //         let insert_rate = insert_history.iter().sum::<u8>() as f64 / insert_history.len() as f64;
    //         //         println!("Step {}: Insertion Acceptance Rate = {:.3}", step + 1, insert_rate);
    //         //     }
    //         //     if delete_attempts > 0 {
    //         //         let delete_rate = delete_history.iter().sum::<u8>() as f64 / delete_history.len() as f64;
    //         //         println!("Step {}: Deletion Acceptance Rate = {:.3}", step + 1, delete_rate);
    //         //     }
    //         //     println!("Step {}: Number of Waters = {}", step + 1, waters.len());
    //         // }

    //         // Save frame
    //         // if (step + 1) % FRAME_INTERVAL == 0 {
    //         //     frames.push(waters.clone());
    //         //     // println!("Frame {} E: {}", step+1, energy);
    //         // }
    //         // let step_elapsed = step_start.elapsed();
    //         // println!("Time taken for 1 step: {} seconds", step_elapsed.as_secs_f64());
    //     }

    //     // let system_energy = energy::get_system_energy(&self.waters, &receptor_atoms);
    //     println!("\n# of waters inserted: {}", self.waters.len());
    //     if trans_attempts > 0 {
    //         let trans_rate = trans_history.iter().sum::<u8>() as f64 / trans_history.len() as f64;
    //         println!("# of Translations accepted: {} out of {} attempts.", trans_accepts, trans_attempts);
    //         println!("Translation Acceptance Rate = {:.3}", trans_rate);
    //     }
    //     if insert_attempts > 0 {
    //         let insert_rate = insert_history.iter().sum::<u8>() as f64 / insert_history.len() as f64;
    //         println!("# of Insertions accepted: {} out of {} attempts.", insert_accepts, insert_attempts);
    //         println!("Insertion Acceptance Rate = {:.3}", insert_rate);
    //     }
    //     if delete_attempts > 0 {
    //         let delete_rate = delete_history.iter().sum::<u8>() as f64 / delete_history.len() as f64;
    //         println!("# of Deletions accepted: {} out of {} attempts.", delete_accepts, delete_attempts);
    //         println!("Deletion Acceptance Rate = {:.3}", delete_rate);
    //     }        
    //     println!("Insertions proposed in {} steps: {}", step_cnt, insert_attempts);
    //     println!("Deletions proposed in {} steps: {}", step_cnt, delete_attempts);
    //     println!("System Energy: {}", energy);
    //     // println!("\tEww: {}", system_energy.0);
    //     // println!("\tEsw: {}", system_energy.1);
    //     frames.push(self.waters.clone());
    //     Ok(frames)
    // }
}