use crate::consts;
use crate::consts::BOLTZMANN_K;
use crate::consts::TEMPERATURE;
use crate::geometry;
use crate::atom::Atom;
use crate::grid::Grid3D;
use crate::grid::ProbeType;
use crate::monte_carlo;
use crate::energy::energy_for_real_water;
use crate::water::WaterMolecule;

use core::f64;
use std::f64::consts::PI;
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
    let mut energy_value = f64::INFINITY;
    // Need to interpolate the oxygen too since we are sampling small movements for this atom too 
    let lj_oxygen = grid.trilinear_interpolation(oxygen_pos, ProbeType::OW);

    let electrostatics_h1 = grid.trilinear_interpolation(h1_pos, ProbeType::HW);
    let electrostatics_h2 = grid.trilinear_interpolation(h2_pos, ProbeType::HW);
    let electrostatics_oxygen = grid.trilinear_interpolation(oxygen_pos, ProbeType::HW);
    if lj_oxygen.is_some() && electrostatics_h1.is_some() && electrostatics_h2.is_some() && electrostatics_oxygen.is_some() {
        energy_value = lj_oxygen.unwrap() + (electrostatics_oxygen.unwrap() * consts::OXYGEN_W_Q) + (electrostatics_h1.unwrap()  * consts::HYDROGEN_W_Q) + (electrostatics_h2.unwrap() * consts::HYDROGEN_W_Q);
    }
    energy_value
}

// pub fn optimize(water: &WaterMolecule, waters_in_system: &Vec<Atom>, grid: &mut Grid3D) -> WaterMolecule {   
//     let res_number = water.get_res_number();
//     let mut rng = rand::thread_rng();
//     let mut update_grids = false;
//     let water_atoms = water.as_vec();
//     let mut original_oxygen_coords = water_atoms[0].coords();
//     let mut original_h1_coords = water_atoms[1].coords();
//     let mut original_h2_coords = water_atoms[2].coords();
    
//     // Monte Carlo parameters
//     let num_steps = 1000;
//     let max_disp = 0.1;
//     // let beta = TEMPERATURE * BOLTZMANN_K;

//     for _i in 0..num_steps {
//         // Propose a move
//         let translation = translate_oxygen(original_oxygen_coords, original_h1_coords, original_h2_coords, max_disp);
//         let rotation = rotate_hydrogens(translation[0], translation[1], translation[2]);

//         // Evaluate energy change
//         let old_energy = get_energy(original_oxygen_coords, original_h1_coords, original_h2_coords, grid);
//         let old_atoms = WaterMolecule::new(original_oxygen_coords, original_h1_coords, original_h2_coords, "".to_string(), res_number);
//         let old_energy = energy_for_real_water(waters_in_system, &old_atoms.as_vec());

//         // let new_energy = get_energy(translation[0], rotation[0], rotation[1], grid);
//         let new_atoms = WaterMolecule::new(translation[0], rotation[0], rotation[1], "".to_string(), res_number);
//         let new_energy = energy_for_real_water(waters_in_system, &new_atoms.as_vec());

//         println!("Old Energy: {old_energy}");
//         println!("New Energy: {new_energy}");
//         // Metropolis acceptance criterion
//         if monte_carlo::boltzmann_acceptance_rejection(&new_energy, &old_energy, &TEMPERATURE, &BOLTZMANN_K) {
//             println!("Accepted!!\n");
//             // Accept the move
//             update_grids = true;
//             original_oxygen_coords = translation[0];
//             original_h1_coords = rotation[0];
//             original_h2_coords = rotation[1];
//         }
//         else {
//             update_grids = false;
//         }
//     }

//     let optimized_water = WaterMolecule::new(original_oxygen_coords, original_h1_coords, original_h2_coords, "".to_string(), 1000000);
//     if update_grids {
//         grid.remove_points(&water.as_vec());
//         grid.update_energies(&optimized_water.as_vec());
//     }
//     optimized_water
// }

pub fn optimize(water: &mut Vec<Atom>, waters_in_system: &Vec<Atom>, grid: &mut Grid3D) -> WaterMolecule {   
    // first element is always oxygen, then the hydrogens!
    let res_number = water[0].residue_number;
    // let mut rng = rand::thread_rng();
    let mut update_grids = false;
    let mut original_oxygen_coords = water[0].coords();
    let mut original_h1_coords = water[1].coords();
    let mut original_h2_coords = water[2].coords();

    // println!("Original coordinates: {:?} {:?} {:?}", original_oxygen_coords, original_h1_coords, original_h2_coords);
    
    // Monte Carlo parameters
    let num_steps = 100;
    let max_disp = 0.2;

    for _i in 0..num_steps {
        // Propose a move
        let translation = translate_oxygen(original_oxygen_coords, original_h1_coords, original_h2_coords, max_disp);
        let rotation = rotate_hydrogens(translation[0], translation[1], translation[2]);

        // Evaluate energy change
        // let old_energy = get_energy(original_oxygen_coords, original_h1_coords, original_h2_coords, grid);
        let old_atoms = WaterMolecule::new(original_oxygen_coords, original_h1_coords, original_h2_coords, "".to_string(), res_number);
        let old_energy = energy_for_real_water(waters_in_system, &old_atoms.as_vec());

        // let new_energy = get_energy(translation[0], rotation[0], rotation[1], grid);
        let new_atoms = WaterMolecule::new(translation[0], rotation[0], rotation[1], "".to_string(), res_number);
        let new_energy = energy_for_real_water(waters_in_system, &new_atoms.as_vec());

        // println!("Old Energy: {old_energy}");
        // println!("New Energy: {new_energy}");
        // Metropolis acceptance criterion
        if monte_carlo::boltzmann_acceptance_rejection(&new_energy, &old_energy, &TEMPERATURE, &BOLTZMANN_K) {
            // println!("Accepted!!\n");
            // Accept the move
            update_grids = true;
            original_oxygen_coords = translation[0];
            original_h1_coords = rotation[0];
            original_h2_coords = rotation[1];
        }
        else {
            update_grids = false;
        }
    }

    let optimized_water = WaterMolecule::new(original_oxygen_coords, original_h1_coords, original_h2_coords, "".to_string(), 1000000);
    if update_grids {
        grid.remove_points(&water);

        water[0].set_coords(original_oxygen_coords);
        water[1].set_coords(original_h1_coords);
        water[2].set_coords(original_h2_coords);
        
        grid.update_energies(&optimized_water.as_vec());
    }

    // println!("After coordinates: {:?} {:?} {:?}", water[0].coords(), water[1].coords(), water[2].coords());
    optimized_water
}
