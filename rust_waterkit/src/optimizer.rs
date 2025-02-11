use crate::geometry;
use crate::grid::Grid3D;
use crate::water::WaterMolecule;

use std::f64::consts::PI;
use rand::prelude::*;

/// The main idea behind the optimizer is that we want to 
/// find the waters for which a position optimization is necessary.
/// The positions to optimize are chosen using Boltzmann wheighted
/// and then for each water a Monte Carlo sampling where water's
/// are sampled by rotating around the oxygen in every direction.
/// The expected result is that the water would maximize the  
/// # of H bonds between both upper and lower layers.

pub enum Perturbation {
    Translation,
    Rotation,
}

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

fn rotate_hydrogens(oxygen_coords: [f64; 3], h1_coords: [f64; 3], h2_coords: [f64; 3], max_angle: f64) -> [[f64; 3]; 2] {
    let mut rng = rand::thread_rng();
    let axis = geometry::normalize(&[rng.gen(), rng.gen(), rng.gen()]);
    let angle = rng.gen_range(-max_angle..max_angle);
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

pub fn optimize(water: &WaterMolecule, grid: &Grid3D) {
    let water_atoms = water.as_vec();
    let original_oxygen_coords = water_atoms[0].coords();
    let original_h1_coords = water_atoms[1].coords();
    let original_h2_coords = water_atoms[2].coords();
    
    // Monte Carlo parameters
    let num_steps = 1000;
    let beta = 1.0;
    let max_disp = 0.1;
    let max_angle = PI / 18.0;

    for _ in 0..num_steps {
        // Propose a move
        let new_atoms = translate_oxygen(original_oxygen_coords, original_h1_coords, original_h2_coords, max_disp);
        let new_hydrogens = rotate_hydrogens(new_atoms[0], new_atoms[1], new_atoms[2], max_angle);
        
        // Evaluate energy change
    }
}

