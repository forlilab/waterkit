use std::f64;
use pyo3::prelude::*;

use crate::{spheric_probe::Sphere, spheric_probe::RADIUS_WATER, utils::FloatRange};

// Helper function to generate a random number in a range
fn rand_in_range(min: &f64, max: &f64) -> f64 {
    min + (max - min) * (rand::random::<f64>())
}

// Basic geometric operations on points
pub fn sum_points(p1: &[f64; 3], p2: &[f64; 3]) -> [f64; 3] {
    let mut retvalue = [0.0; 3];
    retvalue[0] = p1[0] + p2[0]; 
    retvalue[1] = p1[1] + p2[1]; 
    retvalue[2] = p1[2] + p2[2];
    retvalue
}

pub fn subtract_points(p1: &[f64; 3], p2: &[f64; 3]) -> [f64; 3] {
    let mut retvalue = [0.0; 3];
    retvalue[0] = p1[0] - p2[0]; 
    retvalue[1] = p1[1] - p2[1]; 
    retvalue[2] = p1[2] - p2[2];
    retvalue
}

pub fn euclidean_distance(p1: &[f64; 3], p2: &[f64; 3]) -> f64 {
    let distance = ((p1[0] - p2[0]).powi(2) + (p1[1] - p2[1]).powi(2) + (p1[2] - p2[2]).powi(2)).sqrt(); 
    distance
}

pub fn scale_point(p1: &[f64; 3], scalar: &f64) -> [f64; 3] {
    let mut retvalue = [0.0; 3];
    retvalue[0] = p1[0] * scalar;
    retvalue[1] = p1[1] * scalar;
    retvalue[2] = p1[2] * scalar;
    retvalue
}

pub fn normalize(p1: &[f64; 3]) -> [f64; 3] {
    let magnitude = euclidean_distance(p1, &[0.0; 3]);
    let mut retvalue = [0.0; 3];
    retvalue[0] = p1[0] / magnitude;
    retvalue[1] = p1[1] / magnitude;
    retvalue[2] = p1[2] / magnitude;
    retvalue
}

/// Check if a probe sphere is accessible at a given position.
/// A position is accessible if the sphere does not overlap with any surface atom.
pub fn is_accessible(radius: &f64, sphere_coords: &[f64; 3], surface_points: &Vec<[f64; 3]>) -> bool {
    for point in surface_points {
        if &euclidean_distance(&sphere_coords, &point) < radius {
            return false;
        }
    }
    true
}


/// Determine if the starting_point [0.0, 0.0, probe_radius] is accessible
/// otherwise check the surrounding to find a good starting point. 
// pub fn find_valid_start(point: &[f64; 3], radius: &f64, surface_points: &Vec<[f64; 3]>) -> Result<[f64; 3], String> {
//     let directions = [
//         [1.0, 0.0, 0.0],
//         [0.0, 1.0, 0.0],
//         [0.0, 0.0, 1.0],
//         [-1.0, 0.0, 0.0],
//         [0.0, -1.0, 0.0],
//         [0.0, 0.0, -1.0],
//     ];
    
//     for direction in &directions {
//         let probe_center = scale_point(&sum_points(point, direction), radius);
//         if is_accessible(radius, &probe_center, surface_points) {
//             return Ok(probe_center);
//         }
//     }

//     for _ in 0..100 {
//         let negative_radius = -radius;
//         let offset = [
//             rand_in_range(&negative_radius, radius),
//             rand_in_range(&negative_radius, radius),
//             rand_in_range(&negative_radius, radius),
//         ];

//         let probe_center = sum_points(point, &offset);
//         if is_accessible(radius, &probe_center, surface_points) {
//             return Ok(probe_center);
//         }
//     }

//     Err("No valid starting position found to start rolling the sphere on the surface".to_string())
// }

#[pyfunction]
pub fn roll_sphere(surface_points: Vec<[f64; 3]>,
                   step_size: f64) -> Vec<[f64; 3]> {

    let sphere = Sphere::new([0.0, 0.0, 0.0]);
    let mut trajectory = Vec::new();
    // match find_valid_start(&sphere.coords(), sphere.radius(), &surface_points) {
    //     Ok(valid_start) => {
    let surface_points_cloned = surface_points.clone();
    
    for point in surface_points.iter() {
        // if point == &[3.447000026702881, 15.916000366210938, 21.625] {
        // Start the probe at the surface point
        // let center = [0.0, 0.0, 0.0];
        for d_radius in FloatRange::new(0.0, RADIUS_WATER*2.0, 0.7) {
            let probe_center = [point[0]+d_radius, point[1] + d_radius, point[2] + d_radius];
            // Check if the probe is accessible at the initial position
            if is_accessible(&d_radius, &probe_center, &surface_points_cloned) {
                trajectory.push(probe_center);
            }

            // Roll the sphere by moving it in a grid-like manner around the initial point
            for dx in FloatRange::new(-step_size, step_size, 0.5) {
                for dy in FloatRange::new(-step_size, step_size, 0.5) {
                    for dz in FloatRange::new(-step_size, step_size, 0.5) {
                        if dx == 0.0 && dy == 0.0 && dz == 0.0 {
                            continue;
                        }
                        let new_point = [dx, dy, dz];

                        let candidate_position = sum_points(&probe_center, &new_point);

                        if is_accessible(&d_radius, &candidate_position, &surface_points_cloned) {
                            trajectory.push(candidate_position);
                        }
                    }
                }
                // }
            }

        }
    }
    //     },
    //     Err(err) => println!("Error: {}", err),
    // }
    trajectory
}