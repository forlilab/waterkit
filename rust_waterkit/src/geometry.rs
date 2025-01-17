use std::f64;
use pyo3::prelude::*;

use crate::spheric_probe::Sphere;


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

// Add random number generator to pick random x,y,z to start from for the initial probe position.
#[pyfunction]
pub fn roll_sphere(surface_points: Vec<[f64; 3]>,
                   step_size: f64) -> Vec<[f64; 3]> {

    let mut sphere = Sphere::new([0.0, 0.0, 0.0]);
    sphere.update_coords([0.0, 1.0, sphere.radius().clone()]);
    
    let mut trajectory = Vec::new();
    let surface_points_cloned = surface_points.clone();
    let sphere_coords = sphere.coords().clone();

    for point in surface_points.iter() {
        // Start the probe at the surface point
        let probe_center = sum_points(&sphere_coords, point);
        // Check if the probe is accessible at the initial position
        if !is_accessible(sphere.radius(), &probe_center, &surface_points_cloned) {
            continue;
        }

        // Roll the sphere by moving it in a grid-like manner around the initial point
        for dx in [-step_size, 0.0, step_size] {
            for dy in [-step_size, 0.0, step_size] {
                for dz in [-step_size, 0.0, step_size] {
                    if dx == 0.0 && dy == 0.0 && dz == 0.0 {
                        continue;
                    }
                    let new_point = [dx, dy, dz];

                    let candidate_position = sum_points(&probe_center, &new_point);

                    if is_accessible(sphere.radius(), &candidate_position, &surface_points_cloned) {
                        trajectory.push(candidate_position);
                    }
                }
            }
        }

    }
    trajectory 
}