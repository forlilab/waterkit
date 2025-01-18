use std::f64;
use std::fmt::Debug;
use pyo3::prelude::*;

use crate::atom::Atom;
use crate::energy::spheric_energy;
use crate::spheric_probe::RADIUS_WATER;
use crate::utils::FloatRange;

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

trait CheckAccessibility {
    fn is_accessible(self, radius: &f64, sphere_coords: &[f64; 3]) -> bool;
}

// Implement the trait for &Vec<[f64; 3]>
impl CheckAccessibility for &Vec<[f64; 3]> {
    fn is_accessible(self, radius: &f64, sphere_coords: &[f64; 3]) -> bool {
        for point in self {
            if &euclidean_distance(&sphere_coords, &point) < radius {
                return false;
            }
        }
        true
    }
}

impl CheckAccessibility for &Vec<Atom> {
    fn is_accessible(self, radius: &f64, sphere_coords: &[f64; 3]) -> bool {
        for point in self {
            if &euclidean_distance(&sphere_coords, &point.coords()) < radius {
                return false;
            }
        }
        true
    }
}

// A generic function to accept any type that implements PrintValue
fn is_accessible<T: CheckAccessibility + Debug>(value: T, radius: &f64, sphere_coords: &[f64; 3]) -> bool {
    value.is_accessible(radius, sphere_coords)
}

#[pyfunction]
pub fn roll_sphere(surface_points: Vec<[f64; 3]>,
                   step_size: f64) -> Vec<[f64; 3]> {

    let mut trajectory = Vec::new();
    let surface_points_cloned = surface_points.clone();
    
    for point in surface_points.iter() {
        // Start the probe at the surface point
        for d_radius in FloatRange::new(0.0, RADIUS_WATER*2.0, 0.7) {
            let probe_center = [point[0]+d_radius, point[1] + d_radius, point[2] + d_radius];
            // Check if the probe is accessible at the initial position
            if is_accessible(&surface_points_cloned, &d_radius, &probe_center) {
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

                        if is_accessible(&surface_points_cloned, &d_radius, &candidate_position) {
                            trajectory.push(candidate_position);
                        }
                    }
                }
            }

        }
    }
    trajectory
}


#[pyfunction]
pub fn roll_sphere_and_compute_energies(surface_points: Vec<Atom>,
                                        step_size: f64) -> (Vec<f64>, Vec<[f64; 3]>) {

    let mut energies = Vec::new();
    let mut trajectories = Vec::new();

    let surface_points_cloned = surface_points.clone();
    
    for point in surface_points.iter() {
        // Start the probe at the surface point
        let point_coords = point.coords();
        for d_radius in FloatRange::new(0.0, RADIUS_WATER*2.0, 0.7) {
            let probe_center = [point_coords[0] + d_radius, 
                point_coords[1] + d_radius, 
                point_coords[2] + d_radius];
            // Check if the probe is accessible at the initial position
            if is_accessible(&surface_points_cloned, &d_radius, &probe_center) {
                // Compute the energy
                energies.push(spheric_energy(&surface_points_cloned, &probe_center));
                trajectories.push(probe_center);
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

                        if is_accessible(&surface_points_cloned, &d_radius, &probe_center) {
                            // Compute the energy
                            energies.push(spheric_energy(&surface_points_cloned, &candidate_position));
                            trajectories.push(candidate_position);
                        }
                    }
                }
            }

        }
    }
    (energies, trajectories) 
}