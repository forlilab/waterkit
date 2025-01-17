use std::f64;
use pyo3::prelude::*;

use crate::spheric_probe::Sphere;

#[derive(Debug, Clone, Copy)]
#[pyclass]
pub struct Point3D {
    x: f64,
    y: f64,
    z: f64
}

#[pymethods]
impl Point3D {
    #[new]
    pub fn new(x: f64, y: f64, z: f64) -> Point3D {
        let point = Self {
            x: x,
            y: y,
            z: z,
        };
        point
    }

    /// Updates the coordinates of the 3D point
    pub fn update_coords(&mut self, new_coords: Vec<f64>) {
        self.x = new_coords[0];
        self.y = new_coords[1];
        self.z = new_coords[2];
    }

    /// Computes the Euclidean distance between two points.
    pub fn distance(&self, other: &Point3D) -> f64 {
        ((self.x - other.x).powi(2) + (self.y - other.y).powi(2) + (self.z - other.z).powi(2)).sqrt()
    }

    /// Subtracts two points to form a vector.
    pub fn subtract(&self, other: &Point3D) -> Point3D {
        Point3D {
            x: self.x - other.x,
            y: self.y - other.y,
            z: self.z - other.z,
        }
    }

    /// Normalizes a vector to have a magnitude of 1.
    pub fn normalize(&self) -> Point3D {
        let magnitude = self.distance(&Point3D {x: 0.0, y: 0.0, z: 0.0});
        Point3D {
            x: self.x / magnitude,
            y: self.y / magnitude,
            z: self.z / magnitude,
        }
    }

    /// Scales a vector by a scalar value.
    pub fn scale(&self, scalar: f64) -> Point3D {
        Point3D {
            x: self.x * scalar,
            y: self.y * scalar,
            z: self.z * scalar,
        }
    }

    /// Adds two vecotrs.
    pub fn add(&self, other: &Point3D) -> Point3D {
        Point3D {
            x: self.x + other.x,
            y: self.y + other.y,
            z: self.z + other.z,
        }
    }

    pub fn as_vec(&self) -> Vec<f64> {
        vec![self.x, self.y, self.z]
    }
}

#[pyfunction]
pub fn roll_sphere_on_surface(mut sphere: Sphere, 
                              surface_points: Vec<Point3D>, 
                              steps: usize, 
                              step_size: f64) -> Vec<Point3D> {
    
    let mut trajectory = vec![sphere.coords().clone()];
    
    for _ in 0..steps {
        // Find the closest surface point
        let mut closest_point = None;
        let mut min_distance = f64::INFINITY;

        for point in surface_points.iter() {
            let distance = sphere.coords().distance(&point) - sphere.radius();
            if distance < min_distance {
                min_distance = distance;
                closest_point = Some(point);
            }
        }

        // ensure a closest point was found
        if let Some(closest_point) = closest_point {
            // Compute the direction of movement (towards the surface gradient)
            let direction = closest_point
                .subtract(&sphere.coords())
                .normalize()
                .scale(step_size);

            // Update the sphere's position
            sphere.update_coords_from_point3d(sphere.coords().add(&direction));
            trajectory.push(sphere.coords().clone());
        }
        else {
            break; // No valid closest point, stop the simulation
        }
    }
    trajectory
}

/// Check if a probe sphere is accessible at a given position.
/// A position is accessible if the sphere does not overlap with any surface atom.
pub fn is_accessible(sphere: &Sphere, sphere_coords: &Point3D, surface_points: &Vec<Point3D>) -> bool {
    for point in surface_points {
        if &sphere_coords.distance(&point) < sphere.radius() {
            return false;
        }
    }
    true
}

pub fn sum_points(p1: &Point3D, p2: &Point3D) -> Point3D {
    let v1 = p1.as_vec();
    let v2 = p2.as_vec();

    Point3D {
        x: v1[0] + v2[0], 
        y: v1[1] + v2[1], 
        z: v1[2] + v2[2]
    }
}

// Add random number generator to pick random x,y,z to start from for the initial probe position.
#[pyfunction]
pub fn roll_sphere(surface_points: Vec<Point3D>,
                   step_size: f64) -> Vec<Point3D> {

    let mut sphere = Sphere::new(vec![0.0, 0.0, 0.0]);
    sphere.update_coords(vec![0.0, 1.0, sphere.radius().clone()]);
    
    let mut trajectory = Vec::new();
    let surface_points_cloned = surface_points.clone();
    let sphere_coords = sphere.coords().clone();

    for point in surface_points.iter() {
        // Start the probe at the surface point
        let probe_center = sum_points(&sphere_coords, point);
        // Check if the probe is accessible at the initial position
        if !is_accessible(&sphere, &probe_center, &surface_points_cloned) {
            continue;
        }

        // Roll the sphere by moving it in a grid-like manner around the initial point
        for dx in [-step_size, 0.0, step_size] {
            for dy in [-step_size, 0.0, step_size] {
                for dz in [-step_size, 0.0, step_size] {
                    if dx == 0.0 && dy == 0.0 && dz == 0.0 {
                        continue;
                    }
                    let new_point = Point3D::new(dx, dy, dz);

                    let candidate_position = sum_points(&probe_center, &new_point);

                    if is_accessible(&sphere, &candidate_position, &surface_points_cloned) {
                        trajectory.push(candidate_position);
                    }
                }
            }
        }

    }
    trajectory 
}