use pyo3::prelude::*;

use crate::geometry::Point3D;

const RADIUS_WATER: f64 = 1.4;
const EPSILON_WATER: f64 = 0.1521; // According to AMBER


#[derive(Debug, Clone)]
#[pyclass]
pub struct Sphere {
    center_coordinates: Point3D,
    radius: f64, 
    epsilon: f64,
}

#[pymethods]
impl Sphere {
    #[new]
    pub fn new(coordinates: Vec<f64>) -> Sphere {
        let sphere = Self {
            center_coordinates: Point3D::new(coordinates[0], coordinates[1], coordinates[2]),
            radius: RADIUS_WATER,
            epsilon: EPSILON_WATER,           
        };
        sphere
    }

    pub fn update_coords(&mut self, new_coordinates: Vec<f64>) {
        self.center_coordinates.update_coords(new_coordinates);
    }

    pub fn update_coords_from_point3d(&mut self, new_coordinates: Point3D) {
        self.center_coordinates.update_coords(new_coordinates.as_vec());
    }

    pub fn coords(&self) -> Point3D {
        self.center_coordinates
    }

    pub fn radius(&self) -> &f64 {
        &self.radius
    } 

    pub fn epsilon(&self) -> &f64 {
        &self.epsilon
    }
}