use pyo3::prelude::*;

const RADIUS_WATER: f64 = 1.4;
const EPSILON_WATER: f64 = 0.1521; // According to AMBER


#[derive(Debug, Clone)]
#[pyclass]
pub struct Sphere {
    center_coordinates: [f64; 3],
    radius: f64, 
    epsilon: f64,
}

#[pymethods]
impl Sphere {
    #[new]
    pub fn new(coordinates: [f64; 3]) -> Sphere {
        let sphere = Self {
            center_coordinates: coordinates,
            radius: RADIUS_WATER,
            epsilon: EPSILON_WATER,           
        };
        sphere
    }

    pub fn update_coords(&mut self, new_coordinates: [f64; 3]) {
        self.center_coordinates = new_coordinates
    }

    pub fn coords(&self) -> [f64; 3] {
        self.center_coordinates
    }

    pub fn radius(&self) -> &f64 {
        &self.radius
    } 

    pub fn epsilon(&self) -> &f64 {
        &self.epsilon
    }
}