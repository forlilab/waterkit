use pyo3::prelude::*;

pub const RMIN_HALF_WATER: f64 = 1.7682; // TIP3P 
pub const RADIUS_WATER: f64 = 1.4;
pub const EPSILON_WATER: f64 = 0.1521; // According to AMBER