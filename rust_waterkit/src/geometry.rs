use std::f64;

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


// Function to calculate the angle between three points (in radians)
pub fn calculate_angle(a: &[f64; 3], b: &[f64; 3], c: &[f64; 3 ]) -> f64 {
    let ba = [a[0] - b[0], a[1] - b[1], a[2] - b[2]]; // Vector BA
    let bc = [c[0] - b[0], c[1] - b[1], c[2] - b[2]]; // Vector BC
    let dot_product = ba[0] * bc[0] + ba[1] * bc[1] + ba[2] * bc[2];
    let magnitude_ba = (ba[0] * ba[0] + ba[1] * ba[1] + ba[2] * ba[2]).sqrt();
    let magnitude_bc = (bc[0] * bc[0] + bc[1] * bc[1] + bc[2] * bc[2]).sqrt();
    (dot_product / (magnitude_ba * magnitude_bc)).acos()
}

// Function to normalize a vector
pub fn normalize(v: &[f64; 3]) -> [f64; 3] {
    let magnitude = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    [v[0] / magnitude, v[1] / magnitude, v[2] / magnitude]
}