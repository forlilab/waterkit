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

pub fn normalize(p1: &[f64; 3]) -> [f64; 3] {
    let magnitude = euclidean_distance(p1, &[0.0; 3]);
    let mut retvalue = [0.0; 3];
    retvalue[0] = p1[0] / magnitude;
    retvalue[1] = p1[1] / magnitude;
    retvalue[2] = p1[2] / magnitude;
    retvalue
}