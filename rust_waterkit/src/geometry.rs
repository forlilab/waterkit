use std::f64;
use std::f64::consts::PI;

use rayon::vec;

// Basic geometric operations on points
pub fn vector(p1: &[f64; 3], p2: &[f64; 3]) -> [f64; 3] {
    subtract_points(p2, p1)
}

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
    let dx = p1[0] - p2[0];
    let dy = p1[1] - p2[1];
    let dz = p1[2] - p2[2];
    (dx * dx + dy * dy + dz * dz).sqrt()
}

pub fn scale_point(p1: &[f64; 3], scalar: &f64) -> [f64; 3] {
    let mut retvalue = [0.0; 3];
    retvalue[0] = p1[0] * scalar;
    retvalue[1] = p1[1] * scalar;
    retvalue[2] = p1[2] * scalar;
    retvalue
}

pub fn resize_vector(v1: &[f64; 3], length: &f64, origin: &[f64; 3]) -> [f64; 3] {
    let resized_v = normalize(&subtract_points(&v1, &origin));
    let scaled = scale_point(&resized_v, length);
    sum_points(&scaled, origin)
}


// Function to calculate the angle between three points (in radians)
pub fn calculate_angle(a: &[f64; 3], b: &[f64; 3], c: &[f64; 3 ]) -> f64 {
    let ba = [a[0] - b[0], a[1] - b[1], a[2] - b[2]]; // Vector BA
    let bc = [c[0] - b[0], c[1] - b[1], c[2] - b[2]]; // Vector BC
    let dot_product = dot(&ba, &bc);
    let magnitude_ba = (ba[0] * ba[0] + ba[1] * ba[1] + ba[2] * ba[2]).sqrt();
    let magnitude_bc = (bc[0] * bc[0] + bc[1] * bc[1] + bc[2] * bc[2]).sqrt();
    (dot_product / (magnitude_ba * magnitude_bc)).acos()
}

// Function to normalize a vector
pub fn normalize(v: &[f64; 3]) -> [f64; 3] {
    let magnitude = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    [v[0] / magnitude, v[1] / magnitude, v[2] / magnitude]
}


pub fn cross(p1: &[f64; 3], p2: &[f64; 3]) -> [f64; 3] {
    
        [p1[1] * p2[2] - p1[2] * p2[1],
        p1[2] * p2[0] - p1[0] * p2[2],
        p1[0] * p2[1] - p1[1] * p2[0],]
}

/// Compute the dot product of two vectors.
pub fn dot(p1: &[f64; 3], p2: &[f64; 3]) -> f64 {
    p1[0] * p2[0] + p1[1] * p2[1] + p1[2] * p2[2]
}

pub fn atoms_to_move(o: &[f64; 3], points: &[[f64; 3]; 2]) -> [f64; 3] {
    let mid = [(points[0][0] + points[1][0]) / 2., 
        (points[0][1] + points[1][1]) / 2.,
        (points[0][2] + points[1][2]) / 2.];
    let v = vector(o, &mid);
    let normalized_v = normalize(&scale_point(&v, &-1.0));
    sum_points(&o, &normalized_v)
}

pub fn rotate_point(p: &[f64; 3], p1: &[f64; 3], p2: &[f64; 3], angle: f64) -> [f64; 3] {
    // translate the point
    let pn = subtract_points(p, p1);

    // get unit vector for axis p1-p2
    let n = normalize(&subtract_points(p2, p1));

    // Setup rotation matrix
    let c = angle.cos();
    let t = 1. - angle.cos();
    let s = angle.sin();
    let x = n[0];
    let y = n[1];
    let z = n[2];

    let r = [[t*x.powi(2) + c, t*x*y - s*z, t*x*z + s*y],
    [t*x*y + s*z, t*y.powi(2) + c, t*y*z - s*x],
    [t*x*z - s*y, t*y*z + s*x, t*z.powi(2) + c]];

     // Apply rotation
     let ptr = [
        r[0][0] * pn[0] + r[0][1] * pn[1] + r[0][2] * pn[2],
        r[1][0] * pn[0] + r[1][1] * pn[1] + r[1][2] * pn[2],
        r[2][0] * pn[0] + r[2][1] * pn[1] + r[2][2] * pn[2],
    ];

    // Translate back
    [ptr[0] + p1[0], ptr[1] + p1[1], ptr[2] + p1[2]]
    
}
