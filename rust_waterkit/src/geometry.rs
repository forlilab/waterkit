use std::f64;
use std::f64::consts::PI;

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

/// Compute the angle between points `a`, `b`, and `c`.
/// If `degree` is true, the result is returned in degrees; otherwise, in radians.
pub fn get_angle_for_neighbors(a: &[[f64; 3]], b: &[f64; 3], c: &[f64; 3], degree: bool) -> Vec<f64> {
    let bc = [c[0] - b[0], c[1] - b[1], c[2] - b[2]]; // Vector from b to c
    let mut angles = Vec::new();

    for point in a {
        let ba = [point[0] - b[0], point[1] - b[1], point[2] - b[2]]; // Vector from b to a

        // Dot product of ba and bc
        let dot_product = ba[0] * bc[0] + ba[1] * bc[1] + ba[2] * bc[2];

        // Magnitude (norm) of ba and bc
        let norm_ba = (ba[0].powi(2) + ba[1].powi(2) + ba[2].powi(2)).sqrt();
        let norm_bc = (bc[0].powi(2) + bc[1].powi(2) + bc[2].powi(2)).sqrt();

        // Compute cosine of the angle
        let cos_angle = dot_product / (norm_ba * norm_bc);

        // Clip the cosine value to the valid range [-1, 1]
        let cos_angle = cos_angle.clamp(-1.0, 1.0);

        // Compute the angle in radians
        let angle = cos_angle.acos();

        // Convert to degrees if requested
        if degree {
            angles.push(angle * 180.0 / PI);
        } else {
            angles.push(angle);
        }
    }

    angles
}