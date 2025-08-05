use std::f64;
use cubecl::prelude::*;
use cubecl::cube;

// Basic geometric operations on points
#[cube]
pub fn vector<F: Float>(p1: &Array<F>, p2: &Array<F>) -> Array<F> {
    subtract_points(p2, p1)
}

#[cube]
pub fn sum_points<F: Float>(p1: &Array<F>, p2: &Array<F>) -> Array<F> {
    let mut retvalue = Array::new(3);
    retvalue[0] = p1[0] + p2[0]; 
    retvalue[1] = p1[1] + p2[1]; 
    retvalue[2] = p1[2] + p2[2];
    retvalue
}

#[cube]
pub fn subtract_points<F: Float>(p1: &Array<F>, p2: &Array<F>) -> Array<F> {
    let mut retvalue = Array::new(3);
    retvalue[0] = p1[0] - p2[0]; 
    retvalue[1] = p1[1] - p2[1]; 
    retvalue[2] = p1[2] - p2[2];
    retvalue
}

#[cube]
pub fn euclidean_distance<F: Float>(p1: &Array<F>, p2: &Array<F>) -> F {
    let dx = p1[0] - p2[0];
    let dy = p1[1] - p2[1];
    let dz = p1[2] - p2[2];
    F::sqrt(dx * dx + dy *dy + dz * dz)
}

#[cube]
pub fn scale_point<F: Float>(p1: &Array<F>, scalar: F) -> Array<F> {
    let mut retvalue = Array::new(3);
    retvalue[0] = p1[0] * scalar;
    retvalue[1] = p1[1] * scalar;
    retvalue[2] = p1[2] * scalar;
    retvalue
}

#[cube]
pub fn resize_vector<F: Float>(v1: &Array<F>, length: F, origin: &Array<F>) -> Array<F> {
    let resized_v = normalize(&subtract_points(&v1, &origin));
    let scaled = scale_point(&resized_v, length);
    sum_points(&scaled, origin)
}


#[cube]
pub fn caclulate_angle_to_apply<F: Float>(actual_angle: &F, expected_angle: &F) -> F {
    let mut possible_angles: Array<F> = Array::new(3);
    possible_angles[0] = *expected_angle - *actual_angle;
    possible_angles[1] = *expected_angle - *actual_angle + F::new(360.0);
    possible_angles[2] = *expected_angle - *actual_angle - F::new(360.0);

    // Find the index of the minimum absolute value
    let mut min_index: i32 = 0;
    let mut min_value = F::abs(possible_angles[0]);

    let abs_0 = F::abs(possible_angles[0]);
    let abs_1 = F::abs(possible_angles[1]);
    let abs_2 = F::abs(possible_angles[2]);

    if abs_0 <= abs_1 && abs_0 <= abs_2 {
        possible_angles[0]
    } else if abs_1 <= abs_0 && abs_1 <= abs_2 {
        possible_angles[1]
    } else {
        possible_angles[2]
    }
}

// Function to normalize a vector
#[cube]
pub fn normalize<F: Float>(v: &Array<F>) -> Array<F> {
    let magnitude = F::sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
    let mut retvalue = Array::new(3);
    retvalue[0] = v[0] / magnitude;
    retvalue[1] = v[1] / magnitude;
    retvalue[2] = v[2] / magnitude;
    retvalue
}

#[cube]
pub fn cross<F: Float>(p1: &Array<F>, p2: &Array<F>) -> Array<F> {
    let mut retvalue = Array::new(3);
    retvalue[0] = p1[1] * p2[2] - p1[2] * p2[1];
    retvalue[1] = p1[2] * p2[0] - p1[0] * p2[2];
    retvalue[2] = p1[0] * p2[1] - p1[1] * p2[0];
    retvalue
}

/// Compute the dot product of two vectors.
#[cube]
pub fn dot<F: Float>(p1: &Array<F>, p2: &Array<F>) -> F {
    p1[0] * p2[0] + p1[1] * p2[1] + p1[2] * p2[2]
}