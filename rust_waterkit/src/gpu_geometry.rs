use std::f64;
use cubecl::prelude::*;
use cubecl::cube;

// Basic geometric operations on points
#[cube]
pub fn add<F: Float>(v1: &Array<F>, v2: &Array<F>) -> Array<F> {
    let mut retvalue = Array::new(3);
    retvalue[0] = v1[0] + v2[0];
    retvalue[1] = v1[1] + v2[1];
    retvalue[2] = v1[2] + v2[2];
    retvalue
}

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

/// Rotation around a point and axis
#[cube]
pub fn rodrigues_rotation<F: Float>(
    point: &Array<F>,           // [x, y, z] - point to rotate
    axis: &Array<F>,            // [x, y, z] - rotation axis (should be normalized)
    angle: F,                   // rotation angle in radians
    pivot: &Array<F>,           // [x, y, z] - pivot point (oxygen position)
    result: &mut Array<F>       // [x, y, z] - output rotated point
) {
    let cos_theta = F::cos(angle);
    let sin_theta = F::sin(angle);
    
    let v_rel = subtract_points(point, pivot);
    let term1 = scale_point(&v_rel, cos_theta);
    let term2 = scale_point(&cross(&axis, &v_rel), sin_theta);
    let term3 = scale_point(&scale_point(&axis, dot(axis, &v_rel)), (F::new(1.0) - cos_theta));
    let summed = sum_points(&sum_points(&sum_points(&term1, &term2), &term3), &pivot);
    result[0] = summed[0];
    result[1] = summed[1];
    result[2] = summed[2];
}

// /// Rotation around a point and axis
// #[cube]
// pub fn rodrigues_rotation<F: Float>(
//     point: &Array<F>,           // [x, y, z] - point to rotate
//     axis: &Array<F>,            // [x, y, z] - rotation axis (should be normalized)
//     angle: F,                   // rotation angle in radians
//     pivot: &Array<F>,           // [x, y, z] - pivot point (oxygen position)
//     result: &mut Array<F>       // [x, y, z] - output rotated point
// ) {
//     let cos_theta = F::cos(angle);
//     let sin_theta = F::sin(angle);
    
//     // Calculate point relative to pivot
//     let v_rel_x = point[0] - pivot[0];
//     let v_rel_y = point[1] - pivot[1]; 
//     let v_rel_z = point[2] - pivot[2];
    
//     // Calculate axis cross product with v_rel: k × v_rel
//     let cross_x = axis[1] * v_rel_z - axis[2] * v_rel_y;
//     let cross_y = axis[2] * v_rel_x - axis[0] * v_rel_z;
//     let cross_z = axis[0] * v_rel_y - axis[1] * v_rel_x;
    
//     // Calculate axis dot product with v_rel: k · v_rel
//     let dot_product = axis[0] * v_rel_x + axis[1] * v_rel_y + axis[2] * v_rel_z;
    
//     // Rodrigues' rotation formula: v_rot = v*cos(θ) + (k×v)*sin(θ) + k*(k·v)*(1-cos(θ))
//     // Term 1: v_rel * cos(θ)
//     let term1_x = v_rel_x * cos_theta;
//     let term1_y = v_rel_y * cos_theta;
//     let term1_z = v_rel_z * cos_theta;
    
//     // Term 2: (k × v_rel) * sin(θ)
//     let term2_x = cross_x * sin_theta;
//     let term2_y = cross_y * sin_theta;
//     let term2_z = cross_z * sin_theta;
    
//     // Term 3: k * (k · v_rel) * (1 - cos(θ))
//     let one_minus_cos = F::new(1.0) - cos_theta;
//     let term3_x = axis[0] * dot_product * one_minus_cos;
//     let term3_y = axis[1] * dot_product * one_minus_cos;
//     let term3_z = axis[2] * dot_product * one_minus_cos;
    
//     // Sum all terms and add back the pivot point
//     result[0] = term1_x + term2_x + term3_x + pivot[0];
//     result[1] = term1_y + term2_y + term3_y + pivot[1];
//     result[2] = term1_z + term2_z + term3_z + pivot[2];
// }