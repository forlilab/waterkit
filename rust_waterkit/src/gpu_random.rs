use std::f32::EPSILON;
use cubecl::prelude::*;

// 1. Basic XorShift32 (corrected)
#[cube]
pub fn xorshift32(state: &mut u32) -> u32 {
    let mut x = *state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    *state = x;
    x
}

// 2. Linear Congruential Generator (LCG) - corrected
#[cube]
pub fn lcg_next(state: &mut u32) -> u32 {
    // Manual wrapping multiply and add
    *state = (*state * 1103515245u32) + 12345u32;
    *state
}

// 3. Convert to float in range [0, 1)
#[cube]
pub fn random_float(rng_state: &mut u32) -> f32 {
    let rand_int = xorshift32(rng_state);
    // Convert to float in range [0, 1)
    (rand_int >> 8) as f32 / 16777216.0 // 2^24
}

// 4. Random float in custom range [min, max) - same as before
#[cube]
pub fn random_range(rng_state: &mut u32, min: f32, max: f32) -> f32 {
    random_float(rng_state) * (max - min) + min
}

// 5. Random integer in range [0, max) - same as before
#[cube]
pub fn random_int_range(rng_state: &mut u32, max: u32) -> u32 {
    xorshift32(rng_state) % max
}

// 6. Simplified Box-Muller transform
#[cube]
pub fn random_normal_simple<F: Float>(rng_state: &mut u32, mean: F, std_dev: F) -> F {
    let eps = F::cast_from(EPSILON);
    let random_float_1 = random_float(rng_state);
    let random_float_2 = random_float(rng_state);
    // Simple Box-Muller without trying to cache spare value
    let u1 = F::cast_from(random_float_1);
    let u2 = F::cast_from(random_float_2);
    
    // Avoid log(0) by ensuring u1 > epsilon
    let u1_safe = F::max(u1, eps);
    
    // Box-Muller transform
    let mag = std_dev * F::sqrt(F::cast_from(-2.0) * F::log(u1_safe));
    let angle = F::new(2.0) * F::cast_from(3.14159265359) * u2;
    mag * F::sin(angle) + mean
}

// 7. Manual left rotation for 32-bit integers
#[cube]
fn rotate_left_32(value: u32, n: u32) -> u32 {
    (value << n) | (value >> (32 - n))
}

// 8. Simplified xoshiro128
#[cube]
pub fn xoshiro128_next(state: &mut Array<u32>) -> u32 {
    let result = rotate_left_32(state[1] * 5u32, 7) * 9u32;
    let t = state[1] << 9;
    
    state[2] ^= state[0];
    state[3] ^= state[1];
    state[1] ^= state[2];
    state[0] ^= state[3];
    state[2] ^= t;
    state[3] = rotate_left_32(state[3], 11);
    
    result
}

// 9. Thread-safe initialization
#[cube]
pub fn initialize_rng_states(
    base_seed: u32,
    rng_states: &mut Array<u32>
) {
    let thread_id = ABSOLUTE_POS;
    
    // Each thread gets a unique seed
    let mut seed = base_seed + (thread_id as u32);
    
    // Warm up the RNG
    for _ in 0..10 {
        seed = xorshift32(&mut seed);
    }
    
    rng_states[thread_id] = seed;
}

// 10. Initialize xoshiro128 state array
// #[cube]
// pub fn initialize_xoshiro128_states(
//     base_seed: u32,
//     states: &mut Array<Array<u32>>  // Each thread gets 4 u32s
// ) {
//     let thread_id = CUBE_POS_X;
    
//     // Generate 4 seed values using simple LCG
//     let mut seed = base_seed + (thread_id as u32);
    
//     for i in 0..4 {
//         seed = seed * 1103515245u32 + 12345u32;
//         states[thread_id][i] = seed;
//     }
// }

// 11. Alternative normal distribution using central limit theorem
#[cube]
pub fn random_normal_clt(rng_state: &mut u32, mean: f32, std_dev: f32) -> f32 {
    // Sum of 12 uniform random numbers approximates normal distribution
    let mut sum = 0.0f32;
    for _ in 0..12 {
        sum += random_float(rng_state);
    }
    // Normalize to standard normal, then scale and shift
    (sum - 6.0) * std_dev + mean
}

// 12. Simple hash function for seed generation
#[cube]
pub fn hash_combine(a: u32, b: u32) -> u32 {
    let mut result = a;
    result ^= b + 0x9e3779b9u32 + (result << 6) + (result >> 2);
    result
}