#![allow(warnings)]

pub mod utils;
pub mod consts;
pub mod gpu_energy;
pub mod geometry;
pub mod vina_ff;
pub mod anchor_point;
pub mod rotatable_bond;
pub mod atom;
pub mod water;
pub mod waterkit_system;
pub mod energy;
pub mod monte_carlo;
pub mod optimizer;
pub mod setup;
pub mod sampling;
pub mod grid;
pub mod waterkit;
pub mod gcmc;
pub mod gpu_gcmc;
pub mod gpu_geometry;
pub mod gpu_gcmc_moves;
pub mod replica_exchange;
pub mod python_wrapper;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_trilinear_interpolation() {
        let spacing = 1.0;
        let size = (3.0, 3.0, 3.0);
        let center = [1.5, 1.5, 1.5];
        
        let mut grid = grid::Grid3D::new(size, spacing, center);
    
        // Manually assign energy values to a small 2x2x2 cube for easy validation
        let test_indices = [
            (0, 0, 0, 1.0),  // c000
            (0, 0, 1, 2.0),  // c001
            (0, 1, 0, 3.0),  // c010
            (0, 1, 1, 4.0),  // c011
            (1, 0, 0, 5.0),  // c100
            (1, 0, 1, 6.0),  // c101
            (1, 1, 0, 7.0),  // c110
            (1, 1, 1, 8.0),  // c111
        ];

        for data in grid.data.iter() {
            println!("Point at index: {} - {:?} - {}", data.index, data.coords, data.energy_oda);
        }
    
        for &(i, j, k, energy) in &test_indices {
            let index = i + 4 * (j + 4 * k); // Manually mapping indices
            println!("Fetching energy at i={}, j={}, k={} -> index {}", i, j, k, index);
            grid.data[index].energy_oda = energy;
        }
        
        // Test a point in the middle of the cell (should return the average energy)
        let test_point = [0.5, 0.5, 0.5];  // Midpoint of (0,0,0) and (1,1,1)
        let interpolated_value = grid.trilinear_interpolation(test_point, grid::ProbeType::HW).unwrap();
    
        // Compute expected value manually
        let expected_value = (1.0 + 2.0 + 3.0 + 4.0 + 5.0 + 6.0 + 7.0 + 8.0) / 8.0;
    
        println!(
            "Interpolated value: {:.3}, Expected value: {:.3}",
            interpolated_value, expected_value
        );
    
        assert!(
            (interpolated_value - expected_value).abs() < 1e-6,
            "Interpolation is incorrect!"
        );
    
        println!("Trilinear interpolation test passed!");
    }
}