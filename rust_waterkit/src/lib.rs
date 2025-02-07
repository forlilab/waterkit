pub mod utils;
pub mod consts;
pub mod geometry;
pub mod vina_ff;
pub mod anchor_point;
pub mod atom;
pub mod water;
pub mod energy;
pub mod monte_carlo;
pub mod setup;
pub mod sampling;
pub mod grid;
pub mod waterkit;
pub mod python_wrapper;

#[cfg(test)]
mod tests {
    use super::*;
    use rayon::prelude::*;
    #[test]
    fn test_interpolation() {
        let mut grid = grid::Grid3D::new((2.0, 2.0, 2.0), 1.0, [4.0, 4.0, 4.0]);
        let mut e = 1.0;
        for point in grid.all_points_mut() {
            point.energy = e;
            e += 1.; 
        }
        let interpolated = grid.trilinear_interpolation(&[0.5, 0.5, 0.5]);
        assert_eq!(interpolated, 4.5);

    }
}