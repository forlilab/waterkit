use crate::atom::Atom;
use crate::grid::Grid3D;
use crate::vina_ff;
use crate::energy;

use pyo3::pyfunction;
use rayon::prelude::*;



/// In the setup we pre-compute the 3 grid necessary
/// to compute the eneregies:
///     - O_DA grid for spherical Oxygen
///     - OW grid for TIP3P Oxygen (LJ only)
///     - Q grid for electrostatics (Probe with partial charge of +1)
#[pyfunction]
pub fn setup_system(receptor: Vec<Atom>,
    x_size: f64, 
    y_size: f64, 
    z_size: f64, 
    spacing: f64, 
    center: [f64; 3],) -> [Grid3D; 3] {
    let oda_grid = setup_oda_grid(&receptor, x_size, y_size, z_size, spacing, center);
    let ow_grid = setup_ow_grid(&receptor, x_size, y_size, z_size, spacing, center);
    let elec_grid = setup_q_grid(&receptor, x_size, y_size, z_size, spacing, center);

    [oda_grid, ow_grid, elec_grid]
}

pub fn setup_oda_grid(
    receptor_points: &Vec<Atom>,
    x_size: f64,
    y_size: f64,
    z_size: f64,
    spacing: f64,
    center: [f64; 3]) -> Grid3D {

    let mut grid = Grid3D::new((x_size, y_size, z_size), spacing, center);
    grid.all_points_mut()
    .par_iter_mut()
    .for_each(|point| {
    let energy = vina_ff::vina_energy(receptor_points, &point.coords);
        point.energy =  energy;
    });

    grid.build_kdtree();
    grid
}

pub fn setup_ow_grid(
    receptor_points: &Vec<Atom>,
    x_size: f64,
    y_size: f64,
    z_size: f64,
    spacing: f64,
    center: [f64; 3]) -> Grid3D {
    
    let mut grid = Grid3D::new((x_size, y_size, z_size), spacing, center);  
    grid.all_points_mut()
    .par_iter_mut()
    .for_each(|point| {
        point.energy =  energy::get_ow_energy(receptor_points, &point.coords);
    });

    grid.build_kdtree();
    grid
}

pub fn setup_q_grid(
    receptor_points: &Vec<Atom>,
    x_size: f64,
    y_size: f64,
    z_size: f64,
    spacing: f64,
    center: [f64; 3]) -> Grid3D {
    
    let mut grid = Grid3D::new((x_size, y_size, z_size), spacing, center);  
    grid.all_points_mut()
    .par_iter_mut()
    .for_each(|point| {
        point.energy =  energy::get_q_energy(receptor_points, &point.coords);
    });

    grid.build_kdtree();
    grid
}