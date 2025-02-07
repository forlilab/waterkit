use crate::atom::Atom;
use crate::consts::ELECTROSTATICS_CUTOFF;
use crate::grid::Grid3D;
use crate::vina_ff;
use crate::geometry;
use crate::consts;
use crate::energy;
use rayon::prelude::*;

fn get_ow_energy(atoms_1: &Vec<Atom>, sphere_center: &[f64; 3]) -> f64{
    let mut total_energy = 0.0;
    for atom_1 in atoms_1.iter() {
        let atom_1_coords = atom_1.coords();

        // Calculate distance avoiding division by 0
        let distance = f64::max(geometry::euclidean_distance(&atom_1_coords,
            sphere_center), 1e-8_f64);

        if distance < ELECTROSTATICS_CUTOFF {

            if atom_1.atom_type() != &"HW" {
                let lj_energy = energy::lennard_jones_rmin_half(atom_1.epsilon(),
                &consts::TIP3P_EPSILON, 
                &distance,
                atom_1.rmin_half(), 
                &consts::RMIN_HALF_WATER);
                total_energy += lj_energy;
            }
        }
    }
    total_energy
}

fn get_q_energy(atoms_1: &Vec<Atom>, sphere_center: &[f64; 3]) -> f64 {
    let mut total_energy = 0.0;
    for atom_1 in atoms_1.iter() {
        let atom_1_coords = atom_1.coords();

        // Calculate distance avoiding division by 0
        let distance = f64::max(geometry::euclidean_distance(&atom_1_coords,
            sphere_center), 1e-8_f64);

        if distance < ELECTROSTATICS_CUTOFF {

            let electrostatics = energy::coulomb_energy(atom_1.charge(), &1.0, &distance);
            total_energy += electrostatics;
        }
    }
    total_energy
}

/// In the setup we pre-compute the 3 grid necessary
/// to compute the eneregies:
///     - O_DA grid for spherical Oxygen
///     - OW grid for TIP3P Oxygen (LJ only)
///     - Q grid for electrostatics (Probe with partial charge of +1)
pub fn setup_system(receptor: &Vec<Atom>,
    x_size: f64, 
    y_size: f64, 
    z_size: f64, 
    spacing: f64, 
    center: [f64; 3],) -> Vec<Grid3D> {
    let oda_grid = setup_oda_grid(receptor, x_size, y_size, z_size, spacing, center);
    let ow_grid = setup_ow_grid(receptor, x_size, y_size, z_size, spacing, center);
    let elec_grid = setup_q_grid(receptor, x_size, y_size, z_size, spacing, center);

    vec![oda_grid, ow_grid, elec_grid]
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
        point.energy =  get_ow_energy(receptor_points, &point.coords);
    });
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
        point.energy =  get_q_energy(receptor_points, &point.coords);
    });
    grid
}