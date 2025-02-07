use std::time::Instant;

use pyo3::prelude::*;
use rayon::prelude::*;

use crate::anchor_point::AnchorPoint;
use crate::atom::Atom;
use crate::grid::Grid3D;
use crate::sampling::{sample, sample_using_grids, sample_waters_with_grids};
use crate::setup::setup_system;


fn run_single_waterkit(receptor_points: &Vec<Atom>, 
    water_configurations: &Vec<[f64; 6]>, 
    anchor_points: &Vec<AnchorPoint>, 
    mut grid: Grid3D) -> Vec<Atom> {
        // let start_time = Instant::now();
        let mut receptor_map = receptor_points.clone();
        
        let mut mutable_anchor_points = anchor_points.clone();

        for _ in 0..4 {
            sample(&mut grid, &mut receptor_map, &mut mutable_anchor_points, &water_configurations);
        }
        // let elapsed_time = start_time.elapsed();
        // println!("Time taken for one map: {:?}", elapsed_time);
        receptor_map
}

fn run_single_waterkit_with_grids(receptor_points: &Vec<Atom>, 
    water_configurations: &Vec<[f64; 6]>, 
    anchor_points: &Vec<AnchorPoint>, 
    mut grid_oda: Grid3D,
    mut grid_ow: Grid3D,
    mut grid_elec: Grid3D) -> Vec<Atom> {
        // let start_time = Instant::now();
        let mut receptor_map = receptor_points.clone();
        
        let mut mutable_anchor_points = anchor_points.clone();

        for _ in 0..3 {
            sample_using_grids(&mut grid_oda,
                &mut grid_ow,
                &mut grid_elec,
                 &mut receptor_map,
                  &mut mutable_anchor_points,
                   &water_configurations);
        }
        // let elapsed_time = start_time.elapsed();
        // println!("Time taken for one map: {:?}", elapsed_time);
        receptor_map
}

#[pyfunction]
pub fn run_waterkit(receptor_points: Vec<Atom>, 
    water_configurations: Vec<[f64; 6]>, 
    anchor_points: Vec<AnchorPoint>, 
    x_size: f64, 
    y_size: f64, 
    z_size: f64, 
    spacing: f64, 
    center: [f64; 3],
    epochs: usize,
    use_grids: bool) -> Vec<Vec<Atom>> {
    let start_time = Instant::now();
    
    let grids = setup_system(&receptor_points, x_size, y_size, z_size, spacing, center);
    let grid_oda = grids[0].clone();
    let grid_ow = grids[1].clone();
    let grid_elec = grids[2].clone();

    if !use_grids {
        let results: Vec<Vec<Atom>> = (0..epochs)
            .into_par_iter()
            .map(|_| {
                run_single_waterkit(
                    &receptor_points.clone(),
                    &water_configurations.clone(),
                    &anchor_points.clone(),
                    grid_oda.clone()
                )
            })
            .collect();
        let elapsed_time = start_time.elapsed();
        println!("Time taken for {} maps: {:?}", epochs, elapsed_time);
        return results;
    } else {
        let results: Vec<Vec<Atom>> = (0..epochs)
            .into_par_iter()
            .map(|_| {
                run_single_waterkit_with_grids(
                    &receptor_points.clone(),
                    &water_configurations.clone(),
                    &anchor_points.clone(),
                    grid_oda.clone(),
                    grid_ow.clone(),
                    grid_elec.clone(),
                )
            })
            .collect();
        let elapsed_time = start_time.elapsed();
        println!("Time taken for {} maps: {:?}", epochs, elapsed_time);
        return results;
    }
}
