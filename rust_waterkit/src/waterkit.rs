use core::f64;
use std::io::Write;
use std::fs::OpenOptions;
use pyo3::prelude::*;
use rand::seq::SliceRandom;
use rand::thread_rng;
use rayon::prelude::*;

use crate::anchor_point::AnchorPoint;
use crate::atom::Atom;
use crate::grid::Grid3D;
use crate::grid::ProbeType;
use crate::consts;
use crate::monte_carlo;
use crate::optimizer::optimize;
use crate::energy;
use crate::sampling::sample;
use crate::sampling::sample_using_grids;

pub fn to_pdb(atoms: &Vec<Atom>, fname: &str) {
    let mut cnt = 0;
    let mut h_index = 1;
    for (index, point) in atoms.iter().enumerate() {
        if index % 3 == 0 {
            cnt += 1;
        }
        let coordinates = point.coords();
        let mut line = String::new();
        let mut h_type = "";
        if point.atom_type() == "HW" {
            if h_index < 2 {
                h_type = "H1";
                h_index += 1;
            }
            else {
                h_type = "H2";
                h_index = 1;
            }
            line = format!(
                "{:<6}{:>5} {:^4} {:>3} {:1}{:>4}    {:>8.3}{:>8.3}{:>8.3}{:>6.2}{:>6.2}          {:>2}\n",
                "ATOM",
                index,
                h_type,
                "HOH",
                "A",
                cnt,
                coordinates[0],
                coordinates[1],
                coordinates[2],
                0.0,
                0.0,
                "H"
            );
        }
        else {
            line = format!(
                "{:<6}{:>5} {:^4} {:>3} {:1}{:>4}    {:>8.3}{:>8.3}{:>8.3}{:>6.2}{:>6.2}          {:>2}\n",
                "ATOM",
                index,
                "O",
                "HOH",
                "A",
                cnt,
                coordinates[0],
                coordinates[1],
                coordinates[2],
                0.0,
                0.0,
                "O"
            );
        }
        let mut f = OpenOptions::new()
        .append(true)
        .create(true) // Optionally create the file if it doesn't already exist
        .open(fname)
        .expect("Unable to open file");
        
        f.write_all(line.as_bytes()).expect("Unable to write data");
        // fs::write(fname, line).expect("Unable to write file");
    }
}


fn run_single_waterkit(receptor_points: &Vec<Atom>, 
    water_configurations: &Vec<[f64; 6]>, 
    anchor_points: &Vec<AnchorPoint>, 
    mut grid: Grid3D) -> Vec<Atom> {
        // let start_time = Instant::now();
        let mut receptor_map = receptor_points.clone();
        
        let mut mutable_anchor_points = anchor_points.clone();

        // while mutable_anchor_points.len() > 0 {
        for _i in 0..3 {
            sample(&mut grid, &mut receptor_map, &mut mutable_anchor_points, &water_configurations);
        }
        // let elapsed_time = start_time.elapsed();
        // println!("Time taken for one map: {:?}", elapsed_time);
        receptor_map
}


fn run_single_waterkit_with_grids(receptor_points: &Vec<Atom>, 
    water_configurations: &Vec<[f64; 6]>, 
    anchor_points: &Vec<AnchorPoint>, 
    mut grid: Grid3D,
    epoch: usize) -> Vec<Atom> {
        // let start_time = Instant::now();
        let mut receptor_map = receptor_points.clone();
        
        let mut mutable_anchor_points = anchor_points.clone();
        let mut new_water_molecules = Vec::new();
        for _i in 0..2 {
            sample_using_grids(&mut grid,&mut receptor_map, &mut mutable_anchor_points, &water_configurations, &mut new_water_molecules);
        }

        let waters: Vec<Atom> = receptor_map.iter().filter(|x| x.atom_type() == "OW" || x.atom_type() == "HW").cloned().collect();
        to_pdb(&waters, &format!("test/water_{epoch}_unoptimized.pdb"));
        // Need to get energies for the waters placed and select the ones to perturb based on an 
        // inverted boltzmann wheighted choice
        let mut water_energies = Vec::new();
        for water in new_water_molecules.iter() {
            let atoms = water.as_vec();
            let waters_in_system: Vec<Atom> = receptor_map.iter().filter(|x| !atoms.contains(x)).cloned().collect();
            water_energies.push(energy::energy_for_real_water(&waters_in_system, &atoms));         
        }
        
        let mut optimized_waters: Vec<Atom> = Vec::new();
        let steps = water_energies.len();
        let indices = monte_carlo::inverted_boltzmann_choices(&water_energies, Some(steps));
        for index in  indices {
            // Select a water molecule to modify
            let new_water = new_water_molecules[index].clone();
            let waters_in_system: Vec<Atom> = receptor_map.iter().filter(|x| !new_water.as_vec().contains(x)).cloned().collect();
            let optimized_water = optimize(&new_water, &waters_in_system,  &mut grid);
            for a in optimized_water.as_vec() {
                optimized_waters.push(a.clone());
            }
        }

        to_pdb(&optimized_waters, &format!("test/water_{epoch}_optimized.pdb"));

        receptor_map
}

#[pyfunction]
pub fn run_waterkit(receptor_points: Vec<Atom>, 
    water_configurations: Vec<[f64; 6]>, 
    anchor_points: Vec<AnchorPoint>, 
    grid: Grid3D,
    epochs: usize,
    use_grids: bool) -> Vec<Vec<Atom>> {

    let mut results = Vec::new();
    // println!("Starting main waterkit");
    if !use_grids {
        results.push(run_single_waterkit(&receptor_points.clone(), &water_configurations.clone(), &anchor_points.clone(), grid.clone()));
    } else {
        results.push(run_single_waterkit_with_grids(
            &receptor_points.clone(),
            &water_configurations.clone(),
            &anchor_points.clone(),
            grid.clone(),
            epochs
        ));
    }
    return results;
}

#[pyfunction]
pub fn run_parallel_waterkit(receptor_points: Vec<Atom>, 
    water_configurations: Vec<[f64; 6]>, 
    anchor_points: Vec<AnchorPoint>, 
    grid: Grid3D,
    epochs: usize,
    use_grids: bool) {

    (0..epochs).into_par_iter()
        .for_each(|epoch| {
            if !use_grids {
                run_single_waterkit(&receptor_points.clone(), &water_configurations.clone(), &anchor_points.clone(), grid.clone());
            } else {
                run_single_waterkit_with_grids(
                    &receptor_points.clone(),
                    &water_configurations.clone(),
                    &anchor_points.clone(),
                    grid.clone(),
                    epoch
                );
            }
        });
}
