use std::io::Write;
use std::time::Instant;
use std::fs::{self, OpenOptions};
use pyo3::prelude::*;
use rayon::prelude::*;

use crate::anchor_point::AnchorPoint;
use crate::atom::Atom;
use crate::grid::Grid3D;
use crate::sampling::{sample, sample_using_grids, sample_waters_with_grids};
use crate::setup::setup_system;

pub fn to_pdb(atoms: &Vec<&Atom>, fname: &str) {
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
        for _ in 0..3 {
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
    mut grid_elec: Grid3D,
    epoch: usize) -> Vec<Atom> {
        // let start_time = Instant::now();
        let mut receptor_map = receptor_points.clone();
        
        let mut mutable_anchor_points = anchor_points.clone();

        // while mutable_anchor_points.len() > 0 {
        for i in 0..2 {
            sample_using_grids(&mut grid_oda, &mut grid_ow, &mut grid_elec,&mut receptor_map, &mut mutable_anchor_points, &water_configurations);
        }
        // let elapsed_time = start_time.elapsed();
        // println!("Time taken for one map: {:?}", elapsed_time);
        let waters: Vec<&Atom> = receptor_map.iter().filter(|x| x.atom_type() == "OW" || x.atom_type() == "HW").collect();
        println!("Waters: {}", waters.len());
        // println!("{:?}", waters);
        to_pdb(&waters, &format!("test/water_{epoch}.pdb"));
        receptor_map
}

#[pyfunction]
pub fn run_waterkit(receptor_points: Vec<Atom>, 
    water_configurations: Vec<[f64; 6]>, 
    anchor_points: Vec<AnchorPoint>, 
    grids: [Grid3D; 3],
    epochs: usize,
    use_grids: bool) -> Vec<Vec<Atom>> {

    let mut results = Vec::new();
    println!("Starting main waterkit");
    if !use_grids {
        results.push(run_single_waterkit(&receptor_points.clone(), &water_configurations.clone(), &anchor_points.clone(), grids[0].clone()));
    } else {
        results.push(run_single_waterkit_with_grids(
            &receptor_points.clone(),
            &water_configurations.clone(),
            &anchor_points.clone(),
            grids[0].clone(),
            grids[1].clone(),
            grids[2].clone(),
            epochs
        ));
    }
    return results;
}
