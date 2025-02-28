use core::f64;
use std::collections::HashSet;
use std::time::SystemTime;
use pyo3::prelude::*;
use rand::seq::SliceRandom;
use rand::thread_rng;
use rand::Rng;
use rayon::prelude::*;

use crate::anchor_point::AnchorPoint;
use crate::atom::Atom;
use crate::energy::energy_for_real_water;
use crate::grid::Grid3D;
use crate::grid::ProbeType;
use crate::consts;
use crate::optimizer::optimize;
use crate::optimizer::optimize_using_grids;
use crate::sampling::sample_using_grids;
use crate::setup::setup_grid;
use crate::utils::to_pdb;
use crate::water::WaterMolecule;


fn run_single_waterkit_with_grids(receptor_points: &[Atom], 
    water_configurations: &Vec<[f64; 6]>, 
    anchor_points: &[AnchorPoint], 
    mut grid: Grid3D,
    epoch: usize) -> Vec<Atom> {
    let mut receptor_map = receptor_points.to_vec();
    let mut last_residue_number = receptor_map.iter().map(|n| n.residue_number).max().unwrap_or(1);

    let mut mutable_anchor_points = anchor_points.to_vec();
    let mut new_water_molecules = Vec::new();
    for _i in 0..3 {
        sample_using_grids(&mut grid, &mut mutable_anchor_points, &water_configurations, &mut new_water_molecules, &mut last_residue_number);
    }

    // let waters: Vec<Atom> = receptor_map.iter().filter(|x| x.atom_type() == "OW" || x.atom_type() == "HW").cloned().collect();
    let mut waters: Vec<Atom> = Vec::with_capacity(new_water_molecules.len() * 3);
    // for w in new_water_molecules.iter() {
    //     waters.extend(w.as_vec().into_iter());
    //     receptor_map.extend(w.as_vec().into_iter());
    // } 
    // utils::to_pdb(&waters, &format!("test/water_{epoch}_unoptimized.pdb"));
    // waters

    // let optimized_waters = optimize_water_network(&mut receptor_map, &new_water_molecules, &mut grid, &format!("test/water_{epoch}_optimized.pdb"));
    // let optimized_waters = optimize_water_nw_with_grids(&mut new_water_molecules, &mut grid);
    // optimized_waters
    optimize_water_nw_with_grids(&mut new_water_molecules, &mut grid);
    for w in new_water_molecules.into_iter() {
        waters.extend(w.as_vec().into_iter());
    }
    waters
}

pub fn optimize_water_nw_with_grids(new_waters: &mut Vec<WaterMolecule>, grid: &mut Grid3D) {
    // let mut optimized_waters = Vec::new();
    let mut rng = thread_rng();

    // Standard approach
    // new_waters.shuffle(&mut rng);
    // for water in new_waters.into_iter() {
    //     let water_atoms = optimize_using_grids(&water.as_vec(), grid).as_vec();
    //     water_atoms
    //         .into_iter()
    //         .for_each(|a| optimized_waters.push(a));
    // } 
    for step in 0..1000 {
        let index = rng.gen_range(0..new_waters.len()); // Generate a random index
        let water = &new_waters[index];
        let new_water = optimize_using_grids(water, grid);
        new_waters[index] = new_water;
    }
}

pub fn optimize_water_network(receptor_map: &mut Vec<Atom>, new_waters: &Vec<WaterMolecule>, grid: &mut Grid3D, filename: &str) -> Vec<Atom> {
    // let start = SystemTime::now();
    let mut rng = thread_rng();
    let mut waters_res_number: Vec<i32> = new_waters.iter()
        .map(|w| w.get_res_number())
        .collect::<HashSet<_>>() // Collect unique values first
        .into_iter()
        .collect();

    waters_res_number.shuffle(&mut rng);
    for res_number in waters_res_number.iter() {
        // println!("{}", res_number);
        let mut filtered = Vec::new();

        receptor_map
            .retain(|atom| {
                if atom.residue_number == *res_number {
                    filtered.push(atom.clone());
                    false
                } else {
                    true
                }
            });

        // Here I optimize
        optimize(&mut filtered, &receptor_map, grid);

        receptor_map.extend(filtered);
    }
    
    let mut waters = Vec::new();
    receptor_map.retain(|x| {
        if x.atom_type() == "OW" || x.atom_type() == "HW" {
            waters.push(x.clone());
            false // Remove this element from receptor_map
        } else {
            true // Keep this element in receptor_map
        }
    });

    // let end = SystemTime::now();
    // let duration = end.duration_since(start).unwrap();
    // println!("MC Optimization took {} ms", duration.as_millis());
    
    waters

    // to_pdb(&receptor_map.iter().filter(|x| x.atom_type() == "OW" || x.atom_type() == "HW").cloned().collect::<Vec<Atom>>(), filename);
}


#[pyfunction]
pub fn run_parallel_waterkit(receptor_points: Vec<Atom>, 
    water_configurations: Vec<[f64; 6]>, 
    anchor_points: Vec<AnchorPoint>, 
    grid: Grid3D,
    epochs: usize) {

    let waters: Vec<Vec<Atom>> = (0..epochs).into_par_iter()
        .map(|epoch| run_single_waterkit_with_grids(
                &receptor_points.clone(),
                &water_configurations.clone(),
                &anchor_points.clone(),
                grid.clone(),
                epoch
            )).collect();
            
    println!("Done sampling...saving results!");
    
    waters.par_iter().enumerate()
        .for_each(|(idx, system)| to_pdb(&system, &format!("/data/phd/waterkit/rust_waterkit/test/water_{idx}_optimized.pdb")));
    // waters.par_iter().enumerate()
    //     .for_each(|(idx, system)| to_pdb(&system, &format!("/data/phd/waterkit/rust_waterkit/test/water_{idx}_unoptimized.pdb")));
}


#[pyfunction]
pub fn get_energies_for_system(receptor_points: Vec<Atom>, 
    waters: Vec<[Atom; 3]>, center: [f64; 3], x: f64, y: f64, z: f64) {
    
    let grid_receptor = setup_grid(&receptor_points, x, y, z, 0.375, center);

    for (index, water) in waters.iter().enumerate() {
        let mut points = receptor_points.clone();
        // println!("Points before: {}", points.len());
        let water_atoms = vec![water[0].clone(), water[1].clone(), water[2].clone()];
        let waters_excluding: Vec<&[Atom; 3]> = waters.iter().filter(|x| &water != x).collect();
        let mut waters_network = Vec::new();
        for excluded_w in waters_excluding {
            for excluded_atom in excluded_w {
                points.push(excluded_atom.clone());
                waters_network.push(excluded_atom.clone());
            }
        }
        // println!("Waters network: {}", waters_network.len());
        // println!("Points after: {}", points.len());
        let use_grids = true;

        if use_grids {
            // let grid_receptor_and_w = setup_grid(&points, 21.0, 24.0, 26.0, 0.375, [71.5, 73.1, 243.4]);
            let grid_receptor_and_w = setup_grid(&points, x, y, z, 0.375, center);

            let oxygen = water_atoms[0].clone();
            println!("{:?}", oxygen.coords());
            let h1 = water_atoms[1].clone();
            let h2 = water_atoms[2].clone();
            // println!("{}", receptor_points.len());
            // let mut energy = energy_for_real_water(&points, &vec![oxygen.clone()]);
            let mut energy = grid_receptor_and_w.trilinear_interpolation(oxygen.coords(), ProbeType::OW).unwrap();
            println!("Total LJ: {}", energy);
            let e_elec = grid_receptor_and_w.trilinear_interpolation(oxygen.coords(), ProbeType::HW).unwrap() * consts::OXYGEN_W_Q;
            println!("Total Coulomb: {}", e_elec);
            energy += e_elec;
            println!("{index} {index} {energy} O (rec+wat)");

            // let mut energy_rec = energy_for_real_water(&receptor_points, &vec![oxygen.clone()]);
            let mut energy_rec = grid_receptor.trilinear_interpolation(oxygen.coords(), ProbeType::OW).unwrap();
            energy_rec += grid_receptor.trilinear_interpolation(oxygen.coords(), ProbeType::HW).unwrap() * consts::OXYGEN_W_Q;
            println!("{index} {index} {energy_rec} O (just rec)");

            // let e = energy_for_real_water(&points, &vec![h1.clone()]);
            let e = grid_receptor_and_w.trilinear_interpolation(h1.coords(), ProbeType::HW).unwrap() * consts::HYDROGEN_W_Q;
            println!("Total Coulomb: {}", e);
            println!("{index} {index} {e} H (rec+wat)");
            energy += e;

            // let er = energy_for_real_water(&receptor_points, &vec![h1.clone()]);
            let er = grid_receptor.trilinear_interpolation(h1.coords(), ProbeType::HW).unwrap() * consts::HYDROGEN_W_Q;
            println!("{index} {index} {er} H (just rec)");
            energy_rec += er;

            // let e = energy_for_real_water(&points, &vec![h2.clone()]);
            let e = grid_receptor_and_w.trilinear_interpolation(h2.coords(), ProbeType::HW).unwrap() * consts::HYDROGEN_W_Q;
            println!("Total Coulomb: {}", e);
            println!("{index} {index} {e} H (rec+wat)");
            energy += e;

            // let er = energy_for_real_water(&receptor_points, &vec![h2.clone()]);
            let er = grid_receptor.trilinear_interpolation(h2.coords(), ProbeType::HW).unwrap() * consts::HYDROGEN_W_Q;
            println!("{index} {index} {er} H (just rec)");
            energy_rec += er;

            println!("{index} {index} {energy} HOH (rec+wat)");
            println!("{index} {index} {energy_rec} HOH (just rec)");
            println!();
        } else {
            // println!("waters network: {:?}", waters_network);
            // println!("Target water: {:?}", water_atoms);

            let oxygen = water_atoms[0].clone();
            println!("{:?}", oxygen.coords());
            let h1 = water_atoms[1].clone();
            let h2 = water_atoms[2].clone();
            println!("{}", receptor_points.len());
            let mut energy = energy_for_real_water(&points, &vec![oxygen.clone()]);
            println!("{index} {index} {energy} O (rec+wat)");

            let mut energy_w = energy_for_real_water(&waters_network, &vec![oxygen.clone()]);
            println!("{index} {index} {energy_w} O (just wat)");

            let mut energy_rec = energy_for_real_water(&receptor_points, &vec![oxygen.clone()]);
            println!("{index} {index} {energy_rec} O (just rec)");

            let e = energy_for_real_water(&points, &vec![h1.clone()]);
            println!("{index} {index} {e} H (rec+wat)");
            energy += e;

            let ew = energy_for_real_water(&waters_network, &vec![h1.clone()]);
            println!("{index} {index} {ew} H (just wat)");
            energy_w += ew;

            let er = energy_for_real_water(&receptor_points, &vec![h1.clone()]);
            println!("{index} {index} {er} H (just rec)");
            energy_rec += er;

            let e = energy_for_real_water(&points, &vec![h2.clone()]);
            println!("{index} {index} {e} H (rec+wat)");
            energy += e;

            let ew = energy_for_real_water(&waters_network, &vec![h2.clone()]);
            println!("{index} {index} {ew} H (just wat)");
            energy_w += ew;

            let er = energy_for_real_water(&receptor_points, &vec![h2.clone()]);
            println!("{index} {index} {er} H (just rec)");
            energy_rec += er;

            println!("{index} {index} {energy} HOH (rec+wat)");
            println!("{index} {index} {energy_w} HOH (just wat)");
            println!("{index} {index} {energy_rec} HOH (just rec)");
            // println!();
        }
        // println!("{:?}", water);
        // println!("{:?}\n", water_atoms);
        // let energy = energy_for_real_water(&points, &water_atoms);
        // println!("Energy for water: {}", energy);
    }

}
