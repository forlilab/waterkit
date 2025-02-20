use core::f64;
use std::io::Write;
use std::fs::OpenOptions;
use pyo3::prelude::*;
use rand::seq::SliceRandom;
use rand::thread_rng;
use rayon::prelude::*;

use crate::anchor_point::AnchorPoint;
use crate::atom::Atom;
use crate::energy::energy_for_real_water;
use crate::grid::Grid3D;
use crate::grid::ProbeType;
use crate::consts;
use crate::monte_carlo;
use crate::optimizer::optimize;
use crate::energy;
use crate::sampling::sample;
use crate::sampling::sample_using_grids;
use crate::setup::setup_grid;

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
        .create(true)// Optionally create the file if it doesn't already exist
        .append(true)
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
        // let mut receptor_map = receptor_points.clone();
        let mut receptor_map = Vec::new();
        let mut mutable_anchor_points = anchor_points.clone();

        // select 16 random samples of waters configurations
        // let mut rng = thread_rng();
        // let num_samples = 16;
        // let sampled_configurations = water_configurations.choose_multiple(&mut rng, num_samples).cloned().collect();

        // while mutable_anchor_points.len() > 0 {
        for _i in 0..3 {
            println!("Epoch {}", _i);
            // println!("Receptor: {}", receptor_map.len());
            sample(&mut grid, &mut receptor_map, &mut mutable_anchor_points, &water_configurations);
            // println!("Receptor: {}\n", receptor_map.len());
            let waters: Vec<Atom> = receptor_map.iter().filter(|x| x.atom_type() == "OW" || x.atom_type() == "HW").cloned().collect();
            to_pdb(&waters, &format!("test/water_{_i}_unoptimized.pdb"));
        }
        let waters: Vec<Atom> = receptor_map.iter().filter(|x| x.atom_type() == "OW" || x.atom_type() == "HW").cloned().collect();
        to_pdb(&waters, &format!("test/water_unoptimized.pdb"));
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
        for _i in 0..3 {
            sample_using_grids(&mut grid,&mut receptor_map, &mut mutable_anchor_points, &water_configurations, &mut new_water_molecules);
        }

        // let waters: Vec<Atom> = receptor_map.iter().filter(|x| x.atom_type() == "OW" || x.atom_type() == "HW").cloned().collect();
        let mut waters = Vec::new();
        for w in new_water_molecules.iter() {
            for a in w.as_vec() {
                waters.push(a.clone());
            }
        } 
        to_pdb(&waters, &format!("test/water_{epoch}_unoptimized.pdb"));
        // Need to get energies for the waters placed and select the ones to perturb based on an 
        // inverted boltzmann wheighted choice
        let mut water_energies = Vec::new();
        for water in new_water_molecules.iter() {
            let atoms = water.as_vec();
            let waters_in_system: Vec<Atom> = receptor_map.iter().filter(|x| !atoms.contains(x)).cloned().collect();
            water_energies.push(energy::energy_for_real_water(&waters_in_system, &atoms));         
        }
        
        // let mut optimized_waters: Vec<Atom> = Vec::new();
        // let steps = water_energies.len();
        // let indices = monte_carlo::inverted_boltzmann_choices(&water_energies, Some(steps));
        // // println!("# of choices: {} out of {} total waters.", indices.len(), steps);
        // for (index, water_energy) in  water_energies.iter().enumerate() {
        //     // Select a water molecule to modify
        //     let new_water = new_water_molecules[index].clone();
        //     let waters_in_system: Vec<Atom> = receptor_map.iter().filter(|x| !new_water.as_vec().contains(x)).cloned().collect();
        //     if indices.contains(&index) {
        //         let optimized_water = optimize(&new_water, &waters_in_system,  &mut grid);
        //         for a in optimized_water.as_vec() {
        //             optimized_waters.push(a.clone());
        //         }
        //     } else {
        //         for a in new_water.as_vec() {
        //             optimized_waters.push(a.clone());
        //         }
        //     }
        // }

        // to_pdb(&optimized_waters, &format!("test/water_{epoch}_optimized.pdb"));

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

    // for point in grid.all_points() {
    //     println!("{} {} {} {}", point.energy_hw, point.coords[0], point.coords[1], point.coords[2]);
    // }

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
        let use_grids = false;

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
            energy += grid_receptor_and_w.trilinear_interpolation(oxygen.coords(), ProbeType::HW).unwrap() * consts::OXYGEN_W_Q;
            println!("{index} {index} {energy} O (rec+wat)");

            // let mut energy_rec = energy_for_real_water(&receptor_points, &vec![oxygen.clone()]);
            let mut energy_rec = grid_receptor.trilinear_interpolation(oxygen.coords(), ProbeType::OW).unwrap();
            energy_rec += grid_receptor.trilinear_interpolation(oxygen.coords(), ProbeType::HW).unwrap() * consts::OXYGEN_W_Q;
            println!("{index} {index} {energy_rec} O (just rec)");

            // let e = energy_for_real_water(&points, &vec![h1.clone()]);
            let e = grid_receptor_and_w.trilinear_interpolation(h1.coords(), ProbeType::HW).unwrap() * consts::HYDROGEN_W_Q;
            println!("{index} {index} {e} H (rec+wat)");
            energy += e;

            // let er = energy_for_real_water(&receptor_points, &vec![h1.clone()]);
            let er = grid_receptor.trilinear_interpolation(h1.coords(), ProbeType::HW).unwrap() * consts::HYDROGEN_W_Q;
            println!("{index} {index} {er} H (just rec)");
            energy_rec += er;

            // let e = energy_for_real_water(&points, &vec![h2.clone()]);
            let e = grid_receptor_and_w.trilinear_interpolation(h2.coords(), ProbeType::HW).unwrap() * consts::HYDROGEN_W_Q;
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
            // println!("{}", receptor_points.len());
            let mut energy = energy_for_real_water(&points, &vec![oxygen.clone()]);
            println!("{index} {index} {energy} O (rec+wat)");

            // let mut energy_w = energy_for_real_water(&waters_network, &vec![oxygen.clone()]);
            // println!("{index} {index} {energy_w} O (just wat)");

            let mut energy_rec = energy_for_real_water(&receptor_points, &vec![oxygen.clone()]);
            println!("{index} {index} {energy_rec} O (just rec)");

            let e = energy_for_real_water(&points, &vec![h1.clone()]);
            println!("{index} {index} {e} H (rec+wat)");
            energy += e;

            // let ew = energy_for_real_water(&waters_network, &vec![h1.clone()]);
            // println!("{index} {index} {ew} H (just wat)");
            // energy_w += ew;

            let er = energy_for_real_water(&receptor_points, &vec![h1.clone()]);
            println!("{index} {index} {er} H (just rec)");
            energy_rec += er;

            let e = energy_for_real_water(&points, &vec![h2.clone()]);
            println!("{index} {index} {e} H (rec+wat)");
            energy += e;

            // let ew = energy_for_real_water(&waters_network, &vec![h2.clone()]);
            // println!("{index} {index} {ew} H (just wat)\n");
            // energy_w += ew;

            let er = energy_for_real_water(&receptor_points, &vec![h2.clone()]);
            println!("{index} {index} {er} H (just rec)");
            energy_rec += er;

            println!("{index} {index} {energy} HOH (rec+wat)");
            // println!("{index} {index} {energy_w} HOH (just wat)\n");
            println!("{index} {index} {energy_rec} HOH (just rec)\n");
            // println!();
        }
        // println!("{:?}", water);
        // println!("{:?}\n", water_atoms);
        // let energy = energy_for_real_water(&points, &water_atoms);
        // println!("Energy for water: {}", energy);
    }

}
