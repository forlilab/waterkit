use rand::Rng;
use rayon::prelude::*;

use crate::anchor_point::AnchorPoint;
use crate::{geometry, monte_carlo as mc};
use crate::grid::{Grid3D, GridPoint, ProbeType};
use crate::consts::*;
use crate::atom::Atom;
use crate::water::WaterMolecule;
use crate::energy::energy_for_real_water;

fn optimize_placement_order_grid(grid: &Grid3D, points: &Vec<AnchorPoint>) -> Vec<AnchorPoint> {
    let mut energies = Vec::new();
    let mut min_points = Vec::new();
    let mut decisions = Vec::new();

    for point in points.iter() {
        // println!("{:?}", point);
        let mut min = 2.5;
        let mut max = 3.6;
        if point.hb_type() == "donor" {
            min -= 1.0;
            max -= 1.0;
        }
        let neighbors = grid.get_neighbors_within_distance_and_angle(point.anchor_point(), point.anchor_vectors(), max, min);
        // let neighbors = grid.get_neighbors_within_distance(point.anchor_point(), max, min);
        // println!("Neighbors: {:?}", neighbors.len());
        if neighbors.len() > 0 {
            let mut min_point = neighbors.first().cloned().unwrap().clone();
            for neigh in neighbors.into_iter() {
                if neigh.energy_oda < min_point.energy_oda {
                    min_point = neigh.clone();
                }
            }
            let energy;
            if min_point.energy_oda == 0. { energy = f64::INFINITY; }
            else { energy = min_point.energy_oda;}
            energies.push(energy);
            min_points.push(point);
        }
        
    }
    // println!("Energies: {:?}", energies);
    let order = mc::boltzmann_choices(&energies, Some(energies.len()));
    if order.len() > 0 {
        for order_idx in order.into_iter() {
            if mc::boltzmann_acceptance_rejection(&energies[order_idx], 
                &BOLTZMANN_ENERGY_CUTOFF, 
                &TEMPERATURE, 
            &BOLTZMANN_K) {
                if !decisions.contains(min_points[order_idx]) {
                    decisions.push(min_points[order_idx].clone());
                }
            }
        }
    }
    decisions
}


fn optimize_poistion_grid(grid: &Grid3D, point: &AnchorPoint) -> [f64; 3] {
    let max_disp = 0.375/2.;
    let mut min = 2.5;
    let mut max = 3.6;
    if point.hb_type() == "donor" {
        min -= 1.0;
        max -= 1.0;
    }
    let neighbors = grid.get_neighbors_within_distance_and_angle(point.anchor_point(), point.anchor_vectors(), max, min);
    // let neighbors = grid.get_neighbors_within_distance(point.anchor_point(), max, min);
    let energies: Vec<f64> = neighbors.iter().map(|x| x.energy_oda).collect();
    let choice = mc::boltzmann_choices(&energies, None);
    if choice.len() > 0 {
        let idx = choice.first().unwrap().clone();
        let new_point = neighbors[idx].clone();
        // return new_point.coords;
        let mut rng = rand::thread_rng();
        let displacement = [rng.gen_range(-max_disp..max_disp), 
            rng.gen_range(-max_disp..max_disp),
            rng.gen_range(-max_disp..max_disp)];
        
        return geometry::sum_points(&new_point.coords, &displacement);
    }
    *point.anchor_point()
}


pub fn sample(grid: &mut Grid3D, receptor_points: &mut Vec<Atom>, anchor_points: &mut Vec<AnchorPoint>, water_configurations: &Vec<[f64; 6]>) -> bool {
    
    let mut new_anchor_points: Vec<AnchorPoint> = Vec::new();
    let placement: bool = false;
    let mut receptor_points_on_the_grid = Vec::new();
    for point in anchor_points.into_iter() {
        if grid.in_box(point.anchor_point()){
            receptor_points_on_the_grid.push(point.clone());
        }
    }

    // Sample with Boltzmann the neighbors and the actual point and based on Metropolis 
    // acceptance criteria then these are the starting anchor points
    // let start_time = Instant::now();
    let decisions = optimize_placement_order_grid(grid, &receptor_points_on_the_grid);
    for (_idx, decision) in decisions.iter().enumerate() {
        let new_point = optimize_poistion_grid(grid, decision);
        let point_energy = grid.trilinear_interpolation(new_point, ProbeType::ODa).unwrap();
        if mc::boltzmann_acceptance_rejection(&point_energy, &BOLTZMANN_ENERGY_CUTOFF, &TEMPERATURE, &BOLTZMANN_K) {
            // Now we build explicit water
            let (placed, mut water) = sample_real_waters(&new_point, water_configurations, receptor_points);
            if placed {
                water.guess_new_hydrogen_bonds();
                for hb in water.hydrogen_bonds().into_iter() {
                    new_anchor_points.push(hb);
                }
                // let atoms_to_update = water.as_vec();
                grid.update_energies(&receptor_points);
            }
        }
    }
    anchor_points.clear();
    for p in new_anchor_points.into_iter() {
        anchor_points.push(p);
    }
    placement
}

pub fn sample_using_grids(grid: &mut Grid3D,  
        receptor_points: &mut Vec<Atom>, 
        anchor_points: &mut Vec<AnchorPoint>, 
        water_configurations: &Vec<[f64; 6]>,
        new_water_molecules: &mut Vec<WaterMolecule>) -> bool {
    
    let mut new_anchor_points = Vec::new();
    let placement: bool = false;
    let mut receptor_points_on_the_grid = Vec::new();
    for point in anchor_points.into_iter() {
        if grid.in_box(point.anchor_point()){
            receptor_points_on_the_grid.push(point.clone());
        }
    }
    // Sample with Boltzmann the neighbors and the actual point and based on Metropolis 
    // acceptance criteria then these are the starting anchor points
    let decisions = optimize_placement_order_grid(grid, &receptor_points_on_the_grid);
    for (_idx, decision) in decisions.iter().enumerate() {
        let new_point = optimize_poistion_grid(grid, decision);
        let point_energy = grid.trilinear_interpolation(new_point, ProbeType::ODa).unwrap();
        if mc::boltzmann_acceptance_rejection(&point_energy, &BOLTZMANN_ENERGY_CUTOFF, &TEMPERATURE, &BOLTZMANN_K) {
            // Now we build explicit water
            let (placed, mut water) = sample_waters_with_grids(&new_point, water_configurations, receptor_points, grid);

            if placed {
                water.guess_new_hydrogen_bonds();
                for hb in water.hydrogen_bonds().into_iter() {
                    new_anchor_points.push(hb);
                }
                new_water_molecules.push(water.clone());
                let atoms_to_update = water.as_vec();
                grid.update_energies(&atoms_to_update);
            }
        }
    }
    anchor_points.clear();
    for p in new_anchor_points.into_iter() {
        anchor_points.push(p);
    }
    placement
}


pub fn sample_real_waters(oxygen_atom: &[f64; 3],
    water_configurations: &Vec<[f64; 6]>,
    receptor_points: &mut Vec<Atom>) -> (bool, WaterMolecule) {    
    // Let's parallelize
    let oxygen_position = oxygen_atom;
    // println!("O {} {} {}", oxygen_position[0], oxygen_position[1], oxygen_position[2]);
    // let possible_results: Vec<(WaterMolecule, f64)> = water_configurations
    //     .par_iter()
    //     .map(|configuration| {
    //         // H1 in position
    //         let h1_coords: [f64; 3] = [
    //             configuration[0] + oxygen_position[0],
    //             configuration[1] + oxygen_position[1],
    //             configuration[2] + oxygen_position[2],
    //         ];
    //         // H2 in position
    //         let h2_coords: [f64; 3] = [
    //             configuration[3] + oxygen_position[0],
    //             configuration[4] + oxygen_position[1],
    //             configuration[5] + oxygen_position[2],
    //         ];

    //         // Create water molecule
    //         let water = WaterMolecule::new(oxygen_position.clone(), 
    //             h1_coords.clone(),
    //             h2_coords.clone());

    //         // Compute energy
    //         let energy_value = energy_for_real_water(&receptor_points, &water.as_vec());
    //         // println!("Energy: {}", energy_value);
    //         (water, energy_value)
    //     })
    //     .collect();
    let mut possible_results: Vec<(WaterMolecule, f64)> =  Vec::new(); 
    for configuration in water_configurations {
        // H1 in position
        let h1_coords: [f64; 3] = [
            configuration[0] + oxygen_position[0],
            configuration[1] + oxygen_position[1],
            configuration[2] + oxygen_position[2],
        ];
        // H2 in position
        let h2_coords: [f64; 3] = [
            configuration[3] + oxygen_position[0],
            configuration[4] + oxygen_position[1],
            configuration[5] + oxygen_position[2],
        ];

        // Create water molecule
        let water = WaterMolecule::new(oxygen_position.clone(), 
            h1_coords.clone(),
            h2_coords.clone());
        // water.to_xyz();
        // Compute energy
        let energy_value = energy_for_real_water(&receptor_points, &water.as_vec());
        // println!("Energy: {}", energy_value);
        possible_results.push((water, energy_value));
    }
    let possible_waters_energies: Vec<f64> = possible_results.iter().map(|(_, e)| *e).collect();
    // let probabilities = mc::boltzmann_probabilities(&possible_waters_energies);
    // println!("Probabilities: \n{:?}", probabilities);
    let choice = mc::boltzmann_choices(&possible_waters_energies, None);
    if choice.len() > 0 {
        let value = choice.first().unwrap().clone();
        if mc::boltzmann_acceptance_rejection(&possible_results[value].1,
            &BOLTZMANN_ENERGY_CUTOFF,
            &TEMPERATURE,
            &BOLTZMANN_K) {
            // println!("Chosen energy - {}\n", possible_waters_energies[value]);
            receptor_points.extend(possible_results[value].0.as_vec());
            // for atom in possible_results[value].0.as_vec() {
            //     receptor_points.push(atom.clone());
            // }
            return (true, possible_results[value].0.clone());

        }
        else {
            return (false, WaterMolecule::new([0., 0., 0.], [0., 0., 0.], [0., 0., 0.]));
        }
    }
    else {
        return (false, WaterMolecule::new([0., 0., 0.], [0., 0., 0.], [0., 0., 0.]));
    }
}

pub fn sample_waters_with_grids(oxygen_atom: &[f64; 3],
    water_configurations: &Vec<[f64; 6]>,
    receptor_points: &mut Vec<Atom>,
    grid: &Grid3D) -> (bool, WaterMolecule) {
    let mut possible_results: Vec<(WaterMolecule, f64)> = Vec::new();
    let oxygen_position = oxygen_atom;
    // let lj_oxygen = grid.get_nearest_neighbor(&oxygen_position).unwrap().energy_oda;
    let lj_oxygen = grid.trilinear_interpolation(*oxygen_position, ProbeType::OW).unwrap();
    for configuration in water_configurations {
            // H1 in position
            let h1_coords: [f64; 3] = [
                configuration[0] + oxygen_position[0],
                configuration[1] + oxygen_position[1],
                configuration[2] + oxygen_position[2],
            ];
            // H2 in position
            let h2_coords: [f64; 3] = [
                configuration[3] + oxygen_position[0],
                configuration[4] + oxygen_position[1],
                configuration[5] + oxygen_position[2],
            ];

            // Create water molecule
            let water = WaterMolecule::new(oxygen_position.clone(), h1_coords, h2_coords);

            // Compute energy
            // Interpolation baby!
            let electrostatics_h1 = grid.trilinear_interpolation(h1_coords, ProbeType::HW);
            let electrostatics_h2 = grid.trilinear_interpolation(h2_coords, ProbeType::HW);
            let electrostatics_oxygen = grid.trilinear_interpolation(*oxygen_position, ProbeType::HW);
            
            if electrostatics_h1.is_some() && electrostatics_h2.is_some() && electrostatics_oxygen.is_some() {
                // println!("OW: {}\nO_Q: {}\nH1_Q: {}\nH2_Q: {}\n\n", 
                //     lj_oxygen, 
                //     electrostatics_oxygen.unwrap() * OXYGEN_W_Q, 
                //     electrostatics_h1.unwrap()  * HYDROGEN_W_Q, 
                //     electrostatics_h2.unwrap()  * HYDROGEN_W_Q);
                
                let energy_value = lj_oxygen + (electrostatics_oxygen.unwrap() * OXYGEN_W_Q) + (electrostatics_h1.unwrap()  * HYDROGEN_W_Q) + (electrostatics_h2.unwrap() * HYDROGEN_W_Q);
                possible_results.push((water, energy_value));
            } else {
                possible_results.push((water, f64::INFINITY));
            }
    }
    let possible_waters_energies: Vec<f64> = possible_results.iter().map(|(_, e)| *e).collect();
    // println!("{:?}", possible_waters_energies);
    // let probabilities = mc::boltzmann_probabilities(&possible_waters_energies);
    // println!("{:?}", probabilities);
    let choice = mc::boltzmann_choices(&possible_waters_energies, None);
    if choice.len() > 0 {
        let value = choice.first().unwrap().clone();
        if mc::boltzmann_acceptance_rejection(&possible_waters_energies[value],
            &BOLTZMANN_ENERGY_CUTOFF,
            &TEMPERATURE,
            &BOLTZMANN_K) {
            // for atom in possible_results[value].0.as_vec() {
            //     receptor_points.push(atom.clone());
            // }
            return (true, possible_results[value].0.clone());

        }
        else {
            return (false, WaterMolecule::new([0., 0., 0.], [0., 0., 0.], [0., 0., 0.]));
        }
    }
    else {
        return (false, WaterMolecule::new([0., 0., 0.], [0., 0., 0.], [0., 0., 0.]));
    }
}