
use core::f64;

use pyo3::prelude::*;
use kdtree::distance::squared_euclidean;
use kdtree::KdTree;

use crate::atom::Atom;
use crate::energy;
use crate::consts;
use crate::geometry;
use crate::water::WaterMolecule;

pub enum ProbeType {
    ODa,
    OW,
    HW,
}


#[derive(Clone, Debug)]
#[pyclass]
pub struct GridPoint {
    pub index: usize,
    pub coords: [f64; 3],
    pub energy_oda: f64,
    pub energy_ow: f64,
    pub energy_hw: f64,
}

impl PartialEq for GridPoint {
    fn eq(&self, other: &Self) -> bool {
        self.coords == other.coords
    }
}

impl Eq for GridPoint { }

#[derive(Clone, Debug)]
#[pyclass]
pub struct Grid3D {
    pub data: Vec<GridPoint>,
    
    // boundaries
    x_min: f64,
    y_min: f64,
    z_min: f64,
    x_max: f64,
    y_max: f64,
    z_max: f64,

    spacing: f64, 

    // tree: RTree<GridPoint>,
    pub kdtree: KdTree<f64, usize, [f64; 3]>,

    // updated: bool,
}

#[pymethods]
impl Grid3D {
    #[new]
    pub fn new(size: (f64, f64, f64), spacing: f64, center: [f64; 3]) -> Self {
        let (width, height, depth) = size;
        let cx = center[0];
        let cy =  center[1];
        let cz = center[2];
    
        // Calculate boundaries
        let x_min = cx - width / 2.0;
        let x_max = cx + width / 2.0;
        let y_min = cy - height / 2.0;
        let y_max = cy + height / 2.0;
        let z_min = cz - depth / 2.0;
        let z_max = cz + depth / 2.0;
    
        // Calculate the number of points along each dimension
        let x_points = ((x_max - x_min) / spacing).ceil() as usize + 1;
        let y_points = ((y_max - y_min) / spacing).ceil() as usize + 1;
        let z_points = ((z_max - z_min) / spacing).ceil() as usize + 1;
    
        // Preallocate the vector
        let mut data = Vec::with_capacity(x_points * y_points * z_points);
        let kdtree: KdTree<f64, usize, [f64; 3]> = KdTree::new(3);

        // Generate points using iterators
        // let mut counter = 0;
        let mut index = 0;
        for x in (0..x_points).map(|i| x_min + i as f64 * spacing) {
            for y in (0..y_points).map(|j| y_min + j as f64 * spacing) {
                for z in (0..z_points).map(|k| z_min + k as f64 * spacing) {
                    let coords = [x, y, z];
                    data.push(GridPoint {
                        index: index,
                        coords,
                        energy_oda: f64::INFINITY,
                        energy_ow: f64::INFINITY,
                        energy_hw: f64::INFINITY,
                    });
                    index += 1;
                }
            }
        }
    
        Grid3D {
            data,
            x_min,
            y_min,
            z_min,
            x_max,
            y_max,
            z_max,
            spacing,
            kdtree,
        }
    }
}

impl Grid3D {
    
    pub fn update_energies(&mut self, new_points: &Vec<Atom>) { 
        // let radius_sq = (12.0_f64).powf(2.); // Use squared distance for efficiency

        // for atom in new_points {
        //     // Find grid points within radius of this atom
        //     let neighbors = self.kdtree.within(
        //         &atom.coords(),
        //         radius_sq,
        //         &kdtree::distance::squared_euclidean,
        //     ).unwrap();

        //     for (_dist_sq, &idx) in neighbors {
        //         let point = &mut self.data[idx];
        //         let (oda_energy, ow_energy, q_energy) = energy::update_grid_energies(&vec![atom.clone()], &point.coords);
        //         point.energy_oda += oda_energy;
        //         point.energy_ow += ow_energy;
        //         point.energy_hw += q_energy;
        //     }
        // }
        for point in self.all_points_mut() {
            let (oda_energy, ow_energy, q_energy) = energy::update_grid_energies(new_points, &point.coords);
            point.energy_oda += oda_energy;
            point.energy_ow += ow_energy;
            point.energy_hw += q_energy;
        }
    }

    pub fn remove_points(&mut self, points_to_remove: &Vec<Atom>) {
        // let radius_sq = (12.0_f64).powf(2.);

        // for atom in points_to_remove {
        //     let neighbors = self.kdtree.within(
        //         &atom.coords(),
        //         radius_sq,
        //         &kdtree::distance::squared_euclidean,
        //     ).unwrap();

        //     for (_dist_sq, &idx) in neighbors {
        //         let point = &mut self.data[idx];
        //         let (oda_energy, ow_energy, q_energy) = energy::update_grid_energies(&vec![atom.clone()], &point.coords);
        //         point.energy_oda -= oda_energy;
        //         point.energy_ow -= ow_energy;
        //         point.energy_hw -= q_energy;
        //     }
        // }
        
        for point in self.all_points_mut() {
            let (oda_energy, ow_energy, q_energy) = energy::update_grid_energies(points_to_remove, &point.coords);
            point.energy_oda -= oda_energy;
            point.energy_ow -= ow_energy;
            point.energy_hw -= q_energy;
        }
    }


    pub fn all_points(&self) -> &Vec<GridPoint> {
        &self.data
    }

    pub fn all_points_mut(&mut self) -> &mut Vec<GridPoint>{
        &mut self.data
    }

    pub fn in_box(&self, point: &[f64; 3]) -> bool {
        (self.x_min <= point[0] && point[0] <= self.x_max) && 
        (self.y_min <= point[1] && point[1] <= self.y_max) && 
        (self.z_min <= point[2] && point[2] <= self.z_max) 
    }

    pub fn build_kdtree(&mut self) {
        let mut tree = KdTree::new(3);
        for (idx, point) in self.data
            .iter()
            .enumerate() {
                let _ = tree.add(point.coords, idx);
        }
        self.kdtree = tree;
    }

    pub fn is_close_to_edge(&self, xyz: &[[f64; 3]], distance: f64) -> bool {
        xyz.iter().any(|&[x, y, z]| {
            let x_close = (self.x_min - x).abs() <= distance || (self.x_max - x).abs() <= distance;
            let y_close = (self.y_min - y).abs() <= distance || (self.y_max - y).abs() <= distance;
            let z_close = (self.z_min - z).abs() <= distance || (self.z_max - z).abs() <= distance;
            x_close || y_close || z_close
        })
    }

    pub fn get_nearest_neighbor(&self, query_point: &[f64; 3]) -> Option<&GridPoint> {
        if self.in_box(query_point) {
            let nearest = self.kdtree
                .nearest(query_point, 
                    1,
                &squared_euclidean::<f64>,
            ).unwrap();
            if let Some((_distance, &index)) = nearest.first() {
                let nearest_point = &self.data[index];
                Some(nearest_point)
            }
            else {
                None
            } 
        } else {
            None
        }
    }

    pub fn get_neighbors_within_distance(&self, query_point:&[f64; 3], max: f64, min: f64) -> Vec<&GridPoint> {
        let min_distance: f64 = min.powf(2.0); // Minimum distance in angstroms
        let max_distance: f64 = max.powf(2.0); // Maximum distance in angstroms

        let within_max_distance = self.kdtree
            .within(query_point, max_distance, &squared_euclidean)
            .unwrap();

        let in_range: Vec<_> = within_max_distance
            .into_iter()
            .filter(|&(distance, _)| distance >= min_distance) // Compare squared distances
            .collect();
        
        let mut neighbor_points = Vec::new();
        for (_distance, &index) in in_range {
            if !self.is_close_to_edge(&[self.data[index].coords], 1.) {
                neighbor_points.push(&self.data[index]);
            }
        }
        neighbor_points
    }

    pub fn get_neighbors_within_distance_and_angle(&self, anchor_xyz: &[f64; 3], vector_xyz: &[f64; 3], max: f64, min: f64) -> Vec<&GridPoint> {
        let min_distance: f64 = min.powf(2.0); // Minimum distance in angstroms
        let max_distance: f64 = max.powf(2.0); // Maximum distance in angstroms

        let within_max_distance = self.kdtree
            .within(anchor_xyz, max_distance, &squared_euclidean)
            .unwrap();

        let in_range: Vec<_> = within_max_distance
            .into_iter()
            .filter(|&(distance, _)| distance >= min_distance) // Compare squared distances
            .collect();
        
        let mut neighbor_points = Vec::new();
        for (_distance, &index) in in_range {
            if geometry::calculate_angle(&self.data[index].coords, anchor_xyz, vector_xyz).to_degrees() >= 90.0  && !self.is_close_to_edge(&[self.data[index].coords], 1.){
                neighbor_points.push(&self.data[index]);
            }
        }
        neighbor_points
    }


    /// Perform trilinear interpolation at a point (x, y, z)
    pub fn trilinear_interpolation(&self, point: [f64; 3], probe_type: ProbeType) -> Option<f64> {
        let (x, y, z) = (point[0], point[1], point[2]);

        // Check if the point is within the grid boundaries
        if !self.in_box(&[x, y, z]) {
            return None; // Point is outside the grid
        }

        // Find the indices of the grid cell containing the point
        let i = ((x - self.x_min) / self.spacing).floor() as usize;
        let j = ((y - self.y_min) / self.spacing).floor() as usize;
        let k = ((z - self.z_min) / self.spacing).floor() as usize;

        // Get the coordinates of the cell's corners
        let x0 = self.x_min + i as f64 * self.spacing;
        let y0 = self.y_min + j as f64 * self.spacing;
        let z0 = self.z_min + k as f64 * self.spacing;

        // Get the values at the 8 corners of the cell
        let c000 = self.get_energy_at(i, j, k, &probe_type)?;
        let c001 = self.get_energy_at(i, j, k + 1, &probe_type)?;
        let c010 = self.get_energy_at(i, j + 1, k, &probe_type)?;
        let c011 = self.get_energy_at(i, j + 1, k + 1, &probe_type)?;
        let c100 = self.get_energy_at(i + 1, j, k, &probe_type)?;
        let c101 = self.get_energy_at(i + 1, j, k + 1, &probe_type)?;
        let c110 = self.get_energy_at(i + 1, j + 1, k, &probe_type)?;
        let c111 = self.get_energy_at(i + 1, j + 1, k + 1, &probe_type)?;

        // Compute the weights
        let xd = (x - x0) / self.spacing;
        let yd = (y - y0) / self.spacing;
        let zd = (z - z0) / self.spacing;

        // Interpolate along x
        let c00 = c000 * (1.0 - xd) + c100 * xd;
        let c01 = c001 * (1.0 - xd) + c101 * xd;
        let c10 = c010 * (1.0 - xd) + c110 * xd;
        let c11 = c011 * (1.0 - xd) + c111 * xd;

        // Interpolate along y
        let c0 = c00 * (1.0 - yd) + c10 * yd;
        let c1 = c01 * (1.0 - yd) + c11 * yd;

        // Interpolate along z
        let c = c0 * (1.0 - zd) + c1 * zd;

        Some(c)
    }

    /// Helper function to get the energy at a specific grid point (i, j, k)
    fn get_energy_at(&self, i: usize, j: usize, k: usize, probe_type: &ProbeType) -> Option<f64> {
        let x_points = ((self.x_max - self.x_min) / self.spacing).ceil() as usize + 1;
        let y_points = ((self.y_max - self.y_min) / self.spacing).ceil() as usize + 1;
        let z_points = ((self.z_max - self.z_min) / self.spacing).ceil() as usize + 1;
    
        // Ensure indices are within valid range
        if i >= x_points || j >= y_points || k >= z_points {
            return None;
        }
        // Compute correct 1D index
        let index = i * (y_points * z_points) + j * z_points + k;
        // let index = i + x_points * (j + y_points * k);
        if index < self.data.len() {
            match probe_type {
                ProbeType::ODa => Some(self.data[index].energy_oda),

                ProbeType::OW => Some(self.data[index].energy_ow),
                
                // Interpolation is only for electrostatics
                ProbeType::HW => Some(self.data[index].energy_hw)
            }
        } else {
            None
        }
    }
}

pub fn get_systems_energy(grid: &Grid3D, water_atoms: &Vec<WaterMolecule>) -> f64 {
    let mut total_energy = 0.0;

    for water in water_atoms {
        let atoms = water.as_vec();
        let oxygen = atoms[0].coords();
        let h1 = atoms[1].coords();
        let h2 = atoms[2].coords();
        let oda = grid.trilinear_interpolation(oxygen, ProbeType::ODa).unwrap();
        // println!("ODA: {oda}");
        let lj_oxygen = grid.get_nearest_neighbor(&oxygen).unwrap().energy_ow;
        // println!("LJ {lj_oxygen}");
        let electrostatics_h1 = grid.trilinear_interpolation(h1, ProbeType::HW);
        let electrostatics_h2 = grid.trilinear_interpolation(h2, ProbeType::HW);
        let electrostatics_oxygen = grid.trilinear_interpolation(oxygen, ProbeType::HW);

        if electrostatics_h1.is_none() || electrostatics_h2.is_none() || electrostatics_oxygen.is_none() {
            continue; // Skip invalid configurations
        }
        // println!("LJ: {lj_oxygen}");
        // println!("Q O: {}", electrostatics_oxygen.unwrap() * consts::OXYGEN_W_Q_TIP3PFB);
        // println!("Q H1: {}", electrostatics_h1.unwrap() * consts::HYDROGEN_W_Q_TIP3PFB);
        // println!("Q H2: {}", electrostatics_oxygen.unwrap() * consts::HYDROGEN_W_Q_TIP3PFB);
        
        let energy_value = lj_oxygen
            + electrostatics_oxygen.unwrap() * consts::OXYGEN_W_Q_TIP3PFB
            + electrostatics_h1.unwrap() * consts::HYDROGEN_W_Q_TIP3PFB
            + electrostatics_h2.unwrap() * consts::HYDROGEN_W_Q_TIP3PFB;
        // println!("Water's energy: {energy_value}");
        total_energy += energy_value;    
    }
    total_energy
}