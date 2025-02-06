use core::f64;
use std::num::NonZeroUsize;

use pyo3::ffi::PyBUF_MAX_NDIM;
use rand::seq::SliceRandom;
use rayon::prelude::*;
use kdtree::distance::squared_euclidean;
use kdtree::KdTree;

use crate::atom::Atom;
// use crate::energy::spheric_energy;
use crate::geometry::{self, resize_vector};
use crate::consts::ELECTROSTATICS_CUTOFF;
use crate::vina_ff::vina_energy;
use crate::water::WaterMolecule;

#[derive(Clone, Debug)]
pub struct GridPoint {
    pub index: usize,
    pub coords: [f64; 3],
    pub energy: f64,
}

impl PartialEq for GridPoint {
    fn eq(&self, other: &Self) -> bool {
        self.coords == other.coords
    }
}

impl Eq for GridPoint { }

#[derive(Clone, Debug)]
pub struct Grid3D {
    data: Vec<GridPoint>,
    x_size: f64,
    y_size: f64,
    z_size: f64,
    
    // center
    c_x: f64,
    c_y: f64,
    c_z: f64,
    
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

impl Grid3D {
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
                        energy: f64::INFINITY,
                    });
                    // println!("H {} {} {}", x, y, z);
                    index += 1;
                }
            }
        }
    
        Grid3D {
            data,
            x_size: size.0,
            y_size: size.1,
            z_size: size.2,
            c_x: cx,
            c_y: cy,
            c_z: cz,
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

    pub fn update_grid_energies(
        &mut self,
        receptor_points: &Vec<Atom>) {
        self.all_points_mut()
            .par_iter_mut()
            .for_each(|point| {
                let energy = vina_energy(receptor_points, &point.coords);
                point.energy = energy;
            });
    }

    pub fn update_energies(&mut self, new_points: &Vec<Atom>) {
        self.all_points_mut()
            .par_iter_mut()
            .for_each(|point| {
                let energy = vina_energy(new_points, &point.coords);
                point.energy += energy;
            });
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

    pub fn get_nearest_neighbor(&mut self, query_point: &[f64; 3]) -> Option<&GridPoint> {
        if self.in_box(query_point) {
            let nearest = self.kdtree
                .nearest(query_point, 
                    1,
                &squared_euclidean::<f64>,
            ).unwrap();
            if let Some((distance, &index)) = nearest.first() {
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

        // Query all points within the maximum distance (3.6 Å)
        let within_max_distance = self.kdtree
            .within(query_point, max_distance, &squared_euclidean)
            .unwrap();
        // println!("Points found: {}", within_max_distance.len());

        // Filter out points that are closer than the minimum distance (2.5 Å)
        let in_range: Vec<_> = within_max_distance
            .into_iter()
            .filter(|&(distance, _)| distance >= min_distance) // Compare squared distances
            .collect();
        // println!("Points found: {}", in_range.len());
        
        let mut neighbor_points = Vec::new();
        for (distance, &index) in in_range {
            neighbor_points.push(&self.data[index]);
        }
        // println!("# Neighbors found: {}", neighbor_points.len());
        neighbor_points
    }

    pub fn get_neighbors_within_distance_and_angle(&self, anchor_xyz: &[f64; 3], vector_xyz: &[f64; 3], max: f64, min: f64) -> Vec<&GridPoint> {
        let min_distance: f64 = min.powf(2.0); // Minimum distance in angstroms
        let max_distance: f64 = max.powf(2.0); // Maximum distance in angstroms

        // Query all points within the maximum distance (3.6 Å)
        let within_max_distance = self.kdtree
            .within(anchor_xyz, max_distance, &squared_euclidean)
            .unwrap();
        // println!("Points found: {}", within_max_distance.len());

        // Filter out points that are closer than the minimum distance (2.5 Å)
        let in_range: Vec<_> = within_max_distance
            .into_iter()
            .filter(|&(distance, _)| distance >= min_distance) // Compare squared distances
            .collect();
        // println!("Points found: {}", in_range.len());
        
        let mut neighbor_points = Vec::new();
        for (distance, &index) in in_range {
            if geometry::calculate_angle(&self.data[index].coords, anchor_xyz, vector_xyz).to_degrees() >= 90.0 {
                neighbor_points.push(&self.data[index]);
            }
        }
        // println!("# Neighbors found: {}", neighbor_points.len());
        neighbor_points
    }
}
