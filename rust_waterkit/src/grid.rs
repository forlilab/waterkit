
use std::collections::HashSet;

use rayon::prelude::*;

use kiddo::{float::kdtree::KdTree, SquaredEuclidean};

use crate::{atom::Atom, energy::spheric_energy, utils::{SHELL_LIMIT, WATER_LIMIT}};


#[derive(Clone, Debug)]
pub struct GridPoint {
    pub index: usize,
    pub coords: [f64; 3],
    pub energy: f64,
    pub updated: bool,
}

impl PartialEq for GridPoint {
    fn eq(&self, other: &Self) -> bool {
        self.coords == other.coords
    }
}

impl Eq for GridPoint { }

pub struct Grid3D {
    pub data: Vec<GridPoint>,
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
    tree: KdTree<f32, u32, 3, 300000, u32>,

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
        let tree = KdTree::new();
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
                        energy: 0.0,
                        updated: false,
                    });
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
            tree,
            // updated: false,
        }
    }

    pub fn update_grid_energies(
        &mut self,
        receptor_points: &Vec<Atom>) {
        self.all_points_mut()
            .par_iter_mut()
            // .filter(|point| !to_exclude.contains(&point.coords))
            .for_each(|point| {
                let energy = spheric_energy(receptor_points, &point.coords);
                point.energy = energy;
                point.updated = false;
            });
        // self.updated = true;
    }

    pub fn get(&self, x: f64, y: f64, z: f64) -> Option<&GridPoint> {
        if !self.in_box(&[x, y, z]) {
            return None
        }
        let i = ((x - self.x_min) / self.spacing).round() as usize;
        let j = ((y - self.y_min) / self.spacing).round() as usize;
        let k = ((z - self.z_min) / self.spacing).round() as usize;

        let y_points = ((self.y_max - self.y_min) / self.spacing).ceil() as usize + 1;
        let z_points = ((self.z_max - self.z_min) / self.spacing).ceil() as usize + 1;

        let index = i * y_points * z_points + j * z_points + k;

        self.data.get(index) // Safely access the data point
    }

    pub fn set(&mut self, x: f64, y: f64, z: f64, energy: f64) {
        if self.in_box(&[x, y, z]) {
            let i = ((x - self.x_min) / self.spacing).round() as usize;
            let j = ((y - self.y_min) / self.spacing).round() as usize;
            let k = ((z - self.z_min) / self.spacing).round() as usize;

            let y_points = ((self.y_max - self.y_min) / self.spacing).ceil() as usize + 1;
            let z_points = ((self.z_max - self.z_min) / self.spacing).ceil() as usize + 1;

            let index = i * y_points * z_points + j * z_points + k;
            self.data[index].energy = energy;
        }
    }

    pub fn in_box(&self, point: &[f64; 3]) -> bool {
        (self.x_min <= point[0] && point[0] <= self.x_max) && 
        (self.y_min <= point[1] && point[1] <= self.y_max) && 
        (self.z_min <= point[2] && point[2] <= self.z_max) 

    }

    pub fn all_points_mut(&mut self) -> &mut [GridPoint] {
        &mut self.data
    }

    pub fn all_points(&self) -> &[GridPoint] {
        &self.data
    }

    pub fn all_points_as_vec(&self) -> &Vec<GridPoint> {
        &self.data
    }

    pub fn extract_energies_and_coordinates_parallel(&self) -> (Vec<f64>, Vec<[f64; 3]>) {
        let energies: Vec<f64> = self.data.par_iter().map(|point| point.energy).collect();
        let coordinates: Vec<[f64; 3]> = self.data.par_iter().map(|point| point.coords).collect();
        (energies, coordinates)
    }

    pub fn get_energies_for_points(&self, points: &Vec<[f64; 3]>) -> Vec<f64> {
        let mut energies: Vec<f64> = Vec::new();
        for point in points {
            energies.push(self.get(point[0], point[1], point[1]).unwrap().energy);

        }
        energies
    }

    // Returns the list of points within the specified min and max from the anchor_point
    pub fn get_neighbor_for_point(&self, anchor_point: &[f64; 3]) -> Vec<GridPoint> {
        let min = 2.5f32.powf(2.0);
        let max = 3.6f32.powf(2.0);

        let mut points_for_shell = Vec::new();
        let coords = anchor_point;
        let tmp_max: Vec<u32> = self.tree
            .within::<kiddo::SquaredEuclidean>(&[coords[0] as f32, coords[1] as f32, coords[2] as f32], max)
            .into_iter()
            .map(|n| n.item)
            .collect();
        let tmp_min: Vec<u32> = self.tree
            .within::<kiddo::SquaredEuclidean>(&[coords[0] as f32, coords[1] as f32, coords[2] as f32], min)
            .into_iter()
            .map(|n| n.item)
            .collect();
        // println!("Max: {} Min: {}", tmp_max.len(), tmp_min.len());
        let points: Vec<u32> = difference(&tmp_max, &tmp_min);
        for p in points {
            points_for_shell.push(self.data[p as usize].clone());
        }
        // println!("Grid updated: {}", self.updated);
        // for point in points_for_shell.clone() {
        //     println!("Point updated: {}", point.updated);
        // }
        points_for_shell
    }

    pub fn get_neighbors(&self, receptor_points_in_box: &Vec<[f64; 3]>) -> Vec<GridPoint>{
        let min = 2.5f32.powf(2.0);
        let max = 3.6f32.powf(2.0);

        let mut points_for_shell = Vec::new();
        for point in receptor_points_in_box {
            let coords = &point;
            let tmp_max: Vec<u32> = self.tree
                .within::<kiddo::SquaredEuclidean>(&[coords[0] as f32, coords[1] as f32, coords[2] as f32], max)
                .into_iter()
                .map(|n| n.item)
                .collect();
            let tmp_min: Vec<u32> = self.tree
                .within::<kiddo::SquaredEuclidean>(&[coords[0] as f32, coords[1] as f32, coords[2] as f32], min)
                .into_iter()
                .map(|n| n.item)
                .collect();
            // println!("Max: {} Min: {}", tmp_max.len(), tmp_min.len());
            let points: Vec<u32> = difference(&tmp_max, &tmp_min);
            for p in points {
                points_for_shell.push(self.data[p as usize].clone());
            }
        }
        points_for_shell
    }

    pub fn get_nearest(&self, query_point: &[f64; 3]) -> GridPoint {
        let q_point = [query_point[0] as f32, query_point[1] as f32, query_point[2] as f32];
        let nearest = self.tree.nearest_one::<kiddo::SquaredEuclidean>(&q_point);
        self.data[nearest.item as usize].clone()
    }

    pub fn set_possible_points(&mut self) {
        let mut tree: KdTree<f32, u32, 3, 300000, u32> = KdTree::new();
        let all_points = self.all_points();

        for (idx, gridpoint) in all_points.iter().enumerate() {
            let coords = &gridpoint.coords;
            tree.add(&[coords[0] as f32, coords[1] as f32, coords[2] as f32], idx as u32);
        }
        self.tree = tree;
        println!("Created the first tree");
    }

    pub fn get_receptor_points_in_grid(&self, receptor_map: &Vec<Atom>) -> Vec<Atom> {
        let mut grid_points = Vec::new();
        for atom in receptor_map {
            let coords = atom.coords();
            if self.in_box(&coords) {
                // println!("{:?}", coords);
                grid_points.push(atom.clone());
            }
        }
        grid_points
    } 

    pub fn get_anchor_points_in_grid(&self, receptor_map: &Vec<[f64; 3]>) -> Vec<[f64; 3]> {
        let mut grid_points = Vec::new();
        for atom in receptor_map {
            if self.in_box(atom) {
                // println!("{:?}", coords);
                grid_points.push(atom.clone());
            }
        }
        grid_points
    } 
}

pub fn update_grid_energies(
    source_grid: &Grid3D,
    target_grid: &mut Grid3D,
) {
    // Ensure dimensions match
    assert_eq!(source_grid.data.len(), target_grid.data.len(), "Grids must have the same size");

    target_grid
        .all_points_mut()
        .par_iter_mut()
        .zip(source_grid.all_points().par_iter())
        .for_each(|(target_point, source_point)| {
            target_point.energy += source_point.energy;
        });
}

fn difference<T: Eq + std::hash::Hash + Clone>(vec1: &[T], vec2: &[T]) -> Vec<T> {
    let set2: HashSet<_> = vec2.iter().collect();
    vec1.iter()
        .filter(|item| !set2.contains(item))
        .cloned()
        .collect()
}