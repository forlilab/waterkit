
use std::collections::HashSet;

use rayon::prelude::*;

use kiddo::{float::kdtree::KdTree, SquaredEuclidean};

use crate::{atom::Atom, utils::{SHELL_LIMIT, WATER_LIMIT}};


#[derive(Clone, Debug)]
pub struct GridPoint {
    pub coords: [f64; 3],
    pub energy: f64,
}

impl PartialEq for GridPoint {
    fn eq(&self, other: &Self) -> bool {
        self.coords == other.coords
    }
}

impl Eq for GridPoint { }

// impl RTreeObject for GridPoint {
//     type Envelope = AABB<[f64; 3]>;

//     fn envelope(&self) -> Self::Envelope {
//         AABB::from_point(self.coords)
//     }
// }

// impl PointDistance for GridPoint {
//     fn distance_2(&self, point: &[f64; 3]) -> f64 {
//         let dx = self.coords[0] - point[0];
//         let dy = self.coords[1] - point[1];
//         let dz = self.coords[2] - point[2];
//         dx * dx + dy * dy + dz * dz
//     }

//     fn contains_point(&self, point: &[f64; 3]) -> bool {
//         self.coords == *point
//     }
// }

pub struct Grid3D {
    data: Vec<GridPoint>,
    possible_points: Vec<GridPoint>,
    allowed_points: Vec<GridPoint>,
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
    // tree: RTree<GridPoint>,
    tree: KdTree<f32, u32, 3, 300000, u32>,
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
        let possible_points = Vec::new();
        let allowed_points = Vec::new();
        let tree = KdTree::new();
        // Generate points using iterators
        // let mut counter = 0;
        for x in (0..x_points).map(|i| x_min + i as f64 * spacing) {
            for y in (0..y_points).map(|j| y_min + j as f64 * spacing) {
                for z in (0..z_points).map(|k| z_min + k as f64 * spacing) {
                    let coords = [x, y, z];
                    data.push(GridPoint {
                        coords,
                        energy: 0.0,
                    });
                    // tree.add(&coords, counter).unwrap();
                    // counter += 1;
                }
            }
        }
    
        Grid3D {
            data,
            possible_points,
            allowed_points,
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
            tree,
        }
    }

    pub fn get(&self, x: f64, y: f64, z: f64) -> &GridPoint {
        let index = self.index(x, y, z) as usize;
        self.data.get(index).unwrap()
    }

    pub fn set(&mut self, x: f64, y: f64, z: f64, energy: f64) {
        if self.in_box(&[x, y, z]) {
            let index = self.index(x, y, z) as usize;
            self.data[index].energy = energy;
            if self.allowed_points.len() > 0 {
                self.allowed_points[index].energy = energy;
            }
        }
    }

    pub fn in_box(&self, point: &[f64; 3]) -> bool {
        (self.x_min <= point[0] && point[0] <= self.x_max) && 
        (self.y_min <= point[1] && point[1] <= self.y_max) && 
        (self.z_min <= point[2] && point[2] <= self.z_max) 

    }

    fn index(&self, x: f64, y: f64, z: f64) -> f64 {
        x * self.x_size * (y + self.y_size * self.z_size)
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

    pub fn allowed_points(&self) -> &[GridPoint] {
        &self.allowed_points
    }

    pub fn possible_points(&self) -> &[GridPoint] {
        &self.possible_points
    }

    pub fn extract_energies_and_coordinates_parallel(&self) -> (Vec<f64>, Vec<[f64; 3]>) {
        let energies: Vec<f64> = self.data.par_iter().map(|point| point.energy).collect();
        let coordinates: Vec<[f64; 3]> = self.data.par_iter().map(|point| point.coords).collect();
        (energies, coordinates)
    }

    pub fn extract_energies_and_coordinates_for_allowed(&self) -> (Vec<f64>, Vec<[f64; 3]>) {
        let energies: Vec<f64> = self.allowed_points.par_iter().map(|point| point.energy).collect();
        let coordinates: Vec<[f64; 3]> = self.allowed_points.par_iter().map(|point| point.coords).collect();
        (energies, coordinates)
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
            points_for_shell.push(self.possible_points[p as usize].clone());
        }
        points_for_shell
    }

    pub fn get_neighbors(&mut self, receptor_points_in_box: &Vec<[f64; 3]>) -> Vec<GridPoint>{
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
                points_for_shell.push(self.possible_points[p as usize].clone());
            }
        }
        points_for_shell
    }

    pub fn set_possible_points(&mut self, receptor_points_in_box: &Vec<[f64; 3]>) {
        // Build the tree of the whole grid first to select only the points within
        // 12 A from the receptor points that are in the grid.
        // Those points are going to be the tree of the grid.
        let mut tree: KdTree<f32, u32, 3, 300000, u32> = KdTree::new();
        let mut possible_points = Vec::new();
        let radius = 24.;
        let mut tmp_tree: KdTree<f32, u32, 3, 300000, u32> = KdTree::new();
        let all_points = self.all_points_mut();

        for (idx, gridpoint) in all_points.iter().enumerate() {
            let coords = &gridpoint.coords;
            tmp_tree.add(&[coords[0] as f32, coords[1] as f32, coords[2] as f32], idx as u32);
        }

        println!("Created the first tree");
        
        let mut tree_points = HashSet::new();
        for point in receptor_points_in_box.iter() {
            let coords = &point;
            let tmp = tmp_tree
                .within::<kiddo::SquaredEuclidean>(&[coords[0] as f32, coords[1] as f32, coords[2] as f32], radius)
                .into_iter()
                .map(|n| n.item)
                .collect::<Vec<u32>>();
            for n in tmp {
                tree_points.insert(n);
            }
        }

        for (idx, gridpoint_index) in tree_points.iter().enumerate() {
            let gridpoint = all_points[*gridpoint_index as usize].clone();
            let coords = gridpoint.coords;
            tree.add(&[coords[0] as f32, coords[1] as f32, coords[2] as f32], idx as u32);
            possible_points.push(gridpoint);
        }
        self.tree = tree;
        self.possible_points = possible_points;
    }

    // pub fn set_all_possible_points(&mut self, receptor_map: &Vec<Atom>) {
    //     let mut surface_points = Vec::new();
    //     for atom in receptor_map {
    //         surface_points.push(
    //             GridPoint {
    //                 coords: atom.coords(),
    //                 energy: 0.0,
    //             }
    //         )
    //     }
    //     self.tree = RTree::bulk_load(surface_points);
    //     let mut max_allowed_points = Vec::new();
    //     for point in self.all_points() {
    //         if self.tree
    //             .locate_within_distance(point.coords, WATER_LIMIT)
    //             .next()
    //             .is_some()
    //             {
    //                 max_allowed_points.push(point.clone());
    //             }
    //     }
    //     self.possible_points = max_allowed_points;
    //     println!("# of possible points {:?}", self.possible_points.len());
    // }

    // pub fn set_allowed_points(&mut self) {
    //     let mut nearby_points = Vec::new();

    //     for point in self.possible_points() {
    //         if self.tree
    //             .locate_within_distance(point.coords, SHELL_LIMIT)
    //             .next()
    //             .is_some()
    //             {
    //                 nearby_points.push(point.clone());
    //             }
    //     }

    //     self.allowed_points = nearby_points;
    //     // println!("# of allowed points: {}", self.allowed_points.len());
    // }

    // pub fn update_allowed_points(&mut self, new_point: &[f64; 3]) {
    //     let p = self.get(new_point[0], new_point[1], new_point[2]).clone();
    //     self.allowed_points.retain(|x| x != &p);
    //     // println!("# of allowed points: {}", self.allowed_points.len());
    // }

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