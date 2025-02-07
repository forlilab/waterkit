use core::f64;
use rayon::prelude::*;
use kdtree::distance::squared_euclidean;
use kdtree::KdTree;

use crate::atom::Atom;
use crate::energy;
use crate::geometry;
use crate::vina_ff;

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


    pub fn update_energies_oda(&mut self, new_points: &Vec<Atom>) {
        self.all_points_mut()
            .par_iter_mut()
            .for_each(|point| {
                point.energy += vina_ff::vina_energy(new_points, &point.coords);
            });
    }

    pub fn update_energies_ow(&mut self, new_points: &Vec<Atom>) {
        self.all_points_mut()
            .par_iter_mut()
            .for_each(|point| {
                point.energy +=  energy::get_ow_energy(new_points, &point.coords);
            });
    }

    pub fn update_energies_elec(&mut self, new_points: &Vec<Atom>) {
        self.all_points_mut()
            .par_iter_mut()
            .for_each(|point| {
                point.energy +=  energy::get_q_energy(new_points, &point.coords);
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

    pub fn get_nearest_neighbor(&self, query_point: &[f64; 3]) -> Option<&GridPoint> {
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

        let within_max_distance = self.kdtree
            .within(query_point, max_distance, &squared_euclidean)
            .unwrap();

        let in_range: Vec<_> = within_max_distance
            .into_iter()
            .filter(|&(distance, _)| distance >= min_distance) // Compare squared distances
            .collect();
        
        let mut neighbor_points = Vec::new();
        for (distance, &index) in in_range {
            neighbor_points.push(&self.data[index]);
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
        for (distance, &index) in in_range {
            if geometry::calculate_angle(&self.data[index].coords, anchor_xyz, vector_xyz).to_degrees() >= 90.0 {
                neighbor_points.push(&self.data[index]);
            }
        }
        neighbor_points
    }


    pub fn trilinear_interpolation(
        &self,
        point: &[f64; 3],
    ) -> f64 {
        let x = point[0];
        let y = point[1];
        let z = point[2];
        
        let nx = self.x_size as usize;
        let ny = self.y_size as usize;
        let nz = self.z_size as usize;

        // Ensure the point is within the grid bounds
        // assert!(x >= 0.0 && x <= (nx - 1) as f64);
        // assert!(y >= 0.0 && y <= (ny - 1) as f64);
        // assert!(z >= 0.0 && z <= (nz - 1) as f64);
    
        // Find the lower corner of the cell containing (x, y, z)
        let x0 = x.floor() as usize;
        let y0 = y.floor() as usize;
        let z0 = z.floor() as usize;
    
        // Ensure we don't go out of bounds
        let x1 = (x0 + 1).min(nx - 1);
        let y1 = (y0 + 1).min(ny - 1);
        let z1 = (z0 + 1).min(nz - 1);
    
        // Fractional parts for interpolation
        let xd = x - x0 as f64;
        let yd = y - y0 as f64;
        let zd = z - z0 as f64;
    
        // Helper function to get the value at (i, j, k) in the grid
        let get_value = |i: usize, j: usize, k: usize| self.data[i + j * nx + k * nx * ny].energy;
    
        // Interpolate along the x-axis
        let c00 = get_value(x0, y0, z0) * (1.0 - xd) + get_value(x1, y0, z0) * xd;
        let c01 = get_value(x0, y0, z1) * (1.0 - xd) + get_value(x1, y0, z1) * xd;
        let c10 = get_value(x0, y1, z0) * (1.0 - xd) + get_value(x1, y1, z0) * xd;
        let c11 = get_value(x0, y1, z1) * (1.0 - xd) + get_value(x1, y1, z1) * xd;
    
        // Interpolate along the y-axis
        let c0 = c00 * (1.0 - yd) + c10 * yd;
        let c1 = c01 * (1.0 - yd) + c11 * yd;
    
        // Interpolate along the z-axis
        c0 * (1.0 - zd) + c1 * zd
    }

    pub fn to_pdb(&self) {
        for point in self.all_points() {
            // let line = format!(
            //     "{:<6}{:>5} {:^4} {:>3} {:1}{:>4}    {:>8.3}{:>8.3}{:>8.3}{:>6.2}{:>6.2}          {:>2}",
            //     "ATOM",
            //     point.index,
            //     "H",
            //     "HOH",
            //     "A",
            //     1,
            //     point.coords[0],
            //     point.coords[1],
            //     point.coords[2],
            //     0.0,
            //     point.energy,
            //     "H"
            // );
            println!("{} {}, {}, {}", point.energy, point.coords[0], point.coords[1], point.coords[2]);
        }
    }

}