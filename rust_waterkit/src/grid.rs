use rayon::prelude::*;

#[derive(Clone, Debug)]
pub struct GridPoint {
    pub coords: [f64; 3],
    pub energy: f64,
}
pub struct Grid3D {
    data: Vec<GridPoint>,
    x_size: f64,
    y_size: f64,
    z_size: f64,
}

impl Grid3D {
    /// Creates a new grid with initialized coordinates based on grid indices
    /// Creates a new grid with float dimensions and spacing
    pub fn new(x_size: f64, y_size: f64, z_size: f64, spacing: f64, center: [f64; 3]) -> Self {
        let mut data = Vec::new();

        let half_x = (x_size - 1.0) / 2.0 * spacing;
        let half_y = (y_size - 1.0) / 2.0 * spacing;
        let half_z = (z_size - 1.0) / 2.0 * spacing;

        let x_steps = (x_size / spacing).ceil() as usize;
        let y_steps = (y_size / spacing).ceil() as usize;
        let z_steps = (z_size / spacing).ceil() as usize;

        for z in 0..z_steps {
            for y in 0..y_steps {
                for x in 0..x_steps {
                    let coords = [
                        center[0] + (x as f64 * spacing - half_x),
                        center[1] + (y as f64 * spacing - half_y),
                        center[2] + (z as f64 * spacing - half_z),
                    ];
                    data.push(GridPoint {
                        coords,
                        energy: 0.0,
                    });
                }
            }
        }

        Grid3D {
            data,
            x_size,
            y_size,
            z_size,
        }
    }

    pub fn get(&self, x: f64, y: f64, z: f64) -> Option<&GridPoint> {
        if x < self.x_size && y < self.y_size && z < self.z_size {
            let index = self.index(x, y, z) as usize;
            self.data.get(index)
        } else {
            None
        }
    }

    pub fn set(&mut self, x: f64, y: f64, z: f64, energy: f64) {
        if x < self.x_size && y < self.y_size && z < self.z_size {
            let index = self.index(x, y, z) as usize;
            self.data[index].energy = energy;
        }
    }

    fn index(&self, x: f64, y: f64, z: f64) -> f64 {
        z * self.x_size * self.y_size + y * self.x_size + x
    }

    pub fn all_points_mut(&mut self) -> &mut [GridPoint] {
        &mut self.data
    }

    pub fn all_points(&self) -> &[GridPoint] {
        &self.data
    }


    pub fn extract_energies_and_coordinates_parallel(&self) -> (Vec<f64>, Vec<[f64; 3]>) {
        let energies: Vec<f64> = self.data.par_iter().map(|point| point.energy).collect();
        let coordinates: Vec<[f64; 3]> = self.data.par_iter().map(|point| point.coords).collect();
        (energies, coordinates)
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