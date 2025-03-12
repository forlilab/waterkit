use pyo3::prelude::*;
use crate::geometry;

#[derive(Clone, Debug, PartialEq)]
#[pyclass]
pub struct RotatableBond {
    atom_i_xyz: [f64; 3],
    atom_j_xyz: [f64; 3],
    atom_k_xyz: [f64; 3],
    atom_l_xyz: [f64; 3]
}

#[pymethods]
impl RotatableBond {
    #[new]
    pub fn new(atom_i_xyz: [f64; 3], 
        atom_j_xyz: [f64; 3],
        atom_k_xyz: [f64; 3],
        atom_l_xyz: [f64; 3]) -> RotatableBond {
        Self {
            atom_i_xyz: atom_i_xyz,
            atom_j_xyz: atom_j_xyz,
            atom_k_xyz: atom_k_xyz,
            atom_l_xyz: atom_l_xyz
        }
    }

    pub fn atom_i_xyz(&self) -> [f64; 3] {
        self.atom_i_xyz
    }

    pub fn atom_j_xyz(&self) -> [f64; 3] {
        self.atom_j_xyz
    }

    pub fn atom_k_xyz(&self) -> [f64; 3] {
        self.atom_k_xyz
    }

    pub fn atom_l_xyz(&self) -> [f64; 3] {
        self.atom_l_xyz
    }

    pub fn sample_rotatable_hydrogen(&self, original_anchor_vector: [f64; 3]) -> (Vec<[f64; 3]>, Vec<[f64; 3]>) {
        // Sample every 10 degrees
        let n_rotations = (360.0f64 / 10.0).floor() as i32 - 1;
        let mut sampled_anchor_points = Vec::with_capacity((n_rotations + 1) as usize);
        let mut sampled_anchor_vectors = Vec::with_capacity((n_rotations + 1) as usize);
        let rotation = 10.0f64.to_radians();
        let current_angle = geometry::dihedral(&self.atom_i_xyz,
            &self.atom_j_xyz,
            &self.atom_k_xyz,
            &self.atom_l_xyz);
        
        let p1 = self.atom_k_xyz;
        let p2 = self.atom_j_xyz;
        let mut anchor_point = self.atom_i_xyz;
        let mut anchor_vector = original_anchor_vector.clone();
        sampled_anchor_points.push(anchor_vector);
        sampled_anchor_vectors.push(anchor_vector);
        for i in 0..n_rotations {
            let new_anchor_point = geometry::rotate_point(&anchor_point, &p1, &p2, rotation);
            let new_anchor_vector = geometry::rotate_point(&anchor_vector, &p1, &p2, rotation);
            anchor_point = new_anchor_point;
            anchor_vector = new_anchor_vector;
            sampled_anchor_points.push(new_anchor_point);
            sampled_anchor_vectors.push(new_anchor_vector);
        }

        (sampled_anchor_points, sampled_anchor_vectors)
        // println!("{}\n", sampled_positions.len());
        // for position in sampled_positions {
        //     println!("H {} {} {}", position[0], position[1], position[2]);
        // }
    }
}