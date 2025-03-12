use pyo3::prelude::*;

use crate::rotatable_bond::RotatableBond;

#[derive(Clone, Debug, PartialEq)]

#[pyclass]
pub struct AnchorPoint {
    idx: usize,
    hb_type: String,
    anchor_xyz: [f64; 3],
    vector_xyz: [f64; 3],
    disordered_hydrogens: Option<RotatableBond>
}

#[pymethods]
impl AnchorPoint {
    #[new]
    pub fn new(idx: usize, hb_type: String, anchor_xyz: [f64; 3], vector_xyz: [f64; 3], disorderd_hydrogens: Option<RotatableBond>) -> AnchorPoint {
        Self {
            idx: idx,
            hb_type: hb_type.clone(),
            anchor_xyz: anchor_xyz.clone(),
            vector_xyz: vector_xyz.clone(),
            disordered_hydrogens: disorderd_hydrogens
        }
    }

    pub fn get_idx(&self) -> usize {
        self.idx
    }

    pub fn anchor_vectors(&self) -> &[f64; 3] {
        &self.vector_xyz
    }

    pub fn anchor_point(&self) -> &[f64; 3] {
        &self.anchor_xyz
    }

    pub fn hb_type(&self) -> &String {
        &self.hb_type
    }

    pub fn disordered_hydrogens(&self) -> Option<RotatableBond> {
        self.disordered_hydrogens.clone()
    }

    pub fn set_anchor_point_xyz(&mut self, xyz: [f64; 3]) {
        self.anchor_xyz = xyz
    }

    pub fn set_anchor_vector_xyz(&mut self, xyz: [f64; 3]) {
        self.vector_xyz = xyz;
    }
}