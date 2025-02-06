use pyo3::prelude::*;

#[derive(Clone, Debug)]
#[pyclass]
pub struct AnchorPoint {
    hb_type: String,
    anchor_xyz: [f64; 3],
    vector_xyz: Vec<[f64; 3]>
}

#[pymethods]
impl AnchorPoint {
    #[new]
    pub fn new(hb_type: String, anchor_xyz: [f64; 3], vector_xyz: Vec<[f64; 3]>) -> AnchorPoint {
        Self {
            hb_type: hb_type.clone(),
            anchor_xyz: anchor_xyz.clone(),
            vector_xyz: vector_xyz.clone()
        }
    }

    pub fn anchor_vectors(&self) -> &Vec<[f64; 3]> {
        &self.vector_xyz
    }

    pub fn anchor_point(&self) -> &[f64; 3] {
        &self.anchor_xyz
    }

    pub fn hb_type(&self) -> &String {
        &self.hb_type
    }
}