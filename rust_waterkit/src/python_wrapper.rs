use pyo3::prelude::*;

use crate::atom::Atom;
use crate::geometry::{roll_sphere_on_surface, roll_sphere, Point3D};
use crate::spheric_probe::Sphere;
use crate::energy::energy;



#[pymodule(name = "rust_waterkit")]
fn rust_waterkit(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Point3D>()?;
    m.add_class::<Atom>()?;
    m.add_class::<Sphere>()?;
    m.add_wrapped(wrap_pyfunction!(roll_sphere_on_surface))?;
    m.add_wrapped(wrap_pyfunction!(roll_sphere))?;
    m.add_wrapped(wrap_pyfunction!(energy))?;
    Ok(())
}