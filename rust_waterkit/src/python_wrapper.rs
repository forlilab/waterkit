use pyo3::prelude::*;

use crate::atom::Atom;
use crate::geometry::{roll_sphere, roll_sphere_and_compute_energies};
use crate::energy::energy;



#[pymodule(name = "rust_waterkit")]
fn rust_waterkit(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Atom>()?;
    m.add_wrapped(wrap_pyfunction!(roll_sphere))?;
    m.add_wrapped(wrap_pyfunction!(energy))?;
    m.add_wrapped(wrap_pyfunction!(roll_sphere_and_compute_energies))?;
    Ok(())
}