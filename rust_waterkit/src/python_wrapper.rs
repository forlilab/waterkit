use pyo3::prelude::*;

use crate::atom::Atom;
use crate::water::WaterMolecule;
use crate::geometry::roll_sphere_and_compute_energies;
use crate::waterkit::run_waterkit;


#[pymodule(name = "rust_waterkit")]
fn rust_waterkit(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Atom>()?;
    m.add_class::<WaterMolecule>()?;
    // m.add_wrapped(wrap_pyfunction!(roll_sphere_and_compute_energies))?;
    m.add_wrapped(wrap_pyfunction!(run_waterkit))?;
    Ok(())
}