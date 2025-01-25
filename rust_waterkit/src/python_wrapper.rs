use pyo3::prelude::*;

use crate::atom::Atom;
use crate::water::WaterMolecule;
use crate::waterkit::{run_waterkit, run_waterkit_simple, get_map, test_allowed_points};


#[pymodule(name = "rust_waterkit")]
fn rust_waterkit(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Atom>()?;
    m.add_class::<WaterMolecule>()?;
    m.add_wrapped(wrap_pyfunction!(run_waterkit))?;
    m.add_wrapped(wrap_pyfunction!(run_waterkit_simple))?;
    m.add_wrapped(wrap_pyfunction!(get_map))?;
    m.add_wrapped(wrap_pyfunction!(test_allowed_points))?;
    Ok(())
}