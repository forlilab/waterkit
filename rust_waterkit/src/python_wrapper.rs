use pyo3::prelude::*;

use crate::atom::Atom;
use crate::water::WaterMolecule;
use crate::waterkit::{run_waterkit, get_map};


#[pymodule(name = "rust_waterkit")]
fn rust_waterkit(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Atom>()?;
    m.add_class::<WaterMolecule>()?;
    m.add_wrapped(wrap_pyfunction!(run_waterkit))?;
    m.add_wrapped(wrap_pyfunction!(get_map))?;
    Ok(())
}