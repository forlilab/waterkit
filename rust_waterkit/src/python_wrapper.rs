use pyo3::prelude::*;

use crate::atom::Atom;
use crate::grid::Grid3D;
use crate::grid::GridPoint;
use crate::anchor_point::AnchorPoint;
use crate::setup::setup_system;
use crate::waterkit::run_waterkit;
use crate::waterkit::run_parallel_waterkit;

#[pymodule(name = "rust_waterkit")]
fn rust_waterkit(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Atom>()?;
    m.add_class::<AnchorPoint>()?;
    m.add_class::<Grid3D>()?;
    m.add_class::<GridPoint>()?;
    m.add_wrapped(wrap_pyfunction!(setup_system))?;
    m.add_wrapped(wrap_pyfunction!(run_waterkit))?;
    m.add_wrapped(wrap_pyfunction!(run_parallel_waterkit))?;
    Ok(())
}