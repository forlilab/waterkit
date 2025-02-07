use pyo3::prelude::*;

use crate::atom::Atom;
use crate::anchor_point::AnchorPoint;
use crate::waterkit::run_waterkit;


#[pymodule(name = "rust_waterkit")]
fn rust_waterkit(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Atom>()?;
    m.add_class::<AnchorPoint>()?;
    m.add_wrapped(wrap_pyfunction!(run_waterkit))?;
    Ok(())
}