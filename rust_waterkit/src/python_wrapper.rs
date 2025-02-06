use pyo3::prelude::*;

use crate::atom::Atom;
use crate::anchor_point::AnchorPoint;
use crate::waterkit::{get_map, get_shells, run_waterkit, run_waterkit_simple, save_shell_points_with_energies, test_allowed_points, order_ap, test_new_aps, get_energy_for_water};


#[pymodule(name = "rust_waterkit")]
fn rust_waterkit(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Atom>()?;
    m.add_class::<AnchorPoint>()?;
    // m.add_class::<WaterMolecule>()?;
    m.add_wrapped(wrap_pyfunction!(run_waterkit))?;
    m.add_wrapped(wrap_pyfunction!(run_waterkit_simple))?;
    m.add_wrapped(wrap_pyfunction!(get_map))?;
    m.add_wrapped(wrap_pyfunction!(get_shells))?;
    m.add_wrapped(wrap_pyfunction!(test_allowed_points))?;
    m.add_wrapped(wrap_pyfunction!(save_shell_points_with_energies))?;
    m.add_wrapped(wrap_pyfunction!(order_ap))?;
    m.add_wrapped(wrap_pyfunction!(test_new_aps))?;
    m.add_wrapped(wrap_pyfunction!(get_energy_for_water))?;
    Ok(())
}