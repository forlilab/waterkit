use crate::consts;
use cubecl::prelude::*;

#[cube(launch_unchecked)]
fn compute_interactions<F: Float>(
    x: &Array<F>,
    y: &Array<F>,
    z: &Array<F>,
    rmin_half: &Array<F>,
    epsilon: &Array<F>,
    charges: &Array<F>,
    energies: &mut Array<F>,
    k_coulomb: F,
    cutoff_sq: F,
    #[comptime] num_atoms: u32
) {
    let idx = ABSOLUTE_POS;
    if idx >= num_atoms {
        terminate!();
    }

    let mut energy = F::new(0.0);

    for j in (idx + 1)..num_atoms {
        let dx = x[idx] - x[j];
        let dy = y[idx] - y[j];
        let dz = z[idx] - z[j];
        let r2 = dx * dx + dy * dy + dz * dz;

        if r2 < cutoff_sq {
            let r = F::sqrt(r2);

            let rmin = rmin_half[idx] + rmin_half[j];
            let epsilon_2 = (epsilon[idx] * epsilon[j]);
            let epsilon = F::sqrt(epsilon_2);
            let rmin_over_r = rmin / r;
            let lj = epsilon * (F::powf(rmin_over_r, F::new(12.0)) - (F::new(2.0) * F::powf(rmin_over_r, F::new(6.0))));

            let coulomb = k_coulomb * charges[idx] * charges[j] / r;
            energy += lj + coulomb;
        }
    }

    energies[ABSOLUTE_POS] = energy;
}


pub fn compute_energy<R: Runtime>(
    device: &R::Device,
    x: &[f32],
    y: &[f32],
    z: &[f32],
    sigmas: &[f32],
    epsilons: &[f32],
    charges: &[f32],
) -> f64 {
    let client = R::client(device);
    let num_atoms = x.len();

    let x_buffer = client.create(f32::as_bytes(&x));
    let y_buffer = client.create(f32::as_bytes(&y));
    let z_buffer = client.create(f32::as_bytes(&z));
    let sigma_buffer = client.create(f32::as_bytes(&sigmas));
    let epsilon_buffer = client.create(f32::as_bytes(&epsilons));
    let charge_buffer = client.create(f32::as_bytes(&charges));
    let energy_buffer = client.empty(num_atoms * core::mem::size_of::<f32>());

    let threads_per_workgroup = 256;
    let num_workgroups = (num_atoms as u32 + threads_per_workgroup - 1) / threads_per_workgroup;

    unsafe{
            compute_interactions::launch_unchecked::<f32, R>(
                &client,
                CubeCount::Static(num_workgroups, 1, 1),
                CubeDim::new(threads_per_workgroup, 1, 1),
                ArrayArg::from_raw_parts::<f32>(&x_buffer, num_atoms, 1),
                ArrayArg::from_raw_parts::<f32>(&y_buffer, num_atoms, 1),
                ArrayArg::from_raw_parts::<f32>(&z_buffer, num_atoms, 1),
                ArrayArg::from_raw_parts::<f32>(&sigma_buffer, num_atoms, 1),
                ArrayArg::from_raw_parts::<f32>(&epsilon_buffer, num_atoms, 1),
                ArrayArg::from_raw_parts::<f32>(&charge_buffer, num_atoms, 1),
                ArrayArg::from_raw_parts::<f32>(&energy_buffer, num_atoms, 1),
                ScalarArg { elem: consts::K_E as f32 },
                ScalarArg { elem: consts::ELECTROSTATICS_CUTOFF.powi(2) as f32 },
                num_atoms as u32,
        );
    }

    let bytes = client.read_one(energy_buffer.binding());
    let energies: Vec<f32> = f32::from_bytes(&bytes).to_vec();
    // energies.iter().sum()
    let f64_energies = energies.iter().map(|&v| v as f64).sum();
    f64_energies
}