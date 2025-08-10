use crate::{atom::Atom, consts, water};
use cubecl::prelude::*;

const NUM_FEATURES: u32 = 7;

#[cube]
pub fn lennard_jones_rmin_half(epsilon_1: f32, epsilon_2: f32, dist: f32, rmin_half1: f32, rmin_half2: f32) -> f32 {
    let rmin = rmin_half1 + rmin_half2;
    let epsilon = f32::sqrt(epsilon_1 * epsilon_2);
    let lj = epsilon * (f32::powf(rmin / dist, 12.0) - (2.0 * f32::powf(rmin / dist, 6.0)));
    lj
}

/// Calculate the Coulomb interaction energy.
/// Parameters:
///     q1 (&f64): Charge of the first atom (in e).
///     q2 (&f64): Charge of the second atom (in e).
///     r (&f64): Distance between two atoms (in angstroms).
///
/// Returns:
///     f64: Coulomb energy (in kcal/mol).
#[cube]
pub fn coulomb_energy<F: Float>(q1: F, q2: F, r: F) -> F {
    let k_e = F::new(332.0636); // Electrostatic constant in kcal·Å/(mol·e^2)
    // let dielectric = 1.0; // Dielectric constant of the medium (default: 1.0)
    let coulomb = k_e * (q1 * q2) / r;
    coulomb
}

#[cube(launch_unchecked)]
fn compute_interactions<F: Float>(
    atoms: &Array<F>,
    target_water: &Array<F>,
    hw_mapping_atoms: &Array<i32>,
    hw_mapping_water: &Array<i32>,
    energies: &mut Array<F>
) {
    let idx = ABSOLUTE_POS;
    if idx >= atoms.len() / NUM_FEATURES {  // Fixed: >= instead of >
        terminate!()
    }


    let cutoff_sq = F::new(12.0*12.0);
    let mut energy = F::new(0.0);

    // Fixed: multiply idx by NUM_FEATURES to get correct base index
    let base_idx = idx * NUM_FEATURES;
    let x = atoms[base_idx];
    let y = atoms[base_idx + 1];
    let z = atoms[base_idx + 2];
    let charge = atoms[base_idx + 3];
    let epsilon = atoms[base_idx + 4];  // Make sure this matches your packing order
    let rmin_half = atoms[base_idx + 5];
    let resnum = atoms[base_idx + 6];
    
    for j in 0..(target_water.len() / NUM_FEATURES) {
        let start_idx = j * NUM_FEATURES;
        // Fixed: Read from target_water instead of atoms
        let wat_x = target_water[start_idx];
        let wat_y = target_water[start_idx + 1];
        let wat_z = target_water[start_idx + 2];
        let wat_charge = target_water[start_idx + 3];
        let wat_epsilon = target_water[start_idx + 4];
        let wat_rmin_half = target_water[start_idx + 5];
        let wat_resnum = target_water[start_idx + 6];
        
        if resnum != wat_resnum {
            let dx = x - wat_x;
            let dy = y - wat_y;
            let dz = z - wat_z;
            let r2 = dx * dx + dy * dy + dz * dz;

            // if r2 < cutoff_sq {
                let r = F::sqrt(r2);
                let rmin_half1 = rmin_half;
                let rmin_half2 = wat_rmin_half;
                let epsilon1 = epsilon;
                let epsilon2 = wat_epsilon;
                let charge1 = charge;
                let charge2 = wat_charge;
                let rmin = rmin_half1 + rmin_half2;
                // let sigma_mixed = (sigma1 + sigma2) * F::new(0.5);
                let epsilon_mixed = F::sqrt(epsilon1 * epsilon2);
                
                let mut lj = F::new(0.0);
                
                // No HW in VdW
                if hw_mapping_atoms[idx] != 1 && hw_mapping_water[j] != 1 {
                    lj = epsilon * (F::powf((rmin / r), F::new(12.0)) - (F::new(2.0) * F::powf((rmin / r), F::new(6.0))));
                    // lj = F::new(4.0) * epsilon_mixed * (sigma_over_r12 - sigma_over_r6);
                }
                let coulomb = F::new(332.0636) * charge1 * charge2 / r;
                energy += lj + coulomb;
            // }
        }
    }

    energies[idx] = energy;
}


pub fn compute_energy<R: Runtime>(
    device: &R::Device,
    atoms: &Vec<Atom>,
    water_atoms: &Vec<Atom>,
    hw_mapping_system: &Vec<i32>,
    hw_mapping_water: &Vec<i32>
) -> f64 {
    unsafe {
        let client: ComputeClient<<R as Runtime>::Server, <R as Runtime>::Channel> = R::client(device);
        let num_atoms = atoms.len();
        let mut system_atoms = Vec::with_capacity(atoms.len() * 7);
        let mut target_water_atoms = Vec::with_capacity(water_atoms.len() * 7);

        for atom in atoms {
            let coords = atom.coords();
            system_atoms.push(coords[0] as f32);
            system_atoms.push(coords[1] as f32);
            system_atoms.push(coords[2] as f32);
            system_atoms.push(atom.charge() as f32);
            system_atoms.push(atom.epsilon() as f32);
            system_atoms.push(atom.rmin_half() as f32);
            system_atoms.push(atom.residue_number as f32);
        }

        for atom in water_atoms {
            let coords = atom.coords();
            target_water_atoms.push(coords[0] as f32);
            target_water_atoms.push(coords[1] as f32);
            target_water_atoms.push(coords[2] as f32);
            target_water_atoms.push(atom.charge() as f32);
            target_water_atoms.push(atom.epsilon() as f32);
            target_water_atoms.push(atom.rmin_half() as f32);
            target_water_atoms.push(atom.residue_number as f32);
        }

        let system_atoms_handle = client.create(f32::as_bytes(&system_atoms));
        let target_water_handle = client.create(f32::as_bytes(&target_water_atoms));
        let hw_mapping_system_handle = client.create(i32::as_bytes(&hw_mapping_system));
        let hw_mapping_water_handle = client.create(i32::as_bytes(&hw_mapping_water));
        let energies = client.empty(num_atoms * core::mem::size_of::<f32>());

        let threads_per_workgroup = 256;
        let num_workgroups = (num_atoms as u32 + threads_per_workgroup - 1) / threads_per_workgroup;


        compute_interactions::launch_unchecked::<f32, R>(
            &client,
            CubeCount::Static(num_workgroups, 1, 1),
            CubeDim::new(threads_per_workgroup,1, 1),
            ArrayArg::from_raw_parts::<f32>(&system_atoms_handle, num_atoms * 7, 1),
            ArrayArg::from_raw_parts::<f32>(&target_water_handle, water_atoms.len() * 7, 1),
            ArrayArg::from_raw_parts::<i32>(&hw_mapping_system_handle, num_atoms, 1),
            ArrayArg::from_raw_parts::<i32>(&hw_mapping_water_handle, water_atoms.len(), 1),
            ArrayArg::from_raw_parts::<f32>(&energies, num_atoms, 1),
        );

        let bytes = client.read_one(energies.clone().binding());
        // let output = f32::from_bytes(&bytes);
        // println!("{:?}", output);
        // let bytes = client.read_one(energies.clone().binding());
        let output: Vec<f32> = f32::from_bytes(&bytes).to_vec();
        let gpu_energy: f64 = output.iter().map(|&v| v as f64).sum();
        gpu_energy
    }
}