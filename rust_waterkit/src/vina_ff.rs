use crate::geometry;
use crate::atom::Atom;
use crate::consts;

pub fn vina_repulsion(distance: &f64, vina_rij: &f64) -> f64 {
    if distance < vina_rij {
        return (distance - vina_rij).powi(2);
    } else {
        return 0.0
    }
}

pub fn vina_gauss1(distance: &f64, vina_rij: &f64) -> f64 {
    (-(((distance - vina_rij) / consts::VINA_GAUSS1_SIGMA).powi(2))).exp()
}

pub fn vina_gauss2(distance: &f64, vina_rij: &f64) -> f64 {
    (-(((distance - (vina_rij + consts::VINA_GAUSS2_OFFSET)) / consts::VINA_GAUSS2_SIGMA).powi(2))).exp()
}

pub fn vina_hb(distance: &f64, vina_rij: &f64) -> f64 {
    if distance < &(vina_rij + consts::VINA_HB_H1) {
        return 1.0;
    } else if distance < vina_rij {
        return (distance - vina_rij) / consts::VINA_HB_H1;
    } else {
        return 0.0;
    }
}

pub fn vina_energy(atoms_1: &Vec<Atom>, sphere_center: &[f64; 3]) -> f64 {
    let mut total_energy = 0.0;
    for atom_1 in atoms_1.iter() {
        let atom_1_coords = atom_1.coords();

        // Calculate distance avoiding division by 0
        let r = f64::max(geometry::euclidean_distance(&atom_1_coords,
            sphere_center), 1e-8_f64);

        if r < consts::VINA_DISTANCE_CUTOFF {
            if atom_1.is_heavy_atom() {
                // println!("{}", atom_1.atom_type());
                let rijs = atom_1.vina_rij() + consts::VINA_O_RIJ;
                let vg1 = vina_gauss1(&r, &rijs) * consts::VINA_GAUSS1_W;
                let vg2 = vina_gauss2(&r, &rijs) * consts::VINA_GAUSS2_W;
                let rep = vina_repulsion(&r, &rijs) * consts::VINA_REPULSION_W;
                // println!()
                total_energy += vg1 + vg2 + rep;
                
                // only donors or acceptors contribute to this term
                if atom_1.is_vina_acceptor() || atom_1.is_vina_donor() {
                    let hb = vina_hb(&r, &rijs) * consts::VINA_HB_W;
                    total_energy += hb;
                }
            }
        }
    }
    total_energy
}