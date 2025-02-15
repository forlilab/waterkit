use crate::anchor_point::AnchorPoint;
use crate::atom::Atom;
use crate::geometry;
use crate::consts;

#[derive(Clone, Debug)]
// #[pyclass]
/// Water Molecule defined by three atoms.
/// Electrostatics and charges are assigned
/// according to TIP3P forcefield.
pub struct WaterMolecule {
    oxygen: Atom,
    hydrogen_1: Atom,
    hydrogen_2: Atom,
    hydrogen_bonds: Vec<AnchorPoint>,
}

// #[pymethods]
impl WaterMolecule {
    // #[new]
    pub fn new(oxygen_coords: [f64; 3], 
        hydrogen_1_coords: [f64; 3],
        hydrogen_2_coords: [f64; 3],) -> Self {
            let oxygen = Atom::new("OW".to_string(),
                "0".to_string(),
                oxygen_coords,
                consts::RMIN_HALF_WATER,
                consts::TIP3P_EPSILON,
                consts::OXYGEN_W_Q,
                1.7,
                true,
                true

            );
            let hydrogen_1 = Atom::new("HW".to_string(),
                "1".to_string(),
                hydrogen_1_coords,
                0.0,
                0.0,
                0.4170,
                0.0,
                false,
                false
            );
            let hydrogen_2 = Atom::new("HW".to_string(),
                "2".to_string(),
                hydrogen_2_coords,
                0.0,
                0.0,
                0.4170,
                0.0,
                false,
                false
            );
            WaterMolecule {
                oxygen: oxygen,
                hydrogen_1: hydrogen_1,
                hydrogen_2: hydrogen_2,
                hydrogen_bonds: Vec::new()
            }
        }
    pub fn as_vec(&self) -> Vec<Atom> {
        let atoms = vec![self.oxygen.clone(), self.hydrogen_1.clone(), self.hydrogen_2.clone()];
        atoms
    }

    pub fn hydrogen_bonds(&self) -> Vec<AnchorPoint> {
        self.hydrogen_bonds.clone()
    } 

    pub fn guess_new_hydrogen_bonds(
        &mut self,
    ) {

        let angle_lp: f64 = 109.47;
        let hb_length = 2.8;
        
        let water_atoms = self.as_vec();
        let oxygen_atom = water_atoms[0].clone();
        let oxygen_xyz = oxygen_atom.coords();
        let h1 = water_atoms[1].clone();
        let h1_xyz = h1.coords();
        let h2 = water_atoms[2].clone();
        let h2_xyz = h2.coords();

        let angle_lp1 = (angle_lp / 2.0).to_radians();
        let angle_lp2 = -angle_lp1;

        let v = geometry::atoms_to_move(&oxygen_xyz, &[h1_xyz, h2_xyz]);
        let r = geometry::sum_points(&oxygen_xyz, &geometry::normalize(&geometry::vector(&h1_xyz, &h2_xyz)));
        let lp1_xyz = geometry::rotate_point(&v, &oxygen_xyz, &r, angle_lp1);
        let lp1_resized = geometry::resize_vector(&lp1_xyz, &hb_length, &oxygen_xyz);
        let lp2_xyz = geometry::rotate_point(&v, &oxygen_xyz, &r, angle_lp2);
        let lp2_resized = geometry::resize_vector(&lp2_xyz, &hb_length, &oxygen_xyz);

        let ap1 = AnchorPoint::new("acceptor".to_string(), oxygen_xyz, lp1_resized);
        self.hydrogen_bonds.push(ap1);

        let ap2 = AnchorPoint::new("acceptor".to_string(), oxygen_xyz, lp2_resized);
        self.hydrogen_bonds.push(ap2);

        let r_h1 = geometry::resize_vector(&h1.coords(), &2.8, &oxygen_atom.coords());
        let ap3 = AnchorPoint::new("donor".to_string(), oxygen_xyz, r_h1);
        self.hydrogen_bonds.push(ap3);

        let r_h2 = geometry::resize_vector(&h2.coords(), &2.8, &oxygen_atom.coords());
        let ap4 = AnchorPoint::new("donor".to_string(), oxygen_xyz, r_h2);
        self.hydrogen_bonds.push(ap4);
    }
}

