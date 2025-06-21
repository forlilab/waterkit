use crate::anchor_point::AnchorPoint;
use crate::atom::Atom;
use crate::geometry;
use crate::consts;

#[derive(Clone, Debug, PartialEq)]
// #[pyclass]
/// Water Molecule defined by three atoms.
/// Electrostatics and charges are assigned
/// according to TIP3P forcefield.
pub struct WaterMolecule {
    pub layer_id: usize,
    pub oxygen: Atom,
    pub hydrogen_1: Atom,
    pub hydrogen_2: Atom,
    hydrogen_bonds: Vec<AnchorPoint>,
    res_number: usize,
    energy: f64,
}

// #[pymethods]
impl WaterMolecule {
    // #[new]
    pub fn new(oxygen_coords: [f64; 3], 
                hydrogen_1_coords: [f64; 3],
                hydrogen_2_coords: [f64; 3],
                chain: String,
                resnumber: usize) -> Self {
            let oxygen = Atom::new("OW".to_string(),
                format!("{chain}:HOH:{resnumber}:0"),
                oxygen_coords,
                consts::RMIN_HALF_WATER,
                consts::TIP3P_EPSILON,
                consts::OXYGEN_W_Q,
                1.7,
                true,
                true

            );
            let hydrogen_1 = Atom::new("HW".to_string(),
            format!("{chain}:HOH:{resnumber}:H1"),
                hydrogen_1_coords,
                0.0,
                0.0,
                consts::HYDROGEN_W_Q,
                0.0,
                false,
                false
            );
            let hydrogen_2 = Atom::new("HW".to_string(),
            format!("{chain}:HOH:{resnumber}:H2"),
                hydrogen_2_coords,
                0.0,
                0.0,
                consts::HYDROGEN_W_Q,
                0.0,
                false,
                false
            );
            let res_number: usize = oxygen.residue_number;
            WaterMolecule {
                layer_id: 0,
                oxygen: oxygen,
                hydrogen_1: hydrogen_1,
                hydrogen_2: hydrogen_2,
                hydrogen_bonds: Vec::new(),
                res_number: res_number,
                energy: 0.0
            }
        }
    pub fn as_vec(&self) -> Vec<Atom> {
        let atoms = vec![self.oxygen.clone(), self.hydrogen_1.clone(), self.hydrogen_2.clone()];
        atoms
    }

    pub fn set_layer_id(&mut self, layer_id: usize) {
        self.layer_id = layer_id
    }

    pub fn set_res_number(&mut self, resnumber: usize) {
        self.res_number = resnumber;
        self.oxygen.set_residue_number(resnumber); 
        self.hydrogen_1.set_residue_number(resnumber);
        self.hydrogen_2.set_residue_number(resnumber);
    }
    
    pub fn get_res_number(&self) -> usize {
        self.res_number
    }

    pub fn hydrogen_bonds(&self) -> Vec<AnchorPoint> {
        self.hydrogen_bonds.clone()
    } 

    pub fn to_xyz(&self) {
        let atoms = self.as_vec();
        println!("3\n\n");
        println!("O {} {} {}", atoms[0].coords()[0], atoms[0].coords()[1], atoms[0].coords()[2]);
        println!("H {} {} {}", atoms[1].coords()[0], atoms[1].coords()[1], atoms[1].coords()[2]);
        println!("H {} {} {}", atoms[2].coords()[0], atoms[2].coords()[1], atoms[2].coords()[2])
    }

    pub fn update_coords(&mut self, oxygen_coords: [f64; 3], 
        hydrogen_1_coords: [f64; 3],
        hydrogen_2_coords: [f64; 3]) {
            self.oxygen.set_coords(oxygen_coords);
            self.hydrogen_1.set_coords(hydrogen_1_coords);
            self.hydrogen_2.set_coords(hydrogen_2_coords);

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

        let ap1 = AnchorPoint::new(0, "donor".to_string(), oxygen_xyz, lp1_resized, None);
        self.hydrogen_bonds.push(ap1);

        let ap2 = AnchorPoint::new(1, "donor".to_string(), oxygen_xyz, lp2_resized, None);
        self.hydrogen_bonds.push(ap2);

        let r_h1 = geometry::resize_vector(&h1.coords(), &2.8, &oxygen_atom.coords());
        let ap3 = AnchorPoint::new(2, "acceptor".to_string(), oxygen_xyz, r_h1, None);
        self.hydrogen_bonds.push(ap3);

        let r_h2 = geometry::resize_vector(&h2.coords(), &2.8, &oxygen_atom.coords());
        let ap4 = AnchorPoint::new(3, "acceptor".to_string(), oxygen_xyz, r_h2, None);
        self.hydrogen_bonds.push(ap4);
    }

    pub fn set_energy(&mut self, energy: f64) {
        self.energy = energy
    }

    pub fn get_energy(&self) -> f64 {
        self.energy
    }
}

