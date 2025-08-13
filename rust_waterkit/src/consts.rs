use crate::water::WaterMolecule;
use lazy_static::lazy_static;
use std::collections::HashMap;

pub struct WATER_CONSTANTS {
    pub RMIN_HALF_WATER: f64,
    pub EPSILON_WATER: f64,
    pub OXYGEN_W_Q: f64,
    pub HYDROGEN_W_Q: f64,
    pub WATER_MODEL: [[f64; 3]; 3]
}

const TIP3P_CONSTANTS: WATER_CONSTANTS = WATER_CONSTANTS {
    RMIN_HALF_WATER: 1.7682,
    EPSILON_WATER: 0.15210325,
    OXYGEN_W_Q: -0.8340,
    HYDROGEN_W_Q: 0.4170,
    WATER_MODEL: [[0.000, 0.000, 0.000], 
                  [0.000, 0.756, 0.586], 
                  [0.000, -0.756, 0.585]]
};

const TIP3PFB_CONSTANTS: WATER_CONSTANTS = WATER_CONSTANTS {
    RMIN_HALF_WATER: 1.7835723,
    EPSILON_WATER: 0.15586604,
    OXYGEN_W_Q: -0.8484,
    HYDROGEN_W_Q: 0.4242,
    WATER_MODEL: [[0.000, 0.000, -0.018], 
                  [0.000, 0.761, 0.595], 
                  [0.000, -0.761, 0.594],]
};

macro_rules! hashmap {
    ($( $key: expr => $val: expr ),*) => {{
         let mut map = ::std::collections::HashMap::new();
         $( map.insert($key, $val); )*
         map
    }}
}

lazy_static! {
    pub static ref WATER_PARAMS: HashMap<&'static str, WATER_CONSTANTS> = hashmap![
        "TIP3P" => TIP3P_CONSTANTS,
        "TIP3PFB" => TIP3PFB_CONSTANTS
    ];
}

pub const WATER_FF: &'static str = "TIP3P";

pub const BOLTZMANN_K: f64 = 0.0019872041; // Boltzmann constant (kcal/mol)
pub const TEMPERATURE: f64 = 300.; // Temperature used for Boltzmann sampling (K)
pub const BOLTZMANN_ENERGY_CUTOFF: f64 = 0.;
pub const K_E: f64 = 332.0636;

// // TIP3P
pub const ELECTROSTATICS_CUTOFF: f64 = 12.; // Electrostatics distance cutoff
// pub const RMIN_HALF_WATER_TIP3P: f64 = 1.7682; // TIP3P
// pub const TIP3P_EPSILON: f64 = 0.15210325;
// pub const OXYGEN_W_Q_TIP3P: f64 = -0.8340;
// pub const HYDROGEN_W_Q_TIP3P: f64 = 0.4170;

// // TIP3PFB
// pub const RMIN_HALF_WATER_TIP3PFB: f64 = 1.7835723; // TIP3P
// pub const TIP3PFB_EPSILON: f64 = 0.15586604;
// pub const OXYGEN_W_Q_TIP3PFB: f64 = -0.8484;
// pub const HYDROGEN_W_Q_TIP3PFB: f64 = 0.4242;


// VINA
pub const VINA_GAUSS1_SIGMA: f64 = 0.5;
pub const VINA_GAUSS2_SIGMA: f64 = 2.0;
pub const VINA_GAUSS2_OFFSET: f64 = 3.0;
pub const VINA_HB_H1: f64 = -0.7;
pub const VINA_O_RIJ: f64 = 1.7;
pub const VINA_DISTANCE_CUTOFF: f64 = 8.0;

// VINA WEIGHTS
pub const VINA_GAUSS1_W: f64 = -0.035579;
pub const VINA_REPULSION_W: f64 = 0.840245;
pub const VINA_HB_W: f64 = -0.587439;
pub const VINA_GAUSS2_W: f64 = -0.005156;
// pub const VINA_HYDROPHOBIC_W: f64 = -0.035069;

pub const USE_DIELECTRIC: bool = false;

// GPU
pub const MAX_N_WATERS: u32 = 500;
pub const ATOM_FEATURES: u32 = 7;

// 0 -> oxygen, 1 -> hydrogen_1, 2 -> hydrogen_2, 3 -> resnum
pub const WATER_SIZE: u32 = 4 * 3;

pub const INSERTION_X_IDX: u32 = 0;
pub const INSERTION_Y_IDX: u32 = 1;
pub const INSERTION_Z_IDX: u32 = 2;
pub const TRANSLATION_X_IDX: u32 = 3;
pub const TRANSLATION_Y_IDX: u32 = 4;
pub const TRANSLATION_Z_IDX: u32 = 5;
pub const ROT_AXIS_X_IDX: u32 = 6;
pub const ROT_AXIS_Y_IDX: u32 = 7;
pub const ROT_AXIS_Z_IDX: u32 = 8;
pub const ROTATION_ANGLE: u32 = 9;
pub const WATER_TARGET_IDX: u32 = 10;
pub const ACCEPTANCE_IDX: u32 = 11;

// MC 
pub const KB: f32 = 0.0019872041; // Boltzmann constant in kcal/mol/K
pub const KT: f32 = KB * TEMPERATURE as f32;
pub const STANDARD_VOLUME: f32 = 30.345; // Volume per water molecule in bulk
pub const BETA: f32 = 1.0 / KT;
pub const CHEMICAL_POTENTIAL: f32 = -6.09; // Chemical potential in kcal/mol
// const LAMBDA: f64 = 0.145; // Thermal de Broglie wavelength in Å (simplified)
pub const GCMC_STEPS: usize = 400000; // Total Monte Carlo steps -> original = 800000

// Replicas
pub const CHEMICAL_POTENTIALS: [f64; 17] = [-10.0, -9.75, -9.50, -9.25, -9.0, -8.75, -8.5, -8.25, -8.0, -7.75, -7.5, -7.25, -7.0, -6.75, -6.5, -6.25, -6.09];
pub const RE_STEPS: usize = 100;
pub const EXCHANGE_RATE: usize = 1000;
