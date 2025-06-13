//pub const RADIUS_WATER: f64 = 1.4;
//pub const EPSILON_WATER: f64 = 0.1521; // According to AMBER
pub const BOLTZMANN_K: f64 = 0.0019872041; // Boltzmann constant (kcal/mol)
pub const TEMPERATURE: f64 = 300.; // Temperature used for Boltzmann sampling (K)
pub const BOLTZMANN_ENERGY_CUTOFF: f64 = 0.;

// TIP3P
pub const ELECTROSTATICS_CUTOFF: f64 = 12.; // Electrostatics distance cutoff
pub const RMIN_HALF_WATER: f64 = 1.7682; // TIP3P
pub const TIP3P_EPSILON: f64 = 0.15210325;
pub const OXYGEN_W_Q: f64 = -0.8340;
pub const HYDROGEN_W_Q: f64 = 0.4170;

// TIP3PFB
pub const RMIN_HALF_WATER_TIP3PFB: f64 = 1.7835723; // TIP3P
pub const EPSILON_TIP3PFB: f64 = 0.15586604;
pub const OXYGEN_W_Q_TIP3PFB: f64 = -0.8484;
pub const HYDROGEN_W_Q_TIP3PFB: f64 = 0.4242;


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