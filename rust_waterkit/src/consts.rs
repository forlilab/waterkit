//pub const RADIUS_WATER: f64 = 1.4;
//pub const EPSILON_WATER: f64 = 0.1521; // According to AMBER
pub const BOLTZMANN_K: f64 = 0.0019872041; // Boltzmann constant (kcal/mol)
pub const TEMPERATURE: f64 = 300.; // Temperature used for Boltzmann sampling (K)
pub const BOLTZMANN_ENERGY_CUTOFF: f64 = 0.;
//pub const SHELL_LIMIT: f64 = 2.8 * 2.;
//pub const WATER_LIMIT: f64 = 2.8 * 4.; 
//pub const MIN_SHELL_DISTANCE: f64 = 1.5;
//pub const MAX_SHELL_DISTANCE: f64 = 3.6;

// TIP3P
pub const ELECTROSTATICS_CUTOFF: f64 = 12.; // Electrostatics distance cutoff
pub const RMIN_HALF_WATER: f64 = 1.7682; // TIP3P
pub const TIP3P_EPSILON: f64 = 0.6364;
pub const OXYGEN_W_Q: f64 = -0.8340;
pub const HYDROGEN_W_Q: f64 = 0.4170;


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
