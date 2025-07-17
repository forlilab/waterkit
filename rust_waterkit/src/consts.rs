//pub const RADIUS_WATER: f64 = 1.4;
//pub const EPSILON_WATER: f64 = 0.1521; // According to AMBER
pub const BOLTZMANN_K: f64 = 0.0019872041; // Boltzmann constant (kcal/mol)
pub const TEMPERATURE: f64 = 300.; // Temperature used for Boltzmann sampling (K)
pub const BOLTZMANN_ENERGY_CUTOFF: f64 = 0.;

// TIP3P
pub const ELECTROSTATICS_CUTOFF: f64 = 12.; // Electrostatics distance cutoff
pub const RMIN_HALF_WATER_TIP3P: f64 = 1.7682; // TIP3P
pub const TIP3P_EPSILON: f64 = 0.15210325;
pub const OXYGEN_W_Q_TIP3P: f64 = -0.8340;
pub const HYDROGEN_W_Q_TIP3P: f64 = 0.4170;

// TIP3PFB
pub const RMIN_HALF_WATER_TIP3PFB: f64 = 1.7835723; // TIP3P
pub const TIP3PFB_EPSILON: f64 = 0.15586604;
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

// MC 
pub const KB: f64 = 0.0019872041; // Boltzmann constant in kcal/mol/K
pub const KT: f64 = KB * TEMPERATURE;
pub const STANDARD_VOLUME: f64 = 30.345; // Volume per water molecule in bulk
pub const BETA: f64 = 1.0 / (KB * TEMPERATURE);
pub const CHEMICAL_POTENTIAL: f64 = -6.09; // Chemical potential in kcal/mol
// const LAMBDA: f64 = 0.145; // Thermal de Broglie wavelength in Å (simplified)
pub const GCMC_STEPS: usize = 80000; // Total Monte Carlo steps -> original = 800000

// Replicas
pub const CHEMICAL_POTENTIALS: [f64; 17] = [-10.0, -9.75, -9.50, -9.25, -9.0, -8.75, -8.5, -8.25, -8.0, -7.75, -7.5, -7.25, -7.0, -6.75, -6.5, -6.25, -6.09];
pub const RE_STEPS: usize = 100;
pub const EXCHANGE_RATE: usize = 1000;