pub const ELECTROSTATICS_CUTOFF: f64 = 12.; // Electrostatics distance cutoff
pub const RMIN_HALF_WATER: f64 = 1.7682; // TIP3P 
pub const RADIUS_WATER: f64 = 1.4;
pub const EPSILON_WATER: f64 = 0.1521; // According to AMBER
// pub const EPSILON_WATER: f64 = 0.6364;
pub const BOLTZMANN_K: f64 = 0.0019872041; // Boltzmann constant (kcal/mol)
pub const TEMPERATURE: f64 = 300.; // Temperature used for Boltzmann sampling (K)
pub const BOLTZMANN_ENERGY_CUTOFF: f64 = 0.;
pub const SHELL_LIMIT: f64 = 2.8 * 2.;
pub const WATER_LIMIT: f64 = 2.8 * 4.; 
pub const MIN_SHELL_DISTANCE: f64 = 2.5;
pub const MAX_SHELL_DISTANCE: f64 = 3.6;

// Helper to generate ranges with float step_size
pub struct FloatRange {
    current: f64,
    end: f64,
    step: f64,
} 

impl FloatRange {
    pub fn new(start: f64, end: f64, step: f64) -> Self {
        Self {current: start, end, step}
    }
}

impl Iterator for FloatRange {
    type Item = f64;

    fn next(&mut self) -> Option<Self::Item> {
        if self.current <= self.end {
            let value = self.current;
            self.current += self.step;
            Some(value)
        } else {
            None
        }
    }
}