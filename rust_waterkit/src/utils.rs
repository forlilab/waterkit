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