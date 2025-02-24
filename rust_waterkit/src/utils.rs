use std::fs::OpenOptions;
use std::io::{BufWriter, Write};

use crate::atom::Atom;

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

pub fn to_pdb(atoms: &Vec<Atom>, fname: &str) {
    let mut cnt = 0;
    let mut h_index = 1;
    let f = std::fs::File::create(fname).expect("unable to create file");
    let mut f = BufWriter::new(f);
    for (index, point) in atoms.iter().enumerate() {
        if index % 3 == 0 {
            cnt += 1;
        }
        let coordinates = point.coords();
        let mut line = String::new();
        let mut h_type = "";
        if point.atom_type() == "HW" {
            if h_index < 2 {
                h_type = "H1";
                h_index += 1;
            }
            else {
                h_type = "H2";
                h_index = 1;
            }
            line = format!(
                "{:<6}{:>5} {:^4} {:>3} {:1}{:>4}    {:>8.3}{:>8.3}{:>8.3}{:>6.2}{:>6.2}          {:>2}\n",
                "ATOM",
                index,
                h_type,
                "HOH",
                "A",
                cnt,
                coordinates[0],
                coordinates[1],
                coordinates[2],
                0.0,
                0.0,
                "H"
            );
        }
        else {
            line = format!(
                "{:<6}{:>5} {:^4} {:>3} {:1}{:>4}    {:>8.3}{:>8.3}{:>8.3}{:>6.2}{:>6.2}          {:>2}\n",
                "ATOM",
                index,
                "O",
                "HOH",
                "A",
                cnt,
                coordinates[0],
                coordinates[1],
                coordinates[2],
                0.0,
                0.0,
                "O"
            );
        }
        // let mut f = OpenOptions::new()
        // .create(true)// Optionally create the file if it doesn't already exist
        // .open(fname)
        // .expect("Unable to open file");
        
        
        write!(f, "{}", line).expect("Unable to write data");
        // fs::write(fname, line).expect("Unable to write file");
    }
}