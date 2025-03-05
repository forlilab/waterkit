use std::io::{BufWriter, Write};
use std::time::SystemTime;
use plotters::prelude::*;

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

pub fn timeit<F: Fn() -> T, T>(f: F) -> T {
    let start = SystemTime::now();
    let result = f();
    let end = SystemTime::now();
    let duration = end.duration_since(start).unwrap();
    println!("it took {} seconds", duration.as_secs());
    result
}


pub fn plot_optimization(data: &Vec<f64>, waters: &Vec<f64>, receptor: &Vec<f64>, min: f64, max: f64, n_steps: usize, o_steps: usize) -> Result<(), Box<dyn std::error::Error>>{
    let outpath = format!("plot_{n_steps}_{o_steps}.png");
    let root = BitMapBackend::new(&outpath, (1280, 720)).into_drawing_area();
    root.fill(&WHITE)?;

    let mut chart = ChartBuilder::on(&root)
        .caption(format!("Simulated Annealing Optimization {n_steps} waters sampled and optimized for {o_steps} steps"), ("sans-serif", 20))
        .margin((1).percent())
        .x_label_area_size(30)
        .y_label_area_size(60)
        .build_cartesian_2d(0..data.len(), min..max)?;

    chart.configure_mesh()
    .x_desc("Optimization steps (100)")
    .y_desc("System's Energy (kcal/mol)")
    .draw()?;

    chart.draw_series(LineSeries::new(
        data.iter().enumerate().map(|(i, &v)| (i, v)),
        &GREEN,
    ))?
    .label("Total E")
    .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &GREEN));

    chart.draw_series(LineSeries::new(
        waters.iter().enumerate().map(|(i, &v)| (i, v)),
        &BLUE,
    ))?
    .label("Eww")
    .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &BLUE));

    chart.draw_series(LineSeries::new(
        receptor.iter().enumerate().map(|(i, &v)| (i, v)),
        &RED,
    ))?
    .label("Esw")
    .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &RED));

    chart.configure_series_labels().draw()?;
    
    Ok(())
}