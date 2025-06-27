use std::error::Error;
use std::io::{BufWriter, Write};
use std::time::SystemTime;
use plotters::prelude::*;

use crate::atom::Atom;

pub fn round(value: f64) -> f64 {
    (value * 1000.0).round() / 1000.0
}

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

pub fn to_pdb(atoms: &Vec<Atom>, fname: &str, energies: Option<Vec<f64>>) {
    let mut cnt = 0;
    let mut h_index = 1;
    let f = std::fs::File::create(fname).expect("unable to create file");
    let mut f = BufWriter::new(f);
    let include_energies = energies.is_some();
    for (index, point) in atoms.iter().enumerate() {
        if index % 3 == 0 {
            cnt += 1;
        }
        let coordinates = point.coords();
        let mut line = String::new();
        let mut h_type = "";
        let mut energy = 0.0;
        if include_energies {
            energy = energies.as_ref().unwrap()[cnt-1];
        }
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
                energy,
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
                energy,
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
    write!(f, "{}", format!("TER")).expect("Unable to write data");
}


pub fn receptor_to_pdb(atoms: &Vec<Atom>, fname: &str) {
    let mut cnt = 0;
    let mut h_index = 1;
    let f = std::fs::File::create(fname).expect("unable to create file");
    let mut f = BufWriter::new(f);
    for (index, point) in atoms.iter().enumerate() {
        let coordinates = point.coords();
        let mut line = String::new();
        let atom_type = point.atom_type();
        let splitted: Vec<&str> = point.atom_id().split(':').collect();
        let chain = splitted[0];
        let resname = splitted[1];
        let resid = splitted[2];
        line = format!(
            "{:<6}{:>5} {:^4} {:>3} {:1}{:>4}    {:>8.3}{:>8.3}{:>8.3}{:>6.2}{:>6.2}          {:>2}\n",
            "ATOM",
            index,
            atom_type,
            resname,
            chain,
            resid,
            coordinates[0],
            coordinates[1],
            coordinates[2],
            0.0,
            0.0,
            atom_type
        );        
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

pub fn plot_acceptance_rate(data: Vec<(usize, f64)>, num_iterations: usize, end_sa: usize) -> Result<(), Box<dyn std::error::Error>> {
    // Create scatter plot
    let output_file = "monte_carlo_acceptance.png";
    let root = BitMapBackend::new(output_file, (800, 600)).into_drawing_area();
    root.fill(&WHITE)?;

    let mut chart = ChartBuilder::on(&root)
        .caption("Monte Carlo Acceptance Rate vs. Iteration", ("sans-serif", 20))
        .margin(10)
        .x_label_area_size(40)
        .y_label_area_size(40)
        .build_cartesian_2d(0..num_iterations, 0.0..100.0)?;

    chart
        .configure_mesh()
        .x_desc("Iteration")
        .y_desc("Acceptance Rate")
        .axis_desc_style(("sans-serif", 15))
        .draw()?;

    // Plot scatter points
    chart.draw_series(
        data.iter()
            .map(|&(iter, rate)| Circle::new((iter, rate), 0.1, BLUE.filled())),
    )?;

    // Draw vertical red dashed line at specified epoch
    chart
        .draw_series(LineSeries::new(
            vec![(end_sa, 0.0), (end_sa, 100.0)],
            ShapeStyle {
                color: RED.into(),
                filled: false,
                stroke_width: 1,
            }
        ))?
        .label(format!("SA eneded at Epoch {}", end_sa))
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], ShapeStyle{
            color: RED.into(),
            filled: false,
            stroke_width: 2,
        }));

    chart
        .configure_series_labels()
        .border_style(BLACK)
        .background_style(WHITE.mix(0.8))
        .draw()?;

    root.present()?;

    println!("Scatter plot saved to '{}'", output_file);

    Ok(())
}

// Function to plot energies
pub fn plot_energies(energies: &[f64], output_file: &str) -> Result<(), Box<dyn Error>> {
    // Create a drawing area (800x600 pixels)
    let root = BitMapBackend::new(output_file, (800, 600)).into_drawing_area();
    
    // Fill background with white
    root.fill(&WHITE)?;

    // Create chart context
    let min_energy = energies.iter().fold(f64::INFINITY, |a, &b| a.min(b));
    let max_energy = energies.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
    
    let mut chart = ChartBuilder::on(&root)
        .caption("Energy Plot", ("sans-serif", 40).into_font())
        .margin(10)
        .x_label_area_size(40)
        .y_label_area_size(60)
        .build_cartesian_2d(0f64..(energies.len() as f64), min_energy..max_energy)?;

    // Configure the chart
    chart
        .configure_mesh()
        .x_desc("Index")
        .y_desc("Energy")
        .axis_desc_style(("sans-serif", 20))
        .draw()?;

    // Plot the energy data as a line series
    chart.draw_series(LineSeries::new(
        energies.iter().enumerate().map(|(i, &e)| (i as f64, e)),
        &BLUE,
    ))?
    .label("Energy")
    .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &BLUE));

    // Configure the legend
    chart
        .configure_series_labels()
        .border_style(&BLACK)
        .background_style(&WHITE.mix(0.8))
        .draw()?;

    // Finalize the plot
    root.present()?;
    Ok(())
}