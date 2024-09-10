use std::env;
use std::path::Path;
use tch::Device;
use jessica_smpl_lib::model::data::DataModel;
use indicatif::{ProgressBar, ProgressStyle};
use log::{error, info};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = env::args().collect();
    if args.len() != 3 {
        error!("Usage: {} <input_pickle_file> <output_rust_file>", args[0]);
        std::process::exit(1);
    }

    let input_path = Path::new(&args[1]);
    let output_path = Path::new(&args[2]);

    // Check if input file exists
    if !input_path.exists() {
        error!("Error: Input file '{}' does not exist.", input_path.display());
        std::process::exit(1);
    }

    info!("SMPL Model Converter");
    info!("====================");

    let device = if tch::Cuda::is_available() {
        info!("CUDA is available. Using GPU for processing.");
        Device::Cuda(0)
    } else {
        info!("CUDA is not available. Using CPU for processing.");
        Device::Cpu
    };

    // Create a progress bar
    let pb = ProgressBar::new(100);
    pb.set_style(ProgressStyle::default_bar()
        .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}% {msg}")
        .unwrap()
        .progress_chars("#>-"));

    pb.set_message("Loading pickle file...");
    pb.set_position(0);

    // Convert pickle file to Rust format
    let data = DataModel::from_pickle_file(input_path, device)?;
    pb.set_position(50);
    pb.set_message("Processing data...");

    // Save to Rust format
    pb.set_message("Saving to Rust format...");
    data.save_to_file(output_path)?;
    data.log_tensor_sizes();
    pb.set_position(100);
    pb.finish_with_message("Conversion complete!");

    // Compare file sizes
    let input_size = std::fs::metadata(input_path)?.len();
    let output_size = std::fs::metadata(output_path)?.len();

    info!("\nFile Size Comparison:");
    info!("Input file:  {} bytes", input_size);
    info!("Output file: {} bytes", output_size);

    let size_diff = (output_size as f64 - input_size as f64) / input_size as f64 * 100.0;
    info!("Size difference: {:.2}%", size_diff);

    info!("\nConversion successful! SMPL model data saved to '{}'", output_path.display());

    Ok(())
}