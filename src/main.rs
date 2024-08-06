use fast_snarf::filter;
use tch::{Tensor, Kind, Device};

fn example_filter() -> Tensor {
    // Ensure CUDA is available
    assert!(tch::Cuda::is_available(), "CUDA is not available");

    // Set up dimensions
    let batch_size = 2;
    let num_points = 1000;
    let num_init = 10;

    // Create a random tensor for x on GPU
    let x = Tensor::randn(&[batch_size, num_points, num_init, 3], (Kind::Float, Device::Cuda(0)));

    // Create a random boolean mask on GPU
    let mask = Tensor::rand(&[batch_size, num_points, num_init], (Kind::Float, Device::Cuda(0)))
        .gt(0.5) // Convert to boolean tensor
        .to_kind(Kind::Bool);

    // Call the filter function
    let output = filter(&x, &mask);

    // Print shapes for verification
    println!("x shape: {:?}", x.size());
    println!("mask shape: {:?}", mask.size());
    println!("output shape: {:?}", output.size());

    // Return the output tensor
    output
}

fn main() {
    let result = example_filter();
    println!("Filter completed successfully");

    println!("Number of true values in result: {}", result.sum(Kind::Int64).int64_value(&[]));
}