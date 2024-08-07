use fast_snarf::filter;
use tch::{Tensor, Kind, Device};

fn example_filter() -> Tensor {
    if !tch::Cuda::is_available() {
        panic!("CUDA is not available. This program requires a CUDA-capable GPU.")
    }

    let batch_size = 2;
    let num_points = 1000;
    let num_init = 10;

    let x = Tensor::rand(&[batch_size, num_points, num_init, 3], (Kind::Float, Device::Cuda(0)));
    let mask = Tensor::rand(&[batch_size, num_points, num_init], (Kind::Float, Device::Cuda(0))).gt(0.5);

    
    let output = filter(&x, &mask);
    
    output
}

fn main() {
    let result = example_filter();
    println!("Filter completed successfully");

    println!("Number of true values in result: {}", result.sum(Kind::Int64).int64_value(&[]));
}