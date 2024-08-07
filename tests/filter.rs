use fast_snarf::filter;
use tch::{Device, Kind, Tensor};

// Helper function to create a random float tensor
fn create_random_float_tensor(shape: &[i64], device: Device) -> Tensor {
    Tensor::rand(shape, (Kind::Float, device))
}

// Helper function to create a random boolean tensor
fn create_random_bool_tensor(shape: &[i64], device: Device) -> Tensor {
    // Bool tensor rand is not implemented? Ok, whatever
    let float_tensor = Tensor::rand(shape, (Kind::Float, device));
    float_tensor.ge(0.5)} // cancer

#[test]
fn test_filter_basic_functionality() {
    let device = Device::Cuda(0);
    let x = create_random_float_tensor(&[2, 10, 5, 3], device);
    let mask = create_random_bool_tensor(&[2, 10, 5], device);

    let result = filter(&x, &mask);

    assert_eq!(result.size(), mask.size());
    assert_eq!(result.kind(), Kind::Bool);
    assert_eq!(result.device(), device);
}

#[test]
#[should_panic(expected = "x must be a float tensor")]
fn test_filter_non_float_x() {
    let device = Device::Cuda(0);
    let x = create_random_bool_tensor(&[2, 10, 5, 3], device);
    let mask = create_random_bool_tensor(&[2, 10, 5], device);

    let _ = filter(&x, &mask);
}

#[test]
#[should_panic(expected = "mask must be a boolean tensor")]
fn test_filter_non_bool_mask() {
    let device = Device::Cuda(0);
    let x = create_random_float_tensor(&[2, 10, 5, 3], device);
    let mask = create_random_float_tensor(&[2, 10, 5], device);

    let _ = filter(&x, &mask);
}

#[test]
#[should_panic(expected = "Input tensor must be on a CUDA device")]
fn test_filter_non_cuda_tensor() {
    let x = create_random_float_tensor(&[2, 10, 5, 3], Device::Cpu);
    let mask = create_random_bool_tensor(&[2, 10, 5], Device::Cpu);

    let _ = filter(&x, &mask);
}

#[test]
#[should_panic(expected = "Input tensors must be on the same device")]
fn test_filter_different_devices() {
    let x = create_random_float_tensor(&[2, 10, 5, 3], Device::Cuda(0));
    let mask = create_random_bool_tensor(&[2, 10, 5], Device::Cpu);

    let _ = filter(&x, &mask);
}

#[test]
fn test_filter_all_true_mask() {
    let device = Device::Cuda(0);
    let x = create_random_float_tensor(&[2, 10, 5, 3], device);
    let mask = Tensor::ones(&[2, 10, 5], (Kind::Bool, device));

    let result = filter(&x, &mask);

    // The result should have at least one true value per group
    assert!(result.sum(Kind::Int64).ge(10).all().int64_value(&[]) > 0);
}

#[test]
fn test_filter_all_false_mask() {
    let device = Device::Cuda(0);
    let x = create_random_float_tensor(&[2, 10, 5, 3], device);
    let mask = Tensor::zeros(&[2, 10, 5], (Kind::Bool, device));

    let result = filter(&x, &mask);

    // The result should be all false
    assert!(result.eq(0).all().int64_value(&[]) > 0);
}

#[test]
fn test_filter_proximity() {
    let device = Device::Cuda(0);
    let mut x = Tensor::zeros(&[1, 1, 5, 3], (Kind::Float, device));
    x.select(2, 0).copy_(&Tensor::from_slice(&[0.0, 0.0, 0.0]));
    x.select(2, 1).copy_(&Tensor::from_slice(&[1.0, 0.0, 0.0]));
    x.select(2, 2).copy_(&Tensor::from_slice(&[0.0, 1.0, 0.0]));
    x.select(2, 3).copy_(&Tensor::from_slice(&[0.0, 0.0, 1.0]));
    x.select(2, 4).copy_(&Tensor::from_slice(&[0.0, 0.0, 0.00009])); // Very close to the first point
    let mask = Tensor::ones(&[1, 1, 5], (Kind::Bool, device));
    let result = filter(&x, &mask);
    let expected = Tensor::from_slice(&[false, true, true, true, true]).reshape(&[1, 1, 5]).to_device(device);

    let result_cpu = result.to_device(Device::Cpu);
    let expected_cpu = expected.to_device(Device::Cpu);

    println!("Result values:");
    for i in 0..5 {
        println!("{}: {}", i, result_cpu.get(0).get(0).get(i).int64_value(&[]) == 1);
    }

    println!("Expected values:");
    for i in 0..5 {
        println!("{}: {}", i, expected_cpu.get(0).get(0).get(i).int64_value(&[]) == 1);
    }

    assert_eq!(result, expected);
}

#[test]
fn test_filter_proximity_with_mask() {
    let device = Device::Cuda(0);
    let mut x = Tensor::zeros(&[1, 1, 5, 3], (Kind::Float, device));
    x.select(2, 0).copy_(&Tensor::from_slice(&[0.0, 0.0, 0.0]));
    x.select(2, 1).copy_(&Tensor::from_slice(&[0.00005, 0.0, 0.0]));
    x.select(2, 2).copy_(&Tensor::from_slice(&[1.0, 0.0, 0.0]));
    x.select(2, 3).copy_(&Tensor::from_slice(&[0.0, 1.0, 0.0]));
    x.select(2, 4).copy_(&Tensor::from_slice(&[0.0, 0.0, 1.0]));
    let mask = Tensor::from_slice(&[true, false, true, true, true]).reshape(&[1, 1, 5]).to_device(device);
    let result = filter(&x, &mask);
    let expected = Tensor::from_slice(&[true, false, true, true, true]).reshape(&[1, 1, 5]).to_device(device);
    assert_eq!(result, expected);
}