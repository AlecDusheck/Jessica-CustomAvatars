use fast_snarf::precompute::precompute;
use tch::{Device, Tensor};

#[test]
fn test_precompute_correct_shape_and_type() {
    let device = Device::Cuda(0);
    let batch_size = 2;

    let voxel_w = Tensor::rand(&[batch_size, 24, 8, 32, 32], (tch::Kind::Float, device));
    let tfs = Tensor::rand(&[batch_size, 24, 3, 4], (tch::Kind::Float, device));
    let mut voxel_d = Tensor::rand(&[batch_size, 3, 8, 32, 32], (tch::Kind::Float, device));
    let mut voxel_j = Tensor::rand(&[batch_size, 9, 8, 32, 32], (tch::Kind::Float, device));
    let offset = Tensor::rand(&[batch_size, 1, 3], (tch::Kind::Float, device));
    let scale = Tensor::rand(&[batch_size, 1, 3], (tch::Kind::Float, device));


    // Create new tensors with the same shape and data type as the original tensors
    let mut voxel_d_clone = Tensor::zeros_like(&voxel_d);
    let mut voxel_j_clone = Tensor::zeros_like(&voxel_j);

    // Clone the original tensors into the new tensors
    let _ = voxel_d.clone(&mut voxel_d_clone);
    let _ = voxel_j.clone(&mut voxel_j_clone);

    precompute(&voxel_w, &tfs, &mut voxel_d, &mut voxel_j, &offset, &scale);

    // Assert that the output tensors have the expected shapes
    assert_eq!(voxel_d.size(), [batch_size, 3, 8, 32, 32]);
    assert_eq!(voxel_j.size(), [batch_size, 9, 8, 32, 32]);

    // Assert that the output tensors have the expected data types
    assert_eq!(voxel_d.kind(), tch::Kind::Float);
    assert_eq!(voxel_j.kind(), tch::Kind::Float);

    // Assert that the output tensors are different from the input tensors
    assert_ne!(voxel_d, voxel_d_clone);
    assert_ne!(voxel_j, voxel_j_clone);

    // Assert that the output tensors are on the same device as the input tensors
    assert_eq!(voxel_d.device(), Device::Cuda(0));
    assert_eq!(voxel_j.device(), Device::Cuda(0));
}

#[test]
fn test_precompute_edge_cases() {
    let device = Device::Cuda(0);

    // Test with zero-sized tensors
    let voxel_w = Tensor::rand(&[0, 24, 8, 32, 32], (tch::Kind::Float, device));
    let tfs = Tensor::rand(&[0, 24, 3, 4], (tch::Kind::Float, device));
    let mut voxel_d = Tensor::rand(&[0, 3, 8, 32, 32], (tch::Kind::Float, device));
    let mut voxel_j = Tensor::rand(&[0, 9, 8, 32, 32], (tch::Kind::Float, device));
    let offset = Tensor::rand(&[0, 1, 3], (tch::Kind::Float, device));
    let scale = Tensor::rand(&[0, 1, 3], (tch::Kind::Float, device));
    precompute(&voxel_w, &tfs, &mut voxel_d, &mut voxel_j, &offset, &scale);
    // Assert that the output tensors have the expected shapes
    assert_eq!(voxel_d.size(), [0, 3, 8, 32, 32]);
    assert_eq!(voxel_j.size(), [0, 9, 8, 32, 32]);
    
}

#[test]
fn test_precompute_different_sizes() {
    let device = Device::Cuda(0);

    // Test with small tensors
    let voxel_w = Tensor::rand(&[1, 24, 2, 4, 4], (tch::Kind::Float, device));
    let tfs = Tensor::rand(&[1, 24, 3, 4], (tch::Kind::Float, device));
    let mut voxel_d = Tensor::rand(&[1, 3, 2, 4, 4], (tch::Kind::Float, device));
    let mut voxel_j = Tensor::rand(&[1, 9, 2, 4, 4], (tch::Kind::Float, device));
    let offset = Tensor::rand(&[1, 1, 3], (tch::Kind::Float, device));
    let scale = Tensor::rand(&[1, 1, 3], (tch::Kind::Float, device));
    precompute(&voxel_w, &tfs, &mut voxel_d, &mut voxel_j, &offset, &scale);
    // Assert that the output tensors have the expected shapes
    assert_eq!(voxel_d.size(), [1, 3, 2, 4, 4]);
    assert_eq!(voxel_j.size(), [1, 9, 2, 4, 4]);

    // Test with large tensors
    let voxel_w = Tensor::rand(&[4, 24, 16, 64, 64], (tch::Kind::Float, device));
    let tfs = Tensor::rand(&[4, 24, 3, 4], (tch::Kind::Float, device));
    let mut voxel_d = Tensor::rand(&[4, 3, 16, 64, 64], (tch::Kind::Float, device));
    let mut voxel_j = Tensor::rand(&[4, 9, 16, 64, 64], (tch::Kind::Float, device));
    let offset = Tensor::rand(&[4, 1, 3], (tch::Kind::Float, device));
    let scale = Tensor::rand(&[4, 1, 3], (tch::Kind::Float, device));
    precompute(&voxel_w, &tfs, &mut voxel_d, &mut voxel_j, &offset, &scale);
    // Assert that the output tensors have the expected shapes
    assert_eq!(voxel_d.size(), [4, 3, 16, 64, 64]);
    assert_eq!(voxel_j.size(), [4, 9, 16, 64, 64]);
}