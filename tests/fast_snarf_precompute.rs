use fast_snarf::precompute::precompute;
use tch::{Device, Tensor};

#[test]
fn test_precompute_correct_shape_and_type() {
    let device = Device::Cuda(0);
    let batch_size = 2;
    
    let voxel_w = Tensor::rand(&[batch_size, 200000, 9, 3], (tch::Kind::Float, device));
    let tfs = Tensor::rand(&[batch_size, 200000, 3, 3], (tch::Kind::Float, device));
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