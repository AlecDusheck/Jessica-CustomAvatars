use knn_points::knn::{knn_backward_cuda, knn_idx_cuda};
use tch::{Device, Kind, Tensor};

#[test]
fn test_knn_forward_l2_norm_v0() {
    let device = Device::Cuda(0);
    let batch_size = 2;
    let p1_size = 3;
    let p2_size = 4;
    let dim = 5;
    let k = 2;

    let p1 = Tensor::randn(&[batch_size, p1_size, dim], (Kind::Float, device));
    let p2 = Tensor::randn(&[batch_size, p2_size, dim], (Kind::Float, device));
    let lengths1 = Tensor::from_slice(&[p1_size as i64, p1_size as i64]).to_device(device);
    let lengths2 = Tensor::from_slice(&[p2_size as i64, p2_size as i64]).to_device(device);

    let mut idxs = Tensor::zeros(&[batch_size, p1_size, k], (Kind::Int64, device));
    let mut dists = Tensor::zeros(&[batch_size, p1_size, k], (Kind::Float, device));

    knn_idx_cuda(&p1, &p2, &lengths1, &lengths2, /* norm= */ 2, k as i64, /* version= */ 0, &mut idxs, &mut dists);

    // Check that idxs and dists have the expected shape
    assert_eq!(idxs.size(), &[batch_size, p1_size, k]);
    assert_eq!(dists.size(), &[batch_size, p1_size, k]);
}

#[test]
fn test_knn_forward_l1_norm_v1() {
    let device = Device::Cuda(0);
    let batch_size = 2;
    let p1_size = 3;
    let p2_size = 4;
    let dim = 5;
    let k = 2;

    let p1 = Tensor::randn(&[batch_size, p1_size, dim], (Kind::Float, device));
    let p2 = Tensor::randn(&[batch_size, p2_size, dim], (Kind::Float, device));
    let lengths1 = Tensor::from_slice(&[p1_size as i64, p1_size as i64]).to_device(device);
    let lengths2 = Tensor::from_slice(&[p2_size as i64, p2_size as i64]).to_device(device);

    let mut idxs = Tensor::zeros(&[batch_size, p1_size, k], (Kind::Int64, device));
    let mut dists = Tensor::zeros(&[batch_size, p1_size, k], (Kind::Float, device));

    knn_idx_cuda(&p1, &p2, &lengths1, &lengths2, /* norm= */ 1, k as i64, /* version= */ 1, &mut idxs, &mut dists);

    // Check that idxs and dists have the expected shape
    assert_eq!(idxs.size(), &[batch_size, p1_size, k]);
    assert_eq!(dists.size(), &[batch_size, p1_size, k]);
}

#[test]
fn test_knn_forward_single_batch() {
    let device = Device::Cuda(0);
    let batch_size = 1;
    let p1_size = 3;
    let p2_size = 4;
    let dim = 5;
    let k = 2;

    let p1 = Tensor::randn(&[batch_size, p1_size, dim], (Kind::Float, device));
    let p2 = Tensor::randn(&[batch_size, p2_size, dim], (Kind::Float, device));
    let lengths1 = Tensor::from_slice(&[p1_size as i64]).to_device(device);
    let lengths2 = Tensor::from_slice(&[p2_size as i64]).to_device(device);

    let mut idxs = Tensor::zeros(&[batch_size, p1_size, k], (Kind::Int64, device));
    let mut dists = Tensor::zeros(&[batch_size, p1_size, k], (Kind::Float, device));

    knn_idx_cuda(&p1, &p2, &lengths1, &lengths2, /* norm= */ 2, k as i64, /* version= */ 0, &mut idxs, &mut dists);

    // Check that idxs and dists have the expected shape
    assert_eq!(idxs.size(), &[batch_size, p1_size, k]);
    assert_eq!(dists.size(), &[batch_size, p1_size, k]);
}

#[test]
fn test_knn_backward_l2_norm() {
    let device = Device::Cuda(0);
    let batch_size = 2;
    let p1_size = 3;
    let p2_size = 4;
    let dim = 5;
    let k = 2;

    let p1 = Tensor::randn(&[batch_size, p1_size, dim], (Kind::Float, device));
    let p2 = Tensor::randn(&[batch_size, p2_size, dim], (Kind::Float, device));
    let lengths1 = Tensor::from_slice(&[p1_size as i64, p1_size as i64]).to_device(device);
    let lengths2 = Tensor::from_slice(&[p2_size as i64, p2_size as i64]).to_device(device);
    let idxs = Tensor::randint(p2_size as i64, &[batch_size, p1_size, k], (Kind::Int64, device));
    let grad_dists = Tensor::ones(&[batch_size, p1_size, k], (Kind::Float, device));

    let mut grad_p1 = Tensor::zeros(&[batch_size, p1_size, dim], (Kind::Float, device));
    let mut grad_p2 = Tensor::zeros(&[batch_size, p2_size, dim], (Kind::Float, device));

    knn_backward_cuda(&p1, &p2, &lengths1, &lengths2, &idxs, /* norm= */ 2, &grad_dists, &mut grad_p1, &mut grad_p2);

    // Check that grad_p1 and grad_p2 have the expected shape
    assert_eq!(grad_p1.size(), &[batch_size, p1_size, dim]);
    assert_eq!(grad_p2.size(), &[batch_size, p2_size, dim]);
}

#[test]
fn test_knn_backward_l1_norm() {
    let device = Device::Cuda(0);
    let batch_size = 2;
    let p1_size = 3;
    let p2_size = 4;
    let dim = 5;
    let k = 2;

    let p1 = Tensor::randn(&[batch_size, p1_size, dim], (Kind::Float, device));
    let p2 = Tensor::randn(&[batch_size, p2_size, dim], (Kind::Float, device));
    let lengths1 = Tensor::from_slice(&[p1_size as i64, p1_size as i64]).to_device(device);
    let lengths2 = Tensor::from_slice(&[p2_size as i64, p2_size as i64]).to_device(device);
    let idxs = Tensor::randint(p2_size as i64, &[batch_size, p1_size, k], (Kind::Int64, device));
    let grad_dists = Tensor::ones(&[batch_size, p1_size, k], (Kind::Float, device));

    let mut grad_p1 = Tensor::zeros(&[batch_size, p1_size, dim], (Kind::Float, device));
    let mut grad_p2 = Tensor::zeros(&[batch_size, p2_size, dim], (Kind::Float, device));

    knn_backward_cuda(&p1, &p2, &lengths1, &lengths2, &idxs, /* norm= */ 1, &grad_dists, &mut grad_p1, &mut grad_p2);

    // Check that grad_p1 and grad_p2 have the expected shape  
    assert_eq!(grad_p1.size(), &[batch_size, p1_size, dim]);
    assert_eq!(grad_p2.size(), &[batch_size, p2_size, dim]);
}