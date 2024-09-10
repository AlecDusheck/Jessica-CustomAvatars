use fast_snarf::fuse::fuse;
use tch::{Device, Tensor};

#[test]
fn test_fuse_valid_input() {
    let device = Device::Cuda(0);
    let batch_size = 2;

    let mut x = Tensor::randn(&[batch_size, 200000, 9, 3], (tch::Kind::Float, device));
    let xd_tgt = Tensor::randn(&[batch_size, 200000, 3], (tch::Kind::Float, device));
    let grid = Tensor::randn(&[batch_size, 3, 8, 32, 32], (tch::Kind::Float, device));
    let grid_j_inv = Tensor::randn(&[batch_size, 9, 8, 32, 32], (tch::Kind::Float, device));
    let tfs = Tensor::randn(&[batch_size, 24, 4, 4], (tch::Kind::Float, device));
    let bone_ids = Tensor::randint(9, &[9], (tch::Kind::Int, device));
    let mut j_inv = Tensor::randn(&[batch_size, 200000, 9, 3, 3], (tch::Kind::Float, device));
    let mut is_valid = Tensor::ones(&[batch_size, 200000, 9], (tch::Kind::Bool, device));
    let offset = Tensor::randn(&[batch_size, 1, 3], (tch::Kind::Float, device));
    let scale = Tensor::randn(&[batch_size, 1, 3], (tch::Kind::Float, device));

    let align_corners = true;
    let cvg_threshold = 1e-5;
    let dvg_threshold = 1e5;

    fuse(
        &mut x,
        &xd_tgt,
        &grid,
        &grid_j_inv,
        &tfs,
        &bone_ids,
        align_corners,
        &mut j_inv,
        &mut is_valid,
        &offset,
        &scale,
        cvg_threshold,
        dvg_threshold,
    );

    assert_eq!(x.size(), &[batch_size, 200000, 9, 3]);
    assert_eq!(j_inv.size(), &[batch_size, 200000, 9, 3, 3]);
    assert_eq!(is_valid.size(), &[batch_size, 200000, 9]);
}

#[test]
#[should_panic(expected = "assertion `left == right` failed: x dimension 3 is 4, expected 3\n  left: 4\n right: 3")]
fn test_fuse_invalid_x_shape() {
    let device = Device::Cuda(0);
    let batch_size = 2;

    let mut x = Tensor::randn(&[batch_size, 200000, 9, 4], (tch::Kind::Float, device));
    let xd_tgt = Tensor::randn(&[batch_size, 200000, 3], (tch::Kind::Float, device));
    let grid = Tensor::randn(&[batch_size, 3, 8, 32, 32], (tch::Kind::Float, device));
    let grid_j_inv = Tensor::randn(&[batch_size, 9, 8, 32, 32], (tch::Kind::Float, device));
    let tfs = Tensor::randn(&[batch_size, 24, 4, 4], (tch::Kind::Float, device));
    let bone_ids = Tensor::randint(9, &[9], (tch::Kind::Int, device));
    let mut j_inv = Tensor::randn(&[batch_size, 200000, 9, 3, 3], (tch::Kind::Float, device));
    let mut is_valid = Tensor::ones(&[batch_size, 200000, 9], (tch::Kind::Bool, device));
    let offset = Tensor::randn(&[batch_size, 1, 3], (tch::Kind::Float, device));
    let scale = Tensor::randn(&[batch_size, 1, 3], (tch::Kind::Float, device));

    let align_corners = true;
    let cvg_threshold = 1e-5;
    let dvg_threshold = 1e5;

    fuse(
        &mut x,
        &xd_tgt,
        &grid,
        &grid_j_inv,
        &tfs,
        &bone_ids,
        align_corners,
        &mut j_inv,
        &mut is_valid,
        &offset,
        &scale,
        cvg_threshold,
        dvg_threshold,
    );
}

#[test]
#[should_panic(expected = "bone_ids: Expected tensor kind `Int`, got `Float`")]
fn test_fuse_invalid_bone_ids_type() {
    let device = Device::Cuda(0);
    let batch_size = 2;

    let mut x = Tensor::randn(&[batch_size, 200000, 9, 3], (tch::Kind::Float, device));
    let xd_tgt = Tensor::randn(&[batch_size, 200000, 3], (tch::Kind::Float, device));
    let grid = Tensor::randn(&[batch_size, 3, 8, 32, 32], (tch::Kind::Float, device));
    let grid_j_inv = Tensor::randn(&[batch_size, 9, 8, 32, 32], (tch::Kind::Float, device));
    let tfs = Tensor::randn(&[batch_size, 24, 4, 4], (tch::Kind::Float, device));
    let bone_ids = Tensor::randint(9, &[9], (tch::Kind::Float, device));
    let mut j_inv = Tensor::randn(&[batch_size, 200000, 9, 3, 3], (tch::Kind::Float, device));
    let mut is_valid = Tensor::ones(&[batch_size, 200000, 9], (tch::Kind::Bool, device));
    let offset = Tensor::randn(&[batch_size, 1, 3], (tch::Kind::Float, device));
    let scale = Tensor::randn(&[batch_size, 1, 3], (tch::Kind::Float, device));

    let align_corners = true;
    let cvg_threshold = 1e-5;
    let dvg_threshold = 1e5;

    fuse(
        &mut x,
        &xd_tgt,
        &grid,
        &grid_j_inv,
        &tfs,
        &bone_ids,
        align_corners,
        &mut j_inv,
        &mut is_valid,
        &offset,
        &scale,
        cvg_threshold,
        dvg_threshold,
    );
}

#[test]
fn test_fuse_align_corners_false() {
    let device = Device::Cuda(0);
    let batch_size = 2;

    let mut x = Tensor::randn(&[batch_size, 200000, 9, 3], (tch::Kind::Float, device));
    let xd_tgt = Tensor::randn(&[batch_size, 200000, 3], (tch::Kind::Float, device));
    let grid = Tensor::randn(&[batch_size, 3, 8, 32, 32], (tch::Kind::Float, device));
    let grid_j_inv = Tensor::randn(&[batch_size, 9, 8, 32, 32], (tch::Kind::Float, device));
    let tfs = Tensor::randn(&[batch_size, 24, 4, 4], (tch::Kind::Float, device));
    let bone_ids = Tensor::randint(9, &[9], (tch::Kind::Int, device));
    let mut j_inv = Tensor::randn(&[batch_size, 200000, 9, 3, 3], (tch::Kind::Float, device));
    let mut is_valid = Tensor::ones(&[batch_size, 200000, 9], (tch::Kind::Bool, device));
    let offset = Tensor::randn(&[batch_size, 1, 3], (tch::Kind::Float, device));
    let scale = Tensor::randn(&[batch_size, 1, 3], (tch::Kind::Float, device));

    let align_corners = false;
    let cvg_threshold = 1e-5;
    let dvg_threshold = 1e5;

    fuse(
        &mut x,
        &xd_tgt,
        &grid,
        &grid_j_inv,
        &tfs,
        &bone_ids,
        align_corners,
        &mut j_inv,
        &mut is_valid,
        &offset,
        &scale,
        cvg_threshold,
        dvg_threshold,
    );

    assert_eq!(x.size(), &[batch_size, 200000, 9, 3]);
    assert_eq!(j_inv.size(), &[batch_size, 200000, 9, 3, 3]);
    assert_eq!(is_valid.size(), &[batch_size, 200000, 9]);
}

#[test]
fn test_fuse_different_thresholds() {
    let device = Device::Cuda(0);
    let batch_size = 2;

    let mut x = Tensor::randn(&[batch_size, 200000, 9, 3], (tch::Kind::Float, device));
    let xd_tgt = Tensor::randn(&[batch_size, 200000, 3], (tch::Kind::Float, device));
    let grid = Tensor::randn(&[batch_size, 3, 8, 32, 32], (tch::Kind::Float, device));
    let grid_j_inv = Tensor::randn(&[batch_size, 9, 8, 32, 32], (tch::Kind::Float, device));
    let tfs = Tensor::randn(&[batch_size, 24, 4, 4], (tch::Kind::Float, device));
    let bone_ids = Tensor::randint(9, &[9], (tch::Kind::Int, device));
    let mut j_inv = Tensor::randn(&[batch_size, 200000, 9, 3, 3], (tch::Kind::Float, device));
    let mut is_valid = Tensor::ones(&[batch_size, 200000, 9], (tch::Kind::Bool, device));
    let offset = Tensor::randn(&[batch_size, 1, 3], (tch::Kind::Float, device));
    let scale = Tensor::randn(&[batch_size, 1, 3], (tch::Kind::Float, device));

    let align_corners = true;
    let cvg_threshold = 1e-3;
    let dvg_threshold = 1e3;

    fuse(
        &mut x,
        &xd_tgt,
        &grid,
        &grid_j_inv,
        &tfs,
        &bone_ids,
        align_corners,
        &mut j_inv,
        &mut is_valid,
        &offset,
        &scale,
        cvg_threshold,
        dvg_threshold,
    );

    assert_eq!(x.size(), &[batch_size, 200000, 9, 3]);
    assert_eq!(j_inv.size(), &[batch_size, 200000, 9, 3, 3]);
    assert_eq!(is_valid.size(), &[batch_size, 200000, 9]);
}