use tch::Tensor;
use jessica_utils::tensor::{validate_tensor, validate_tensor_type};

pub fn fuse(
    x: &mut Tensor,
    xd_tgt: &Tensor,
    grid: &Tensor,
    grid_j_inv: &Tensor,
    tfs: &Tensor,
    bone_ids: &Tensor,
    align_corners: bool,
    j_inv: &mut Tensor,
    is_valid: &mut Tensor,
    offset: &Tensor,
    scale: &Tensor,
    cvg_threshold: f32,
    dvg_threshold: f32,
) -> () {
    let batch_size = x.size()[0];
    let n_point = x.size()[1];
    let n_init = bone_ids.size()[0];

    // Perform dimension checks
    validate_tensor(&x, &[batch_size, n_point, n_init, 3], "x");
    validate_tensor(xd_tgt, &[batch_size, n_point, 3], "xd_tgt");
    validate_tensor(grid, &[batch_size, 3, 8, 32, 32], "grid");
    validate_tensor(grid_j_inv, &[batch_size, 9, 8, 32, 32], "grid_J_inv");
    validate_tensor(tfs, &[batch_size, 24, 4, 4], "tfs");
    validate_tensor(bone_ids, &[n_init], "bone_ids");
    validate_tensor(&j_inv, &[batch_size, n_point, n_init, 3, 3], "J_inv");
    validate_tensor(&is_valid, &[batch_size, n_point, n_init], "is_valid");
    validate_tensor(offset, &[batch_size, 1, 3], "offset");
    validate_tensor(scale, &[batch_size, 1, 3], "scale");


    validate_tensor_type(&x, tch::Kind::Float, "x");
    validate_tensor_type(xd_tgt, tch::Kind::Float, "xd_tgt");
    validate_tensor_type(grid, tch::Kind::Float, "grid");
    validate_tensor_type(grid_j_inv, tch::Kind::Float, "grid_J_inv");
    validate_tensor_type(tfs, tch::Kind::Float, "tfs");
    validate_tensor_type(bone_ids, tch::Kind::Int, "bone_ids");
    validate_tensor_type(&j_inv, tch::Kind::Float, "J_inv");
    validate_tensor_type(&is_valid, tch::Kind::Bool, "is_valid");
    validate_tensor_type(offset, tch::Kind::Float, "offset");
    validate_tensor_type(scale, tch::Kind::Float, "scale");

    // Check CUDA availability
    assert!(tch::Cuda::is_available(), "CUDA is not available");

    let device = x.device();
    assert!(device.is_cuda(), "Input tensors must be on a CUDA device");

    // Call CUDA function
    unsafe {
        crate::cuda::c_fuse(
            x.as_mut_ptr(),
            xd_tgt.as_ptr(),
            grid.as_ptr(),
            grid_j_inv.as_ptr(),
            tfs.as_ptr(),
            bone_ids.as_ptr(),
            align_corners,
            j_inv.as_mut_ptr(),
            is_valid.as_mut_ptr(),
            offset.as_ptr(),
            scale.as_ptr(),
            cvg_threshold,
            dvg_threshold,
        );
    }
}