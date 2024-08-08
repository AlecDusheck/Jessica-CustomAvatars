use tch::Tensor;
use crate::tensor_utils::validate_tensor;

pub fn fuse(
    mut x: Tensor,
    xd_tgt: &Tensor,
    grid: &Tensor,
    grid_j_inv: &Tensor,
    tfs: &Tensor,
    bone_ids: &Tensor,
    align_corners: bool,
    mut j_inv: Tensor,
    mut is_valid: Tensor,
    offset: &Tensor,
    scale: &Tensor,
    cvg_threshold: f32,
    dvg_threshold: f32,
) -> (Tensor, Tensor, Tensor) {
    let batch_size = x.size()[0];

    // Perform dimension checks
    validate_tensor(&x, &[batch_size, 200000, 9, 3], "x");
    validate_tensor(xd_tgt, &[batch_size, 200000, 3], "xd_tgt");
    validate_tensor(grid, &[batch_size, 3, 8, 32, 32], "grid");
    validate_tensor(grid_j_inv, &[batch_size, 9, 8, 32, 32], "grid_J_inv");
    validate_tensor(tfs, &[batch_size, 24, 4, 4], "tfs");
    validate_tensor(bone_ids, &[9], "bone_ids");
    validate_tensor(&j_inv, &[batch_size, 200000, 9, 3, 3], "J_inv");  // Expecting 5D
    validate_tensor(&is_valid, &[batch_size, 200000, 9], "is_valid");
    validate_tensor(offset, &[batch_size, 1, 3], "offset");
    validate_tensor(scale, &[batch_size, 1, 3], "scale");

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

    (x, j_inv, is_valid)
}