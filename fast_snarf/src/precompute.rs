use tch::Tensor;
use crate::tensor_utils::validate_tensor;
use crate::tensor_utils::validate_tensor_type;

pub fn precompute(
    voxel_w: &Tensor,
    tfs: &Tensor,
    voxel_d: &mut Tensor,
    voxel_j: &mut Tensor,
    offset: &Tensor,
    scale: &Tensor,
) -> () {
    let batch_size = voxel_w.size()[0];
    let depth = voxel_w.size()[2];
    let height = voxel_w.size()[3];
    let width = voxel_w.size()[4];

    // Perform dimension checks
    validate_tensor(voxel_w, &[batch_size, 24, depth, height, width], "voxel_w");
    validate_tensor(tfs, &[batch_size, 24, 3, 4], "tfs");
    validate_tensor(voxel_d, &[batch_size, 3, depth, height, width], "voxel_d");
    validate_tensor(voxel_j, &[batch_size, 9, depth, height, width], "voxel_J");
    validate_tensor(offset, &[batch_size, 1, 3], "offset");
    validate_tensor(scale, &[batch_size, 1, 3], "scale");

    validate_tensor_type(voxel_w, tch::Kind::Float, "voxel_w");
    validate_tensor_type(tfs, tch::Kind::Float, "tfs");
    validate_tensor_type(voxel_d, tch::Kind::Float, "voxel_d");
    validate_tensor_type(voxel_j, tch::Kind::Float, "voxel_J");
    validate_tensor_type(offset, tch::Kind::Float, "offset");
    validate_tensor_type(scale, tch::Kind::Float, "scale");

    // Check CUDA availability
    assert!(tch::Cuda::is_available(), "CUDA is not available");
    let device = voxel_w.device();
    assert!(device.is_cuda(), "Input tensors must be on a CUDA device");

    // Call CUDA function
    unsafe {
        crate::cuda::c_precompute(
            voxel_w.as_ptr(),
            tfs.as_ptr(),
            voxel_d.as_mut_ptr(),
            voxel_j.as_mut_ptr(),
            offset.as_ptr(),
            scale.as_ptr(),
        );
    }
}