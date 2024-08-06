use tch::{Tensor};
use crate::cuda::launch_filter;

pub fn filter(x: &Tensor, mask: &Tensor) -> Tensor {
    assert_eq!(x.device().is_cuda(), true, "x must be a CUDA tensor");
    assert_eq!(mask.device().is_cuda(), true, "mask must be a CUDA tensor");

    let x_size = x.size();
    let mask_size = mask.size();

    // Check dimensions
    assert_eq!(x_size.len(), 4, "x must be 4-dimensional (B, N, n_init, 3)");
    assert_eq!(mask_size.len(), 3, "mask must be 3-dimensional (B, N, n_init)");
    assert_eq!(x_size[0], mask_size[0], "Batch size mismatch");
    assert_eq!(x_size[1], mask_size[1], "N mismatch");
    assert_eq!(x_size[2], mask_size[2], "n_init mismatch");
    assert_eq!(x_size[3], 3, "Last dimension of x must be 3");

    let output = Tensor::zeros(&mask_size, (mask.kind(), mask.device()));
    let b = mask_size[0];
    let n = mask_size[1];
    let n_init = mask_size[2];

    unsafe {
        launch_filter(
            output.data_ptr() as *mut u8,
            x.data_ptr() as *const u8,
            mask.data_ptr() as *const u8,
            b,
            n,
            n_init,
            x.kind() as i32,
        );
    }
    output
}