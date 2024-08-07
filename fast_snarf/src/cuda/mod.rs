use tch::{Kind};
use tch::Tensor;
use torch_sys::*;

#[link(name = "c_filter", kind = "static")]
extern "C" {
    fn c_filter(output: *mut C_tensor, x: *const C_tensor, mask: *const C_tensor) -> ();
}

pub fn filter(x: &Tensor, mask: &Tensor) -> Tensor {
    assert_eq!(x.kind(), Kind::Float, "x must be a float tensor");
    assert_eq!(mask.kind(), Kind::Bool, "mask must be a boolean tensor");

    tch::Cuda::synchronize(0);

    if !tch::Cuda::is_available() {
        panic!("CUDA is not available")
    }

    let device = x.device();
    if !device.is_cuda() {
        panic!("Input tensor must be on a CUDA device");
    }

    if x.device() != mask.device() {
        panic!("Input tensors must be on the same device");
    }

    let mask_size = mask.size();
    let mut output = Tensor::zeros(&mask_size, (mask.kind(), mask.device()));

    unsafe {
        c_filter(output.as_mut_ptr(), x.as_ptr(), mask.as_ptr());
    };

    tch::Cuda::synchronize(0);

    output
}