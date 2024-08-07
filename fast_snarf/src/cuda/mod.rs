use torch_sys::*;

#[link(name = "c_filter", kind = "static")]
extern "C" {
    pub fn c_filter(output: *mut C_tensor, x: *const C_tensor, mask: *const C_tensor) -> ();
}
