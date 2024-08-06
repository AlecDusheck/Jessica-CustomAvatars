use tch::Tensor;

#[link(name = "filter_cuda", kind = "static")]
extern "C" {
    pub(crate) fn launch_filter(
        output: *mut u8,
        x: *const u8,
        mask: *const u8,
        b: i64,
        n: i64,
        n_init: i64,
        scalar_type: i32,
    );
}