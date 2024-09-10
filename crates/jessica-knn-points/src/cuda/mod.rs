use torch_sys::*;

#[link(name = "c_knn", kind = "static")]
extern "C" {
    pub fn c_knn_idx_cuda(
        p1: *const C_tensor,
        p2: *const C_tensor,
        lengths1: *const C_tensor,
        lengths2: *const C_tensor,
        norm: i32,
        k: i32,
        version: i32,
        idxs: *mut C_tensor,
        dists: *mut C_tensor,
    ) -> ();

    pub fn c_knn_backward_cuda(
        p1: *const C_tensor,
        p2: *const C_tensor,
        lengths1: *const C_tensor,
        lengths2: *const C_tensor,
        idxs: *const C_tensor,
        norm: i32,
        grad_dists: *const C_tensor,
        grad_p1: *mut C_tensor,
        grad_p2: *mut C_tensor,
    ) -> ();
}