use torch_sys::*;

#[link(name = "c_filter", kind = "static")]
extern "C" {
    pub fn c_filter(output: *mut C_tensor, x: *const C_tensor, mask: *const C_tensor) -> ();
}

#[link(name = "c_fuse", kind = "static")]
extern "C" {
    pub fn c_fuse(
        x: *mut C_tensor,
        xd_tgt: *const C_tensor,
        grid: *const C_tensor,
        grid_j_inv: *const C_tensor,
        tfs: *const C_tensor,
        bone_ids: *const C_tensor,
        align_corners: bool,
        j_inv: *mut C_tensor,
        is_valid: *mut C_tensor,
        offset: *const C_tensor,
        scale: *const C_tensor,
        cvg_threshold: f32,
        dvg_threshold: f32,
    ) -> ();
}


#[link(name = "c_precompute", kind = "static")]
extern "C" {
    pub fn c_precompute(
        voxel_w: *const C_tensor,
        tfs: *const C_tensor,
        voxel_d: *mut C_tensor,
        voxel_j: *mut C_tensor,
        offset: *const C_tensor,
        scale: *const C_tensor,
    ) -> ();
}