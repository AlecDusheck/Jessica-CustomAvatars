use torch_sys::*;

#[link(name = "c_raymarcher", kind = "static")]
extern "C" {
    pub fn c_raymarch_test(
        rays_o: *const C_tensor,
        rays_d: *const C_tensor,
        nears: *mut C_tensor,
        fars: *const C_tensor,
        alives: *const C_tensor,
        density_grid: *const C_tensor,
        scale: *const C_tensor,
        offset: *const C_tensor,
        step_size: *const C_tensor,
        n_steps: i32,
        pts: *mut C_tensor,
        deltas: *mut C_tensor,
        depths: *mut C_tensor,
    ) -> ();

    pub fn c_raymarch_train(
        rays_o: *const C_tensor,
        rays_d: *const C_tensor,
        nears: *const C_tensor,
        fars: *const C_tensor,
        density_grid: *const C_tensor,
        scale: *const C_tensor,
        offset: *const C_tensor,
        step_size: *const C_tensor,
        n_steps: i32,
        depths: *mut C_tensor,
    ) -> ();

    pub fn c_composite_test(
        rgb_vals: *const C_tensor,
        sigma_vals: *const C_tensor,
        delta_vals: *const C_tensor,
        depth_vals: *const C_tensor,
        alive_indices: *const C_tensor,
        color: *mut C_tensor,
        depth: *mut C_tensor,
        no_hit: *mut C_tensor,
        thresh: f32,
    ) -> ();
}
