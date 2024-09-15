use crate::cuda;
use jessica_utils::tensor::{validate_tensor, validate_tensor_type};
use tch::{Cuda, Device, Kind, Tensor};

pub fn raymarch_test(
    rays_o: &Tensor,
    rays_d: &Tensor,
    nears: &mut Tensor,
    fars: &Tensor,
    alives: &Tensor,
    density_grid: &Tensor,
    scale: &Tensor,
    offset: &Tensor,
    step_size: &Tensor,
    n_steps: i32,
    pts: &mut Tensor,
    deltas: &mut Tensor,
    depths: &mut Tensor,
) {
    let batch_size = alives.size()[0];
    let grid_size = density_grid.size()[0];

    // Perform dimension checks
    validate_tensor(rays_o, &[batch_size, 3], "rays_o");
    validate_tensor(rays_d, &[batch_size, 3], "rays_d");
    validate_tensor(nears, &[batch_size], "nears");
    validate_tensor(fars, &[batch_size], "fars");
    validate_tensor(alives, &[batch_size], "alives");
    validate_tensor(density_grid, &[grid_size, grid_size, grid_size], "density_grid");
    validate_tensor(scale, &[3], "scale");
    validate_tensor(offset, &[3], "offset");
    validate_tensor(step_size, &[batch_size], "step_size");
    validate_tensor(pts, &[batch_size, n_steps as i64, 3], "pts");
    validate_tensor(deltas, &[batch_size, n_steps as i64], "deltas");
    validate_tensor(depths, &[batch_size, n_steps as i64], "depths");

    // Perform type checks
    validate_tensor_type(rays_o, Kind::Float, "rays_o");
    validate_tensor_type(rays_d, Kind::Float, "rays_d");
    validate_tensor_type(nears, Kind::Float, "nears");
    validate_tensor_type(fars, Kind::Float, "fars");
    validate_tensor_type(alives, Kind::Int64, "alives");
    validate_tensor_type(density_grid, Kind::Bool, "density_grid");
    validate_tensor_type(scale, Kind::Float, "scale");
    validate_tensor_type(offset, Kind::Float, "offset");
    validate_tensor_type(step_size, Kind::Float, "step_size");
    validate_tensor_type(pts, Kind::Float, "pts");
    validate_tensor_type(deltas, Kind::Float, "deltas");
    validate_tensor_type(depths, Kind::Float, "depths");

    // Check CUDA availability
    assert!(Cuda::is_available(), "CUDA is not available");
    let device = rays_o.device();
    assert!(device.is_cuda(), "Input tensors must be on a CUDA device");

    // Call CUDA function
    unsafe {
        cuda::c_raymarch_test(
            rays_o.as_ptr(),
            rays_d.as_ptr(),
            nears.as_mut_ptr(),
            fars.as_ptr(),
            alives.as_ptr(),
            density_grid.as_ptr(),
            scale.as_ptr(),
            offset.as_ptr(),
            step_size.as_ptr(),
            n_steps,
            pts.as_mut_ptr(),
            deltas.as_mut_ptr(),
            depths.as_mut_ptr(),
        );
    }
}

pub fn raymarch_train(
    rays_o: &Tensor,
    rays_d: &Tensor,
    nears: &Tensor,
    fars: &Tensor,
    density_grid: &Tensor,
    scale: &Tensor,
    offset: &Tensor,
    step_size: &Tensor,
    n_steps: i32,
    depths: &mut Tensor,
) {
    let batch_size = rays_o.size()[0];
    let grid_size = density_grid.size()[0];

    // Perform dimension checks
    validate_tensor(rays_o, &[batch_size, 3], "rays_o");
    validate_tensor(rays_d, &[batch_size, 3], "rays_d");
    validate_tensor(nears, &[batch_size], "nears");
    validate_tensor(fars, &[batch_size], "fars");
    validate_tensor(density_grid, &[grid_size, grid_size, grid_size], "density_grid");
    validate_tensor(scale, &[3], "scale");
    validate_tensor(offset, &[3], "offset");
    validate_tensor(step_size, &[batch_size], "step_size");
    validate_tensor(depths, &[batch_size, n_steps as i64], "depths");

    // Perform type checks
    validate_tensor_type(rays_o, Kind::Float, "rays_o");
    validate_tensor_type(rays_d, Kind::Float, "rays_d");
    validate_tensor_type(nears, Kind::Float, "nears");
    validate_tensor_type(fars, Kind::Float, "fars");
    validate_tensor_type(density_grid, Kind::Bool, "density_grid");
    validate_tensor_type(scale, Kind::Float, "scale");
    validate_tensor_type(offset, Kind::Float, "offset");
    validate_tensor_type(step_size, Kind::Float, "step_size");
    validate_tensor_type(depths, Kind::Float, "depths");

    // Check CUDA availability
    assert!(Cuda::is_available(), "CUDA is not available");
    let device = rays_o.device();
    assert!(device.is_cuda(), "Input tensors must be on a CUDA device");

    // Call CUDA function
    unsafe {
        cuda::c_raymarch_train(
            rays_o.as_ptr(),
            rays_d.as_ptr(),
            nears.as_ptr(),
            fars.as_ptr(),
            density_grid.as_ptr(),
            scale.as_ptr(),
            offset.as_ptr(),
            step_size.as_ptr(),
            n_steps,
            depths.as_mut_ptr(),
        );
    }
}

pub fn composite_test(
    rgb_vals: &Tensor,
    sigma_vals: &Tensor,
    delta_vals: &Tensor,
    depth_vals: &Tensor,
    alive_indices: &Tensor,
    color: &mut Tensor,
    depth: &mut Tensor,
    no_hit: &mut Tensor,
    thresh: f32,
) {
    let batch_size = alive_indices.size()[0];
    let n_steps = rgb_vals.size()[1];

    // Perform dimension checks
    validate_tensor(rgb_vals, &[batch_size, n_steps, 3], "rgb_vals");
    validate_tensor(sigma_vals, &[batch_size, n_steps], "sigma_vals");
    validate_tensor(delta_vals, &[batch_size, n_steps], "delta_vals");
    validate_tensor(depth_vals, &[batch_size, n_steps], "depth_vals");
    validate_tensor(alive_indices, &[batch_size], "alive_indices");
    validate_tensor(color, &[batch_size, 3], "color");
    validate_tensor(depth, &[batch_size], "depth");
    validate_tensor(no_hit, &[batch_size], "no_hit");

    // Perform type checks
    validate_tensor_type(rgb_vals, Kind::Float, "rgb_vals");
    validate_tensor_type(sigma_vals, Kind::Float, "sigma_vals");
    validate_tensor_type(delta_vals, Kind::Float, "delta_vals");
    validate_tensor_type(depth_vals, Kind::Float, "depth_vals");
    validate_tensor_type(alive_indices, Kind::Int64, "alive_indices");
    validate_tensor_type(color, Kind::Float, "color");
    validate_tensor_type(depth, Kind::Float, "depth");
    validate_tensor_type(no_hit, Kind::Float, "no_hit");

    // Check CUDA availability
    assert!(Cuda::is_available(), "CUDA is not available");
    let device = rgb_vals.device();
    assert!(device.is_cuda(), "Input tensors must be on a CUDA device");

    // Call CUDA function
    unsafe {
        cuda::c_composite_test(
            rgb_vals.as_ptr(),
            sigma_vals.as_ptr(),
            delta_vals.as_ptr(),
            depth_vals.as_ptr(),
            alive_indices.as_ptr(),
            color.as_mut_ptr(),
            depth.as_mut_ptr(),
            no_hit.as_mut_ptr(),
            thresh,
        );
    }
}
