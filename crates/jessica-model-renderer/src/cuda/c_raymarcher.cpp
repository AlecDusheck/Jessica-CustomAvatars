#include <torch/extension.h>
#include <vector>

std::vector<torch::Tensor> raymarch_test_cuda(const torch::Tensor& rays_o,
                                              const torch::Tensor& rays_d,
                                              torch::Tensor& nears,
                                              const torch::Tensor& fars,
                                              const torch::Tensor& alives,
                                              const torch::Tensor& density_grid,
                                              const torch::Tensor& scale,
                                              const torch::Tensor& offset,
                                              const torch::Tensor& step_size,
                                              const int N_steps,
                                              torch::Tensor& pts,
                                              torch::Tensor& deltas,
                                              torch::Tensor& depths);

extern "C" void c_raymarch_test(
    const torch::Tensor& rays_o,
    const torch::Tensor& rays_d,
    torch::Tensor& nears,
    const torch::Tensor& fars,
    const torch::Tensor& alives,
    const torch::Tensor& density_grid,
    const torch::Tensor& scale,
    const torch::Tensor& offset,
    const torch::Tensor& step_size,
    const int N_steps,
    torch::Tensor& pts,
    torch::Tensor& deltas,
    torch::Tensor& depths) {

    raymarch_test_cuda(rays_o, rays_d, nears, fars, alives, density_grid, scale, offset, step_size, N_steps, pts, deltas, depths);
}

torch::Tensor raymarch_train_cuda(const torch::Tensor& rays_o,
                                  const torch::Tensor& rays_d,
                                  const torch::Tensor& nears,
                                  const torch::Tensor& fars,
                                  const torch::Tensor& density_grid,
                                  const torch::Tensor& scale,
                                  const torch::Tensor& offset,
                                  const torch::Tensor& step_size,
                                  const int N_steps);

extern "C" torch::Tensor c_raymarch_train(
    const torch::Tensor& rays_o,
    const torch::Tensor& rays_d,
    const torch::Tensor& nears,
    const torch::Tensor& fars,
    const torch::Tensor& density_grid,
    const torch::Tensor& scale,
    const torch::Tensor& offset,
    const torch::Tensor& step_size,
    const int N_steps) {

    return raymarch_train_cuda(rays_o, rays_d, nears, fars, density_grid, scale, offset, step_size, N_steps);
}

void composite_test_cuda(const torch::Tensor& rgb_vals,
                         const torch::Tensor& sigma_vals,
                         const torch::Tensor& delta_vals,
                         const torch::Tensor& depth_vals,
                         const torch::Tensor& alive_indices,
                         torch::Tensor& color,
                         torch::Tensor& depth,
                         torch::Tensor& no_hit,
			 float thresh);

extern "C" void c_composite_test(const torch::Tensor& rgb_vals,
                   const torch::Tensor& sigma_vals,
                   const torch::Tensor& delta_vals,
                   const torch::Tensor& depth_vals,
                   const torch::Tensor& alive_indices,
                   torch::Tensor& color,
                   torch::Tensor& depth,
                   torch::Tensor& no_hit,
		   float thresh) {
    composite_test_cuda(rgb_vals, sigma_vals, delta_vals, depth_vals, alive_indices, color, depth, no_hit, thresh);
}