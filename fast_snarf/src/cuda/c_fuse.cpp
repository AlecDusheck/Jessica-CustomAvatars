#include "ATen/Functions.h"
#include "ATen/core/TensorBody.h"
#include <ATen/ATen.h>
#include <torch/extension.h>
#include <vector>

void launch_broyden_kernel(
    torch::Tensor &x, const torch::Tensor &xd_tgt, const torch::Tensor &grid,
    const torch::Tensor &grid_J_inv, const torch::Tensor &tfs,
    const torch::Tensor &bone_ids, bool align_corners, torch::Tensor &J_inv,
    torch::Tensor &is_valid, const torch::Tensor &offset,
    const torch::Tensor &scale, float cvg_threshold, float dvg_threshold);

extern "C" void c_fuse(
    torch::Tensor &x, const torch::Tensor &xd_tgt, const torch::Tensor &grid,
    const torch::Tensor &grid_J_inv, const torch::Tensor &tfs,
    const torch::Tensor &bone_ids, bool align_corners, torch::Tensor &J_inv,
    torch::Tensor &is_valid, const torch::Tensor &offset,
    const torch::Tensor &scale, float cvg_threshold, float dvg_threshold) {
    // Log tensor information
        std::cout << "x: " << x.sizes() << " " << x.dtype() << std::endl;
        std::cout << "xd_tgt: " << xd_tgt.sizes() << " " << xd_tgt.dtype() << std::endl;
        std::cout << "grid: " << grid.sizes() << " " << grid.dtype() << std::endl;
        std::cout << "grid_J_inv: " << grid_J_inv.sizes() << " " << grid_J_inv.dtype() << std::endl;
        std::cout << "tfs: " << tfs.sizes() << " " << tfs.dtype() << std::endl;
        std::cout << "bone_ids: " << bone_ids.sizes() << " " << bone_ids.dtype() << std::endl;
        std::cout << "J_inv: " << J_inv.sizes() << " " << J_inv.dtype() << std::endl;
        std::cout << "is_valid: " << is_valid.sizes() << " " << is_valid.dtype() << std::endl;
        std::cout << "offset: " << offset.sizes() << " " << offset.dtype() << std::endl;
        std::cout << "scale: " << scale.sizes() << " " << scale.dtype() << std::endl;

        // Log scalar values
        std::cout << "align_corners: " << align_corners << std::endl;
        std::cout << "cvg_threshold: " << cvg_threshold << std::endl;
        std::cout << "dvg_threshold: " << dvg_threshold << std::endl;

        // Check if tensors are on CPU or GPU
        std::cout << "x device: " << x.device() << std::endl;
        std::cout << "xd_tgt device: " << xd_tgt.device() << std::endl;

     try {
            launch_broyden_kernel(
                x, xd_tgt, grid,
                grid_J_inv, tfs,
                bone_ids, align_corners, J_inv,
                is_valid, offset,
                scale, cvg_threshold, dvg_threshold);
        } catch (const std::exception& e) {
            std::cerr << "C++ exception caught: " << e.what() << std::endl;
            // You might want to set an error flag or return an error code here
        }
}