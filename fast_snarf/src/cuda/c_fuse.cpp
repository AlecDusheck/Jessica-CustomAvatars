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

     try {
        launch_broyden_kernel(
            x, xd_tgt, grid,
            grid_J_inv, tfs,
            bone_ids, align_corners, J_inv,
            is_valid, offset,
            scale, cvg_threshold, dvg_threshold);
     } catch (const std::exception& e) {
        std::cerr << "C++ exception caught: " << e.what() << std::endl;
        // TODO:Set an error flag or return an error code here
     }
}