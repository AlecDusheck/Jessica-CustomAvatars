#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <iostream>
#include <stdexcept>

void launch_filter(const torch::Tensor &output, const torch::Tensor &x, const torch::Tensor &mask);

extern "C" void c_filter(const torch::Tensor &output, const torch::Tensor &x, const torch::Tensor &mask) {
    launch_filter(output, x, mask);
}
