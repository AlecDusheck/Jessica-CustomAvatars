#include <torch/extension.h>
#include <tuple>

// CUDA implementation
std::tuple<at::Tensor, at::Tensor> KNearestNeighborIdxCuda(
    const at::Tensor& p1,
    const at::Tensor& p2,
    const at::Tensor& lengths1,
    const at::Tensor& lengths2,
    const int norm,
    const int K,
    const int version);

// CUDA implementation
std::tuple<at::Tensor, at::Tensor> KNearestNeighborBackwardCuda(
    const at::Tensor& p1,
    const at::Tensor& p2,
    const at::Tensor& lengths1,
    const at::Tensor& lengths2,
    const at::Tensor& idxs,
    const int norm,
    const at::Tensor& grad_dists);

extern "C" void c_knn_idx_cuda(
     const at::Tensor& p1,
     const at::Tensor& p2,
     const at::Tensor& lengths1,
     const at::Tensor& lengths2,
     const int norm,
     const int K,
     const int version) {
    try {
        KNearestNeighborIdxCuda(p1, p2, lengths1, lengths2, norm, K, version);
    } catch (const std::exception& e) {
         std::cerr << "C++ exception caught: " << e.what() << std::endl;
         //TODO: Set an error flag or return an error code here
    }
}

extern "C" void c_knn_backward_cuda(
    const at::Tensor& p1,
    const at::Tensor& p2,
    const at::Tensor& lengths1,
    const at::Tensor& lengths2,
    const at::Tensor& idxs,
    const int norm,
    const at::Tensor& grad_dists) {
    try {
        KNearestNeighborBackwardCuda(p1, p2, lengths1, lengths2, idxs, norm, grad_dists);
    } catch (const std::exception& e) {
         std::cerr << "C++ exception caught: " << e.what() << std::endl;
         //TODO: Set an error flag or return an error code here
    }
}