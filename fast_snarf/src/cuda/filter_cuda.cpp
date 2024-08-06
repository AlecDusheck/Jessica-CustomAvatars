#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <ATen/cuda/CUDAContext.h>

template <typename scalar_t>
void launch_filter_cuda(
    bool* output,
    const scalar_t* x,
    const bool* mask,
    int64_t B,
    int64_t N,
    int64_t n_init
);

extern "C" void launch_filter(
    void* output,
    const void* x,
    const void* mask,
    int64_t B,
    int64_t N,
    int64_t n_init,
    int scalar_type
) {
    switch(scalar_type) {
        case static_cast<int>(at::ScalarType::Float):
            launch_filter_cuda<float>(
                static_cast<bool*>(output),
                static_cast<const float*>(x),
                static_cast<const bool*>(mask),
                B, N, n_init
            );
            break;
        case static_cast<int>(at::ScalarType::Double):
            launch_filter_cuda<double>(
                static_cast<bool*>(output),
                static_cast<const double*>(x),
                static_cast<const bool*>(mask),
                B, N, n_init
            );
            break;
        case static_cast<int>(at::ScalarType::Half):
            launch_filter_cuda<at::Half>(
                static_cast<bool*>(output),
                static_cast<const at::Half*>(x),
                static_cast<const bool*>(mask),
                B, N, n_init
            );
            break;
        default:
            throw std::runtime_error("Unsupported scalar type");
    }
}