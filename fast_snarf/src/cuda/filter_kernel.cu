#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>
#include <ATen/cuda/detail/KernelUtils.h>

template <typename scalar_t, typename index_t>
__global__ void filter_kernel(
    const index_t nthreads,
    const scalar_t* __restrict__ x,
    const bool* __restrict__ mask,
    bool* __restrict__ output,
    const index_t n_batch,
    const index_t n_point,
    const index_t n_init
) {
    CUDA_KERNEL_LOOP_TYPE(index, nthreads, index_t) {
        const index_t i_batch = index / (n_batch*n_point);
        const index_t i_point = index % (n_batch*n_point);
        for(index_t i = 0; i < n_init; i++) {
            if(!mask[i_batch*n_point*n_init + i_point*n_init + i]){
                output[i_batch*n_point*n_init + i_point*n_init + i] = false;
                continue;
            }
            scalar_t xi0 = x[(i_batch*n_point*n_init + i_point*n_init + i)*3 + 0];
            scalar_t xi1 = x[(i_batch*n_point*n_init + i_point*n_init + i)*3 + 1];
            scalar_t xi2 = x[(i_batch*n_point*n_init + i_point*n_init + i)*3 + 2];
            bool flag = true;
            for(index_t j = i+1; j < n_init; j++){
                if(!mask[i_batch*n_point*n_init + i_point*n_init + j]){
                    continue;
                }
                scalar_t d0 = xi0 - x[(i_batch*n_point*n_init + i_point*n_init + j)*3 + 0];
                scalar_t d1 = xi1 - x[(i_batch*n_point*n_init + i_point*n_init + j)*3 + 1];
                scalar_t d2 = xi2 - x[(i_batch*n_point*n_init + i_point*n_init + j)*3 + 2];
                scalar_t dist = d0*d0 + d1*d1 + d2*d2;
                if(dist<0.0001*0.0001){
                    flag=false;
                    break;
                }
            }
            output[i_batch*n_point*n_init + i_point*n_init + i] = flag;
        }
    }
}

template <typename scalar_t>
void launch_filter_cuda(
    bool* output,
    const scalar_t* x,
    const bool* mask,
    int64_t B,
    int64_t N,
    int64_t n_init
) {
    const int64_t count = B * N;
    const int threads = 512;
    const int blocks = (count + threads - 1) / threads;

    filter_kernel<scalar_t, int64_t><<<blocks, threads>>>(
        count, x, mask, output, B, N, n_init
    );
}

// Explicit instantiations
template void launch_filter_cuda<float>(bool*, const float*, const bool*, int64_t, int64_t, int64_t);
template void launch_filter_cuda<double>(bool*, const double*, const bool*, int64_t, int64_t, int64_t);
template void launch_filter_cuda<at::Half>(bool*, const at::Half*, const bool*, int64_t, int64_t, int64_t);