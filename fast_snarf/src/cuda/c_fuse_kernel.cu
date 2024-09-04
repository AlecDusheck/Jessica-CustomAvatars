#include "ATen/Functions.h"
#include "ATen/core/TensorAccessor.h"
#include "c10/cuda/CUDAException.h"
#include "c10/cuda/CUDAStream.h"

#include <ATen/Dispatch.h>
#include <ATen/core/TensorBase.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/detail/IndexUtils.cuh>
#include <ATen/cuda/detail/KernelUtils.h>
#include <ATen/cuda/detail/TensorInfo.cuh>
#include <c10/macros/Macros.h>
#include <ratio>
#include <vector>

#include <chrono>
using namespace std::chrono;

using namespace at;
using namespace at::cuda::detail;

template <typename scalar_t, typename index_t>
__device__ __forceinline__ void fuse_J_inv_update(const index_t index, scalar_t *J_inv,
                                  scalar_t x0, scalar_t x1, scalar_t x2,
                                  scalar_t g0, scalar_t g1, scalar_t g2) {
  scalar_t J00 = J_inv[3 * 0 + 0];
  scalar_t J01 = J_inv[3 * 0 + 1];
  scalar_t J02 = J_inv[3 * 0 + 2];
  scalar_t J10 = J_inv[3 * 1 + 0];
  scalar_t J11 = J_inv[3 * 1 + 1];
  scalar_t J12 = J_inv[3 * 1 + 2];
  scalar_t J20 = J_inv[3 * 2 + 0];
  scalar_t J21 = J_inv[3 * 2 + 1];
  scalar_t J22 = J_inv[3 * 2 + 2];

  auto c0 = J00 * x0 + J10 * x1 + J20 * x2;
  auto c1 = J01 * x0 + J11 * x1 + J21 * x2;
  auto c2 = J02 * x0 + J12 * x1 + J22 * x2;

  auto s = c0 * g0 + c1 * g1 + c2 * g2;

  auto r0 = -J00 * g0 - J01 * g1 - J02 * g2;
  auto r1 = -J10 * g0 - J11 * g1 - J12 * g2;
  auto r2 = -J20 * g0 - J21 * g1 - J22 * g2;

  J_inv[3 * 0 + 0] += c0 * (r0 + x0) / s;
  J_inv[3 * 0 + 1] += c1 * (r0 + x0) / s;
  J_inv[3 * 0 + 2] += c2 * (r0 + x0) / s;
  J_inv[3 * 1 + 0] += c0 * (r1 + x1) / s;
  J_inv[3 * 1 + 1] += c1 * (r1 + x1) / s;
  J_inv[3 * 1 + 2] += c2 * (r1 + x1) / s;
  J_inv[3 * 2 + 0] += c0 * (r2 + x2) / s;
  J_inv[3 * 2 + 1] += c1 * (r2 + x2) / s;
  J_inv[3 * 2 + 2] += c2 * (r2 + x2) / s;
}

static __forceinline__ __device__ bool within_bounds_3d(int d, int h, int w,
                                                        int D, int H, int W) {
  return d >= 0 && d < D && h >= 0 && h < H && w >= 0 && w < W;
}

template <typename scalar_t>
static __forceinline__ __device__ scalar_t
grid_sampler_unnormalize(scalar_t coord, int size, bool align_corners) {
  if (align_corners) {
    // unnormalize coord from [-1, 1] to [0, size - 1]
    return ((coord + 1.f) / 2) * (size - 1);
  } else {
    // unnormalize coord from [-1, 1] to [-0.5, size - 0.5]
    return ((coord + 1.f) * size - 1) / 2;
  }
}

// Clips coordinates to between 0 and clip_limit - 1
template <typename scalar_t>
static __forceinline__ __device__ scalar_t clip_coordinates(scalar_t in,
                                                            int clip_limit) {
  return ::min(static_cast<scalar_t>(clip_limit - 1),
               ::max(in, static_cast<scalar_t>(0)));
}

template <typename scalar_t>
static __forceinline__ __device__ scalar_t
safe_downgrade_to_int_range(scalar_t x) {
  // -100.0 does not have special meaning. This is just to make sure
  // it's not within_bounds_2d or within_bounds_3d, and does not cause
  // undefined behavior. See #35506.
  if (x > INT_MAX - 1 || x < INT_MIN || !::isfinite(static_cast<double>(x)))
    return static_cast<scalar_t>(-100.0);
  return x;
}

template <typename scalar_t>
static __forceinline__ __device__ scalar_t
compute_coordinates(scalar_t coord, int size, bool align_corners) {
  // clip coordinates to image borders
  // coord = clip_coordinates(coord, size);
  coord = safe_downgrade_to_int_range(coord);
  return coord;
}

template <typename scalar_t>
static __forceinline__ __device__ scalar_t grid_sampler_compute_source_index(
    scalar_t coord, int size, bool align_corners) {
  coord = grid_sampler_unnormalize(coord, size, align_corners);
  coord = compute_coordinates(coord, size, align_corners);
  return coord;
}

template <typename scalar_t, typename index_t>
__device__ void grid_sampler_3d(
    index_t i_batch,
    const TensorInfo<scalar_t, index_t>& input,
    scalar_t grid_x, scalar_t grid_y, scalar_t grid_z,
    const PackedTensorAccessor32<scalar_t, 5>& input_p,
    scalar_t* output,
    bool align_corners) {

    const index_t C = input.sizes[1];
    const index_t inp_D = input.sizes[2];
    const index_t inp_H = input.sizes[3];
    const index_t inp_W = input.sizes[4];

    scalar_t ix = grid_sampler_compute_source_index(grid_x, inp_W, align_corners);
    scalar_t iy = grid_sampler_compute_source_index(grid_y, inp_H, align_corners);
    scalar_t iz = grid_sampler_compute_source_index(grid_z, inp_D, align_corners);

    // For bilinear interpolation, we need the floor of the indices.
    // Using static_cast<index_t> is faster than floor() for positive numbers.
    index_t ix_tnw = static_cast<index_t>(ix);
    index_t iy_tnw = static_cast<index_t>(iy);
    index_t iz_tnw = static_cast<index_t>(iz);

    // Compute the fractional parts for interpolation weights.
    scalar_t dx = ix - ix_tnw;
    scalar_t dy = iy - iy_tnw;
    scalar_t dz = iz - iz_tnw;

    // Precompute complements to reduce operations in the weight calculations.
    scalar_t dx_complement = 1 - dx;
    scalar_t dy_complement = 1 - dy;
    scalar_t dz_complement = 1 - dz;

    // Precompute all weights. This reduces repeated computations in the loop.
    const scalar_t weights[8] = {
      dx_complement * dy_complement * dz_complement,
      dx * dy_complement * dz_complement,
      dx_complement * dy * dz_complement,
      dx * dy * dz_complement,
      dx_complement * dy_complement * dz,
      dx * dy_complement * dz,
      dx_complement * dy * dz,
      dx * dy * dz
    };

    // Precompute index arrays. This simplifies the loop structure and
    // allows for better compiler optimizations.
    const index_t x_indices[2] = {ix_tnw, ix_tnw + 1};
    const index_t y_indices[2] = {iy_tnw, iy_tnw + 1};
    const index_t z_indices[2] = {iz_tnw, iz_tnw + 1};

    #pragma unroll
    for (index_t c = 0; c < C; c++) {
      scalar_t result = 0;
      #pragma unroll
      for (int k = 0; k < 2; k++) {
        #pragma unroll
        for (int j = 0; j < 2; j++) {
          #pragma unroll
          for (int i = 0; i < 2; i++) {
            index_t x = x_indices[i];
            index_t y = y_indices[j];
            index_t z = z_indices[k];
            // Bounds checking is necessary for correct behavior.
            // Keeping it inside the innermost loop ensures we don't miss any out-of-bounds accesses.
            if (within_bounds_3d(z, y, x, inp_D, inp_H, inp_W)) {
              // This is faster than array indexing and uses fewer registers.
              result += input_p[i_batch][c][z][y][x] * weights[(k<<2) | (j<<1) | i];
            }
          }
        }
      }
      output[c] = result;
    }
}

template <typename scalar_t, typename index_t>
C10_LAUNCH_BOUNDS_1(512)
__global__ void broyden_kernel(
    const index_t npoints, const index_t n_batch, const index_t n_point,
    const index_t n_init, TensorInfo<scalar_t, index_t> voxel_ti,
    TensorInfo<scalar_t, index_t> voxel_J_ti,
    PackedTensorAccessor32<scalar_t, 4> x,          // shape=(N, n_point, n_init, 3)
    PackedTensorAccessor32<scalar_t, 3> xd_tgt,     // shape=(N, n_point, 3)
    PackedTensorAccessor32<scalar_t, 5> voxel,      // shape=(N, 3, d, h, w)
    PackedTensorAccessor32<scalar_t, 5> grid_J_inv, // shape=(N, 9, d, h, w)
    PackedTensorAccessor32<scalar_t, 4> tfs,        // shape=(N, n_bone, 4, 4)
    PackedTensorAccessor32<int, 1> bone_ids,        // shape=(n_init)
    PackedTensorAccessor32<scalar_t, 5> J_inv,      // shape=(N, n_point, n_init, 3, 3)
    PackedTensorAccessor32<bool, 3> is_valid,       // shape=(N, n_point, n_init)
    PackedTensorAccessor32<scalar_t, 3> offset,     // shape=(N, 1, 3)
    PackedTensorAccessor32<scalar_t, 3> scale,      // shape=(N, 1, 3)
    float cvg_threshold, float dvg_threshold, int N) {

  index_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= npoints)
    return;

  const index_t i_batch = index / (n_point * n_init);
  const index_t i_point = (index % (n_point * n_init)) / n_init;
  const index_t i_init = (index % (n_point * n_init)) % n_init;

  scalar_t gx[3];
  scalar_t gx_new[3];

  scalar_t xd_tgt_index[3];

  xd_tgt_index[0] = xd_tgt[i_batch][i_point][0];
  xd_tgt_index[1] = xd_tgt[i_batch][i_point][1];
  xd_tgt_index[2] = xd_tgt[i_batch][i_point][2];

  scalar_t x_l[3];

  int i_bone = bone_ids[i_init];
  scalar_t ixd = xd_tgt_index[0] - tfs[i_batch][i_bone][0][3];
  scalar_t iyd = xd_tgt_index[1] - tfs[i_batch][i_bone][1][3];
  scalar_t izd = xd_tgt_index[2] - tfs[i_batch][i_bone][2][3];
  x_l[0] = ixd * tfs[i_batch][i_bone][0][0] + iyd * tfs[i_batch][i_bone][1][0] + izd * tfs[i_batch][i_bone][2][0];
  x_l[1] = ixd * tfs[i_batch][i_bone][0][1] + iyd * tfs[i_batch][i_bone][1][1] + izd * tfs[i_batch][i_bone][2][1];
  x_l[2] = ixd * tfs[i_batch][i_bone][0][2] + iyd * tfs[i_batch][i_bone][1][2] + izd * tfs[i_batch][i_bone][2][2];

  scalar_t J_local[12];
  grid_sampler_3d(i_batch, voxel_J_ti,
                  scale[0][0][0] * (x_l[0] + offset[0][0][0]),
                  scale[0][0][1] * (x_l[1] + offset[0][0][1]),
                  scale[0][0][2] * (x_l[2] + offset[0][0][2]),
                  grid_J_inv, J_local, true);

  scalar_t J_inv_local[9];
  J_inv_local[3 * 0 + 0] = J_local[4 * 0 + 0];
  J_inv_local[3 * 1 + 0] = J_local[4 * 0 + 1];
  J_inv_local[3 * 2 + 0] = J_local[4 * 0 + 2];
  J_inv_local[3 * 0 + 1] = J_local[4 * 1 + 0];
  J_inv_local[3 * 1 + 1] = J_local[4 * 1 + 1];
  J_inv_local[3 * 2 + 1] = J_local[4 * 1 + 2];
  J_inv_local[3 * 0 + 2] = J_local[4 * 2 + 0];
  J_inv_local[3 * 1 + 2] = J_local[4 * 2 + 1];
  J_inv_local[3 * 2 + 2] = J_local[4 * 2 + 2];

  for (int i = 0; i < 10; i++) {
    scalar_t J00 = J_inv_local[3 * 0 + 0];
    scalar_t J01 = J_inv_local[3 * 0 + 1];
    scalar_t J02 = J_inv_local[3 * 0 + 2];
    scalar_t J10 = J_inv_local[3 * 1 + 0];
    scalar_t J11 = J_inv_local[3 * 1 + 1];
    scalar_t J12 = J_inv_local[3 * 1 + 2];
    scalar_t J20 = J_inv_local[3 * 2 + 0];
    scalar_t J21 = J_inv_local[3 * 2 + 1];
    scalar_t J22 = J_inv_local[3 * 2 + 2];

    if (i == 0) {
      gx[0] = J_local[4 * 0 + 0] * x_l[0] + J_local[4 * 0 + 1] * x_l[1] + J_local[4 * 0 + 2] * x_l[2] + J_local[4 * 0 + 3];
      gx[1] = J_local[4 * 1 + 0] * x_l[0] + J_local[4 * 1 + 1] * x_l[1] + J_local[4 * 1 + 2] * x_l[2] + J_local[4 * 1 + 3];
      gx[2] = J_local[4 * 2 + 0] * x_l[0] + J_local[4 * 2 + 1] * x_l[1] + J_local[4 * 2 + 2] * x_l[2] + J_local[4 * 2 + 3];

      gx[0] = gx[0] - xd_tgt_index[0];
      gx[1] = gx[1] - xd_tgt_index[1];
      gx[2] = gx[2] - xd_tgt_index[2];
    } else {
      gx[0] = gx_new[0];
      gx[1] = gx_new[1];
      gx[2] = gx_new[2];
    }

    // update = -J_inv @ gx
    scalar_t u0 = -J00 * gx[0] + -J01 * gx[1] + -J02 * gx[2];
    scalar_t u1 = -J10 * gx[0] + -J11 * gx[1] + -J12 * gx[2];
    scalar_t u2 = -J20 * gx[0] + -J21 * gx[1] + -J22 * gx[2];

    // x += update
    x_l[0] += u0;
    x_l[1] += u1;
    x_l[2] += u2;

    scalar_t ix = scale[0][0][0] * (x_l[0] + offset[0][0][0]);
    scalar_t iy = scale[0][0][1] * (x_l[1] + offset[0][0][1]);
    scalar_t iz = scale[0][0][2] * (x_l[2] + offset[0][0][2]);

    // gx_new = g(x)
    grid_sampler_3d(i_batch, voxel_J_ti, ix, iy, iz, grid_J_inv, J_local, true);

    gx_new[0] = J_local[4 * 0 + 0] * x_l[0] + J_local[4 * 0 + 1] * x_l[1] +
                J_local[4 * 0 + 2] * x_l[2] + J_local[4 * 0 + 3] -
                xd_tgt_index[0];
    gx_new[1] = J_local[4 * 1 + 0] * x_l[0] + J_local[4 * 1 + 1] * x_l[1] +
                J_local[4 * 1 + 2] * x_l[2] + J_local[4 * 1 + 3] -
                xd_tgt_index[1];
    gx_new[2] = J_local[4 * 2 + 0] * x_l[0] + J_local[4 * 2 + 1] * x_l[1] +
                J_local[4 * 2 + 2] * x_l[2] + J_local[4 * 2 + 3] -
                xd_tgt_index[2];

    // convergence checking
    scalar_t norm_gx = gx_new[0] * gx_new[0] + gx_new[1] * gx_new[1] + gx_new[2] * gx_new[2];

    // convergence/divergence criterion
    if (norm_gx < cvg_threshold * cvg_threshold) {

      auto b = 1;
      bool is_valid_ =
          ix >= -b && ix <= b && iy >= -b && iy <= b && iz >= -b && iz <= b;

      is_valid[i_batch][i_point][i_init] = is_valid_;

      if (is_valid_) {
        x[i_batch][i_point][i_init][0] = x_l[0];
        x[i_batch][i_point][i_init][1] = x_l[1];
        x[i_batch][i_point][i_init][2] = x_l[2];

        J_inv[i_batch][i_point][i_init][0][0] = J00;
        J_inv[i_batch][i_point][i_init][0][1] = J01;
        J_inv[i_batch][i_point][i_init][0][2] = J02;
        J_inv[i_batch][i_point][i_init][1][0] = J10;
        J_inv[i_batch][i_point][i_init][1][1] = J11;
        J_inv[i_batch][i_point][i_init][1][2] = J12;
        J_inv[i_batch][i_point][i_init][2][0] = J20;
        J_inv[i_batch][i_point][i_init][2][1] = J21;
        J_inv[i_batch][i_point][i_init][2][2] = J22;
      }
      return;

    } else if (norm_gx > dvg_threshold * dvg_threshold) {
      is_valid[i_batch][i_point][i_init] = false;
      return;
    }

    // delta_x = update
    scalar_t delta_x_0 = u0;
    scalar_t delta_x_1 = u1;
    scalar_t delta_x_2 = u2;

    // delta_gx = gx_new - gx
    scalar_t delta_gx_0 = gx_new[0] - gx[0];
    scalar_t delta_gx_1 = gx_new[1] - gx[1];
    scalar_t delta_gx_2 = gx_new[2] - gx[2];

    fuse_J_inv_update(index, J_inv_local, delta_x_0, delta_x_1, delta_x_2,
                      delta_gx_0, delta_gx_1, delta_gx_2);
  }
}

void launch_broyden_kernel(Tensor &x, const Tensor &xd_tgt, const Tensor &voxel,
                           const Tensor &grid_J_inv, const Tensor &tfs,
                           const Tensor &bone_ids, bool align_corners,
                           Tensor &J_inv, Tensor &is_valid,
                           const Tensor &offset, const Tensor &scale,
                           float cvg_threshold, float dvg_threshold) {

  // calculate #threads required
  int64_t n_batch = xd_tgt.size(0);
  int64_t n_point = xd_tgt.size(1);
  int64_t n_init = bone_ids.size(0);
  int64_t count = n_batch * n_point * n_init;

  if (count > 0) {
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
        x.scalar_type(), "fuse_kernel_cuda", [&] {
          broyden_kernel<<<GET_BLOCKS(count, 512), 512, 0,
                           at::cuda::getCurrentCUDAStream()>>>(
              static_cast<int>(count), static_cast<int>(n_batch),
              static_cast<int>(n_point), static_cast<int>(n_init),
              getTensorInfo<scalar_t, int>(voxel),
              getTensorInfo<scalar_t, int>(grid_J_inv),
              x.packed_accessor32<scalar_t, 4>(),
              xd_tgt.packed_accessor32<scalar_t, 3>(),
              voxel.packed_accessor32<scalar_t, 5>(),
              grid_J_inv.packed_accessor32<scalar_t, 5>(),
              tfs.packed_accessor32<scalar_t, 4>(),
              bone_ids.packed_accessor32<int, 1>(),
              J_inv.packed_accessor32<scalar_t, 5>(),
              is_valid.packed_accessor32<bool, 3>(),
              offset.packed_accessor32<scalar_t, 3>(),
              scale.packed_accessor32<scalar_t, 3>(), cvg_threshold,
              dvg_threshold, 0);
          C10_CUDA_KERNEL_LAUNCH_CHECK();
        });
  }
  cudaDeviceSynchronize();
}