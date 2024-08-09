# Project Jessica Environment Renderer
This project contains the source code to render Project Jessica environments given a compiled model for inference. It is written in Rust to be blazing fast and for reliability. 

This project uses PyTorch and CUDA extensively. Here's what's expected:
- PyTorch installed (`/usr/local/libtorch`)
- CUDA 12.1, other versions are NOT tested and are not targets currently

For building, you'll need:
- C++17 compiler 
- Python3.10 installed (`/usr/include/python3.10`). This version is currently hard corded in multiple `build.rs` files, but it can probably be changed
- `nvcc` version for respective CUDA version
- Rust

![Image of avatar](./media/rotation.gif)

## Note to self on `knn_points`
We must reimplement this. 
We need two CUDA functions? 

- Find them here https://github.com/facebookresearch/pytorch3d/blob/1e0b1d9c727e8d1a11df5c25a0722c3f9e12711b/pytorch3d/csrc/ext.cpp#L45
- Convert this over to Rust: https://github.com/facebookresearch/pytorch3d/blob/main/pytorch3d/ops/knn.py
- Use CUDA functions in this... so fucking nasty