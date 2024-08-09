# Project Jessica Environment Renderer
This project contains the source code to render Project Jessica environments given a compiled model for interference. It is written in Rust to be blazing fast and for reliability. 

This project uses PyTorch and CUDA extensively. Here's what's expected:
- PyTorch installed (`/usr/local/libtorch`)
- CUDA 12.1, other versions are NOT tested and are not targets currently

For building, you'll need:
- C++17 compiler 
- Python3.10 installed (`/usr/include/python3.10`). This version is currently hard corded in multiple `build.rs` files, but it can probably be changed
- `nvcc` version for respective CUDA version

![Image of avatar](./media/rotation.gif)