# Project Jessica Environment Renderer
This project contains the source code to render Project Jessica environments given a compiled model for inference. It is written in Rust to be blazing fast and for reliability. 

## Development
For building, you'll need:
- C++17 compiler (GCC <=12, or whatever your `nvcc` wants)
- Python3.10 installed (`/usr/include/python3.10`). This version is currently hard corded in multiple `build.rs` files, but it can probably be changed. You'll also want the headers
- `nvcc` version for respective CUDA version
- Rust (stable supported)
- CUDA 12.1
- PyTorch 2.3.0 installed (`/usr/local/libtorch`)

We can probably use other Python versions but this is not tested, the 3.10 path is hard-coded. If you'd like to use a different CUDA version, that'll probably work too. You will need to update your PyTorch version to whatever CUDA version you have installed. In addition, you'll need to update the `tch-rs` package across all repos the version for your CUDA install.

### Dev Container
There is also a dev container available for use. The container has all the requirements already installed. If you can't get things to run use this. **Please note that the container requires the `nvidia-container-toolkit` installed on your host machine for CUDA pass-through, in addition to Docker.** 


![Image of avatar](./media/rotation.gif)

## Note about CUDA code
I've done some optimizations to make things compile on some terrible graphics cards.
In particular, the grid sampler was overhauled. If we end up with strange inaccurate results, try reverting the optimizations to see.