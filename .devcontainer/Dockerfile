# Start with CUDA 12.1 base image
FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04

# Avoid prompts from apt
ENV DEBIAN_FRONTEND=noninteractive

# Install basic dependencies
RUN apt-get update && apt-get install -y \
    git curl wget ninja-build software-properties-common \
    build-essential openssl libssl-dev \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Install GCC 12
RUN add-apt-repository ppa:ubuntu-toolchain-r/test && \
    apt-get update && apt-get install -y gcc-12 g++-12 && \
    update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-12 120 --slave /usr/bin/g++ g++ /usr/bin/g++-12 && \
    update-alternatives --set gcc /usr/bin/gcc-12

# Install Python 3.10 and dev headers
RUN add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y python3.10 python3.10-dev python3.10-venv python3-pip python3.10-distutils && \
    rm -rf /var/lib/apt/lists/*

# Set Python 3.10 as the default python version
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 && \
    update-alternatives --set python /usr/bin/python3.10

# Create a non-root user
ARG USERNAME=vscode
ARG USER_UID=1000
ARG USER_GID=$USER_UID

RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME \
    && apt-get update \
    && apt-get install -y sudo \
    && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME

# Create /usr/local/torch and set permissions
RUN mkdir -p /usr/local/torch && \
    chown $USERNAME:$USERNAME /usr/local/torch && \
    chmod 755 /usr/local/torch

# Switch to non-root user
USER $USERNAME

# Install Rust
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/home/$USERNAME/.cargo/bin:${PATH}"

# Install Python deps (global)
# TODO: Pin these?
RUN pip install --no-cache-dir numpy scipy cmake pyyaml typing_extensions ninja

# https://github.com/mattloper/chumpy/issues/55
RUN pip install --no-cache-dir git+https://github.com/mattloper/chumpy

# Set up a working directory
WORKDIR /workspace

# Clone PyTorch repo
RUN git clone --branch v2.3.0 --recursive https://github.com/pytorch/pytorch.git

# Set environment variables for compiling with new C++ ABI
ENV _GLIBCXX_USE_CXX11_ABI=1
ENV MAX_JOBS=3
ENV TORCH_CUDA_ARCH_LIST="8.9"
ENV BUILD_TEST=OFF

# Set up Python virtual environment
RUN python -m venv /home/$USERNAME/venv
ENV PATH="/home/$USERNAME/venv/bin:${PATH}"

# Install PyTorch build dependencies. This is inside the venv
# TODO: Pin these?
RUN pip install --no-cache-dir numpy scipy cmake pyyaml typing_extensions ninja

# Compile PyTorch from source
WORKDIR /workspace/pytorch
RUN python setup.py install --prefix=/usr/local/torch

ENV PYTHONPATH="/usr/local/torch/lib/python3.10/site-packages:${PYTHONPATH}"
ENV LD_LIBRARY_PATH="/usr/local/torch/lib:/usr/local/torch/lib/python3.10/site-packages/torch/lib:${LD_LIBRARY_PATH}"
ENV LIBTORCH="/usr/local/torch/lib/python3.10/site-packages/torch"

# Remove /workspace/pytorch directory if it exists
USER root
RUN rm -rf /workspace/pytorch
USER $USERNAME

# Set up a working directory
WORKDIR /home/$USERNAME

# Diagnostic steps
RUN echo "Checking PyTorch installation:" && \
    echo "Contents of /usr/local/torch:" && \
    ls -l /usr/local/torch && \
    echo "\nContents of /usr/local/torch/lib:" && \
    ls -l /usr/local/torch/lib && \
    echo "\nContents of /usr/local/torch/lib/python3.10/site-packages:" && \
    ls -l /usr/local/torch/lib/python3.10/site-packages && \
    echo "\nChecking for torch directory:" && \
    ls -l /usr/local/torch/lib/python3.10/site-packages/torch && \
    echo "\nPYTHONPATH: $PYTHONPATH" && \
    echo "LD_LIBRARY_PATH: $LD_LIBRARY_PATH" && \
    echo "LIBTORCH: $LIBTORCH" && \
    python -c "import sys; print('Python sys.path:'); print('\n'.join(sys.path))"

# Attempt to import torch and print version
RUN python -c "import torch; print(f'PyTorch version: {torch.__version__}')" || echo "Failed to import torch"

# Verify PyTorch installation (full check)
# TODO: do we need this
RUN python -c "import sys; print(sys.path); import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}'); print(f'cuDNN version: {torch.backends.cudnn.version()}')" || echo "Full PyTorch check failed"

# Final cleanup
USER root
RUN apt-get purge -y --auto-remove -o APT::AutoRemove::RecommendsImportant=false && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/* /root/.cache

# Switch back to non-root user
USER $USERNAME

# Verify Rust, GCC, and PyTorch installation
RUN rustc --version && cargo --version && gcc --version