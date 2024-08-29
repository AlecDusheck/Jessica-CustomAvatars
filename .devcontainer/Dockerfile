# Start with CUDA 12.1 base image
FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04

# Avoid prompts from apt
ENV DEBIAN_FRONTEND=noninteractive

# Install basic dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    wget \
    software-properties-common \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install GCC 12
RUN add-apt-repository ppa:ubuntu-toolchain-r/test && \
    apt-get update && \
    apt-get install -y gcc-12 g++-12 && \
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

# Cache PyTorch download
RUN mkdir -p /usr/local/pytorch-cache && \
    wget -P /usr/local/pytorch-cache https://download.pytorch.org/libtorch/cu121/libtorch-cxx11-abi-shared-with-deps-2.3.0%2Bcu121.zip

RUN apt-get update && apt-get install -y zip

# Install LibTorch (C++ version)
RUN unzip /usr/local/pytorch-cache/libtorch-cxx11-abi-shared-with-deps-2.3.0+cu121.zip -d /usr/local

# Set environment variables for LibTorch
ENV LIBTORCH=/usr/local/libtorch
ENV LD_LIBRARY_PATH=${LIBTORCH}/lib:$LD_LIBRARY_PATH

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

# Install Rust for the vscode user
USER $USERNAME
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/home/$USERNAME/.cargo/bin:${PATH}"

# Set up a working directory
WORKDIR /workspace

# Make sure Rust binaries are in PATH for the vscode user
ENV PATH="/home/$USERNAME/.cargo/bin:${PATH}"

# Verify Rust and GCC installation
RUN rustc --version && cargo --version && gcc --version
