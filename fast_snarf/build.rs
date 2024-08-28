use std::env;
use std::path::PathBuf;

fn main() {
    let cuda = PathBuf::from(env::var("CUDA_PATH").unwrap_or("/usr/local/cuda".to_string()));
    let cuda_include = cuda.join("include");
    let libtorch = PathBuf::from(env::var("LIBTORCH").unwrap_or("/usr/local/libtorch".to_string()));

    println!("cargo:warning=CUDA_PATH: {:?}", cuda);
    println!("cargo:warning=libtorch path: {:?}", libtorch);

    // TODO: remove, was having issues with my PyTorch install
    println!("cargo:warning=libtorch include path: {:?}", libtorch.join("include"));
    println!("cargo:warning=torch/all.h exists: {:?}", libtorch.join("include/torch/all.h").exists());

    println!("cargo:rustc-link-search=native={}", cuda.join("lib64").display());
    println!("cargo:rustc-link-search=native={}", libtorch.join("lib").display());

    // Link libraries
    println!("cargo:rustc-link-lib=stdc++");
    println!("cargo:rustc-link-lib=cudart");
    println!("cargo:rustc-link-lib=c10");
    println!("cargo:rustc-link-lib=torch_cpu");
    println!("cargo:rustc-link-lib=torch");
    println!("cargo:rustc-link-lib=torch_cuda");
    println!("cargo:rustc-link-lib=c10_cuda");
    println!("cargo:rustc-link-lib=gomp");

    let mut build = cc::Build::new();
    build
        .cuda(true)
        .cpp(true)
        .include(&cuda_include)
        .include(&libtorch.join("include"))
        .include(&libtorch.join("include/torch"))
        .include(&libtorch.join("include/torch/csrc/api/include"))
        // TODO: Auto detect Python version and path
        // .include("/usr/include/python3.10")
        .include("/usr/include/python3.12")
        .flag("-std=c++17")
        .flag("--expt-relaxed-constexpr");

    // Set CUDA architecture flags if TORCH_CUDA_ARCH_LIST is set
    if let Ok(arch_list) = env::var("TORCH_CUDA_ARCH_LIST") {
        for arch in arch_list.split(',') {
            build.flag(&format!("-gencode=arch=compute_{0},code=sm_{0}", arch));
        }
    }

    build
        .file("src/cuda/c_filter.cpp")
        .file("src/cuda/c_filter_kernel.cu")
        .compile("c_filter");

    build
        .file("src/cuda/c_fuse.cpp")
        .file("src/cuda/c_fuse_kernel.cu")
        .compile("c_fuse");

    build
        .file("src/cuda/c_precompute.cpp")
        .file("src/cuda/c_precompute_kernel.cu")
        .compile("c_precompute");

    println!("cargo:rerun-if-changed=src/cuda/c_filter.cpp");
    println!("cargo:rerun-if-changed=src/cuda/c_filter_kernel.cu");
    println!("cargo:rerun-if-changed=src/cuda/c_fuse.cpp");
    println!("cargo:rerun-if-changed=src/cuda/c_fuse_kernel.cu");
    println!("cargo:rerun-if-changed=src/cuda/c_precompute.cpp");
    println!("cargo:rerun-if-changed=src/cuda/c_precompute_kernel.cu");

    // Rerun if certain environment variables change
    println!("cargo:rerun-if-env-changed=CUDA_PATH");
    println!("cargo:rerun-if-env-changed=LIBTORCH");
    println!("cargo:rerun-if-env-changed=TORCH_CUDA_ARCH_LIST");
}