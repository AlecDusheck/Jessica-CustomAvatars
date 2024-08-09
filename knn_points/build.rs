use std::env;
use std::path::PathBuf;

fn main() {
    let cuda = PathBuf::from(env::var("CUDA_PATH").unwrap_or("/usr/local/cuda".to_string()));
    let cuda_include = cuda.join("include");
    let libtorch = PathBuf::from("/usr/local/libtorch");

    println!("cargo:warning=CUDA_PATH: {:?}", cuda);
    println!("cargo:warning=libtorch path: {:?}", libtorch);

    println!("cargo:rustc-link-search=native={}", cuda.join("lib64").display());
    println!("cargo:rustc-link-search=native={}", libtorch.join("lib").display());

    // println!("cargo:rustc-link-arg=-Wl,--copy-dt-needed-entries");

    // Link libraries
    println!("cargo:rustc-link-lib=stdc++");
    println!("cargo:rustc-link-lib=cudart");
    println!("cargo:rustc-link-lib=c10");
    println!("cargo:rustc-link-lib=torch_cpu");
    println!("cargo:rustc-link-lib=torch");
    println!("cargo:rustc-link-lib=torch_cuda");
    println!("cargo:rustc-link-lib=c10_cuda");
    println!("cargo:rustc-link-lib=gomp");

    // Compile c_knn.cpp and c_knn.cu
    cc::Build::new()
        .cuda(true)
        .cpp(true)
        .include(&cuda_include)
        .include(&libtorch.join("include"))
        .include("/usr/include/python3.10")
        .file("src/cuda/c_knn.cpp")
        .file("src/cuda/c_knn.cu")
        .flag("-std=c++17")
        .flag("--expt-relaxed-constexpr")
        .compile("c_knn");

    // Specify dependencies
    println!("cargo:rerun-if-changed=src/cuda/c_knn.cpp");
    println!("cargo:rerun-if-changed=src/cuda/c_knn.cu");
    println!("cargo:rerun-if-changed=src/cuda/dispatch.cuh");
    println!("cargo:rerun-if-changed=src/cuda/mink.cuh");
    println!("cargo:rerun-if-changed=src/cuda/index_utils.cuh");
}