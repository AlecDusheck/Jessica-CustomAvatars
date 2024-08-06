use std::env;
use std::path::PathBuf;

fn main() {
    let cuda = PathBuf::from(env::var("CUDA_PATH").unwrap_or("/usr/local/cuda".to_string()));
    let cuda_include = cuda.join("include");

    println!("cargo:rustc-link-search=native={}", cuda.join("lib64").display());
    println!("cargo:rustc-link-search=native=/usr/local/libtorch/lib");
    println!("cargo:rustc-link-lib=cudart");

    let mut build = cc::Build::new();
    build.cuda(true)
        .include(&cuda_include)
        .include("/usr/include/python3.10")
        .include("/usr/local/libtorch/include")
        .file("src/cuda/filter_cuda.cpp")
        .file("src/cuda/filter_kernel.cu")
        .flag("--expt-relaxed-constexpr")
        .flag("-std=c++17");

    build.compile("filter_cuda");
}