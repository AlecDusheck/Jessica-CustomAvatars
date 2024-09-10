use std::env;
use std::path::PathBuf;
use std::process::Command;

fn main() {
    let cuda = PathBuf::from(env::var("CUDA_PATH").unwrap_or("/usr/local/cuda".to_string()));
    let cuda_include = cuda.join("include");
    let libtorch = PathBuf::from(env::var("LIBTORCH").unwrap_or("/usr/local/torch/lib/python3.10/site-packages/torch".to_string()));
    let tiny_cuda_nn = PathBuf::from(env::current_dir().unwrap()).join("tiny-cuda-nn");

    println!("cargo:warning=CUDA_PATH: {:?}", cuda);
    println!("cargo:warning=libtorch path: {:?}", libtorch);
    println!("cargo:warning=tiny-cuda-nn path: {:?}", tiny_cuda_nn);

    let build_path = tiny_cuda_nn.join("build");
    let lib_path = build_path.join("libtiny-cuda-nn.a");

    println!("cargo:warning=Expected tiny-cuda-nn library path: {:?}", lib_path);

    if !lib_path.exists() || env::var("FORCE_TCNN_BUILD").unwrap_or_default() == "true" {
        println!("cargo:warning=Building tiny-cuda-nn...");

        let cmake_config = Command::new("cmake")
            .current_dir(&tiny_cuda_nn)
            .args(&[".", "-B", "build", "-DCMAKE_BUILD_TYPE=Release"])
            .output()
            .expect("Failed to execute cmake");

        println!("cargo:warning=CMake configuration output: {}", String::from_utf8_lossy(&cmake_config.stdout));
        println!("cargo:warning=CMake configuration error: {}", String::from_utf8_lossy(&cmake_config.stderr));

        if !cmake_config.status.success() {
            panic!("Failed to configure tiny-cuda-nn");
        }

        let cmake_build = Command::new("cmake")
            .current_dir(&tiny_cuda_nn)
            .args(&["--build", "build", "--config", "Release"])
            .output()
            .expect("Failed to execute cmake build");

        println!("cargo:warning=CMake build output: {}", String::from_utf8_lossy(&cmake_build.stdout));
        println!("cargo:warning=CMake build error: {}", String::from_utf8_lossy(&cmake_build.stderr));

        if !cmake_build.status.success() {
            panic!("Failed to build tiny-cuda-nn");
        }

        println!("cargo:warning=tiny-cuda-nn build completed.");
    } else {
        println!("cargo:warning=Using existing tiny-cuda-nn build.");
    }

    if !lib_path.exists() {
        panic!("libtiny-cuda-nn.a not found at expected path: {:?}", lib_path);
    }

    println!("cargo:rustc-link-search=native={}", cuda.join("lib64").display());
    println!("cargo:rustc-link-search=native={}", libtorch.join("lib").display());
    println!("cargo:rustc-link-search=native={}", build_path.display());

    // Link libraries
    println!("cargo:rustc-link-lib=stdc++");
    println!("cargo:rustc-link-lib=cudart");
    println!("cargo:rustc-link-lib=c10");
    println!("cargo:rustc-link-lib=torch_cpu");
    println!("cargo:rustc-link-lib=torch");
    println!("cargo:rustc-link-lib=torch_cuda");
    println!("cargo:rustc-link-lib=c10_cuda");
    println!("cargo:rustc-link-lib=gomp");
    println!("cargo:rustc-link-lib=static=tiny-cuda-nn");

    // Compile tcnn.cpp
    cc::Build::new()
        .cuda(true)
        .cpp(true)
        .include(&cuda_include)
        .include(&libtorch.join("include"))
        .include(&libtorch.join("include/torch/csrc/api/include"))
        .include("/usr/include/python3.10")
        .include("src/cpp/include")
        .include(&tiny_cuda_nn.join("include"))
        .file("src/cpp/bindings.cpp")
        .file("src/cpp/tcnn.cpp")
        .flag("-std=c++17")
        .flag("--expt-relaxed-constexpr")
        .flag("-w")
        .compile("rust_tcnn");

    println!("cargo:warning=Build script completed successfully.");

    // Specify dependencies
    println!("cargo:rerun-if-changed=src/cpp/bindings.cpp");
    println!("cargo:rerun-if-changed=src/cpp/tcnn.cpp");
    println!("cargo:rerun-if-changed=tiny-cuda-nn/include");
    println!("cargo:rerun-if-changed=tiny-cuda-nn/src");
    println!("cargo:rerun-if-env-changed=FORCE_TCNN_BUILD");
}