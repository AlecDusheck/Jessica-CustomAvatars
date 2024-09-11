use std::env;
use std::path::PathBuf;
use std::process::Command;

fn main() {
    // Python configuration
    let python_version = "3.10";
    let python_executable = env::var("PYTHON_SYS_EXECUTABLE")
        .unwrap_or_else(|_| format!("/usr/bin/python{}", python_version));

    let output = Command::new(&python_executable)
        .args(&["-c", "import sys; print(sys.prefix)"])
        .output()
        .expect("Failed to execute python");
    let prefix = String::from_utf8(output.stdout).unwrap().trim().to_string();

    let python_libdir = format!("{}/lib", prefix);

    // System PyTorch configuration
    let libtorch = PathBuf::from(env::var("LIBTORCH").unwrap_or("/usr/local/torch/lib/python3.10/site-packages/torch".to_string()));

    // Tell cargo to look for shared libraries
    println!("cargo:rustc-link-search=native={}", python_libdir);
    println!("cargo:rustc-link-search=native={}", libtorch.join("lib").display());

    // Link against Python
    println!("cargo:rustc-link-lib=python{}", python_version);

    // Link against system PyTorch libraries
    println!("cargo:rustc-link-lib=torch");
    println!("cargo:rustc-link-lib=torch_cpu");
    println!("cargo:rustc-link-lib=c10");

    // C++ standard library
    println!("cargo:rustc-link-lib=stdc++");

    // Include directories
    println!("cargo:include={}", libtorch.join("include").display());
    println!("cargo:include={}", libtorch.join("include/torch/csrc/api/include").display());

    // Set environment variables
    println!("cargo:rustc-env=LIBTORCH={}", libtorch.display());
    println!("cargo:rustc-env=LD_LIBRARY_PATH={}:{}", libtorch.join("lib").display(), python_libdir);

    // println!("cargo:warning=Using system PyTorch at: {}", libtorch.display());
    // println!("cargo:warning=Using Python libraries from: {}", python_libdir);
}