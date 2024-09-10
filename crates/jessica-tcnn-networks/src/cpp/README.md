# Quick notes
## `bindings.cpp`
Why is `bindings.cpp` copied from `tiny-cuda-nn`? By default, bindings are NOT built. They are used in the Python bindgen process and are not exposed to CPP for use in libtorch.

Therefore, we must build this file into our own source to use in our libtorch code, which we bindgen to Rust.

This file is entirely unmodified
## `include/pybind11`
`tiny-cuda-nn` contains some Python binding code that we don't need. Since the files are symlinked, we want to avoid modifying them, but we also don't want to include pybind11 in our builds. These are dummy includes that allow our code to compile