# Note regarding tests for `fast_snarf`
Why are they in here? Rust FFI linker is SO WEIRD and will not build the `fast_snarf` library as a executable, which means we can't have tests in there.

However, we can build this as a executable, so we put all the tests in here... idk?