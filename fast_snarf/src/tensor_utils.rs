use tch::Tensor;

#[cfg(debug_assertions)]
pub fn validate_tensor(tensor: &Tensor, expected_dims: &[i64], name: &str) {
    let actual_dims = tensor.size();
    assert_eq!(
        actual_dims.len(),
        expected_dims.len(),
        "{} has {} dimensions, expected {}",
        name,
        actual_dims.len(),
        expected_dims.len()
    );

    for (i, (&actual, &expected)) in actual_dims.iter().zip(expected_dims.iter()).enumerate() {
        assert_eq!(
            actual,
            expected,
            "{} dimension {} is {}, expected {}",
            name,
            i,
            actual,
            expected
        );
    }

    // Additional check to ensure the tensor is contiguous
    assert!(tensor.is_contiguous(), "{} must be contiguous", name);
}

#[cfg(not(debug_assertions))]
fn validate_tensor(tensor: &Tensor, expected_dims: &[i64], name: &str) {
    // Do nothing
    // For release, we assume our tensor sizes are always correct. This is a very computationally expensive function
    // TODO: Is this ok?
}