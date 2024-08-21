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
pub fn validate_tensor(_tensor: &Tensor, _expected_dims: &[i64], _name: &str) {
    // Do nothing
    // For release, we assume our tensor sizes are always correct. This is a very computationally expensive function
}

#[cfg(debug_assertions)]
pub fn validate_tensor_type(tensor: &Tensor, expected_kind: tch::Kind, name: &str) {
    assert_eq!(tensor.kind(), expected_kind, "{name}: Expected tensor kind `{:?}`, got `{:?}`", expected_kind, tensor.kind());
}

#[cfg(not(debug_assertions))]
pub fn validate_tensor_type(_tensor: &Tensor, _expected_kind: tch::Kind, _name: &str) {
    // Do nothing
    // For release, we assume our tensor sizes are always correct. This is a very computationally expensive function
}