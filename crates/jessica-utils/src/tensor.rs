use tch::{IndexOp, Tensor};

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

/// Converts rotation matrices to euler angles (yaw only).
///
/// This function calculates the yaw angle from 3D rotation matrices.
/// It's designed to work with batches of rotation matrices.
///
/// # Arguments
///
/// * `rot_mats` - A tensor of shape `(N, 3, 3)` where N is the batch size,
///                and each 3x3 matrix is a rotation matrix.
///
/// # Returns
///
/// A tensor of shape `(N,)` containing the yaw angles in radians.
///
/// # Note
///
/// This function only calculates the yaw angle. It may not be suitable for
/// extreme cases of euler angles like [0.0, pi, 0.0].
pub fn rot_mat_to_euler(rot_mats: &Tensor) -> Tensor {
    // Calculates rotation matrix to euler angles
    // Careful with extreme cases of euler angles like [0.0, pi, 0.0]

    let sy = Tensor::sqrt(&(
        rot_mats.i((.., 0, 0)).pow(&Tensor::from(2)) +
            rot_mats.i((.., 1, 0)).pow(&Tensor::from(2))
    ));

    Tensor::atan2(&(-rot_mats.i((.., 2, 0))), &sy)
}