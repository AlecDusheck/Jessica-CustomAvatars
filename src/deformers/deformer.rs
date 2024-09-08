use tch::Tensor;

/// Represents a bounding box with minimum and maximum vertices.
pub struct BoundingBox {
    min_vert: Tensor,
    max_vert: Tensor,
}

/// Represents the parameters for the SMPL model.
pub struct SMPLParams {
    pub betas: Tensor,
    pub body_pose: Tensor,
    pub global_orient: Tensor,
    pub transl: Tensor,
}

/// Represents a ray with origin, direction, near and far bounds.
pub struct Rays {
    pub o: Tensor,
    pub d: Tensor,
    pub near: Tensor,
    pub far: Tensor,
}

/// Computes the bounding box from SMPL vertices.
///
/// # Arguments
/// * `vs` - Vertex tensor of shape [1, num_vertices, 3]
/// * `factor` - Scaling factor for the bounding box
///
/// # Returns
/// * `BoundingBox` - The computed bounding box
pub fn get_bbox_from_smpl(vs: &Tensor, factor: f64) -> BoundingBox {
    assert_eq!(vs.size()[0], 1, "Expected batch size of 1 for vertices");

    let (min_vert, _) = vs.min_dim(1, false);
    let (max_vert, _) = vs.max_dim(1, false);

    let c = (&max_vert + &min_vert) / 2.0;
    let s = (&max_vert - &min_vert) / 2.0;
    let (s, _) = s.max_dim(-1, false);
    let s = &s * factor;

    let min_vert = &c - s.unsqueeze(-1);
    let max_vert = &c + s.unsqueeze(-1);

    BoundingBox { min_vert, max_vert }
}

pub trait Deformer {
    /// Deforms points and computes RGB and sigma values for training.
    ///
    /// # Arguments
    /// * `pts` - Points to deform
    /// * `model` - Neural network model to evaluate deformed points
    ///
    /// # Returns
    /// * `(Tensor, Tensor)` - RGB and sigma values
    fn deform_train(&self, pts: &Tensor, model: &impl Fn(&Tensor) -> (Tensor, Tensor)) -> (Tensor, Tensor);
    
    /// Deforms points and computes RGB and sigma values for testing.
    ///
    /// # Arguments
    /// * `pts` - Points to deform
    /// * `model` - Neural network model to evaluate deformed points
    ///
    /// # Returns
    /// * `(Tensor, Tensor)` - RGB and sigma values
    fn deform_test(&self, pts: &Tensor, model: &impl Fn(&Tensor) -> (Tensor, Tensor)) -> (Tensor, Tensor);

    // Calls the appropriate deformation method based on the evaluation mode.
    ///
    /// # Arguments
    /// * `pts` - Points to deform
    /// * `model` - Neural network model to evaluate deformed points
    /// * `eval_mode` - Whether to use evaluation mode
    ///
    /// # Returns
    /// * `(Tensor, Tensor)` - RGB and sigma values
    fn call(&self, pts: &Tensor, model: &impl Fn(&Tensor) -> (Tensor, Tensor), eval_mode: bool) -> (Tensor, Tensor) {
        if eval_mode {
            self.deform_test(pts, model)
        } else {
            self.deform_train(pts, model)
        }
    }
}