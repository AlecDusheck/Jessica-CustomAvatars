use jessica_utils::Model;
use tch::{Kind, Tensor};

/// Represents a bounding box with minimum and maximum vertices.
#[derive(Debug)]
pub struct BoundingBox {
    pub min_vert: Tensor,
    pub max_vert: Tensor,
}

impl BoundingBox {
    pub fn new(min_vert: Tensor, max_vert: Tensor) -> Self {
        BoundingBox { min_vert, max_vert }
    }
}

impl From<Tensor> for BoundingBox {
    fn from(tensor: Tensor) -> Self {
        assert_eq!(tensor.size(), &[2, 3], "Expected tensor shape [2, 3]");
        let min_vert = tensor.slice(0, 0, 1, 1).squeeze();
        let max_vert = tensor.slice(0, 1, 2, 1).squeeze();
        BoundingBox::new(min_vert, max_vert)
    }
}

impl From<BoundingBox> for Tensor {
    fn from(bbox: BoundingBox) -> Self {
        Tensor::cat(&[bbox.min_vert.unsqueeze(0), bbox.max_vert.unsqueeze(0)], 0)
    }
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
    /// Prepares the deformer with SMPL parameters.
    ///
    /// # Arguments
    /// * `smpl_params` - A struct containing SMPL parameters
    fn prepare_deformer(&mut self, smpl_params: &SMPLParams);
    
    /// Deforms points and computes RGB and sigma values for training.
    ///
    /// # Arguments
    /// * `pts` - Points to deform
    /// * `model` - Neural network model to evaluate deformed points
    ///
    /// # Returns
    /// * `(Tensor, Tensor)` - RGB and sigma values
    fn deform_train(&self, pts: &Tensor, model: &Model) -> (Tensor, Tensor);
    
    /// Deforms points and computes RGB and sigma values for testing.
    ///
    /// # Arguments
    /// * `pts` - Points to deform
    /// * `model` - Neural network model to evaluate deformed points
    ///
    /// # Returns
    /// * `(Tensor, Tensor)` - RGB and sigma values
    fn deform_test(&self, pts: &Tensor, model: &Model) -> (Tensor, Tensor);
    
    // Calls the appropriate deformation method based on the evaluation mode.
    ///
    /// # Arguments
    /// * `pts` - Points to deform
    /// * `model` - Neural network model to evaluate deformed points
    /// * `eval_mode` - Whether to use evaluation mode
    ///
    /// # Returns
    /// * `(Tensor, Tensor)` - RGB and sigma values
    fn call(&self, pts: &Tensor, model: &Model, eval_mode: bool) -> (Tensor, Tensor) {
        if eval_mode {
            self.deform_test(pts, model)
        } else {
            self.deform_train(pts, model)
        }
    }
}