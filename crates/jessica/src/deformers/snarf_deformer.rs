use std::f64::consts::PI;
use tch::{nn, Device, IndexOp, Kind, Tensor};
use jessica_fast_snarf::fast_snarf_deformer::ForwardDeformer;
use jessica_smpl_lib::body_models::{SMPLOutput, SMPL};
use crate::deformers::deformer::{get_bbox_from_smpl, BoundingBox, Deformer, SMPLParams};
use jessica_utils::module::ModuleMT;

/// Represents the SNARF deformer.
pub struct SNARFDeformer {
    body_model: SMPL,
    deformer: ForwardDeformer,
    opt: SNARFDeformerOptions,
    tfs_inv_t: Tensor,
    vs_template: Tensor,
    bbox: BoundingBox,
    w2s: Tensor,
    vertices: Tensor,
    tfs: Tensor,
    smpl_outputs: SMPLOutput,
}

/// Represents the options for the SNARF deformer.
#[derive(Debug)]
pub struct SNARFDeformerOptions {
    cano_pose: CanoPose,
    resolution: i64,
}

/// Represents the canonical pose options.
#[derive(Debug)]
pub enum CanoPose {
    DAPose,
    APose,
    Custom(Vec<f64>),
}

/// Returns a predefined rest pose tensor based on the given canonical pose name and device.
///
/// # Arguments
/// * `cano_pose` - The name of the canonical pose ("da_pose" or "a_pose").
/// * `device` - The device to create the tensor on (default: "cuda").
///
/// # Returns
/// * `Tensor` - The predefined rest pose tensor.
fn get_predefined_rest_pose(cano_pose: CanoPose, device: Device) -> Tensor {
    // Create a tensor of zeros with shape [1, 69] on the specified device using tch-rs.
    let body_pose_t = Tensor::zeros(&[1, 69], (Kind::Float, device));

    match cano_pose {
        CanoPose::DAPose => {
            let _ = body_pose_t.i((.., 2)).fill_(PI / 6.0);
            let _ = body_pose_t.i((.., 5)).fill_(-PI / 6.0);
        },
        CanoPose::APose => {
            let _ = body_pose_t.i((.., 2)).fill_(0.2);
            let _ = body_pose_t.i((.., 5)).fill_(-0.2);
            let _ = body_pose_t.i((.., 47)).fill_(-0.8);
            let _ = body_pose_t.i((.., 50)).fill_(0.8);
        }
        _ => panic!("Unknown cano_pose: {:?}", cano_pose)
    }

    body_pose_t
}

impl SNARFDeformer {
    /// Creates a new instance of the SNARF deformer.
    ///
    /// # Arguments
    /// * `p` - Path for storing variables
    /// * `model_path` - Path to the SMPL model
    /// * `gender` - Gender for the SMPL model
    /// * `opt` - The options for the SNARF deformer
    /// * `device` - The device to run computations on
    ///
    /// # Returns
    /// * `Self` - A new instance of the SNARF deformer.
    pub fn new(p: &nn::Path, model_path: &str, gender: &str, opt: SNARFDeformerOptions, device: Device) -> Self {
        // Create an instance of the `SMPL` struct using the `model_path` and `gender`.
        let body_model = SMPL::new(&p.sub("smpl"), Some(&model_path), None, 10, None, None, None, 1, None, gender.to_string(), device);

        let mut deformer = ForwardDeformer::new(&p.sub("deformer"), device);

        let betas = Tensor::zeros(&[1, 10], (Kind::Float, device));
        let body_pose_t = match &opt.cano_pose {
            CanoPose::DAPose => get_predefined_rest_pose(CanoPose::DAPose, device),
            CanoPose::APose => get_predefined_rest_pose(CanoPose::APose, device),
            CanoPose::Custom(pose) => {
                let body_pose_t = Tensor::zeros(&[1, 69], (Kind::Float, device));
                let _ = body_pose_t.i((.., 2)).fill_(pose[0]);
                let _ = body_pose_t.i((.., 5)).fill_(pose[1]);
                let _ = body_pose_t.i((.., 47)).fill_(pose[2]);
                let _ = body_pose_t.i((.., 50)).fill_(pose[3]);
                body_pose_t
            }
        };

        let smpl_outputs = body_model.forward_mt((
                                                     betas.shallow_clone(),
                                                     body_pose_t,
                                                     Tensor::zeros(&[1, 3], (Kind::Float, device)),
                                                     Tensor::zeros(&[1, 3], (Kind::Float, device))
                                                 ), false);

        let bbox = get_bbox_from_smpl(&smpl_outputs.0.slice(0, 0, 1, 1).detach(), 1.2);
        let tfs_inv_t = smpl_outputs.7.shallow_clone();
        let vs_template = smpl_outputs.0.shallow_clone();

        // Call the `switch_to_explicit` method of the `deformer` with the specified arguments.
        deformer.switch_to_explicit(
            opt.resolution,
            &smpl_outputs.0.to_kind(Kind::Float).detach(),
            &body_model.model.weights.shallow_clone().unsqueeze(0).detach(),
        );

        // Initialize the remaining fields with default values.
        let w2s = Tensor::zeros(&[1, 4, 4], (Kind::Float, device));
        let vertices = Tensor::zeros(&[1, body_model.get_num_verts(), 3], (Kind::Float, device));
        let tfs = Tensor::zeros(&[1, 4, 4], (Kind::Float, device));

        Self {
            body_model,
            deformer,
            opt,
            tfs_inv_t,
            vs_template,
            bbox,
            w2s,
            vertices,
            tfs,
            smpl_outputs,
        }
    }

    fn deform(&self, pts: &Tensor, train: bool) -> (Tensor, Tensor) {
        // Reshape the points to match the batch size and number of points.
        // This is equivalent to `pts = pts.reshape(batch_size, -1, 3)` in Python.
        let batch_size = self.vertices.size()[0];
        let pts = pts.reshape(&[batch_size, -1, 3]);

        // Deform the points using the `deformer`.
        // This is equivalent to `pts_cano, others = self.deformer.forward(pts, cond=None, tfs=self.tfs, eval_mode=eval_mode)` in Python.
        let (pts_cano, valid_ids, _j_inv) = tch::no_grad(|| {
            self.deformer.forward_mt((pts.shallow_clone(), self.tfs.shallow_clone()), train)
        });

        // Get the validity mask from the deformation results.
        // This is equivalent to `valid = others["valid_ids"].reshape(point_size, -1)` in Python.
        let point_size = pts.size()[0];
        let valid = valid_ids.reshape(&[point_size, -1]);

        // Return the deformed points and their corresponding validity mask.
        (pts_cano.reshape(&[point_size, -1, 3]), valid)
    }
}

impl Deformer for SNARFDeformer {
    fn prepare_deformer(&mut self, smpl_params: &SMPLParams) {
        let device = smpl_params.betas.device();

        if self.body_model.device != device {
            panic!("`smpl_params` must be on same device as `body_model`")
        }

        let smpl_outputs = self.body_model.forward_mt((
                                                          smpl_params.betas.shallow_clone(),
                                                          smpl_params.body_pose.shallow_clone(),
                                                          smpl_params.global_orient.shallow_clone(),
                                                          smpl_params.transl.shallow_clone()
                                                      ), false);
        
        let s2w = smpl_outputs.6.i((0, 0));
        let w2s = s2w.inverse();
        let tfs = (w2s.unsqueeze(-1) * smpl_outputs.7.to_kind(Kind::Float) * self.tfs_inv_t.shallow_clone()).to_kind(Kind::Float);
        self.deformer.deformer_precompute(&tfs);

        self.w2s = w2s.shallow_clone();
        self.vertices = (smpl_outputs.0.shallow_clone() * w2s.i((.., 0..3, 0..3)).permute(&[0, 2, 1])) + w2s.i((.., .., 0..3, 3));
        self.tfs = tfs;
        self.smpl_outputs = smpl_outputs;
    }

    fn deform_train(&self, pts: &Tensor, model: &impl Fn(&Tensor) -> (Tensor, Tensor)) -> (Tensor, Tensor) {
        // Deform the points to canonical space.
        let (pts_cano_all, valid) = self.deform(pts, true);

        // Initialize tensors for RGB and sigma values.
        // This is equivalent to `rgb_cano = torch.zeros_like(pts_cano_all).float()` and `sigma_cano = -torch.ones_like(pts_cano_all[..., 0]).float() * 1e5` in Python.
        let mut rgb_cano = Tensor::zeros_like(&pts_cano_all).to_kind(Kind::Float);
        let mut sigma_cano = Tensor::ones_like(&pts_cano_all.i((.., .., 0))).to_kind(Kind::Float) * -1e5;

        // Check if there are any valid points.
        if valid.is_nonzero() {
            // Create a boolean mask for valid points.
            let valid_mask = valid.to_kind(Kind::Bool);

            let (rgb_valid, sigma_valid) = tch::autocast(true, || model(&pts_cano_all.masked_select(&valid_mask).reshape(&[-1, 3])));

            // Update the RGB and sigma values for valid points.
            let _ = rgb_cano.index_put_(&[Some(&valid_mask)], &rgb_valid, false);
            let _ = sigma_cano.index_put_(&[Some(&valid_mask)], &sigma_valid, false);

            // Check if the RGB and sigma values are finite.
            let rgb_finite = rgb_cano.isfinite().all_dim(-1, false);
            let sigma_finite = sigma_cano.isfinite();
            let valid = rgb_finite.logical_and(&sigma_finite);

            // Set the RGB and sigma values for invalid points to default values.
            let _ = rgb_cano.index_put_(&[Some(&valid.logical_not())], &Tensor::zeros_like(&rgb_cano), false);
            let _ = sigma_cano.index_put_(&[Some(&valid.logical_not())], &(Tensor::ones_like(&sigma_cano) * -1e5), false);
        }

        // Return the deformed RGB and sigma values.
        (rgb_cano, sigma_cano)
    }

    fn deform_test(&self, pts: &Tensor, model: &impl Fn(&Tensor) -> (Tensor, Tensor)) -> (Tensor, Tensor) {
        // Deform the points to canonical space.
        let (pts_cano_all, valid) = self.deform(pts, false);

        // Initialize tensors for RGB and sigma values.
        let mut rgb_cano = Tensor::zeros_like(&pts_cano_all).to_kind(Kind::Float);
        let mut sigma_cano = Tensor::zeros_like(&pts_cano_all.i((.., .., 0))).to_kind(Kind::Float);

        // Check if there are any valid points.
        if valid.is_nonzero() {
            // Create a boolean mask for valid points.
            let valid_mask = valid.to_kind(Kind::Bool);

            // Apply the deformation model to the valid points.
            let (rgb_valid, sigma_valid) = tch::autocast(true, || model(&pts_cano_all.masked_select(&valid_mask).reshape(&[-1, 3])));

            // Update the RGB and sigma values for valid points.
            let _ = rgb_cano.index_put_(&[Some(&valid_mask)], &rgb_valid, false);
            let _ = sigma_cano.index_put_(&[Some(&valid_mask)], &sigma_valid, false);
        }

        (rgb_cano, sigma_cano)
    }
}