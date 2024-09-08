use tch::{nn, Device, IndexOp, Kind, Tensor};
use knn_points::knn::knn_points;
use three_d_pose_rs_lib::body_models::SMPL;

/// Represents a bounding box with minimum and maximum vertices.
pub struct BoundingBox {
    min_vert: Tensor,
    max_vert: Tensor,
}

/// Computes the bounding box from SMPL vertices.
///
/// # Arguments
/// * `vs` - Vertex tensor of shape [1, num_vertices, 3]
/// * `factor` - Scaling factor for the bounding box
///
/// # Returns
/// * `BoundingBox` - The computed bounding box
fn get_bbox_from_smpl(vs: &Tensor, factor: f64) -> BoundingBox {
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

/// Represents an SMPL deformer for transforming points between world and canonical space.
pub struct SMPLDeformer {
    body_model: SMPL,
    k: i64,
    threshold: f64,
    strategy: String,
    bbox: BoundingBox,
    t_template: Tensor,
    vs_template: Tensor,
    pose_offset_t: Tensor,
    shape_offset_t: Tensor,
    t_inv: Tensor,
    vertices: Tensor,
    w2s: Tensor,
}

impl SMPLDeformer {
    /// Creates a new SMPLDeformer instance.
    ///
    /// # Arguments
    /// * `p` - Path for storing variables
    /// * `model_path` - Path to the SMPL model
    /// * `gender` - Gender for the SMPL model
    /// * `threshold` - Threshold for valid deformations
    /// * `k` - Number of nearest neighbors to consider
    /// * `device` - The device to run computations on
    ///
    /// # Returns
    /// * `SMPLDeformer` - A new instance of SMPLDeformer
    pub fn new(p: &nn::Path, model_path: &str, gender: &str, threshold: f64, k: i64, device: Device) -> Self {
        let body_model = SMPL::new(&p.sub("smpl"), Some(model_path), None, 10, None, None, None, 1, None, gender.to_string(), device);

        let betas = Tensor::zeros(&[1, 10], (Kind::Float, device));
        let body_pose_t = Tensor::zeros(&[1, 69], (Kind::Float, device));
        let _ = body_pose_t.select(1, 2).fill_(std::f64::consts::PI / 6.0);
        let _ = body_pose_t.select(1, 5).fill_(-std::f64::consts::PI / 6.0);

        let smpl_outputs = body_model.forward_mt((
                                                     betas.shallow_clone(),
                                                     body_pose_t,
                                                     Tensor::zeros(&[1, 3], (Kind::Float, device)),
                                                     Tensor::zeros(&[1, 3], (Kind::Float, device))
                                                 ), false);

        let bbox = get_bbox_from_smpl(&smpl_outputs.0.slice(0, 0, 1, 1).detach(), 1.2);
        let t_template = smpl_outputs.7.shallow_clone();
        let vs_template = smpl_outputs.0.shallow_clone();
        let pose_offset_t = smpl_outputs.9.shallow_clone();
        let shape_offset_t = smpl_outputs.8.shallow_clone();

        // Initialize t_inv, vertices, and w2s with dummy tensors
        let t_inv = Tensor::zeros(&[1, 4, 4], (Kind::Float, device));
        let vertices = Tensor::zeros(&[1, body_model.get_num_verts(), 3], (Kind::Float, device));
        let w2s = Tensor::zeros(&[1, 4, 4], (Kind::Float, device));

        SMPLDeformer {
            body_model,
            k,
            threshold,
            strategy: "nearest_neighbor".to_string(),
            bbox,
            t_template,
            vs_template,
            pose_offset_t,
            shape_offset_t,
            t_inv,
            vertices,
            w2s,
        }
    }

    /// Gets the bounding box of the deformed model.
    ///
    /// # Returns
    /// * `BoundingBox` - The bounding box of the deformed model
    pub fn get_bbox_deformed(&self) -> BoundingBox {
        get_bbox_from_smpl(&self.vertices.i((0..1, .., ..)).detach(), 1.2)
    }

    /// Prepares the deformer with SMPL parameters.
    ///
    /// # Arguments
    /// * `smpl_params` - A struct containing SMPL parameters
    pub fn prepare_deformer(&mut self, smpl_params: &SMPLParams) {
        let device = smpl_params.betas.device();
        if self.body_model.device != device {
            panic!("`smpl_params` must be on device")
        }

        let smpl_outputs = self.body_model.forward_mt((
                                                          smpl_params.betas.shallow_clone(),
                                                          smpl_params.body_pose.shallow_clone(),
                                                          smpl_params.global_orient.shallow_clone(),
                                                          smpl_params.transl.shallow_clone()
                                                      ), false);

        let s2w = smpl_outputs.6.i((0, 0));
        let w2s = s2w.inverse();

        let mut t_inv = smpl_outputs.7.to_kind(Kind::Float).inverse().matmul(&s2w.unsqueeze(-1));
        t_inv.i((.., 0..3, 3)) += &(&self.pose_offset_t - &smpl_outputs.9);
        t_inv.i((.., 0..3, 3)) += &(&self.shape_offset_t - &smpl_outputs.8);
        t_inv = self.t_template.matmul(&t_inv);

        self.t_inv = t_inv;
        self.vertices = smpl_outputs.0.matmul(&w2s.i((.., 0..3, 0..3)).permute(&[0, 2, 1])) + w2s.i((.., 0..1, 0..3)).unsqueeze(1);
        self.w2s = w2s;
    }

    /// Transforms rays from world to SMPL coordinate system.
    ///
    /// # Arguments
    /// * `rays` - Ray struct containing origin and direction
    pub fn transform_rays_w2s(&self, rays: &mut Rays) {
        let w2s = &self.w2s;

        rays.o = rays.o.matmul(&w2s.i((.., 0..3, 0..3)).permute(&[0, 2, 1])) + w2s.i((.., .., 0..3, 3));
        rays.d = rays.d.matmul(&w2s.i((.., 0..3, 0..3)).permute(&[0, 2, 1])).to_kind(rays.d.kind());

        let d = rays.o.square().sum_dim_intlist(&[-1], false, Kind::Float).sqrt();

        rays.near = &d - 1.0;
        rays.far = &d + 1.0;
    }

    /// Deforms points to canonical space.
    ///
    /// # Arguments
    /// * `pts` - Points to deform
    ///
    /// # Returns
    /// * `(Tensor, Tensor)` - Deformed points and validity mask
    pub fn deform(&self, pts: &Tensor) -> (Tensor, Tensor) {
        let batch_size = self.vertices.size()[0];
        let pts = pts.reshape(&[batch_size, -1, 3]);

        let knn_result = tch::no_grad(|| {
            knn_points(
                &pts.to_kind(Kind::Float),
                &self.vertices.to_kind(Kind::Float),
                None,
                None,
                2,
                self.k,
                -1,
                false,
                false,
            )
        });
        let dist_sq = knn_result.dists;
        let idx = knn_result.idx;

        let valid = dist_sq.lt(self.threshold.powi(2));
        let idx = idx.squeeze();

        let mut pts_cano = Tensor::zeros_like(&pts).to_kind(Kind::Float);

        for i in 0..batch_size {
            let tv_inv = self.t_inv.i((i, idx.i((i,)))).permute(&[1, 0, 2]);
            let transformed = tv_inv.i((.., 0..3, 0..3)).matmul(&pts.i((i, .., ..)).unsqueeze(-1)).squeeze();
            pts_cano.i((i,)).copy_(&(transformed + tv_inv.i((.., 0..3, 3))));
        }

        (pts_cano.reshape(&[-1, 3]), valid.view(&[-1]))
    }

    /// Deforms points and computes RGB and sigma values for training.
    ///
    /// # Arguments
    /// * `pts` - Points to deform
    /// * `model` - Neural network model to evaluate deformed points
    ///
    /// # Returns
    /// * `(Tensor, Tensor)` - RGB and sigma values
    pub fn deform_train(&self, pts: &Tensor, model: &impl Fn(&Tensor) -> (Tensor, Tensor)) -> (Tensor, Tensor) {
        let (pts_cano, valid) = self.deform(pts);
        let mut rgb_cano = Tensor::zeros(pts.size(), (Kind::Float, pts.device()));
        let mut sigma_cano = Tensor::ones(&[pts.size()[0], pts.size()[1]], (Kind::Float, pts.device())) * -1e5;

        if valid.any() {
            let valid_mask = valid.to_kind(Kind::Bool);
            let (rgb_valid, sigma_valid) = model(&pts_cano.masked_select(&valid_mask).reshape(&[-1, 3]));
            rgb_cano.masked_scatter_(&valid_mask.unsqueeze(-1).expand_as(&rgb_cano), &rgb_valid);
            sigma_cano.masked_scatter_(&valid_mask, &sigma_valid);

            let valid = rgb_cano.isfinite().all_dim(-1, false) & sigma_cano.isfinite();
            rgb_cano.masked_fill_(&valid.logical_not().unsqueeze(-1), 0.0);
            sigma_cano.masked_fill_(&valid.logical_not(), -1e5);
        }

        (rgb_cano, sigma_cano)
    }

    /// Deforms points and computes RGB and sigma values for testing.
    ///
    /// # Arguments
    /// * `pts` - Points to deform
    /// * `model` - Neural network model to evaluate deformed points
    ///
    /// # Returns
    /// * `(Tensor, Tensor)` - RGB and sigma values
    pub fn deform_test(&self, pts: &Tensor, model: &impl Fn(&Tensor) -> (Tensor, Tensor)) -> (Tensor, Tensor) {
        let (pts_cano, valid) = self.deform(pts);
        let mut rgb_cano = Tensor::zeros(pts.size(), (Kind::Float, pts.device()));
        let mut sigma_cano = Tensor::zeros(&[pts.size()[0], pts.size()[1]], (Kind::Float, pts.device()));

        if valid.any() {
            let valid_mask = valid.to_kind(Kind::Bool);
            let (rgb_valid, sigma_valid) = model(&pts_cano.masked_select(&valid_mask).reshape(&[-1, 3]));
            let _ = rgb_cano.masked_scatter_(&valid_mask.unsqueeze(-1).expand_as(&rgb_cano), &rgb_valid);
            let _ = sigma_cano.masked_scatter_(&valid_mask, &sigma_valid);
        }

        (rgb_cano, sigma_cano)
    }

    /// Calls the appropriate deformation method based on the evaluation mode.
    ///
    /// # Arguments
    /// * `pts` - Points to deform
    /// * `model` - Neural network model to evaluate deformed points
    /// * `eval_mode` - Whether to use evaluation mode
    ///
    /// # Returns
    /// * `(Tensor, Tensor)` - RGB and sigma values
    pub fn call(&self, pts: &Tensor, model: &impl Fn(&Tensor) -> (Tensor, Tensor), eval_mode: bool) -> (Tensor, Tensor) {
        if eval_mode {
            self.deform_test(pts, model)
        } else {
            self.deform_train(pts, model)
        }
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