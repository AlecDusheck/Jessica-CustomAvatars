use log::debug;
use tch::{nn, Device, IndexOp, Tensor};
use jessica_utils::module::ModuleMT;
use crate::lbs::{lbs, blend_shapes};
use crate::model::data::DataModel;
use crate::vertex_joints::VertexJointSelector;

/// The number of body joints in the SMPL model.
static NUM_BODY_JOINTS: i64 = 23; // NOT including the root joint

/// The dimension of the shape space in the SMPL model.
static SHAPE_SPACE_DIM: i64 = 300;

pub type SMPLOutput = (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor);
pub type SMPLInput = (Tensor, Tensor, Tensor, Tensor);

/// The SMPL model struct.
pub struct SMPL {
    /// The SMPL model data.
    /// TODO: should this be public?
    pub model: DataModel,

    /// The shape parameters (betas) of the SMPL model.
    betas: Tensor,

    /// The number of shape parameters (betas) used in the model.
    pub num_betas: i64,

    /// The global orientation of the body.
    global_orient: Tensor,

    /// The pose of the body joints.
    body_pose: Tensor,

    /// The translation of the body.
    transl: Tensor,

    /// Local copy of trimmed shapedirs for betas
    shapedirs: Tensor,

    /// The batch size used for creating the member variables.
    batch_size: i64,

    /// The gender of the SMPL model.
    pub gender: String,

    /// The vertex joint selector used to select extra joints from vertices.
    vertex_joint_selector: VertexJointSelector,

    /// Flag for converting the pose to rotation matrices.
    pub pose2rot: bool,

    /// An optional joint mapper for re-ordering the SMPL joints.
    joint_mapper: Option<fn(Tensor) -> Tensor>,

    /// The device the tensors are on
    pub device: Device,
}

impl ModuleMT<SMPLInput, SMPLOutput> for SMPL
{
    /// The forward pass of the SMPL model.
    ///
    /// This function takes the shape parameters (betas), body pose, global orientation, and translation
    /// as input tensors, and returns the output of the SMPL model, which includes the vertices,
    /// joints, transformation matrices, and shape and pose offsets.
    ///
    /// If any of the input tensors are not provided, the default values from the model are used.
    fn forward_mt(
        &self,
        xs: (Tensor, Tensor, Tensor, Tensor),
        _train: bool,
    ) -> (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor) {
        let (betas, body_pose, global_orient, transl) = xs;

        debug!("Input tensor shapes:");
        debug!("betas: {:?}", betas.size());
        debug!("body_pose: {:?}", body_pose.size());
        debug!("global_orient: {:?}", global_orient.size());
        debug!("transl: {:?}", transl.size());

        let batch_size = self.batch_size.max(betas.size()[0]).max(body_pose.size()[0])
            .max(global_orient.size()[0]);

        let global_orient = global_orient.expand(&[batch_size, 3], false);
        let body_pose = body_pose.expand(&[batch_size, NUM_BODY_JOINTS * 3], false);
        let betas = betas.expand(&[batch_size, self.num_betas], false);

        let apply_trans = transl.size() == &[batch_size, 3];

        let full_pose = Tensor::cat(&[global_orient.shallow_clone(), body_pose.shallow_clone()], 1);
        let (vertices, joints, a, t, shape_offsets, pose_offsets) = lbs(
            &betas,
            &full_pose,
            &self.model.v_template,
            &self.shapedirs,
            &self.model.posedirs,
            &self.model.j_regressor,
            &self.model.kintree_table,
            &self.model.weights,
            self.pose2rot,
        );

        let joints = self.vertex_joint_selector.forward_mt((vertices.shallow_clone(), joints), false);

        // Map the joints to the current dataset if a joint mapper is provided
        let joints = match self.joint_mapper {
            Some(mapper) => mapper(joints),
            None => joints,
        };

        let (vertices, global_orient, body_pose, joints, betas, full_pose, a, t, shape_offsets, pose_offsets) = if apply_trans {
            let vertices = vertices + transl.unsqueeze(1);
            let joints = joints + transl.unsqueeze(1);
            (vertices, global_orient, body_pose, joints, betas, full_pose, a, t, shape_offsets, pose_offsets)
        } else {
            (vertices, global_orient, body_pose, joints, betas, full_pose, a, t, shape_offsets, pose_offsets)
        };

        debug!("Output tensor shapes:");
        debug!("vertices: {:?}", vertices.size());
        debug!("joints: {:?}", joints.size());
        debug!("shape_offsets: {:?}", shape_offsets.size());
        debug!("pose_offsets: {:?}", pose_offsets.size());

        (vertices, global_orient, body_pose, joints, betas, full_pose, a, t, shape_offsets, pose_offsets)
    }
}

impl SMPL {
    /// Creates a new SMPL model instance.
    pub fn new(
        p: &nn::Path,
        model_path: Option<&str>,
        betas: Option<Tensor>,
        num_betas: i64,
        global_orient: Option<Tensor>,
        body_pose: Option<Tensor>,
        transl: Option<Tensor>,
        batch_size: i64,
        joint_mapper: Option<fn(Tensor) -> Tensor>,
        gender: String,
        device: Device,
    ) -> Self {
        let mut model = DataModel::new(device);
        if let Some(path) = model_path {
            model = DataModel::load_from_file(path, device).expect("Failed to load model");
        }

        let num_betas = num_betas.min(SHAPE_SPACE_DIM);

        let shapedirs = p.var_copy("shapedirs", &model.shapedirs.i((.., .., ..num_betas)));

        let vertex_joint_selector = VertexJointSelector::new(
            &p,
            &crate::constants::VERTEX_IDS["smplh"],
            true,
            true,
        );

        let global_orient_param = p.var("global_orient", &[batch_size, 3], tch::nn::Init::Const(0.));
        let body_pose_param = p.var("body_pose", &[batch_size, NUM_BODY_JOINTS * 3], tch::nn::Init::Const(0.));
        let transl_param = p.var("transl", &[batch_size, 3], tch::nn::Init::Const(0.));
        let betas_param = p.var("betas", &[batch_size, num_betas], tch::nn::Init::Const(0.));

        let mut smpl = SMPL {
            model,
            betas: betas_param,
            num_betas,
            global_orient: global_orient_param,
            body_pose: body_pose_param,
            transl: transl_param,
            shapedirs,
            batch_size,
            gender,
            vertex_joint_selector,
            pose2rot: true,
            joint_mapper,
            device
        };

        if let Some(global_orient) = global_orient {
            tch::no_grad(|| {
                smpl.global_orient.copy_(&global_orient);
            });
        }

        if let Some(body_pose) = body_pose {
            tch::no_grad(|| {
                smpl.body_pose.copy_(&body_pose);
            });
        }

        if let Some(transl) = transl {
            tch::no_grad(|| {
                smpl.transl.copy_(&transl);
            });
        }

        if let Some(betas) = betas {
            tch::no_grad(|| {
                smpl.betas.copy_(&betas);
            });
        }

        smpl
    }

    /// Returns a string representation of the SMPL model.
    pub fn extra_repr(&self) -> String {
        let msg = vec![
            format!("Gender: {}", self.gender.to_uppercase()),
            format!("Number of joints: {}", self.model.j_regressor.size()[0]),
            format!("Betas: {}", self.num_betas),
        ];
        msg.join("\n")
    }

    /// Returns the number of vertices in the SMPL model.
    pub fn get_num_verts(&self) -> i64 {
        self.model.v_template.size()[0]
    }

    /// Returns the number of faces in the SMPL model.
    pub fn get_num_faces(&self) -> i64 {
        self.model.f.size()[0]
    }

    /// Performs the forward pass for the shape component of the SMPL model.
    ///
    /// This function takes the shape parameters (betas) as input and returns the shaped
    /// vertices, the input betas, and the shaped vertices.
    pub fn forward_shape(&self, betas: Option<Tensor>) -> (Tensor, Tensor, Tensor) {
        let betas = betas.unwrap_or_else(|| self.betas.shallow_clone());
        let v_shaped = &self.model.v_template + blend_shapes(&betas, &self.shapedirs);
        (v_shaped.shallow_clone(), betas, v_shaped)
    }
}