use tch::{nn, Tensor, Kind};
use std::collections::HashMap;
use crate::module_utils::ModuleT;

#[derive(Debug)]
pub struct VertexJointSelector {
    extra_joints_idxs: Tensor,
    num_joints: i64,
}

impl ModuleT<(&Tensor, &Tensor)> for VertexJointSelector {
    type Output = Tensor;
    fn forward_t(&self, xs: (&Tensor, &Tensor), _train: bool) -> Tensor {
        let (vertices, joints) = xs;
        let extra_joints = vertices.index_select(1, &self.extra_joints_idxs);
        Tensor::cat(&[joints, &extra_joints], 1)
    }
}

impl VertexJointSelector {
    pub fn new(
        vs: &nn::Path,
        vertex_ids: &HashMap<String, i64>,
        num_joints: i64,
        use_hands: bool,
        use_feet_keypoints: bool,
    ) -> Self {
        let mut extra_joints_idxs = Vec::new();

        let face_keyp_idxs = vec![
            *vertex_ids.get("nose").unwrap(),
            *vertex_ids.get("reye").unwrap(),
            *vertex_ids.get("leye").unwrap(),
            *vertex_ids.get("rear").unwrap(),
            *vertex_ids.get("lear").unwrap(),
        ];

        extra_joints_idxs.extend_from_slice(&face_keyp_idxs);

        if use_feet_keypoints {
            let feet_keyp_idxs = vec![
                *vertex_ids.get("LBigToe").unwrap(),
                *vertex_ids.get("LSmallToe").unwrap(),
                *vertex_ids.get("LHeel").unwrap(),
                *vertex_ids.get("RBigToe").unwrap(),
                *vertex_ids.get("RSmallToe").unwrap(),
                *vertex_ids.get("RHeel").unwrap(),
            ];

            extra_joints_idxs.extend_from_slice(&feet_keyp_idxs);
        }

        if use_hands {
            let tip_names = vec!["thumb", "index", "middle", "ring", "pinky"];

            for hand_id in &["l", "r"] {
                for tip_name in &tip_names {
                    let key = format!("{}{}", hand_id, tip_name);
                    extra_joints_idxs.push(*vertex_ids.get(&key).unwrap());
                }
            }
        }

        // Convert the extra_joints_idxs vector to a Tensor
        let extra_joints_idxs = Tensor::from_slice(&extra_joints_idxs)
            .to_kind(Kind::Int64)
            .to_device(vs.device());

        // Create a non-trainable tensor
        let extra_joints_idxs = vs.var_copy("extra_joints_idxs", &extra_joints_idxs);

        Self { extra_joints_idxs, num_joints }
    }
}