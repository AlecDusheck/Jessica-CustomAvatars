use tch::{nn, Tensor, Kind};
use std::collections::HashMap;
use jessica_utils::module::ModuleMT;

/// A module that selects and combines joint vertices from a mesh.
#[derive(Debug)]
pub struct VertexJointSelector {
    /// Indices of extra joints to be selected from vertices.
    extra_joints_idxs: Tensor,
}

impl ModuleMT<(Tensor, Tensor), Tensor> for VertexJointSelector {
    /// Performs the forward pass of the VertexJointSelector.
    ///
    /// # Arguments
    ///
    /// * `xs` - A tuple containing:
    ///   - vertices: A tensor of vertex positions.
    ///   - joints: A tensor of existing joint positions.
    /// * `_train` - A boolean indicating whether the module is in training mode (unused in this implementation).
    ///
    /// # Returns
    /// 
    /// A tensor containing the combined joint positions (existing joints + extra joints).
    fn forward_mt(&self, xs: (Tensor, Tensor), _train: bool) -> Tensor {
        let (vertices, joints) = xs;
        let extra_joints = vertices.index_select(1, &self.extra_joints_idxs);
        Tensor::cat(&[joints, extra_joints], 1)
    }
}

impl VertexJointSelector {
    /// Creates a new VertexJointSelector instance.
    ///
    /// # Arguments
    ///
    /// * `vs` - The variable store path for creating tensors.
    /// * `vertex_ids` - A HashMap mapping joint names to their corresponding vertex indices.
    /// * `num_joints` - The number of existing joints.
    /// * `use_hands` - A boolean indicating whether to include hand keypoints.
    /// * `use_feet_keypoints` - A boolean indicating whether to include feet keypoints.
    ///
    /// # Returns
    ///
    /// A new VertexJointSelector instance.
    pub fn new(
        vs: &nn::Path,
        vertex_ids: &HashMap<String, i64>,
        use_hands: bool,
        use_feet_keypoints: bool,
    ) -> Self {
        let mut extra_joints_idxs = Vec::new();

        // Helper function to safely get vertex ID
        let get_vertex_id = |key: &str| -> Option<i64> {
            vertex_ids.get(key).copied()
        };

        // Face keypoints
        for key in &["nose", "reye", "leye", "rear", "lear"] {
            if let Some(id) = get_vertex_id(key) {
                extra_joints_idxs.push(id);
            }
        }

        if use_feet_keypoints {
            for key in &["LBigToe", "LSmallToe", "LHeel", "RBigToe", "RSmallToe", "RHeel"] {
                if let Some(id) = get_vertex_id(key) {
                    extra_joints_idxs.push(id);
                }
            }
        }

        if use_hands {
            let tip_names = ["thumb", "index", "middle", "ring", "pinky"];
            for hand_id in &["l", "r"] {
                for tip_name in &tip_names {
                    let key = format!("{}{}", hand_id, tip_name);
                    if let Some(id) = get_vertex_id(&key) {
                        extra_joints_idxs.push(id);
                    }
                }
            }
        }
        
        // Convert to tensor
        let extra_joints_idxs_data = Tensor::from_slice(&extra_joints_idxs)
            .to_kind(Kind::Int64)
            .to_device(vs.device());

        // Create a new non-trainable variable
        let mut extra_joints_idxs = vs.var("extra_joints_idxs", &extra_joints_idxs_data.size(), nn::Init::Const(0.0));
        tch::no_grad(|| {
            extra_joints_idxs.copy_(&extra_joints_idxs_data);
        });
        let extra_joints_idxs = extra_joints_idxs.to_kind(Kind::Int64).set_requires_grad(false);
        

        Self { extra_joints_idxs }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tch::Device;

    fn setup_vertex_ids() -> HashMap<String, i64> {
        let mut vertex_ids = HashMap::new();
        // Face keypoints
        vertex_ids.insert("nose".to_string(), 0);
        vertex_ids.insert("reye".to_string(), 1);
        vertex_ids.insert("leye".to_string(), 2);
        vertex_ids.insert("rear".to_string(), 3);
        vertex_ids.insert("lear".to_string(), 4);
        // Feet keypoints
        vertex_ids.insert("LBigToe".to_string(), 5);
        vertex_ids.insert("LSmallToe".to_string(), 6);
        vertex_ids.insert("LHeel".to_string(), 7);
        vertex_ids.insert("RBigToe".to_string(), 8);
        vertex_ids.insert("RSmallToe".to_string(), 9);
        vertex_ids.insert("RHeel".to_string(), 10);
        // Hand keypoints
        for hand in &["l", "r"] {
            for finger in &["thumb", "index", "middle", "ring", "pinky"] {
                vertex_ids.insert(format!("{}{}", hand, finger), vertex_ids.len() as i64);
            }
        }
        vertex_ids
    }

    #[test]
    fn test_vertex_joint_selector_all_features() {
        let device = Device::Cpu;
        let vs = nn::VarStore::new(device);
        let vertex_ids = setup_vertex_ids();

        let selector = VertexJointSelector::new(&vs.root(), &vertex_ids, true, true);

        let vertices = Tensor::rand(&[1, 100, 3], (Kind::Float, device));
        let joints = Tensor::rand(&[1, 10, 3], (Kind::Float, device));

        let result = selector.forward_mt((vertices, joints), false);

        // 10 (original) + 5 (face) + 6 (feet) + 10 (hands) = 31
        assert_eq!(result.size(), &[1, 31, 3]);
        assert_eq!(result.kind(), Kind::Float);
    }

    #[test]
    fn test_vertex_joint_selector_no_hands_no_feet() {
        let device = Device::Cpu;
        let vs = nn::VarStore::new(device);
        let vertex_ids = setup_vertex_ids();

        let selector = VertexJointSelector::new(&vs.root(), &vertex_ids, false, false);

        let vertices = Tensor::rand(&[1, 100, 3], (Kind::Float, device));
        let joints = Tensor::rand(&[1, 10, 3], (Kind::Float, device));

        let result = selector.forward_mt((vertices, joints), false);

        // 10 (original) + 5 (face) = 15
        assert_eq!(result.size(), &[1, 15, 3]);
        assert_eq!(result.kind(), Kind::Float);
    }

    #[test]
    fn test_vertex_joint_selector_hands_only() {
        let device = Device::Cpu;
        let vs = nn::VarStore::new(device);
        let vertex_ids = setup_vertex_ids();

        let selector = VertexJointSelector::new(&vs.root(), &vertex_ids, true, false);

        let vertices = Tensor::rand(&[1, 100, 3], (Kind::Float, device));
        let joints = Tensor::rand(&[1, 10, 3], (Kind::Float, device));

        let result = selector.forward_mt((vertices, joints), false);

        // 10 (original) + 5 (face) + 10 (hands) = 25
        assert_eq!(result.size(), &[1, 25, 3]);
        assert_eq!(result.kind(), Kind::Float);
    }

    #[test]
    fn test_vertex_joint_selector_feet_only() {
        let device = Device::Cpu;
        let vs = nn::VarStore::new(device);
        let vertex_ids = setup_vertex_ids();

        let selector = VertexJointSelector::new(&vs.root(), &vertex_ids, false, true);

        let vertices = Tensor::rand(&[1, 100, 3], (Kind::Float, device));
        let joints = Tensor::rand(&[1, 10, 3], (Kind::Float, device));

        let result = selector.forward_mt((vertices, joints), false);

        // 10 (original) + 5 (face) + 6 (feet) = 21
        assert_eq!(result.size(), &[1, 21, 3]);
        assert_eq!(result.kind(), Kind::Float);
    }

    #[test]
    fn test_vertex_joint_selector_extra_joints_idxs_properties() {
        let device = Device::Cpu;
        let vs = nn::VarStore::new(device);
        let vertex_ids = setup_vertex_ids();

        let selector = VertexJointSelector::new(&vs.root(), &vertex_ids, true, true);

        assert_eq!(selector.extra_joints_idxs.kind(), Kind::Int64);
        assert!(!selector.extra_joints_idxs.requires_grad());
        assert_eq!(selector.extra_joints_idxs.size()[0], 21); // 5 (face) + 6 (feet) + 10 (hands)
    }

    #[test]
    fn test_vertex_joint_selector_with_different_batch_sizes() {
        let device = Device::Cpu;
        let vs = nn::VarStore::new(device);
        let vertex_ids = setup_vertex_ids();

        let selector = VertexJointSelector::new(&vs.root(), &vertex_ids, true, true);

        // Test with batch size 1
        let vertices = Tensor::rand(&[1, 100, 3], (Kind::Float, device));
        let joints = Tensor::rand(&[1, 10, 3], (Kind::Float, device));
        let result = selector.forward_mt((vertices, joints), false);
        assert_eq!(result.size(), &[1, 31, 3]);

        // Test with batch size 5
        let vertices = Tensor::rand(&[5, 100, 3], (Kind::Float, device));
        let joints = Tensor::rand(&[5, 10, 3], (Kind::Float, device));
        let result = selector.forward_mt((vertices, joints), false);
        assert_eq!(result.size(), &[5, 31, 3]);
    }
}