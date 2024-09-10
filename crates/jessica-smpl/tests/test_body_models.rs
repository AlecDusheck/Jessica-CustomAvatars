#[cfg(test)]
mod tests {
    use tch::{nn, Device, Tensor};
    use jessica_utils::module::ModuleMT;
    use jessica_smpl_lib::body_models::SMPL;

    // Generates input tensors with the given batch size
    fn generate_inputs(batch_size: i64, betas_size: i64) -> (Tensor, Tensor, Tensor, Tensor) {
        let betas = Tensor::rand(&[batch_size, betas_size], tch::kind::FLOAT_CPU);
        let body_pose = Tensor::rand(&[batch_size, 23 * 3], tch::kind::FLOAT_CPU);
        let global_orient = Tensor::rand(&[batch_size, 3], tch::kind::FLOAT_CPU);
        let transl = Tensor::rand(&[batch_size, 3], tch::kind::FLOAT_CPU);
        (betas, body_pose, global_orient, transl)
    }

    #[test]
    fn test_smpl_model_creation() {
        let device = Device::Cpu;
        let vs1 = nn::VarStore::new(device);
        let vs2 = nn::VarStore::new(device);
        let batch_size = 1;

        // Create SMPL model with default parameters
        let smpl_default = SMPL::new(
            &vs1.root(),
            None,
            None,
            10,
            None,
            None,
            None,
            batch_size,
            None,
            "neutral".to_string(),
            device,
        );

        // Create SMPL model with custom parameters
        let smpl_custom = SMPL::new(
            &vs2.root(),
            None,
            Some(Tensor::rand(&[batch_size, 5], tch::kind::FLOAT_CPU)),
            5,
            Some(Tensor::rand(&[batch_size, 3], tch::kind::FLOAT_CPU)),
            Some(Tensor::rand(&[batch_size, 23 * 3], tch::kind::FLOAT_CPU)),
            Some(Tensor::rand(&[batch_size, 3], tch::kind::FLOAT_CPU)),
            batch_size,
            None,
            "male".to_string(),
            device,
        );

        // Check that the models were created successfully
        assert_eq!(smpl_default.num_betas, 10);
        assert_eq!(smpl_default.gender, "neutral");
        assert_eq!(smpl_custom.num_betas, 5);
        assert_eq!(smpl_custom.gender, "male");
    }

    #[test]
    fn test_smpl_model_loading() {
        let device = Device::Cpu;
        let vs = nn::VarStore::new(device);
        let model_path = "SMPL_NEUTRAL.safetensors";

        // Load the SMPL model from a file
        let smpl_model = SMPL::new(
            &vs.root(),
            Some(model_path),
            None,
            10,
            None,
            None,
            None,
            1,
            None,
            "neutral".to_string(),
            device,
        );

        // Check that the model was loaded successfully
        assert_eq!(smpl_model.model.v_template.size(), &[6890, 3]);
        assert_eq!(smpl_model.num_betas, 10);
    }

    #[test]
    fn test_smpl_forward_pass() {
        let device = Device::Cpu;
        let mut vs = nn::VarStore::new(device);
        let batch_size = 2;
        let num_betas = 10;  // Change this to 10 to match Python default

        // Load the SMPL model
        let smpl_model = SMPL::new(
            &vs.root(),
            Some("SMPL_NEUTRAL.safetensors"),
            None,
            num_betas,
            None,
            None,
            None,
            batch_size,
            None,
            "neutral".to_string(),
            device,
        );

        println!("SMPL model loaded. num_betas: {}", smpl_model.num_betas);

        // Generate input tensors
        let (betas, body_pose, global_orient, transl) = generate_inputs(batch_size, num_betas);

        // Run the forward pass
        let output = smpl_model.forward_mt((betas, body_pose, global_orient, transl), false);

        // Check the output shapes
        assert_eq!(output.0.size(), &[batch_size, 6890, 3]); // vertices
        assert_eq!(output.1.size(), &[batch_size, 3]); // global_orient
        assert_eq!(output.2.size(), &[batch_size, 23 * 3]); // body_pose
        assert_eq!(output.3.size(), &[batch_size, 45, 3]); // joints (after selector)
        assert_eq!(output.4.size(), &[batch_size, num_betas]); // betas
        assert_eq!(output.5.size(), &[batch_size, 24 * 3]); // full_pose
        assert_eq!(output.6.size(), &[batch_size, 24, 4, 4]); // A
        assert_eq!(output.7.size(), &[batch_size, 6890, 4, 4]); // T
        assert_eq!(output.8.size(), &[batch_size, 6890, 3]); // shape_offsets
        assert_eq!(output.9.size(), &[batch_size, 6890, 3]); // pose_offsets
    }

    #[test]
    fn test_smpl_forward_shape() {
        let device = Device::Cpu;
        let mut vs = nn::VarStore::new(device);
        let batch_size = 2;
        let num_betas = 10;

        // Load the SMPL model
        let smpl_model = SMPL::new(
            &vs.root(),
            Some("SMPL_NEUTRAL.safetensors"),
            None,
            num_betas,
            None,
            None,
            None,
            batch_size,
            None,
            "neutral".to_string(),
            device,
        );

        // Generate random betas
        let (betas, body_pose, global_orient, transl) = generate_inputs(batch_size, num_betas);

        // Run the forward shape pass
        let output = smpl_model.forward_shape(Some(betas));

        // Check the output shapes
        assert_eq!(output.0.size(), &[batch_size, 6890, 3]); // vertices
        assert_eq!(output.1.size(), &[batch_size, num_betas]); // betas
        assert_eq!(output.2.size(), &[batch_size, 6890, 3]); // v_shaped
    }
}