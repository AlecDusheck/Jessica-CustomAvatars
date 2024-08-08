use fast_snarf::fuse::fuse;
use tch::{Device, Kind, Tensor};

#[test]
fn test_fuse() {
    // Skip the test if CUDA is not available
    if !tch::Cuda::is_available() {
        println!("CUDA not available, skipping test_fuse");
        return;
    }

    // Create sample input tensors
    use fast_snarf::fuse::fuse;
    use tch::{Device, Kind, Tensor};

    #[test]
    fn test_fuse() {
        // Skip the test if CUDA is not available
        if !tch::Cuda::is_available() {
            println!("CUDA not available, skipping test_fuse");
            return;
        }

        let device = Device::Cuda(0);

        // Create sample input tensors
        let batch_size = 1;
        let n_points = 200000;
        let n_bones = 9;

        let x = Tensor::rand(&[batch_size, n_points, n_bones, 3], (Kind::Float, device));
        let xd_tgt = Tensor::rand(&[batch_size, n_points, 3], (Kind::Float, device));
        let grid = Tensor::rand(&[batch_size, 3, 8, 32, 32], (Kind::Float, device));
        let grid_j_inv = Tensor::rand(&[batch_size, 9, 8, 32, 32], (Kind::Float, device));
        let tfs = Tensor::rand(&[batch_size, 24, 4, 4], (Kind::Float, device));
        let bone_ids = Tensor::from_slice(&[0i64, 1, 2, 3, 4, 5, 6, 7, 8]).to(device);
        let j_inv = Tensor::rand(&[batch_size, n_points, n_bones, 9], (Kind::Float, device));
        let is_valid = Tensor::ones(&[batch_size, n_points, n_bones], (Kind::Bool, device));
        let offset = Tensor::rand(&[batch_size, 1, 3], (Kind::Float, device));
        let scale = Tensor::rand(&[batch_size, 1, 3], (Kind::Float, device));

        let align_corners = true;
        let cvg_threshold = 1e-4;
        let dvg_threshold = 1e2;

        println!("Calling fuse function...");
        // Call the fuse function
        let (x_out, j_inv_out, is_valid_out) = fuse(
            x.shallow_clone(),
            &xd_tgt,
            &grid,
            &grid_j_inv,
            &tfs,
            &bone_ids,
            align_corners,
            j_inv.shallow_clone(),
            is_valid.shallow_clone(),
            &offset,
            &scale,
            cvg_threshold,
            dvg_threshold,
        );

        println!("Fuse function completed successfully.");

        // Check that the output tensors have the correct shapes
        assert_eq!(x_out.size(), x.size());
        assert_eq!(j_inv_out.size(), j_inv.size());
        assert_eq!(is_valid_out.size(), is_valid.size());

        // Check that the output tensors have valid values
        assert_eq!(x_out.isfinite().all().int64_value(&[]), 1);
        assert_eq!(j_inv_out.isfinite().all().int64_value(&[]), 1);
        assert!(is_valid_out.to_kind(Kind::Uint8).sum(Kind::Float).int64_value(&[]) > 0);
    }
}