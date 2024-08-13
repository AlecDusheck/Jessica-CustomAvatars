use knn_points::knn::knn_idx_cuda;
use tch::{Device, IndexOp, Kind, Scalar, Tensor};

pub struct ForwardDeformer {
    soft_blend: i64,
    init_bones: Vec<i64>,
    init_bones_cuda: Tensor,
    global_scale: f64,
    resolution: i64,
    ratio: f64,
    bbox: Tensor,
    scale: Tensor,
    offset: Tensor,
    offset_kernel: Tensor,
    scale_kernel: Tensor,
    lbs_voxel_final: Tensor,
    grid_denorm: Tensor,
}

impl ForwardDeformer {
    pub fn new(device: Device) -> Self {
        let soft_blend = 20;
        let init_bones = vec![0, 1, 2, 4, 5, 10, 11, 12, 15, 16, 17, 18, 19];
        let init_bones_cuda = Tensor::from_slice(&init_bones).to_device(device).to_kind(Kind::Int);
        let global_scale = 1.2;

        ForwardDeformer {
            soft_blend,
            init_bones,
            init_bones_cuda,
            global_scale,
            resolution: 0,
            ratio: 0.0,
            bbox: Tensor::zeros(&[1], (tch::Kind::Float, device)),
            scale: Tensor::zeros(&[1], (tch::Kind::Float, device)),
            offset: Tensor::zeros(&[1], (tch::Kind::Float, device)),
            offset_kernel: Tensor::zeros(&[1], (tch::Kind::Float, device)),
            scale_kernel: Tensor::zeros(&[1], (tch::Kind::Float, device)),
            lbs_voxel_final: Tensor::zeros(&[1], (tch::Kind::Float, device)),
            grid_denorm: Tensor::zeros(&[1], (tch::Kind::Float, device)),
        }
    }

    // pub fn forward(&self, xd: &Tensor, tfs: &Tensor, eval_mode: bool) -> (Tensor, HashMap<String, Tensor>) {
    //     // TODO: Implement forward method
    //     unimplemented!()
    // }
    //
    // pub fn precompute(&mut self, tfs: &Tensor) {
    //     // TODO: Implement precompute method
    //     unimplemented!()
    // }
    //
    // pub fn search(&self, xd: &Tensor, tfs: &Tensor, eval_mode: bool) -> (Tensor, HashMap<String, Tensor>) {
    //     // TODO: Implement search method
    //     unimplemented!()
    // }
    //
    // pub fn broyden_cuda(&self, xd_tgt: &Tensor, voxel: &Tensor, voxel_j_inv: &Tensor, tfs: &Tensor, cvg_thresh: f64, dvg_thresh: f64) -> HashMap<String, Tensor> {
    //     // TODO: Implement broyden_cuda method
    //     unimplemented!()
    // }

    pub fn forward_skinning(&self, xc: &Tensor, tfs: &Tensor, mask: Option<&Tensor>) -> Tensor {
        // Query weights using the canonical points and conditional input
        let weights = self.query_weights(xc);

        // Apply skinning using the masked canonical points, masked weights, and transformation matrices
        if let Some(mask) = mask {
            // The `index` method expects a slice of `Option<Tensor>` as the argument,
            // but we are passing a `&Tensor` directly, causing a type mismatch.
            // To fix this, we need to convert the `mask` tensor into a slice of `Option<Tensor>`.
            let mask_slice = &[Some(mask.shallow_clone())];
            skinning_mask(&xc.index(mask_slice), &weights.index(mask_slice), tfs)
        } else {
            skinning_mask(xc, &weights, tfs)
        }
    }

    pub fn switch_to_explicit(&mut self, resolution: i64, smpl_verts: &Tensor, smpl_weights: &Tensor) {
        self.resolution = resolution;
        let (b, c, d, h, w) = (1, 24, resolution / 4, resolution, resolution);
        self.ratio = h as f64 / d as f64;

        // Define ranges
        let device = smpl_verts.device();
        let x_range = Tensor::linspace(-1., 1., w, (Kind::Float, device))
            .view([1, 1, 1, w])
            .expand(&[1, d, h, w], true);
        let y_range = Tensor::linspace(-1., 1., h, (Kind::Float, device))
            .view([1, 1, h, 1])
            .expand(&[1, d, h, w], true);
        let z_range = Tensor::linspace(-1., 1., d, (Kind::Float, device))
            .view([1, d, 1, 1])
            .expand(&[1, d, h, w], true);

        let grid = Tensor::cat(&[x_range, y_range, z_range], 0)
            .reshape(&[b, 3, -1])
            .permute(&[0, 2, 1]);

        // Find the minimum and maximum values of smpl_verts along dimension 1 using `min` and `max`.
        // Concatenate the minimum and maximum values along dimension 0 using `cat` and move the tensor to the specified device.
        let gt_bbox = Tensor::cat(&[smpl_verts.min_dim(1, false).0, smpl_verts.max_dim(1, false).0], 0).to_device(device);

        // Calculate the scale by subtracting the first element of gt_bbox from the second element, finding the maximum value,
        // dividing by 2, and multiplying by the global_scale.
        let offset = (&gt_bbox.i(0) + &gt_bbox.i(1)).unsqueeze(0).unsqueeze(0) * 0.5; //

        // Calculate the scale by subtracting the first element of gt_bbox from the second element, finding the maximum value,
        // dividing by 2, and multiplying by the global_scale.
        let scale = (&gt_bbox.i(1) - &gt_bbox.i(0)).max() / 2. * self.global_scale;

        // Create a tensor filled with ones that has the same shape as the first element of offset using `ones_like`.
        // Multiply the tensor by the scale value and divide the third element of the first row by the ratio.
        let corner = Tensor::ones_like(&offset.i(0)) * scale.shallow_clone();
        let _ = corner.get(0).get(2).divide_scalar_(self.ratio);

        // Calculate the minimum and maximum vertices by subtracting/adding the corner values from/to the offset and reshaping the tensors.
        let min_vert = (&offset - &corner).reshape(&[1, 3]);
        let max_vert = (&offset + &corner).reshape(&[1, 3]);

        self.bbox = Tensor::cat(&[min_vert, max_vert], 0);
        self.scale = scale.unsqueeze(0).unsqueeze(0);
        self.offset = offset;
        self.offset_kernel = -&self.offset;

        let grid_denorm = self.denormalize(&grid);
        let weights = query_weights_smpl(&grid_denorm, smpl_verts, smpl_weights, resolution).detach().shallow_clone();

        self.lbs_voxel_final = weights.detach();
        self.grid_denorm = grid_denorm;


    }

    fn normalize(&self, x: &Tensor) -> Tensor {
        let mut x_normalized = x.shallow_clone();
        x_normalized -= &self.offset;
        x_normalized /= &self.scale;
        x_normalized.select(1, 2).copy_(&(&x_normalized.select(1, 2) * self.ratio));
        x_normalized
    }

    fn denormalize(&self, x: &Tensor) -> Tensor {
        let mut x_denormalized = x.shallow_clone();
        x_denormalized.select(1, 2).copy_(&(&x_denormalized.select(1, 2) / self.ratio));
        x_denormalized *= &self.scale;
        x_denormalized += &self.offset;
        x_denormalized
    }

    fn query_weights(&self, xc: &Tensor) -> Tensor {
        let shape = xc.size();
        let n = 1;
        let xc = xc.view([1, -1, 3]);

        // Expand the lbs_voxel_final tensor to match the desired shape
        let lbs_voxel_final_expanded = self.lbs_voxel_final.expand(&[n, -1, -1, -1, -1], true);

        // Normalize the input tensor and unsqueeze dimensions
        let xc_normalized = self.normalize(&xc).unsqueeze(-1).unsqueeze(-1);

        // Perform 3D grid sampling using the expanded lbs_voxel_final and normalized input tensor
        let w = lbs_voxel_final_expanded.grid_sampler_3d(
            &xc_normalized,
            0, // 0 for bilinear
            0,  // Use 0 for "zeros" padding mode
            true,  // Align corners
        );

        // Reshape and permute dimensions
        let w = w.squeeze_dim(-1).squeeze_dim(-2).permute(&[0, 2, 1]);

        // Reshape the tensor to match the desired output shape
        w.view([-1, shape[shape.len() - 2], 24])
    }
}

fn bmv(m: &Tensor, v: &Tensor) -> Tensor {
    let v_transposed = v.transpose(-1, -2);
    let v_expanded = v_transposed.expand(&[-1, 3, -1], true);
    (m * v_expanded).sum_dim_intlist(&[-1i64][..], true, Kind::Float)
}

fn skinning_mask(x: &Tensor, w: &Tensor, tfs: &Tensor) -> Tensor {
    let x_h = x.pad(&[0, 1], "constant", 1.0);
    let (p, n) = w.size2().unwrap();
    let w_tf = Tensor::einsum("pn,nij->pij", &[w, &tfs.select(0, 0)], None::<i64>);

    let x_h = x_h.view([p, 1, 4]).expand(&[p, 4, 4], false);
    let x_h = (&w_tf * &x_h).sum_dim_intlist(&[-1i64][..], false, Kind::Float);
    x_h.slice(1, 0, 3, 1)
}

fn query_weights_smpl(x: &Tensor, smpl_verts: &Tensor, smpl_weights: &Tensor, resolution: i64) -> Tensor {
    let b = x.size()[0];
    let n = x.size()[1];
    let m = smpl_verts.size()[1];

    let mut idxs = Tensor::zeros(&[b, n, 30], (Kind::Int64, x.device()));
    let mut dists = Tensor::zeros(&[b, n, 30], (Kind::Float, x.device()));

    let lengths1 = Tensor::from_slice(&[n]).repeat(&[b]);
    let lengths2 = Tensor::from_slice(&[m]).repeat(&[b]);

    knn_idx_cuda(
        x,
        smpl_verts,
        &lengths1,
        &lengths2,
        2,
        30,
        0,
        &mut idxs,
        &mut dists,
    );

    let weights = smpl_weights.select(0, 0).index_select(0, &idxs.view([-1])).view([b, n, 30, 24]);

    let dist = dists.sqrt().clamp(1e-4, 1.0);
    let ws = dist.reciprocal() / dist.reciprocal().sum_dim_intlist(&[-1i64][..], true, Kind::Float);

    let weights_interp = ws.unsqueeze(-1) * weights;
    let weights_sum = weights_interp.sum_dim_intlist(&[-2i64][..], false, Kind::Float);

    let (b, c, d, h, w) = (1, 24, resolution / 4, resolution, resolution);
    let mut weights_reshaped = weights_sum.permute(&[0, 2, 1]).view([b, c, d, h, w]);

    for _ in 0..30 {
        let mut mean = Tensor::zeros_like(&weights_reshaped.select(1, 0).select(1, 0).select(1, 0));

        mean += &weights_reshaped.slice(2, 2, -1, 1).slice(3, 1, -1, 1).slice(4, 1, -1, 1);
        mean += &weights_reshaped.slice(2, 0, -2, 1).slice(3, 1, -1, 1).slice(4, 1, -1, 1);
        mean += &weights_reshaped.slice(2, 1, -1, 1).slice(3, 2, -1, 1).slice(4, 1, -1, 1);
        mean += &weights_reshaped.slice(2, 1, -1, 1).slice(3, 0, -2, 1).slice(4, 1, -1, 1);
        mean += &weights_reshaped.slice(2, 1, -1, 1).slice(3, 1, -1, 1).slice(4, 2, -1, 1);
        mean += &weights_reshaped.slice(2, 1, -1, 1).slice(3, 1, -1, 1).slice(4, 0, -2, 1);

        mean /= 6.0;

        let weights_slice = weights_reshaped.slice(2, 1, -1, 1).slice(3, 1, -1, 1).slice(4, 1, -1, 1);
        let updated_weights = &(&(weights_slice - &mean) * 0.7) + &mean;
        weights_reshaped
            .slice(2, 1, -1, 1)
            .slice(3, 1, -1, 1)
            .slice(4, 1, -1, 1)
            .copy_(&updated_weights);

        let sums = weights_reshaped.sum_dim_intlist(&[1i64][..], true, Kind::Float);
        weights_reshaped /= &sums;
    }

    weights_reshaped.detach()
}