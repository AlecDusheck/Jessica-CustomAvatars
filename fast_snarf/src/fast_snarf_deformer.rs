use std::fmt::{Debug};
use knn_points::knn::knn_idx_cuda;
use tch::{Device, IndexOp, Kind, nn, Tensor};
use tensor_utils::module::ModuleMT;
use crate::filter::filter;
use crate::fuse::fuse;
use crate::precompute::precompute;

#[derive(Debug)]
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
    voxel_d: Tensor,
    voxel_j: Tensor,
}

impl ModuleMT<(Tensor, Tensor), (Tensor, Tensor, Tensor)> for ForwardDeformer  {
    /// Performs the forward pass of the deformer.
    ///
    /// # Arguments
    ///
    /// * `xd` - The deformed points in batch. Shape: [B, N, D]
    /// * `cond` - The conditional input (unused in this implementation).
    /// * `tfs` - The bone transformation matrices. Shape: [B, J, D+1, D+1]
    /// * `train` - A boolean flag indicating whether the function is in train mode.
    ///
    /// # Returns
    ///
    /// A tuple containing:
    /// - `xc` - The canonical correspondences. Shape: [B, N, I, D]
    /// - ...`others` - Other tensors containing additional output tensors (not used in this implementation afaik).
    ///
    /// # Note
    ///
    /// This method assumes version 1 of the deformer.
    fn forward_mt(&self, xs: (Tensor, Tensor), train: bool) -> (Tensor, Tensor, Tensor) {
        let (xd, tfs) = xs;

        let (xc_opt, valid_ids, j_inv) = self.search(&xd, &tfs);

        if !train {
            return (xc_opt, valid_ids, j_inv);
        }

        let xc_opt = xc_opt.detach();

        // Set the values of xc_opt to 0 where valid_ids is false
        let mut xc_opt = xc_opt.detach();
        xc_opt.copy_(&tch::no_grad(|| xc_opt.masked_fill(&valid_ids.logical_not(), 0.)));

        // Extract the shape dimensions from xc_opt
        let (n_batch, n_point, n_init, n_dim) = xc_opt.size4().unwrap();

        let mask = valid_ids.shallow_clone();
        // Perform forward skinning
        let xd_opt = self.forward_skinning(&xc_opt, &tfs, Some(&mask));

        // Get the inverse Jacobian for valid correspondences
        let grad_inv = j_inv.masked_select(&mask);

        // Calculate the correction term
        let correction = &xd_opt - &xd_opt.detach();

        // Apply the inverse Jacobian to the correction term
        let correction = bmv(&-grad_inv, &correction.unsqueeze(-1)).squeeze_dim(-1);

        let mut xc = xc_opt;
        // Apply the correction to valid correspondences
        let _ = xc.masked_scatter_(&mask, &(xc.masked_select(&mask) + correction));

        // Reshape xc to the original shape
        xc = xc.view([n_batch, n_point, n_init, n_dim]);

        (xc, valid_ids, j_inv)
    }
}

impl ForwardDeformer {
    pub fn new(p: &nn::Path, device: Device) -> Self {
        let soft_blend = 20;
        let init_bones = vec![0, 1, 2, 4, 5, 10, 11, 12, 15, 16, 17, 18, 19];
        let init_bones_cuda = p.var_copy("init_bones_cuda", &Tensor::from_slice(&init_bones).to_device(device).to_kind(Kind::Int64));
        let global_scale = 1.2;

        ForwardDeformer {
            soft_blend,
            init_bones,
            init_bones_cuda,
            global_scale,
            resolution: 0,
            ratio: 0.0,
            bbox: p.zeros("bbox", &[1]),
            scale: p.zeros("scale", &[1]),
            offset: p.zeros("offset", &[1]),
            offset_kernel: p.zeros("offset_kernel", &[1]),
            scale_kernel: p.zeros("scale_kernel", &[1]),
            lbs_voxel_final: p.zeros("lbs_voxel_final", &[1]),
            grid_denorm: p.zeros("grid_denorm", &[1]),
            voxel_d: p.zeros("voxel_d", &[1]),
            voxel_j: p.zeros("voxel_j", &[1]),
        }
    }

    /// Precomputes the voxel grid and Jacobian matrix using CUDA.
    ///
    /// # Arguments
    ///
    /// * `tfs` - The bone transformation matrices. Shape: [B, J, D+1, D+1]
    pub fn deformer_precompute(&mut self, tfs: &Tensor) {
        // b, c, d, h, w = tfs.shape[0], 3, self.resolution // 4, self.resolution, self.resolution
        // Extract the dimensions from the `tfs` tensor and the `resolution` field
        let b = tfs.size()[0];
        let c = 3;
        let d = self.resolution / 4;
        let h = self.resolution;
        let w = self.resolution;

        // Create a new tensor for `voxel_d` with the specified shape and device
        self.voxel_d = Tensor::zeros(&[b, c, d, h, w], (tch::Kind::Float, tfs.device()));

        // Create a new tensor for `voxel_j` with the specified shape and device
        self.voxel_j = Tensor::zeros(&[b, 12, d, h, w], (tch::Kind::Float, tfs.device()));

        // Call the `precompute_cuda` function to perform the actual computation
        // The `voxel_d` and `voxel_j` fields are updated in-place by the `precompute_cuda` function
        precompute(
            &self.lbs_voxel_final,
            tfs,
            &mut self.voxel_d,
            &mut self.voxel_j,
            &self.offset_kernel,
            &self.scale_kernel,
        );
    }

    /// Searches for correspondences between deformed points and canonical points.
    ///
    /// # Arguments
    ///
    /// * `xd` - The deformed points in batch. Shape: [B, N, D]
    /// * `cond` - The conditional input (unused in this implementation).
    /// * `tfs` - The bone transformation matrices. Shape: [B, J, D+1, D+1]
    /// * `eval_mode` - A boolean flag indicating whether the function is in evaluation mode.
    ///
    /// # Returns
    ///
    /// A tuple containing:
    /// - `xc_opt` - The optimal canonical correspondences. Shape: [B, N, I, D]
    /// - `valid_ids` - The validity mask for each correspondence. Shape: [B, N, I]
    /// - `j_inv` - The inverse of the Jacobian matrix for each correspondence. Shape: [B, N, I, D, D]
    pub fn search(
        &self,
        xd: &Tensor,
        tfs: &Tensor,
    ) -> (Tensor, Tensor, Tensor) {
        // In Rust, we don't need to explicitly use `torch.no_grad()` as it is used for gradient computation,
        // which is not relevant in this context.
        let result = self.broyden_cuda(xd, &self.voxel_d, &self.voxel_j, tfs, 1e-5, 1e-1);

        result
    }

    /// Performs Broyden's method using CUDA to find the optimal canonical correspondences.
    ///
    /// # Arguments
    ///
    /// * `xd_tgt` - The target deformed points in batch. Shape: [B, N, D]
    /// * `voxel` - The voxel grid. Shape: [B, 3, D, H, W]
    /// * `voxel_j_inv` - The inverse of the Jacobian matrix for each voxel. Shape: [B, 9, D, H, W]
    /// * `tfs` - The bone transformation matrices. Shape: [B, J, D+1, D+1]
    /// * `cvg_thresh` - The convergence threshold.
    /// * `dvg_thresh` - The divergence threshold.
    ///
    /// # Returns
    ///
    /// A tuple containing:
    /// - `xc_init_in` - The optimal canonical correspondences. Shape: [B, N, I, D]
    /// - `is_valid` - The validity mask for each correspondence. Shape: [B, N, I]
    /// - `j_inv_init_in` - The inverse of the Jacobian matrix for each correspondence. Shape: [B, N, I, D, D]
    pub fn broyden_cuda(
        &self,
        xd_tgt: &Tensor,
        voxel: &Tensor,
        voxel_j_inv: &Tensor,
        tfs: &Tensor,
        cvg_thresh: f64,
        dvg_thresh: f64,
    ) -> (Tensor, Tensor, Tensor) {
        // Extract the batch size and number of points from xd_tgt
        let (b, n, _) = xd_tgt.size3().unwrap();

        // Get the number of initial bones from init_bones_cuda
        let n_init = self.init_bones_cuda.size()[0];

        // Create a tensor for initial canonical correspondences
        let mut xc_init_in = Tensor::zeros(&[b, n, n_init, 3], (Kind::Float, xd_tgt.device()));

        // Create a tensor for the inverse Jacobian matrices
        let mut j_inv_init_in = Tensor::zeros(&[b, n, n_init, 3, 3], (Kind::Float, xd_tgt.device()));

        // Create a boolean tensor for validity masks
        let mut is_valid = Tensor::zeros(&[b, n, n_init], (Kind::Bool, xd_tgt.device()));

        // Call the fuse function to perform the CUDA computation
        fuse(
            &mut xc_init_in,
            xd_tgt,
            voxel,
            voxel_j_inv,
            tfs,
            &self.init_bones_cuda,
            true,
            &mut j_inv_init_in,
            &mut is_valid,
            &self.offset_kernel,
            &self.scale_kernel,
            cvg_thresh as f32,
            dvg_thresh as f32,
        );

        // Apply filtering to obtain the validity mask
        let mask = filter(&xc_init_in.shallow_clone(), &is_valid.shallow_clone());

        // Return the optimal canonical correspondences, validity mask, and inverse Jacobian matrices as a tuple
        (xc_init_in, mask, j_inv_init_in)
    }

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
        let (b, d, h, w) = (1, resolution / 4, resolution, resolution);
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
            1, // Use 1 for "border" padding mode
            true,  // Align corners
        );

        let w = w.squeeze_dim(-1).squeeze_dim(-1).permute(&[0, 2, 1]);
        w.view([-1, 24])
    }
}

fn bmv(m: &Tensor, v: &Tensor) -> Tensor {
    let v_transposed = v.transpose(-1, -2);
    let v_expanded = v_transposed.expand(&[-1, 3, -1], true);
    (m * v_expanded).sum_dim_intlist(&[-1i64][..], true, Kind::Float)
}

fn skinning_mask(x: &Tensor, w: &Tensor, tfs: &Tensor) -> Tensor {
    let x_h = x.pad(&[0, 1], "constant", 1.0);
    let (p, _) = w.size2().unwrap();
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