use tch::{Kind, Tensor};
use jessica_utils::tensor::{validate_tensor, validate_tensor_type};

pub struct KNN {
    pub dists: Tensor,
    pub idx: Tensor,
    pub knn: Option<Tensor>,
}

/// K-Nearest Neighbors on point clouds.
///
/// # Arguments
///
/// * `p1` - Tensor of shape (N, P1, D) giving a batch of N point clouds, each containing up to P1 points of dimension D.
/// * `p2` - Tensor of shape (N, P2, D) giving a batch of N point clouds, each containing up to P2 points of dimension D.
/// * `lengths1` - Optional LongTensor of shape (N,) of values in the range [0, P1], giving the length of each pointcloud in p1. Default: None (every cloud has length P1).
/// * `lengths2` - Optional LongTensor of shape (N,) of values in the range [0, P2], giving the length of each pointcloud in p2. Default: None (every cloud has length P2).
/// * `norm` - Integer indicating the norm of the distance. Supports only 1 for L1, 2 for L2. Default: 2.
/// * `K` - Integer giving the number of nearest neighbors to return. Default: 1.
/// * `version` - Which KNN implementation to use in the backend. If version=-1, the correct implementation is selected based on the shapes of the inputs. Default: -1.
/// * `return_nn` - If set to True, returns the K nearest neighbors in p2 for each point in p1. Default: false.
/// * `return_sorted` - Whether to return the nearest neighbors sorted in ascending order of distance. Default: true.
///
/// # Returns
///
/// A struct containing the following fields:
/// * `dists` - Tensor of shape (N, P1, K) giving the squared distances to the nearest neighbors.
/// * `idx` - LongTensor of shape (N, P1, K) giving the indices of the K nearest neighbors from points in p1 to points in p2.
/// * `knn` - Optional Tensor of shape (N, P1, K, D) giving the K nearest neighbors in p2 for each point in p1. Returned if `return_nn` is True.
pub fn knn_points(
    p1: &Tensor,
    p2: &Tensor,
    lengths1: Option<&Tensor>,
    lengths2: Option<&Tensor>,
    norm: i64,
    k: i64,
    version: i64,
    return_nn: bool,
    return_sorted: bool,
) -> KNN {
    assert_eq!(p1.size()[0], p2.size()[0], "pts1 and pts2 must have the same batch dimension.");
    assert_eq!(p1.size()[2], p2.size()[2], "pts1 and pts2 must have the same point dimension.");

    let p1 = p1.contiguous();
    let p2 = p2.contiguous();
    let p1_n = p1.size()[1];
    let p2_n = p2.size()[1];

    let lengths1 = match lengths1 {
        Some(lengths) => lengths.shallow_clone(),
        None => Tensor::full(&[p1.size()[0]], p1_n, (Kind::Int64, p1.device())),
    };

    let lengths2 = match lengths2 {
        Some(lengths) => lengths.shallow_clone(),
        None => Tensor::full(&[p1.size()[0]], p2_n, (Kind::Int64, p1.device())),
    };

   let (p1_dists, p1_idx) = _knn_points(
        &p1,
        &p2,
        &lengths1,
        &lengths2,
        k,
        version,
        norm,
        return_sorted,
    );

    // If return_nn is True, call the knn_gather function to collect the nearest neighbors.
    let p2_nn = if return_nn {
        Some(knn_gather(&p2, &p1_idx, Some(&lengths2)))
    } else {
        None
    };

    KNN {
        dists: p1_dists,
        idx: p1_idx,
        knn: p2_nn,
    }
}

fn _knn_points(
    p1: &Tensor,
    p2: &Tensor,
    lengths1: &Tensor,
    lengths2: &Tensor,
    k: i64,
    version: i64,
    norm: i64,
    return_sorted: bool,
) -> (Tensor, Tensor) {
    assert!((norm == 1) || (norm == 2), "Support for 1 or 2 norm.");

    let mut idx = Tensor::zeros(&[p1.size()[0], p1.size()[1], k], (Kind::Int64, p1.device()));
    let mut dists = Tensor::zeros(&[p1.size()[0], p1.size()[1], k], (Kind::Float, p1.device()));
    knn_idx_cuda(p1, p2, lengths1, lengths2, norm, k, version, &mut idx, &mut dists);

    // If K > 1 and return_sorted is True, sort the distances and indices.
    if k > 1 && return_sorted {
        if lengths2.min().int64_value(&[]) < k {
            let p1 = p1.size()[1];
            let mask = lengths2.unsqueeze(1).lt_tensor(&Tensor::arange(k, (Kind::Int64, dists.device())).unsqueeze(0));
            let mask = mask.unsqueeze(1).expand(&[-1, p1, -1], false);
            let _ = dists.masked_fill_(&mask, std::f64::INFINITY);
            let (sorted_dists, sort_idx) = dists.sort(2, true);
            dists.copy_(&sorted_dists);
            let _ = dists.masked_fill_(&mask, 0.0);
            idx = idx.gather(2, &sort_idx, false);
        } else {
            let (sorted_dists, sort_idx) = dists.sort(2, true);
            dists.copy_(&sorted_dists);
            idx = idx.gather(2, &sort_idx, false);
        }
    }

    (dists, idx)
}

/// A helper function for knn that allows indexing a tensor x with the indices `idx` returned by `knn_points`.
///
/// # Arguments
///
/// * `x` - Tensor of shape (N, M, U) containing U-dimensional features to be gathered.
/// * `idx` - LongTensor of shape (N, L, K) giving the indices returned by `knn_points`.
/// * `lengths` - Optional LongTensor of shape (N,) of values in the range [0, M], giving the length of each example in the batch in x. Default: None (every example has length M).
///
/// # Returns
///
/// * `x_out` - Tensor of shape (N, L, K, U) resulting from gathering the elements of x with idx.
fn knn_gather(
    x: &Tensor,
    idx: &Tensor,
    lengths: Option<&Tensor>,
) -> Tensor {
    let (n, m, u) = (x.size()[0], x.size()[1], x.size()[2]);
    let (_n, l, k) = (idx.size()[0], idx.size()[1], idx.size()[2]);

    assert_eq!(n, _n, "x and idx must have same batch dimension.");

    let lengths = match lengths {
        Some(lengths) => lengths.shallow_clone(),
        None => Tensor::full(&[n], m, (Kind::Int64, x.device())),
    };

    let idx_expanded = idx.unsqueeze(-1).expand(&[-1, -1, -1, u], false);

    let mut x_out = x.unsqueeze(2).expand(&[-1, -1, k, -1], false).gather(1, &idx_expanded, false);

    let needs_mask = lengths.min().int64_value(&[]) < k;
    if needs_mask {
        let mask = lengths.unsqueeze(1).lt_tensor(&Tensor::arange(k, (Kind::Int64, x.device())).unsqueeze(0));
        let mask = mask.unsqueeze(1).expand(&[-1, l, -1], false);
        let mask = mask.unsqueeze(-1).expand(&[-1, -1, -1, u], false);
        let _ = x_out.masked_fill_(&mask, 0.0);
    }

    x_out
}

pub fn knn_idx_cuda(
    p1: &Tensor,
    p2: &Tensor,
    lengths1: &Tensor,
    lengths2: &Tensor,
    norm: i64,
    k: i64,
    version: i64,
    idxs: &mut Tensor,
    dists: &mut Tensor,
) {
    let batch_size = p1.size()[0];
    let p1_size = p1.size()[1];
    let p2_size = p2.size()[1];
    let dim = p1.size()[2];

    // Perform dimension checks
    validate_tensor(p1, &[batch_size, p1_size, dim], "p1");
    validate_tensor(p2, &[batch_size, p2_size, dim], "p2");
    validate_tensor(lengths1, &[batch_size], "lengths1");
    validate_tensor(lengths2, &[batch_size], "lengths2");
    validate_tensor(idxs, &[batch_size, p1_size, k], "idxs");
    validate_tensor(dists, &[batch_size, p1_size, k], "dists");

    validate_tensor_type(p1, tch::Kind::Float, "p1");
    validate_tensor_type(p2, tch::Kind::Float, "p2");
    validate_tensor_type(lengths1, tch::Kind::Int64, "lengths1");
    validate_tensor_type(lengths2, tch::Kind::Int64, "lengths2");
    validate_tensor_type(idxs, tch::Kind::Int64, "idxs");
    validate_tensor_type(dists, tch::Kind::Float, "dists");

    // Check CUDA availability
    assert!(tch::Cuda::is_available(), "CUDA is not available");
    let device = p1.device();
    assert!(device.is_cuda(), "Input tensors must be on a CUDA device");

    // Call CUDA function
    unsafe {
        crate::cuda::c_knn_idx_cuda(
            p1.as_ptr(),
            p2.as_ptr(),
            lengths1.as_ptr(),
            lengths2.as_ptr(),
            norm as i32,
            k as i32,
            version as i32,
            idxs.as_mut_ptr(),
            dists.as_mut_ptr(),
        );
    }
}

pub fn knn_backward_cuda(
    p1: &Tensor,
    p2: &Tensor,
    lengths1: &Tensor,
    lengths2: &Tensor,
    idxs: &Tensor,
    norm: i64,
    grad_dists: &Tensor,
    grad_p1: &mut Tensor,
    grad_p2: &mut Tensor,
) {
    let batch_size = p1.size()[0];
    let p1_size = p1.size()[1];
    let p2_size = p2.size()[1];
    let dim = p1.size()[2];
    let k = idxs.size()[2];

    // Perform dimension checks
    validate_tensor(p1, &[batch_size, p1_size, dim], "p1");
    validate_tensor(p2, &[batch_size, p2_size, dim], "p2");
    validate_tensor(lengths1, &[batch_size], "lengths1");
    validate_tensor(lengths2, &[batch_size], "lengths2");
    validate_tensor(idxs, &[batch_size, p1_size, k], "idxs");
    validate_tensor(grad_dists, &[batch_size, p1_size, k], "grad_dists");
    validate_tensor(grad_p1, &[batch_size, p1_size, dim], "grad_p1");
    validate_tensor(grad_p2, &[batch_size, p2_size, dim], "grad_p2");

    validate_tensor_type(p1, tch::Kind::Float, "p1");
    validate_tensor_type(p2, tch::Kind::Float, "p2");
    validate_tensor_type(lengths1, tch::Kind::Int64, "lengths1");
    validate_tensor_type(lengths2, tch::Kind::Int64, "lengths2");
    validate_tensor_type(idxs, tch::Kind::Int64, "idxs");
    validate_tensor_type(grad_dists, tch::Kind::Float, "grad_dists");
    validate_tensor_type(grad_p1, tch::Kind::Float, "grad_p1");
    validate_tensor_type(grad_p2, tch::Kind::Float, "grad_p2");

    // Check CUDA availability
    assert!(tch::Cuda::is_available(), "CUDA is not available");
    let device = p1.device();
    assert!(device.is_cuda(), "Input tensors must be on a CUDA device");

    // Call CUDA function
    unsafe {
        crate::cuda::c_knn_backward_cuda(
            p1.as_ptr(),
            p2.as_ptr(),
            lengths1.as_ptr(),
            lengths2.as_ptr(),
            idxs.as_ptr(),
            norm as i32,
            grad_dists.as_ptr(),
            grad_p1.as_mut_ptr(),
            grad_p2.as_mut_ptr(),
        );
    }
}