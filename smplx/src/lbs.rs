use tch::{IndexOp, Kind, Tensor};
use tensor_utils::tensor::rot_mat_to_euler;

/// Compute the faces, barycentric coordinates for the dynamic landmarks
///
/// To do so, we first compute the rotation of the neck around the y-axis
/// and then use a pre-computed look-up table to find the faces and the
/// barycentric coordinates that will be used.
///
/// Parameters
/// ----------
/// vertices: Tensor of shape BxVx3, dtype = torch.float32
///     The tensor of input vertices
/// pose: Tensor of shape Bx(Jx3), dtype = torch.float32
///     The current pose of the body model
/// dynamic_lmk_faces_idx: Tensor of shape L, dtype = torch.long
///     The look-up table from neck rotation to faces
/// dynamic_lmk_b_coords: Tensor of shape Lx3, dtype = torch.float32
///     The look-up table from neck rotation to barycentric coordinates
/// neck_kin_chain: Vec<i64>
///     A vector that contains the indices of the joints that form the
///     kinematic chain of the neck.
/// pose2rot: bool, optional, default = true
///     Flag on whether to convert the input pose tensor to rotation matrices.
///
/// Returns
/// -------
/// (Tensor, Tensor)
///     dyn_lmk_faces_idx: Tensor of shape BxL, dtype = torch.long
///         A tensor that contains the indices of the faces that
///         will be used to compute the current dynamic landmarks.
///     dyn_lmk_b_coords: Tensor of shape BxL, dtype = torch.float32
///         A tensor that contains the barycentric coordinates that
///         will be used to compute the current dynamic landmarks.
fn find_dynamic_lmk_idx_and_bcoords(
    vertices: &Tensor,
    pose: &Tensor,
    dynamic_lmk_faces_idx: &Tensor,
    dynamic_lmk_b_coords: &Tensor,
    neck_kin_chain: &[i64],
    pose2rot: bool,
) -> (Tensor, Tensor) {
    let dtype = vertices.kind();
    let batch_size = vertices.size()[0];
    
    let rot_mats = if pose2rot {
        let aa_pose = pose.reshape(&[batch_size, -1, 3]).i(&*neck_kin_chain);
        batch_rodrigues(&aa_pose.reshape(&[-1, 3])).reshape(&[batch_size, -1, 3, 3])
    } else {
        pose.reshape(&[batch_size, -1, 3, 3]).i(&*neck_kin_chain)
    };
    
    let mut rel_rot_mat = Tensor::eye(3, (dtype, vertices.device()))
        .unsqueeze(0)
        .repeat(&[batch_size, 1, 1]);
    
    for idx in 0..neck_kin_chain.len() {
        rel_rot_mat = rot_mats.i((.., idx as i64)).bmm(&rel_rot_mat);

    }
    
    // We use std::f64::consts::PI for numpy's np.pi equivalent.
    let y_rot_angle = (-rot_mat_to_euler(&rel_rot_mat) * 180.0 / std::f64::consts::PI)
        .clamp_max(39.0)
        .round()
        .to_kind(Kind::Int64);
    
    let neg_mask = y_rot_angle.lt(0).to_kind(Kind::Int64);
    let mask = y_rot_angle.lt(-39).to_kind(Kind::Int64);

    let neg_vals = &mask * 78 + (1 - &mask) * (39 - &y_rot_angle);
    let y_rot_angle = &neg_mask * &neg_vals + (1 - &neg_mask) * &y_rot_angle;
    
    let dyn_lmk_faces_idx = dynamic_lmk_faces_idx.i(&y_rot_angle);
    let dyn_lmk_b_coords = dynamic_lmk_b_coords.i(&y_rot_angle);

    (dyn_lmk_faces_idx, dyn_lmk_b_coords)
}

/// Calculates landmarks by barycentric interpolation
///
/// Parameters
/// ----------
/// vertices: Tensor of shape BxVx3, dtype = torch.float32
///     The tensor of input vertices
/// faces: Tensor of shape Fx3, dtype = torch.long
///     The faces of the mesh
/// lmk_faces_idx: Tensor of shape L, dtype = torch.long
///     The tensor with the indices of the faces used to calculate the
///     landmarks.
/// lmk_bary_coords: Tensor of shape Lx3, dtype = torch.float32
///     The tensor of barycentric coordinates that are used to interpolate
///     the landmarks
///
/// Returns
/// -------
/// Tensor of shape BxLx3, dtype = torch.float32
///     The coordinates of the landmarks for each mesh in the batch
fn vertices2landmarks(
    vertices: &Tensor,
    faces: &Tensor,
    lmk_faces_idx: &Tensor,
    lmk_bary_coords: &Tensor,
) -> Tensor {
    let batch_size = vertices.size()[0];
    let num_verts = vertices.size()[1];
    let device = vertices.device();
    
    // index_select() and view() are equivalent in tch.
    let lmk_faces = faces.i(&lmk_faces_idx.view([-1])).view([batch_size, -1, 3]);
    
    // arange() and view() are equivalent in tch. We use & for tensor addition.
    let lmk_faces = &lmk_faces
        + &Tensor::arange(batch_size, (Kind::Int64, device)).view([-1, 1, 1])
        * num_verts;

    // We use & to index the vertices tensor with lmk_faces.
    let lmk_vertices = vertices.view([-1, 3]).i(&lmk_faces).view([batch_size, -1, 3, 3]);
    
    let landmarks =
        Tensor::einsum("blfi,blf->bli", &[&lmk_vertices, &lmk_bary_coords], None::<i64>);

    landmarks
}

/// Performs Linear Blend Skinning with the given shape and pose parameters
///
/// Parameters
/// ----------
/// betas: Tensor of shape BxNB
///     The tensor of shape parameters
/// pose: Tensor of shape Bx(J + 1) * 3
///     The pose parameters in axis-angle format
/// v_template: Tensor of shape BxVx3
///     The template mesh that will be deformed
/// shapedirs: Tensor of shape 1xNB
///     The tensor of PCA shape displacements
/// posedirs: Tensor of shape Px(V * 3)
///     The pose PCA coefficients
/// j_regressor: Tensor of shape JxV
///     The regressor array that is used to calculate the joints from
///     the position of the vertices
/// parents: Tensor of shape J
///     The array that describes the kinematic tree for the model
/// lbs_weights: Tensor of shape N x V x (J + 1)
///     The linear blend skinning weights that represent how much the
///     rotation matrix of each part affects each vertex
/// pose2rot: bool, optional, default = true
///     Flag on whether to convert the input pose tensor to rotation
///     matrices. If False, then the pose tensor should already contain
///     rotation matrices and have a size of Bx(J + 1)x9
///
/// Returns
/// -------
/// (Tensor, Tensor)
///     verts: Tensor of shape BxVx3
///         The vertices of the mesh after applying the shape and pose
///         displacements.
///     joints: Tensor of shape BxJx3
///         The joints of the model
fn lbs(
    betas: &Tensor,
    pose: &Tensor,
    v_template: &Tensor,
    shapedirs: &Tensor,
    posedirs: &Tensor,
    j_regressor: &Tensor,
    parents: &Tensor,
    lbs_weights: &Tensor,
    pose2rot: bool,
) -> (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor) {
    let batch_size = betas.size()[0].max(pose.size()[0]);
    let device = betas.device();
    let dtype = betas.kind();

    // Add shape contribution
    let shape_offsets = blend_shapes(betas, shapedirs);
    let v_shaped = v_template + shape_offsets.shallow_clone();

    // Get the joints
    let j = vertices2joints(j_regressor, &v_shaped);

    // Add pose blend shapes
    let ident = Tensor::eye(3, (dtype, device));

    let (rot_mats, pose_offsets) = if pose2rot {
        let rot_mats = batch_rodrigues(&pose.view([-1, 3])).view([batch_size, -1, 3, 3]);
        let pose_feature = (&rot_mats.i((.., 1..)) - &ident).view([batch_size, -1]);
        (rot_mats, Tensor::matmul(&pose_feature, posedirs).view([batch_size, -1, 3]))
    } else {
        let pose_feature = &pose.i((.., 1..)).view([batch_size, -1, 3, 3]) - &ident;
        let rot_mats = pose.view([batch_size, -1, 3, 3]);
        (rot_mats, Tensor::matmul(&pose_feature.view([batch_size, -1]), posedirs).view([batch_size, -1, 3]))
    };

    let v_posed = &pose_offsets + &v_shaped;

    // Get the global joint location
    let (j_transformed, a) = batch_rigid_transform(&rot_mats, &j, parents, dtype);

    // Do skinning:
    let w = lbs_weights.unsqueeze(0).expand(&[batch_size, -1, -1], false);
    let num_joints = j_regressor.size()[0];
    let t = Tensor::matmul(&w, &a.view([batch_size, num_joints, 16]))
        .view([batch_size, -1, 4, 4]);
    
    let homogen_coord = Tensor::ones(&[batch_size, v_posed.size()[1], 1], (dtype, device));
    let v_posed_homo = Tensor::cat(&[&v_posed, &homogen_coord], 2);
    let v_homo = Tensor::matmul(&t, &v_posed_homo.unsqueeze(-1));

    let verts = v_homo.i((.., .., ..3, 0));

    (verts, j_transformed, a, t, shape_offsets, pose_offsets)
}

/// Calculates the 3D joint locations from the vertices
///
/// Parameters
/// ----------
/// j_regressor: Tensor of shape JxV
///     The regressor array that is used to calculate the joints from the
///     position of the vertices
/// vertices: Tensor of shape BxVx3
///     The tensor of mesh vertices
///
/// Returns
/// -------
/// Tensor of shape BxJx3
///     The location of the joints
fn vertices2joints(j_regressor: &Tensor, vertices: &Tensor) -> Tensor {
    Tensor::einsum("bik,ji->bjk", &[vertices, j_regressor], None::<i64>)
}

/// Calculates the per vertex displacement due to the blend shapes
///
/// Parameters
/// ----------
/// betas: Tensor of shape Bx(num_betas)
///     Blend shape coefficients
/// shape_disps: Tensor of shape Vx3x(num_betas)
///     Blend shapes
///
/// Returns
/// -------
/// Tensor of shape BxVx3
///     The per-vertex displacement due to shape deformation
fn blend_shapes(betas: &Tensor, shape_disps: &Tensor) -> Tensor {
    // Displacement[b, m, k] = sum_{l} betas[b, l] * shape_disps[m, k, l]
    // i.e. Multiply each shape displacement by its corresponding beta and
    // then sum them.
    Tensor::einsum("bl,mkl->bmk", &[betas, shape_disps], None::<i64>)
}

/// Calculates the rotation matrices for a batch of rotation vectors
///
/// Parameters
/// ----------
/// rot_vecs: Tensor of shape Nx3
///     Array of N axis-angle vectors
///
/// Returns
/// -------
/// Tensor of shape Nx3x3
///     The rotation matrices for the given axis-angle parameters
fn batch_rodrigues(rot_vecs: &Tensor) -> Tensor {
    let batch_size = rot_vecs.size()[0];
    let device = rot_vecs.device();
    let dtype = rot_vecs.kind();

    let angle = (&(rot_vecs + 1e-8)).norm();
    let rot_dir = rot_vecs / &angle;
    
    let cos = Tensor::unsqueeze(&angle.cos(), 1);
    let sin = Tensor::unsqueeze(&angle.sin(), 1);

    let split = rot_dir.split(1, 1);
    let rx = split.get(0).unwrap();
    let ry = split.get(1).unwrap();
    let rz = split.get(2).unwrap();
    
    let mut k = Tensor::zeros(&[batch_size, 3, 3], (dtype, device));

    let zeros = Tensor::zeros(&[batch_size, 1], (dtype, device));
    k = Tensor::cat(&[&zeros, &-rz, &ry, &rz, &zeros, &-rx, &-ry, &rx, &zeros], 1)
        .view([batch_size, 3, 3]);

    let ident = Tensor::eye(3, (dtype, device)).unsqueeze(0);
    let rot_mat = &ident + &(&sin * &k) + &((1 - &cos) * &Tensor::bmm(&k, &k));

    rot_mat
}

/// Creates a batch of transformation matrices
///
/// Parameters
/// ----------
/// r: Tensor of shape Bx3x3
///     Array of a batch of rotation matrices
/// t: Tensor of shape Bx3x1
///     Array of a batch of translation vectors
///
/// Returns
/// -------
/// Tensor of shape Bx4x4
///     Transformation matrix
fn transform_mat(r: &Tensor, t: &Tensor) -> Tensor {
    // cat() is equivalent in tch.
    // pad() is equivalent in tch, with `value` argument at the end.
    Tensor::cat(&[r.pad(&[0, 0, 0, 1], "constant", 0.0),
        t.pad(&[0, 0, 0, 1], "constant", 1.0)], 2)
}

/// Applies a batch of rigid transformations to the joints
///
/// Parameters
/// ----------
/// rot_mats: Tensor of shape BxNx3x3
///     Tensor of rotation matrices
/// joints: Tensor of shape BxNx3
///     Locations of joints
/// parents: Tensor of shape BxN
///     The kinematic tree of each object
/// dtype: Dtype, optional
///     The data type of the created tensors, the default is torch.float32
///
/// Returns
/// -------
/// (Tensor, Tensor)
///     posed_joints: Tensor of shape BxNx3
///         The locations of the joints after applying the pose rotations
///     rel_transforms: Tensor of shape BxNx4x4
///         The relative (with respect to the root joint) rigid transformations
///         for all the joints
fn batch_rigid_transform(
    rot_mats: &Tensor,
    joints: &Tensor,
    parents: &Tensor,
    dtype: Kind,
) -> (Tensor, Tensor) {
    let joints = joints.unsqueeze(-1);
    
    let mut rel_joints = joints.copy();
    let _ = rel_joints.i((.., 1..)).subtract_(&joints.i((.., &parents.i(1..))));
    
    let transforms_mat = transform_mat(
        &rot_mats.reshape([-1, 3, 3]),
        &rel_joints.reshape([-1, 3, 1])).reshape([-1, joints.size()[1], 4, 4]);

    let mut transform_chain = vec![transforms_mat.i((.., 0))];
    for i in 1..parents.size()[0] {
        // Subtract the joint location at the rest pose
        // No need for rotation, since it's identity when at rest
        let curr_res = transform_chain[parents.i(i).int64_value(&[]) as usize].matmul(&transforms_mat.i((.., i)));
        transform_chain.push(curr_res);
    }

    let transforms = Tensor::stack(&transform_chain, 1);

    // The last column of the transformations contains the posed joints
    let posed_joints = transforms.i((.., .., ..3, 3));
    let joints_homogen = joints.pad(&[0, 0, 0, 1], "constant", 0.0);
    let rel_transforms = &transforms - &transforms.matmul(&joints_homogen)
        .pad(&[3, 0, 0, 0, 0, 0, 0, 0], "constant", 0.0);

    (posed_joints, rel_transforms)
}

// Test for the `find_dynamic_lmk_idx_and_bcoords` function
#[test]
fn test_find_dynamic_lmk_idx_and_bcoords() {
    // Generate random input tensors with the correct shapes
    let vertices = Tensor::randn(&[2, 100, 3], (tch::Kind::Float, tch::Device::Cpu));
    let pose = Tensor::randn(&[2, 3 * 10], (tch::Kind::Float, tch::Device::Cpu));
    let dynamic_lmk_faces_idx = Tensor::randint(100, &[10], (tch::Kind::Int64, tch::Device::Cpu));
    let dynamic_lmk_b_coords = Tensor::rand(&[10, 3], (tch::Kind::Float, tch::Device::Cpu));
    let neck_kin_chain = vec![0, 1, 2];

    // Call the function with pose2rot set to true
    let (dyn_lmk_faces_idx, dyn_lmk_b_coords) = find_dynamic_lmk_idx_and_bcoords(
        &vertices,
        &pose,
        &dynamic_lmk_faces_idx,
        &dynamic_lmk_b_coords,
        &neck_kin_chain,
        true,
    );

    // Check that the output tensors have the expected shapes
    assert_eq!(dyn_lmk_faces_idx.size(), &[2, 10]);
    assert_eq!(dyn_lmk_b_coords.size(), &[2, 10, 3]);

    // Call the function with pose2rot set to false
    let (dyn_lmk_faces_idx, dyn_lmk_b_coords) = find_dynamic_lmk_idx_and_bcoords(
        &vertices,
        &pose.view([2, 10, 3, 3]),
        &dynamic_lmk_faces_idx,
        &dynamic_lmk_b_coords,
        &neck_kin_chain,
        false,
    );

    // Check that the output tensors have the expected shapes
    assert_eq!(dyn_lmk_faces_idx.size(), &[2, 10]);
    assert_eq!(dyn_lmk_b_coords.size(), &[2, 10, 3]);
}

// Test for the `vertices2landmarks` function
#[test]
fn test_vertices2landmarks() {
    // Generate random input tensors with the correct shapes
    let vertices = Tensor::randn(&[2, 100, 3], (tch::Kind::Float, tch::Device::Cpu));
    let faces = Tensor::randint(100, &[50, 3], (tch::Kind::Int64, tch::Device::Cpu));
    let lmk_faces_idx = Tensor::randint(50, &[10], (tch::Kind::Int64, tch::Device::Cpu));
    let lmk_bary_coords = Tensor::rand(&[10, 3], (tch::Kind::Float, tch::Device::Cpu));

    // Call the function
    let landmarks = vertices2landmarks(
        &vertices,
        &faces,
        &lmk_faces_idx,
        &lmk_bary_coords,
    );

    // Check that the output tensor has the expected shape
    assert_eq!(landmarks.size(), &[2, 10, 3]);
}

// Test for the `lbs` function
#[test]
fn test_lbs() {
    // Generate random input tensors with the correct shapes
    let betas = Tensor::randn(&[2, 10], (tch::Kind::Float, tch::Device::Cpu));
    let pose = Tensor::randn(&[2, 3 * 10], (tch::Kind::Float, tch::Device::Cpu));
    let v_template = Tensor::randn(&[2, 100, 3], (tch::Kind::Float, tch::Device::Cpu));
    let shapedirs = Tensor::randn(&[1, 10], (tch::Kind::Float, tch::Device::Cpu));
    let posedirs = Tensor::randn(&[10, 100 * 3], (tch::Kind::Float, tch::Device::Cpu));
    let j_regressor = Tensor::randn(&[10, 100], (tch::Kind::Float, tch::Device::Cpu));
    let parents = Tensor::randint(10, &[10], (tch::Kind::Int64, tch::Device::Cpu));
    let lbs_weights = Tensor::rand(&[100, 100, 11], (tch::Kind::Float, tch::Device::Cpu));

    // Call the function with pose2rot set to true
    let (verts, joints, a, t, shape_offsets, pose_offsets) = lbs(
        &betas,
        &pose,
        &v_template,
        &shapedirs,
        &posedirs,
        &j_regressor,
        &parents,
        &lbs_weights,
        true,
    );

    // Check that the output tensors have the expected shapes
    assert_eq!(verts.size(), &[2, 100, 3]);
    assert_eq!(joints.size(), &[2, 10, 3]);
    assert_eq!(a.size(), &[2, 10, 4, 4]);
    assert_eq!(t.size(), &[2, 100, 4, 4]);
    assert_eq!(shape_offsets.size(), &[2, 100, 3]);
    assert_eq!(pose_offsets.size(), &[2, 100, 3]);

    // Call the function with pose2rot set to false
    let (verts, joints, a, t, shape_offsets, pose_offsets) = lbs(
        &betas,
        &pose.view([2, 10, 3, 3]),
        &v_template,
        &shapedirs,
        &posedirs,
        &j_regressor,
        &parents,
        &lbs_weights,
        false,
    );

    // Check that the output tensors have the expected shapes
    assert_eq!(verts.size(), &[2, 100, 3]);
    assert_eq!(joints.size(), &[2, 10, 3]);
    assert_eq!(a.size(), &[2, 10, 4, 4]);
    assert_eq!(t.size(), &[2, 100, 4, 4]);
    assert_eq!(shape_offsets.size(), &[2, 100, 3]);
    assert_eq!(pose_offsets.size(), &[2, 100, 3]);
}

// Test for the `vertices2joints` function
#[test]
fn test_vertices2joints() {
    // Generate random input tensors with the correct shapes
    let j_regressor = Tensor::randn(&[10, 100], (tch::Kind::Float, tch::Device::Cpu));
    let vertices = Tensor::randn(&[2, 100, 3], (tch::Kind::Float, tch::Device::Cpu));

    // Call the function
    let joints = vertices2joints(&j_regressor, &vertices);

    // Check that the output tensor has the expected shape
    assert_eq!(joints.size(), &[2, 10, 3]);
}

// Test for the `blend_shapes` function
#[test]
fn test_blend_shapes() {
    // Generate random input tensors with the correct shapes
    let betas = Tensor::randn(&[2, 10], (tch::Kind::Float, tch::Device::Cpu));
    let shape_disps = Tensor::randn(&[100, 3, 10], (tch::Kind::Float, tch::Device::Cpu));

    // Call the function
    let blend_shape = blend_shapes(&betas, &shape_disps);

    // Check that the output tensor has the expected shape
    assert_eq!(blend_shape.size(), &[2, 100, 3]);
}

// Test for the `batch_rodrigues` function
#[test]
fn test_batch_rodrigues() {
    // Generate a random input tensor with the correct shape
    let rot_vecs = Tensor::randn(&[2, 3], (tch::Kind::Float, tch::Device::Cpu));

    // Call the function
    let rot_mat = batch_rodrigues(&rot_vecs);

    // Check that the output tensor has the expected shape
    assert_eq!(rot_mat.size(), &[2, 3, 3]);
}

// Test for the `transform_mat` function
#[test]
fn test_transform_mat() {
    // Generate random input tensors with the correct shapes
    let r = Tensor::randn(&[2, 3, 3], (tch::Kind::Float, tch::Device::Cpu));
    let t = Tensor::randn(&[2, 3, 1], (tch::Kind::Float, tch::Device::Cpu));

    // Call the function
    let t = transform_mat(&r, &t);

    // Check that the output tensor has the expected shape
    assert_eq!(t.size(), &[2, 4, 4]);
}

// Test for the `batch_rigid_transform` function
#[test]
fn test_batch_rigid_transform() {
    // Generate random input tensors with the correct shapes
    let rot_mats = Tensor::randn(&[2, 10, 3, 3], (tch::Kind::Float, tch::Device::Cpu));
    let joints = Tensor::randn(&[2, 10, 3], (tch::Kind::Float, tch::Device::Cpu));
    let parents = Tensor::randint(10, &[10], (tch::Kind::Int64, tch::Device::Cpu));

    // Call the function
    let (posed_joints, rel_transforms) = batch_rigid_transform(
        &rot_mats,
        &joints,
        &parents,
        Kind::Float,
    );

    // Check that the output tensors have the expected shapes
    assert_eq!(posed_joints.size(), &[2, 10, 3]);
    assert_eq!(rel_transforms.size(), &[2, 10, 4, 4]);
}