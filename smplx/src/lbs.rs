use tch::{Device, IndexOp, Kind, Tensor};
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
    let device = vertices.device();
    let batch_size = vertices.size()[0];

    println!("Batch size: {}", batch_size);
    println!("Input pose shape: {:?}", pose.size());

    let rot_mats = if pose2rot {
        let aa_pose = pose.reshape(&[batch_size, -1, 3]).index_select(1, &Tensor::from_slice(neck_kin_chain).to_device(device));
        println!("aa_pose shape: {:?}", aa_pose.size());
        batch_rodrigues(&aa_pose.reshape(&[-1, 3])).reshape(&[batch_size, -1, 3, 3])
    } else {
        pose.index_select(1, &Tensor::from_slice(neck_kin_chain).to_device(device))
    };

    println!("rot_mats shape: {:?}", rot_mats.size());

    let mut rel_rot_mat: Tensor = Tensor::eye(3, (dtype, device))
        .unsqueeze(0)
        .repeat(&[batch_size, 1, 1]);

    for idx in 0..neck_kin_chain.len() {
        rel_rot_mat = rot_mats.i((.., idx as i64)).bmm(&rel_rot_mat);
    }

    println!("rel_rot_mat shape: {:?}", rel_rot_mat.size());

    let mut y_rot_angle: Tensor = (-rot_mat_to_euler(&rel_rot_mat) * 180.0 / std::f64::consts::PI)
        .clamp(-39.0, 39.0)
        .round()
        .to_kind(Kind::Int64);

    println!("y_rot_angle shape: {:?}, values: {:?}", y_rot_angle.size(), y_rot_angle);

    let neg_mask: Tensor = y_rot_angle.lt(0).to_kind(Kind::Int64);
    let mask: Tensor = y_rot_angle.lt(-39).to_kind(Kind::Int64);

    let neg_vals: Tensor = &mask * 78 + (1 - &mask) * (39 - &y_rot_angle);
    y_rot_angle = &neg_mask * &neg_vals + (1 - &neg_mask) * &y_rot_angle;

    // Ensure y_rot_angle is within the valid range
    let max_idx = dynamic_lmk_faces_idx.size()[0] - 1;
    y_rot_angle = y_rot_angle.clamp(0, max_idx);

    println!("Final y_rot_angle shape: {:?}, values: {:?}", y_rot_angle.size(), y_rot_angle);

    let dyn_lmk_faces_idx = dynamic_lmk_faces_idx.index_select(0, &y_rot_angle);
    let dyn_lmk_b_coords = dynamic_lmk_b_coords.index_select(0, &y_rot_angle);

    println!("dyn_lmk_faces_idx shape: {:?}", dyn_lmk_faces_idx.size());
    println!("dyn_lmk_b_coords shape: {:?}", dyn_lmk_b_coords.size());

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
pub fn vertices2landmarks(
    vertices: &Tensor,
    faces: &Tensor,
    lmk_faces_idx: &Tensor,
    lmk_bary_coords: &Tensor
) -> Tensor {
    let (batch_size, num_verts, _) = vertices.size3().unwrap();
    let device = vertices.device();

    let lmk_faces = faces.index_select(0, &lmk_faces_idx.view([-1]))
        .view([-1, 3]);

    let batch_range = Tensor::arange(batch_size, (Kind::Int64, device))
        .view([-1, 1, 1]) * num_verts;

    let lmk_faces_batch = &lmk_faces + &batch_range;

    let lmk_vertices = vertices.view([-1, 3])
        .index_select(0, &lmk_faces_batch.view([-1]))
        .view([batch_size, -1, 3, 3]);

    // Equivalent to einsum('blfi,blf->bli', [lmk_vertices, lmk_bary_coords])
    let landmarks = (&lmk_vertices * lmk_bary_coords.unsqueeze(-1)).sum_dim_intlist(&[-2i64][..], false, Kind::Float);

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
    println!("LBS function input shapes:");
    println!("betas: {:?}", betas.size());
    println!("pose: {:?}", pose.size());
    println!("v_template: {:?}", v_template.size());
    println!("shapedirs: {:?}", shapedirs.size());
    println!("posedirs: {:?}", posedirs.size());
    println!("j_regressor: {:?}", j_regressor.size());
    println!("parents: {:?}", parents.size());
    println!("lbs_weights: {:?}", lbs_weights.size());

    let batch_size = betas.size()[0].max(pose.size()[0]);
    let device = betas.device();
    let dtype = betas.kind();

    // Add shape contribution
    let shape_offsets = blend_shapes(betas, shapedirs);
    println!("shape_offsets: {:?}", shape_offsets.size());
    let v_shaped = v_template + &shape_offsets;
    println!("v_shaped: {:?}", v_shaped.size());

    // Get the joints
    let j = vertices2joints(j_regressor, &v_shaped);
    println!("j: {:?}", j.size());

    // Add pose blend shapes
    let ident = Tensor::eye(3, (dtype, device));

    let (rot_mats, pose_offsets) = if pose2rot {
        let rot_mats = batch_rodrigues(&pose.view([-1, 3])).view([batch_size, -1, 3, 3]);
        let pose_feature = (&rot_mats.i((.., 1..)) - &ident).view([batch_size, -1]);
        let pose_offsets = Tensor::matmul(&pose_feature, posedirs).view([batch_size, -1, 3]);
        (rot_mats, pose_offsets)
    } else {
        let pose_feature = &pose.i((.., 1..)).view([batch_size, -1, 3, 3]) - &ident;
        let rot_mats = pose.view([batch_size, -1, 3, 3]);
        let pose_offsets = Tensor::matmul(&pose_feature.view([batch_size, -1]), posedirs).view([batch_size, -1, 3]);
        (rot_mats, pose_offsets)
    };
    println!("pose_offsets: {:?}", pose_offsets.size());


    let v_posed = &pose_offsets + &v_shaped;
    println!("v_posed: {:?}", v_posed.size());

    // Get the global joint location
    let (j_transformed, a) = batch_rigid_transform(&rot_mats, &j, parents, dtype);
    println!("j_transformed: {:?}", j_transformed.size());
    println!("a: {:?}", a.size());

    // Do skinning:
    let w = lbs_weights.unsqueeze(0).expand(&[batch_size, -1, -1], false);
    println!("w: {:?}", w.size());
    let num_joints = j_regressor.size()[0];
    let t = Tensor::matmul(&w, &a.view([batch_size, num_joints, 16]))
        .view([batch_size, -1, 4, 4]);
    println!("t: {:?}", t.size());

    let homogen_coord = Tensor::ones(&[batch_size, v_posed.size()[1], 1], (dtype, device));
    let v_posed_homo = Tensor::cat(&[&v_posed, &homogen_coord], 2);
    let v_homo = Tensor::matmul(&t, &v_posed_homo.unsqueeze(-1));

    let verts = v_homo.i((.., .., ..3, 0));
    println!("verts: {:?}", verts.size());

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

    let cos = Tensor::unsqueeze(&angle.cos(), -1);
    let sin = Tensor::unsqueeze(&angle.sin(), -1);

    let rx = rot_dir.select(1, 0).unsqueeze(-1);
    let ry = rot_dir.select(1, 1).unsqueeze(-1);
    let rz = rot_dir.select(1, 2).unsqueeze(-1);

    let zeros = Tensor::zeros(&[batch_size, 1], (dtype, device));
    let k = Tensor::cat(&[
        &zeros, &-&rz, &ry.shallow_clone(),
        &rz.shallow_clone(), &zeros, &-&rx,
        &-&ry, &rx.shallow_clone(), &zeros
    ], 1).view([batch_size, 3, 3]);

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

    let rel_joints = joints.shallow_clone();
    let parents_1 = parents.slice(0, 1, parents.size()[0], 1);
    let joint_parents = joints.index_select(1, &parents_1);
    let _ = rel_joints.slice(1, 1, rel_joints.size()[1], 1).subtract_(&joint_parents);

    let transforms_mat = transform_mat(
        &rot_mats.reshape(&[-1, 3, 3]),
        &rel_joints.reshape(&[-1, 3, 1])).reshape(&[-1, joints.size()[1], 4, 4]);

    let mut transform_chain = vec![transforms_mat.i((.., 0))];
    for i in 1..parents.size()[0] {
        let parent_idx = parents.i(i).int64_value(&[]) as usize;
        let curr_res = transform_chain[parent_idx].matmul(&transforms_mat.i((.., i)));
        transform_chain.push(curr_res);
    }

    let transforms = Tensor::stack(&transform_chain, 1);

    let posed_joints = transforms.i((.., .., ..3, 3));

    let joints_homogen = joints.pad(&[0, 0, 0, 1], "constant", 1.0);

    let rel_transforms = transforms.subtract(&transforms.matmul(&joints_homogen)
        .pad(&[3, 0, 0, 0, 0, 0, 0, 0], "constant", 0.0));

    (posed_joints, rel_transforms)
}

#[test]
fn test_find_dynamic_lmk_idx_and_bcoords() {
    let device = tch::Device::Cpu;

    // Generate random input tensors with the correct shapes
    let vertices = Tensor::randn(&[2, 100, 3], (Kind::Float, device));
    let pose = Tensor::randn(&[2, 30], (Kind::Float, device));  // 30 = 3 * 10 (assuming 10 joints)
    let dynamic_lmk_faces_idx = Tensor::randint(100, &[78], (Kind::Int64, device));
    let dynamic_lmk_b_coords = Tensor::rand(&[78, 3], (Kind::Float, device));
    let neck_kin_chain = vec![0, 1, 2];

    println!("Input shapes:");
    println!("vertices: {:?}", vertices.size());
    println!("pose: {:?}", pose.size());
    println!("dynamic_lmk_faces_idx: {:?}", dynamic_lmk_faces_idx.size());
    println!("dynamic_lmk_b_coords: {:?}", dynamic_lmk_b_coords.size());

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
    println!("Output shapes (pose2rot=true):");
    println!("dyn_lmk_faces_idx: {:?}", dyn_lmk_faces_idx.size());
    println!("dyn_lmk_b_coords: {:?}", dyn_lmk_b_coords.size());

    assert_eq!(dyn_lmk_faces_idx.size(), &[2]);
    assert_eq!(dyn_lmk_b_coords.size(), &[2, 3]);

    // Call the function with pose2rot set to false
    // We need to create a new pose tensor with the correct shape for this case
    let pose_matrix = Tensor::randn(&[2, 10, 3, 3], (Kind::Float, device));
    let (dyn_lmk_faces_idx, dyn_lmk_b_coords) = find_dynamic_lmk_idx_and_bcoords(
        &vertices,
        &pose_matrix,
        &dynamic_lmk_faces_idx,
        &dynamic_lmk_b_coords,
        &neck_kin_chain,
        false,
    );

    // Check that the output tensors have the expected shapes
    println!("Output shapes (pose2rot=false):");
    println!("dyn_lmk_faces_idx: {:?}", dyn_lmk_faces_idx.size());
    println!("dyn_lmk_b_coords: {:?}", dyn_lmk_b_coords.size());

    assert_eq!(dyn_lmk_faces_idx.size(), &[2]);
    assert_eq!(dyn_lmk_b_coords.size(), &[2, 3]);
}

// Test for the `vertices2landmarks` function
#[test]
fn test_vertices2landmarks() {
    let device = Device::Cpu;
    let vertices = Tensor::randn(&[2, 10, 3], (Kind::Float, device));
    let faces = Tensor::randint(10, &[20, 3], (Kind::Int64, device));
    let lmk_faces_idx = Tensor::randint(20, &[5], (Kind::Int64, device));
    let lmk_bary_coords = Tensor::rand(&[2, 5, 3], (Kind::Float, device));

    let landmarks = vertices2landmarks(&vertices, &faces, &lmk_faces_idx, &lmk_bary_coords);

    assert_eq!(landmarks.size(), &[2, 5, 3]);
}

#[test]
fn test_lbs() {
    let device = Device::Cpu;
    let batch_size = 2;
    let num_vertices = 6890;
    let num_joints = 24;
    let num_betas = 10;
    let num_pose_params = 24 * 3;  // 24 joints, 3 rotation parameters each

    let betas = Tensor::randn(&[batch_size, num_betas], (Kind::Float, device));
    let pose = Tensor::randn(&[batch_size, num_pose_params], (Kind::Float, device));
    let v_template = Tensor::randn(&[num_vertices, 3], (Kind::Float, device));
    let shapedirs = Tensor::randn(&[num_vertices, 3, num_betas], (Kind::Float, device));
    let posedirs = Tensor::randn(&[num_pose_params - 3, num_vertices * 3], (Kind::Float, device));
    let j_regressor = Tensor::randn(&[num_joints, num_vertices], (Kind::Float, device));
    let parents = Tensor::randint(num_joints, &[num_joints], (Kind::Int64, device));
    let lbs_weights = Tensor::rand(&[num_vertices, num_joints], (Kind::Float, device));

    println!("Input tensor shapes:");
    println!("betas: {:?}", betas.size());
    println!("pose: {:?}", pose.size());
    println!("v_template: {:?}", v_template.size());
    println!("shapedirs: {:?}", shapedirs.size());
    println!("posedirs: {:?}", posedirs.size());
    println!("j_regressor: {:?}", j_regressor.size());
    println!("parents: {:?}", parents.size());
    println!("lbs_weights: {:?}", lbs_weights.size());

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
    println!("Output tensor shapes:");
    println!("verts: {:?}", verts.size());
    println!("joints: {:?}", joints.size());
    println!("a: {:?}", a.size());
    println!("t: {:?}", t.size());
    println!("shape_offsets: {:?}", shape_offsets.size());
    println!("pose_offsets: {:?}", pose_offsets.size());

    assert_eq!(verts.size(), &[2, 6890, 3]);
    assert_eq!(joints.size(), &[2, 24, 3]);
    assert_eq!(a.size(), &[2, 24, 4, 4]);
    assert_eq!(t.size(), &[2, 6890, 4, 4]);
    assert_eq!(shape_offsets.size(), &[2, 6890, 3]);
    assert_eq!(pose_offsets.size(), &[2, 6890, 3]);

    // Call the function with pose2rot set to false
    let pose_matrices = Tensor::randn(&[2, 24, 3, 3], (Kind::Float, Device::Cpu));
    let (verts, joints, a, t, shape_offsets, pose_offsets) = lbs(
        &betas,
        &pose_matrices,
        &v_template,
        &shapedirs,
        &posedirs,
        &j_regressor,
        &parents,
        &lbs_weights,
        false,
    );

    // Check that the output tensors have the expected shapes
    assert_eq!(verts.size(), &[2, 6890, 3]);
    assert_eq!(joints.size(), &[2, 24, 3]);
    assert_eq!(a.size(), &[2, 24, 4, 4]);
    assert_eq!(t.size(), &[2, 6890, 4, 4]);
    assert_eq!(shape_offsets.size(), &[2, 6890, 3]);
    assert_eq!(pose_offsets.size(), &[2, 6890, 3]);
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
    let batch_size = 2;
    let num_joints = 10;

    // Generate random input tensors with the correct shapes
    let rot_mats = Tensor::randn(&[batch_size, num_joints, 3, 3], (Kind::Float, Device::Cpu));
    let joints = Tensor::randn(&[batch_size, num_joints, 3], (Kind::Float, Device::Cpu));

    // Create a valid parent tensor
    let parents: Vec<i64> = vec![0, 0, 0, 0, 1, 2, 3, 4, 5, 6];
    let parents = Tensor::from_slice(&parents).to_kind(Kind::Int64).to_device(Device::Cpu);

    println!("Parents tensor in test:");
    parents.print();

    // Call the function
    let (posed_joints, rel_transforms) = batch_rigid_transform(
        &rot_mats,
        &joints,
        &parents,
        Kind::Float,
    );

    // Check that the output tensors have the expected shapes
    assert_eq!(posed_joints.size(), &[batch_size, num_joints, 3]);
    assert_eq!(rel_transforms.size(), &[batch_size, num_joints, 4, 4]);
}