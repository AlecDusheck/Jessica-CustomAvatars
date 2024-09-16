use tch::{Tensor, Kind};

fn index_vertices_by_faces(vertices_features: &Tensor, faces: &Tensor) -> Tensor {
    assert_eq!(vertices_features.dim(), 3, "vertices_features must have 3 dimensions of shape (batch_size, num_points, knum)");
    assert_eq!(faces.dim(), 2, "faces must have 2 dimensions of shape (num_faces, num_vertices)");
    
    let input = vertices_features.unsqueeze(2).expand(&[-1, -1, faces.size()[1], -1], true);
    let indices = faces.unsqueeze(0).unsqueeze(-1).expand(&[vertices_features.size()[0], -1, -1, vertices_features.size()[2]], true).to_kind(Kind::Int64);
    
    input.gather(1, &indices, false)
}

#[test]
fn test_index_vertices_by_faces() {
    let vertices_features = Tensor::from_slice(&[
        1.0, 2.0, 3.0, 4.0, 5.0, 6.0,
        7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
    ]).reshape(&[2, 3, 2]);
    
    let faces = Tensor::from_slice(&[0, 1, 2, 1, 2, 0]).reshape(&[2, 3]).to_kind(Kind::Int64);
    
    let expected = Tensor::from_slice(&[
        1.0, 2.0, 3.0, 4.0, 5.0, 6.0,
        3.0, 4.0, 5.0, 6.0, 1.0, 2.0,
        7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
        9.0, 10.0, 11.0, 12.0, 7.0, 8.0,
    ]).reshape(&[2, 2, 3, 2]);
    
    let result = index_vertices_by_faces(&vertices_features, &faces);
    
    assert_eq!(result.size(), &[2, 2, 3, 2]);
    assert!(result.allclose(&expected, 1e-5, 1e-8, false));
}