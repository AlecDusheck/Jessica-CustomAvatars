use crate::module::ModuleMT;

pub mod module;
pub mod tensor;
pub mod data;
pub mod var_store;
pub mod mesh_ops;

/// This represents a model that can be trained and used for inference in Jessica, likely a `NeRFNGPNet`
pub type Model = dyn ModuleMT<tch::Tensor, (tch::Tensor, tch::Tensor)>;