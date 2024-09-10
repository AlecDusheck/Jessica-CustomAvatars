use tch::{Device, Kind};
use tch::nn::VarStore;

pub trait ListTensors {
    fn list_tensors(&self);
    fn get_tensor_info(&self) -> Vec<TensorInfo>;
}

impl ListTensors for VarStore {
    fn list_tensors(&self) {
        let variables = self.variables_.lock().unwrap();
        for (name, tensor) in &variables.named_variables {
            println!("Tensor: {}", name);
            println!("  Shape: {:?}", tensor.size());
            println!("  Device: {:?}", tensor.device());
            println!("  Kind: {:?}", tensor.kind());
            println!("  Requires grad: {}", tensor.requires_grad());
            println!();
        }
    }

    fn get_tensor_info(&self) -> Vec<TensorInfo> {
        let variables = self.variables_.lock().unwrap();
        variables.named_variables.iter().map(|(name, tensor)| {
            TensorInfo {
                name: name.clone(),
                shape: tensor.size(),
                device: tensor.device(),
                kind: tensor.kind(),
                requires_grad: tensor.requires_grad(),
            }
        }).collect()
    }
}

#[derive(Debug)]
pub struct TensorInfo {
    pub name: String,
    pub shape: Vec<i64>,
    pub device: Device,
    pub kind: Kind,
    pub requires_grad: bool,
}