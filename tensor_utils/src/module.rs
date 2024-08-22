use tch::{Tensor, Device, Kind};
use crate::data::IterN2;

pub trait TensorCollection: Send {
    fn to_device(&self, device: Device) -> Self;

    fn as_tensor_vec(&self) -> Vec<Tensor>;
    fn from_vec(tensors: Vec<Tensor>) -> Self;

    fn accuracy_for_logits(&self, targets: &Self) -> Tensor;
}

/// `TensorCollection` provides an efficient static sized container for `Tensor` objects.
impl TensorCollection for Tensor {
    fn to_device(&self, device: Device) -> Self {
        self.to_device(device)
    }

    fn as_tensor_vec(&self) -> Vec<Tensor> {
        // Changed the return type from &[Tensor] to Vec<Tensor>
        // to avoid returning a reference to temporary values
        vec![self.shallow_clone()]
    }

    fn from_vec(tensors: Vec<Tensor>) -> Self {
        assert_eq!(tensors.len(), 1, "Expected two tensors");
        tensors[0].shallow_clone()
    }

    fn accuracy_for_logits(&self, targets: &Self) -> Tensor {
        self.argmax(-1, false).eq_tensor(targets).to_kind(Kind::Float).mean(Kind::Float)
    }
}

impl TensorCollection for (Tensor, Tensor) {
    fn to_device(&self, device: Device) -> Self {
        (self.0.to_device(device), self.1.to_device(device))
    }

    fn as_tensor_vec(&self) -> Vec<Tensor> {
        // Changed the return type from &[Tensor] to Vec<Tensor>
        // to avoid returning a reference to temporary values
        vec![self.0.shallow_clone(), self.1.shallow_clone()]
    }

    fn from_vec(tensors: Vec<Tensor>) -> Self {
        assert_eq!(tensors.len(), 2, "Expected two tensors");
        (tensors[0].shallow_clone(), tensors[1].shallow_clone())
    }

    fn accuracy_for_logits(&self, targets: &Self) -> Tensor {
        let acc1 = self.0.argmax(-1, false).eq_tensor(&targets.0).to_kind(Kind::Float).mean(Kind::Float);
        let acc2 = self.1.argmax(-1, false).eq_tensor(&targets.1).to_kind(Kind::Float).mean(Kind::Float);
        (acc1 + acc2) / 2.0
    }
}

impl TensorCollection for (Tensor, Tensor, Tensor,
                           Tensor, Tensor, Tensor) {
    fn to_device(&self, device: Device) -> Self {
        (self.0.to_device(device), self.1.to_device(device), self.2.to_device(device),
         self.3.to_device(device), self.4.to_device(device), self.5.to_device(device))
    }

    fn as_tensor_vec(&self) -> Vec<Tensor> {
        vec![self.0.shallow_clone(), self.1.shallow_clone(), self.2.shallow_clone(),
             self.3.shallow_clone(), self.4.shallow_clone(), self.5.shallow_clone()]
    }

    fn from_vec(tensors: Vec<Tensor>) -> Self {
        assert_eq!(tensors.len(), 6, "Expected six tensors");
        (tensors[0].shallow_clone(), tensors[1].shallow_clone(), tensors[2].shallow_clone(),
         tensors[3].shallow_clone(), tensors[4].shallow_clone(), tensors[5].shallow_clone())
    }

    fn accuracy_for_logits(&self, targets: &Self) -> Tensor {
        let acc1 = self.0.argmax(-1, false).eq_tensor(&targets.0).to_kind(Kind::Float).mean(Kind::Float);
        let acc2 = self.1.argmax(-1, false).eq_tensor(&targets.1).to_kind(Kind::Float).mean(Kind::Float);
        let acc3 = self.2.argmax(-1, false).eq_tensor(&targets.2).to_kind(Kind::Float).mean(Kind::Float);
        let acc4 = self.3.argmax(-1, false).eq_tensor(&targets.3).to_kind(Kind::Float).mean(Kind::Float);
        let acc5 = self.4.argmax(-1, false).eq_tensor(&targets.4).to_kind(Kind::Float).mean(Kind::Float);
        let acc6 = self.5.argmax(-1, false).eq_tensor(&targets.5).to_kind(Kind::Float).mean(Kind::Float);
        (acc1 + acc2 + acc3 + acc4 + acc5 + acc6) / 6.0
    }
}

/// `ModuleMT` is a `tch-rs` `ModuleT` with support for `n` sized input / output tensors
pub trait ModuleMT<Input, Output>: Send
where
    Input: TensorCollection,
    Output: TensorCollection,
{
    fn forward_mt(&self, xs: Input, train: bool) -> Output;

    /// `batch_accuracy_for_logits_mt` supports n sized inputs / outputs as well!
    fn batch_accuracy_for_logits(
        &self,
        xs: Input,
        ys: Output,
        d: Device,
        batch_size: i64,
    ) -> f64 {
        let _no_grad = tch::no_grad_guard();
        let mut sum_accuracy = 0f64;
        let mut sample_count = 0f64;
        for (xs, ys) in IterN2::new(xs.as_tensor_vec().as_slice(), ys.as_tensor_vec().as_slice(), batch_size).return_smaller_last_batch() {
            let size = xs[0].size()[0] as f64;

            let xs_com = Input::from_vec(xs);
            let ys_com = Output::from_vec(ys);

            let acc = self.forward_mt(xs_com.to_device(d), false).accuracy_for_logits(&ys_com.to_device(d));
            sum_accuracy += f64::try_from(&acc).unwrap() * size;
            sample_count += size;
        }
        sum_accuracy / sample_count
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tch::{nn, Device, Tensor, Kind};

    struct TestModule {
        fc: nn::Linear,
    }

    impl TestModule {
        fn new(p: &nn::Path, in_dim: i64, out_dim: i64) -> Self {
            let fc = nn::linear(p, in_dim, out_dim, Default::default());
            Self { fc }
        }
    }

    impl ModuleMT<Tensor, Tensor> for TestModule {
        fn forward_mt(&self, xs: Tensor, _train: bool) -> Tensor {
            xs.apply(&self.fc)
        }
    }

    impl ModuleMT<(Tensor, Tensor), (Tensor, Tensor)> for TestModule {
        fn forward_mt(&self, xs: (Tensor, Tensor), _train: bool) -> (Tensor, Tensor) {
            (xs.0.apply(&self.fc), xs.1.apply(&self.fc))
        }
    }

    #[test]
    fn test_single_tensor_input_output() {
        let device = Device::Cpu;
        let vs = nn::VarStore::new(device);
        let module = TestModule::new(&vs.root(), 10, 5);
        let xs = Tensor::randn(&[64, 10], (Kind::Float, device));
        let ys = Tensor::randint(5, &[64], (Kind::Int64, device));

        let accuracy_mt = module.batch_accuracy_for_logits(
            xs.shallow_clone(),
            ys.shallow_clone(),
            device,
            16,
        );

        println!("Accuracy MT: {}", accuracy_mt);

        let accuracy_std = {
            let _no_grad = tch::no_grad_guard();
            let logits = xs.apply(&module.fc);
            logits.accuracy_for_logits(&ys)
        };

        println!("Accuracy Std: {}", accuracy_std);

        assert!((accuracy_mt - f64::try_from(&accuracy_std).unwrap()).abs() < 1e-6,
                "Accuracy mismatch: MT = {}, Std = {}", accuracy_mt, accuracy_std);
    }

    #[test]
    fn test_tuple_tensor_input_output() {
        let device = Device::Cpu;
        let vs = nn::VarStore::new(device);
        let module = TestModule::new(&vs.root(), 10, 5);
        let xs_0 = Tensor::randn(&[64, 10], (Kind::Float, device));
        let xs_1 = Tensor::randn(&[64, 10], (Kind::Float, device));
        let ys_0 = Tensor::randint(5, &[64], (Kind::Int64, device));
        let ys_1 = Tensor::randint(5, &[64], (Kind::Int64, device));

        let accuracy_mt = module.batch_accuracy_for_logits(
            (xs_0.shallow_clone(), xs_1.shallow_clone()),
            (ys_0.shallow_clone(), ys_1.shallow_clone()),
            device,
            16,
        );

        println!("Accuracy MT: {}", accuracy_mt);

        let accuracy_std = {
            let _no_grad = tch::no_grad_guard();
            let logits_0 = xs_0.apply(&module.fc);
            let logits_1 = xs_1.apply(&module.fc);
            let acc_0 = logits_0.accuracy_for_logits(&ys_0);
            let acc_1 = logits_1.accuracy_for_logits(&ys_1);
            (acc_0 + acc_1) / 2.0
        };

        println!("Accuracy Std: {}", accuracy_std);

        assert!((accuracy_mt - f64::try_from(&accuracy_std).unwrap()).abs() < 1e-6,
                "Accuracy mismatch: MT = {}, Std = {}", accuracy_mt, accuracy_std);
    }
}