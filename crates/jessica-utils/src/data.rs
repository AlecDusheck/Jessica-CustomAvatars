use tch::{Tensor, Device, Kind, TchError};

/// `IterN2` is an iterator over a pair of `Vec<Tensor>` (xs and ys).
/// It allows for batch selection and indexing across all the tensors in the vectors.
/// It is useful for models where the forward pass consumes multiple tensors and outputs multiple tensors.
#[derive(Debug)]
pub struct IterN2 {
    xs: Vec<Tensor>,
    ys: Vec<Tensor>,
    batch_index: i64,
    batch_size: i64,
    total_size: i64,
    device: Device,
    return_smaller_last_batch: bool,
}

impl IterN2 {
    /// Creates a new `IterN2` instance.
    ///
    /// This function takes as input two `Vec<Tensor>` (xs and ys) which must have the same length
    /// and the same first dimension size for all tensors within each vector.
    /// The returned iterator can be used to iterate over mini-batches of data of the specified size.
    /// An error is returned if `xs` and `ys` have different lengths or if any tensor within `xs` or `ys`
    /// has a different first dimension size.
    ///
    /// # Arguments
    ///
    /// * `xs` - A vector of tensors representing the features to be used by the model.
    /// * `ys` - A vector of tensors representing the targets that the model attempts to predict.
    /// * `batch_size` - The size of batches to be returned.
    pub fn f_new(xs: &[Tensor], ys: &[Tensor], batch_size: i64) -> Result<IterN2, TchError> {
        if xs.len() != ys.len() {
            return Err(TchError::Shape(format!(
                "xs and ys have different lengths: xs={}, ys={}",
                xs.len(),
                ys.len()
            )));
        }

        let total_size = xs[0].size()[0];
        for (x, y) in xs.iter().zip(ys.iter()) {
            if x.size()[0] != total_size || y.size()[0] != total_size {
                return Err(TchError::Shape("Inconsistent tensor sizes".to_string()));
            }
        }

        Ok(IterN2 {
            xs: xs.iter().map(|x| x.shallow_clone()).collect(),
            ys: ys.iter().map(|y| y.shallow_clone()).collect(),
            batch_index: 0,
            batch_size,
            total_size,
            device: Device::Cpu,
            return_smaller_last_batch: false,
        })
    }

    /// Creates a new `IterN2` instance.
    ///
    /// This function takes as input two `Vec<Tensor>` (xs and ys) which must have the same length
    /// and the same first dimension size for all tensors within each vector.
    /// The returned iterator can be used to iterate over mini-batches of data of the specified size.
    /// Panics if `xs` and `ys` have different lengths or if any tensor within `xs` or `ys`
    /// has a different first dimension size.
    ///
    /// # Arguments
    ///
    /// * `xs` - A vector of tensors representing the features to be used by the model.
    /// * `ys` - A vector of tensors representing the targets that the model attempts to predict.
    /// * `batch_size` - The size of batches to be returned.
    pub fn new(xs: &[Tensor], ys: &[Tensor], batch_size: i64) -> IterN2 {
        IterN2::f_new(xs, ys, batch_size).unwrap()
    }

    /// Shuffles the dataset.
    ///
    /// The iterator would still run over the whole dataset, but the order in
    /// which elements are grouped in mini-batches is randomized.
    /// The shuffling is applied consistently across all tensors in `xs` and `ys`.
    pub fn shuffle(&mut self) -> &mut IterN2 {
        let index = Tensor::randperm(self.total_size, (Kind::Int64, self.device));
        let shuffled_xs: Vec<Tensor> = self.xs.iter().map(|x| x.index_select(0, &index)).collect();
        let shuffled_ys: Vec<Tensor> = self.ys.iter().map(|y| y.index_select(0, &index)).collect();
        self.xs = shuffled_xs;
        self.ys = shuffled_ys;
        self.batch_index = 0;
        self
    }

    /// Transfers the mini-batches to a specified device.
    #[allow(clippy::wrong_self_convention)]
    pub fn to_device(&mut self, device: Device) -> &mut IterN2 {
        self.device = device;
        self
    }

    /// When set, returns the last batch even if it is smaller than the batch size.
    pub fn return_smaller_last_batch(&mut self) -> &mut IterN2 {
        self.return_smaller_last_batch = true;
        self
    }
}

impl Iterator for IterN2 {
    type Item = (Vec<Tensor>, Vec<Tensor>);

    fn next(&mut self) -> Option<Self::Item> {
        let start = self.batch_index * self.batch_size;
        let end = (start + self.batch_size).min(self.total_size);
        let size = end - start;

        if size <= 0 || (!self.return_smaller_last_batch && size < self.batch_size) {
            return None;
        }

        self.batch_index += 1;
        let xs_batch: Vec<Tensor> = self.xs.iter().map(|x| x.narrow(0, start, size).to_device(self.device)).collect();
        let ys_batch: Vec<Tensor> = self.ys.iter().map(|y| y.narrow(0, start, size).to_device(self.device)).collect();

        Some((xs_batch, ys_batch))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tch::{Device, Kind, Tensor};
    use tch::data::Iter2;

    // Helper function to create test data
    fn create_test_data(num_tensors: usize, size: i64) -> (Vec<Tensor>, Vec<Tensor>) {
        let xs: Vec<Tensor> = (0..num_tensors)
            .map(|_| Tensor::arange(size as f64, (Kind::Float, Device::Cpu)).view([size, 1]))
            .collect();
        let ys: Vec<Tensor> = xs.iter().map(|x| x * 2.0).collect();
        (xs, ys)
    }
    #[test]
    fn test_bigiter2_creation() {
        let (xs, ys) = create_test_data(3, 10);
        let iter = IterN2::new(&xs, &ys, 2);
        assert_eq!(iter.total_size, 10);
        assert_eq!(iter.batch_size, 2);

        // Test creation with mismatched lengths
        let (xs, mut ys) = create_test_data(3, 10);
        ys.pop();
        assert!(IterN2::f_new(&xs, &ys, 2).is_err());

        // Test creation with mismatched sizes
        let (mut xs, ys) = create_test_data(3, 10);
        xs[0] = Tensor::arange(15.0, (Kind::Float, Device::Cpu)).view([15, 1]);
        assert!(IterN2::f_new(&xs, &ys, 2).is_err());
    }

    #[test]
    fn test_bigiter2_iteration() {
        let (xs, ys) = create_test_data(2, 10);
        let iter = IterN2::new(&xs, &ys, 3);
        let batches: Vec<_> = iter.collect();

        assert_eq!(batches.len(), 3); // Expect 3 batches, not 4
        for (i, (batch_xs, batch_ys)) in batches.iter().enumerate() {
            assert_eq!(batch_xs.len(), 2);
            assert_eq!(batch_ys.len(), 2);
            assert_eq!(batch_xs[0].size(), &[3, 1]);
            assert_eq!(batch_ys[0].size(), &[3, 1]);
            assert_eq!(batch_xs[0].double_value(&[0, 0]), (i * 3) as f64);
            assert_eq!(batch_ys[0].double_value(&[0, 0]), (i * 3) as f64 * 2.0);
        }
    }

    #[test]
    fn test_bigiter2_smaller_last_batch() {
        let (xs, ys) = create_test_data(2, 10);
        let mut iter = IterN2::new(&xs, &ys, 3);
        iter.return_smaller_last_batch();
        let batches: Vec<_> = iter.collect();

        assert_eq!(batches.len(), 4);
        assert_eq!(batches.last().unwrap().0[0].size(), &[1, 1]);
    }

    #[test]
    fn test_bigiter2_shuffle() {
        let (xs, ys) = create_test_data(2, 10);
        let mut iter = IterN2::new(&xs, &ys, 2);
        iter.shuffle();

        let mut all_elements = Vec::new();
        for (batch_xs, _) in iter {
            all_elements.extend(Vec::<f64>::try_from(&batch_xs[0].view(-1)).unwrap());
        }
        all_elements.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let expected: Vec<f64> = (0..10).map(|x| x as f64).collect();
        assert_eq!(all_elements, expected);
    }

    #[test]
    fn test_bigiter2_to_device() {
        let (xs, ys) = create_test_data(2, 10);
        let mut iter = IterN2::new(&xs, &ys, 2);
        iter.to_device(Device::Cpu);

        for (batch_xs, batch_ys) in iter {
            assert_eq!(batch_xs[0].device(), Device::Cpu);
            assert_eq!(batch_ys[0].device(), Device::Cpu);
        }
    }

    #[test]
    fn test_bigiter2_consistency_with_iter2() {
        let (xs, ys) = create_test_data(1, 10);

        println!("Original X: {:?}", (0..xs[0].size()[0]).map(|i| xs[0].double_value(&[i, 0])).collect::<Vec<f64>>());
        println!("Original Y: {:?}", (0..ys[0].size()[0]).map(|i| ys[0].double_value(&[i, 0])).collect::<Vec<f64>>());

        // Set a fixed seed for reproducibility
        tch::manual_seed(42);
        let mut big_iter = IterN2::new(&xs, &ys, 3);
        big_iter.shuffle();

        tch::manual_seed(42);
        let mut normal_iter = Iter2::new(&xs[0], &ys[0], 3);
        normal_iter.shuffle();

        let big_batches: Vec<_> = big_iter.collect();
        let normal_batches: Vec<_> = normal_iter.collect();

        assert_eq!(big_batches.len(), normal_batches.len());
        for ((big_xs, big_ys), (normal_x, normal_y)) in big_batches.iter().zip(normal_batches.iter()) {
            println!("BigIter2 X: {:?}", (0..big_xs[0].size()[0]).map(|i| big_xs[0].double_value(&[i, 0])).collect::<Vec<f64>>());
            println!("Iter2 X: {:?}", (0..normal_x.size()[0]).map(|i| normal_x.double_value(&[i, 0])).collect::<Vec<f64>>());
            println!("BigIter2 Y: {:?}", (0..big_ys[0].size()[0]).map(|i| big_ys[0].double_value(&[i, 0])).collect::<Vec<f64>>());
            println!("Iter2 Y: {:?}", (0..normal_y.size()[0]).map(|i| normal_y.double_value(&[i, 0])).collect::<Vec<f64>>());
            assert!(big_xs[0].allclose(normal_x, 1e-5, 1e-8, false), "X tensors are not close");
            assert!(big_ys[0].allclose(normal_y, 1e-5, 1e-8, false), "Y tensors are not close");
        }
    }

    #[test]
    fn test_bigiter2_multiple_tensors() {
        let (xs, ys) = create_test_data(3, 10);
        let iter = IterN2::new(&xs, &ys, 2);

        for (batch_xs, batch_ys) in iter {
            assert_eq!(batch_xs.len(), 3);
            assert_eq!(batch_ys.len(), 3);
            for i in 0..3 {
                let xs_i = batch_xs[i].shallow_clone();
                assert_eq!(xs_i.size(), &[2, 1]);
                assert_eq!(batch_ys[i].size(), &[2, 1]);
                assert_eq!(batch_ys[i], xs_i * 2.0);
            }
        }
    }
}