use tch::{Tensor, Device, Kind, TchError};
use tch::IndexOp;

/// `BigIter2` is an iterator over a pair of `Vec<Tensor>` (xs and ys).
/// It allows for batch selection and indexing across all the tensors in the vectors.
/// It is useful for models where the forward pass consumes multiple tensors and outputs multiple tensors.
#[derive(Debug)]
pub struct BigIter2 {
    xs: Vec<Tensor>,
    ys: Vec<Tensor>,
    batch_index: i64,
    batch_size: i64,
    total_size: i64,
    device: Device,
    return_smaller_last_batch: bool,
}

impl BigIter2 {
    /// Creates a new `BigIter2` instance.
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
    pub fn f_new(xs: &[Tensor], ys: &[Tensor], batch_size: i64) -> Result<BigIter2, TchError> {
        // Check if xs and ys have the same length
        if xs.len() != ys.len() {
            return Err(TchError::Shape(format!(
                "xs and ys have different lengths: xs={}, ys={}",
                xs.len(),
                ys.len()
            )));
        }

        // Check if all tensors in xs have the same first dimension size
        let total_size = xs[0].size()[0];
        for x in xs {
            if x.size()[0] != total_size {
                return Err(TchError::Shape(format!(
                    "tensors in xs have different first dimension sizes"
                )));
            }
        }

        // Check if all tensors in ys have the same first dimension size
        for y in ys {
            if y.size()[0] != total_size {
                return Err(TchError::Shape(format!(
                    "tensors in ys have different first dimension sizes"
                )));
            }
        }

        Ok(BigIter2 {
            xs: xs.iter().map(|x| x.shallow_clone()).collect(),
            ys: ys.iter().map(|y| y.shallow_clone()).collect(),
            batch_index: 0,
            batch_size,
            total_size,
            device: Device::Cpu,
            return_smaller_last_batch: false,
        })
    }

    /// Creates a new `BigIter2` instance.
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
    pub fn new(xs: &[Tensor], ys: &[Tensor], batch_size: i64) -> BigIter2 {
        BigIter2::f_new(xs, ys, batch_size).unwrap()
    }

    /// Shuffles the dataset.
    ///
    /// The iterator would still run over the whole dataset, but the order in
    /// which elements are grouped in mini-batches is randomized.
    /// The shuffling is applied consistently across all tensors in `xs` and `ys`.
    pub fn shuffle(&mut self) -> &mut BigIter2 {
        let index = Tensor::randperm(self.total_size, (Kind::Int64, self.device));
        for x in &mut self.xs {
            *x = x.index_select(0, &index);
        }
        for y in &mut self.ys {
            *y = y.index_select(0, &index);
        }
        self
    }

    /// Transfers the mini-batches to a specified device.
    #[allow(clippy::wrong_self_convention)]
    pub fn to_device(&mut self, device: Device) -> &mut BigIter2 {
        self.device = device;
        self
    }

    /// When set, returns the last batch even if it is smaller than the batch size.
    pub fn return_smaller_last_batch(&mut self) -> &mut BigIter2 {
        self.return_smaller_last_batch = true;
        self
    }
}

impl Iterator for BigIter2 {
    type Item = (Vec<Tensor>, Vec<Tensor>);

    fn next(&mut self) -> Option<Self::Item> {
        let start = self.batch_index * self.batch_size;
        let size = std::cmp::min(self.batch_size, self.total_size - start);
        if size <= 0 || (!self.return_smaller_last_batch && size < self.batch_size) {
            None
        } else {
            self.batch_index += 1;
            let xs_batch: Vec<Tensor> = self.xs.iter().map(|x| x.i((start, start + size)).to_device(self.device)).collect();
            let ys_batch: Vec<Tensor> = self.ys.iter().map(|y| y.i((start, start + size)).to_device(self.device)).collect();
            Some((xs_batch, ys_batch))
        }
    }
}