use pyo3::prelude::*;
use pyo3::types::{IntoPyDict, PyBytes, PyModule, PyString};
use tch::{Tensor, Kind};
use std::path::Path;
use std::fs::File;
use std::io::Read;

pub fn read_file_to_buffer<P: AsRef<Path>>(path: P) -> PyResult<Vec<u8>> {
    let mut file = File::open(path).expect("Failed to open file");
    let mut buffer = Vec::new();
    file.read_to_end(&mut buffer).expect("Failed to read file");
    Ok(buffer)
}

pub fn convert_to_tensor(obj: &PyAny, numpy: &PyModule, vs: &tch::nn::Path, dtype: Kind) -> PyResult<Tensor> {
    let np_array = if obj.hasattr("r")? {
        obj.getattr("r")?.call_method0("copy")?
    } else {
        obj
    };

    let shape: Vec<i64> = np_array.getattr("shape")?.extract::<Vec<usize>>()?.iter().map(|&x| x as i64).collect();

    let flattened: Vec<f32> = numpy.call_method1("array", (np_array,))?.call_method1("astype", ("float32",))?.call_method0("ravel")?.extract()?;

    let tensor = Tensor::from_slice(&flattened).reshape(&shape).to_kind(dtype).to_device(vs.device());
    Ok(tensor)
}

pub fn convert_sparse_tensor(obj: &PyAny, scipy: &PyModule, vs: &tch::nn::Path, dtype: Kind) -> PyResult<Tensor> {
    let sparse = scipy.getattr("sparse")?;
    let coo_matrix = sparse.call_method1("coo_matrix", (obj,))?;

    let values: Vec<f32> = coo_matrix.getattr("data")?.extract()?;
    let row: Vec<i64> = coo_matrix.getattr("row")?.extract()?;
    let col: Vec<i64> = coo_matrix.getattr("col")?.extract()?;
    let shape: Vec<i64> = coo_matrix.getattr("shape")?.extract::<Vec<usize>>()?.iter().map(|&x| x as i64).collect();

    let indices = Tensor::stack(&[
        Tensor::from_slice(&row).to_device(vs.device()),
        Tensor::from_slice(&col).to_device(vs.device()),
    ], 0);

    let values_tensor = Tensor::from_slice(&values).to_device(vs.device()).to_kind(dtype);

    let tensor = Tensor::sparse_coo_tensor_indices_size(
        &indices,
        &values_tensor,
        &shape,
        (dtype, vs.device()),
        false
    );
    Ok(tensor)
}

pub fn from_pickle_file<P: AsRef<Path>>(path: P) -> PyResult<PyObject> {
    pyo3::prepare_freethreaded_python();

    Python::with_gil(|py| {
        let pickle = py.import("pickle")?;

        let buffer = read_file_to_buffer(&path)?;
        let py_bytes = PyBytes::new(py, &buffer);
        let encoding = PyString::new(py, "latin1");

        let kwargs = [("encoding", encoding)].into_py_dict(py);
        let obj: PyObject = pickle.call_method("loads", (py_bytes,), Some(kwargs))?.extract()?;

        Ok(obj)
    })
}