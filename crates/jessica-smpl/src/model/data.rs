use std::fs::File;
use std::io::{BufReader, BufWriter};
use std::path::Path;
use log::{debug, info};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use tch::{Tensor, Device, Kind};
use tch::nn::VarStore;
use crate::model::python;
use crate::model::python::{convert_sparse_tensor, convert_to_tensor};

#[derive(Debug)]
pub struct DataModel {
    pub vs: VarStore,
    pub j_regressor_prior: Tensor,
    pub f: Tensor,
    pub j_regressor: Tensor,
    pub kintree_table: Tensor,
    pub j: Tensor,
    pub weights_prior: Tensor,
    pub weights: Tensor,
    pub posedirs: Tensor,
    pub v_template: Tensor,
    pub shapedirs: Tensor,
    pub bs_style: String,
    pub bs_type: String,
}

impl DataModel {
    pub fn new(device: Device) -> Self {
        let vs = VarStore::new(device);
        let root = vs.root();

        Self {
            j_regressor_prior: root.zeros("j_regressor_prior", &[24, 6890]),
            f: root.zeros("f", &[13776, 3]),
            j_regressor: root.zeros("j_regressor", &[24, 6890]),
            kintree_table: root.zeros("kintree_table", &[2, 24]),
            j: root.zeros("j", &[24, 3]),
            weights_prior: root.zeros("weights_prior", &[6890, 24]),
            weights: root.zeros("weights", &[6890, 24]),
            posedirs: root.zeros("posedirs", &[6890, 3, 207]),
            v_template: root.zeros("v_template", &[6890, 3]),
            shapedirs: root.zeros("shapedirs", &[6890, 3, 300]),
            bs_style: String::new(),
            bs_type: String::new(),
            vs,
        }
    }

    pub fn from_pickle_file<P: AsRef<Path>>(path: P, device: Device) -> PyResult<Self> {
        Python::with_gil(|py| {
            let numpy = py.import("numpy")?;
            let scipy = py.import("scipy")?;

            let obj: PyObject = python::from_pickle_file(path)?;

            let result = Self::from_py_object(obj, py, numpy, scipy, device);
            result
        })
    }

    pub fn from_py_object(obj: PyObject, py: Python, numpy: &PyModule, scipy: &PyModule, device: Device) -> PyResult<Self> {
        let dict: &PyDict = obj.extract(py)?;
        let vs = VarStore::new(device);
        let root = vs.root();
        let dtype = Kind::Float;

        let mut smpl_data = Self::new(device);

        let update_tensor = |key: &str, tensor: &mut Tensor| -> PyResult<()> {
            debug!("Processing {}...", key);
            let item = dict.get_item(key).unwrap().unwrap_or_else(|| panic!("{} not found", key));

            let new_tensor = if key == "J_regressor" || key == "J_regressor_prior" {
                convert_sparse_tensor(item, scipy, &root, dtype)?
            } else {
                convert_to_tensor(item, numpy, &root, dtype)?
            };

            *tensor = new_tensor;
            Ok(())
        };

        update_tensor("J_regressor_prior", &mut smpl_data.j_regressor_prior)?;
        update_tensor("f", &mut smpl_data.f)?;
        update_tensor("J_regressor", &mut smpl_data.j_regressor)?;
        update_tensor("kintree_table", &mut smpl_data.kintree_table)?;
        update_tensor("J", &mut smpl_data.j)?;
        update_tensor("weights_prior", &mut smpl_data.weights_prior)?;
        update_tensor("weights", &mut smpl_data.weights)?;
        update_tensor("posedirs", &mut smpl_data.posedirs)?;
        update_tensor("v_template", &mut smpl_data.v_template)?;
        update_tensor("shapedirs", &mut smpl_data.shapedirs)?;

        smpl_data.bs_style = dict.get_item("bs_style")?.expect("bs_style not found").to_string();
        smpl_data.bs_type = dict.get_item("bs_type")?.expect("bs_type not found").to_string();

        Ok(smpl_data)
    }

    pub fn save_to_file<P: AsRef<Path>>(&self, path: P) -> Result<(), Box<dyn std::error::Error>> {
        self.vs.save(path.as_ref())?;

        let additional_data = serde_json::json!({
            "bs_style": self.bs_style,
            "bs_type": self.bs_type,
        });

        let file = File::create(path.as_ref().with_extension("json"))?;
        let writer = BufWriter::new(file);
        serde_json::to_writer(writer, &additional_data)?;

        Ok(())
    }

    pub fn load_from_file<P: AsRef<Path>>(path: P, device: Device) -> Result<Self, Box<dyn std::error::Error>> {
        let mut vs = VarStore::new(device);
        vs.load(path.as_ref())?;
        let root = vs.root();

        let json_path = path.as_ref().with_extension("json");
        let file = File::open(json_path)?;
        let reader = BufReader::new(file);
        let additional_data: serde_json::Value = serde_json::from_reader(reader)?;

        let result = Self {
            shapedirs: root.get("shapedirs").unwrap_or_else(|| root.zeros("shapedirs", &[6890, 3, 300])),
            bs_style: additional_data["bs_style"].as_str().unwrap_or("lbs").to_string(),
            f: root.get("f").unwrap_or_else(|| root.zeros("f", &[13776, 3])),
            v_template: root.get("v_template").unwrap_or_else(|| root.zeros("v_template", &[6890, 3])),
            j_regressor: root.get("j_regressor").unwrap_or_else(|| root.zeros("j_regressor", &[24, 6890])),
            posedirs: root.get("posedirs").unwrap_or_else(|| root.zeros("posedirs", &[6890, 3, 207])),
            kintree_table: root.get("kintree_table").unwrap_or_else(|| root.zeros("kintree_table", &[2, 24])),
            j: root.get("j").unwrap_or_else(|| root.zeros("j", &[24, 3])),
            weights_prior: root.get("weights_prior").unwrap_or_else(|| root.zeros("weights_prior", &[6890, 24])),
            weights: root.get("weights").unwrap_or_else(|| root.zeros("weights", &[6890, 24])),
            j_regressor_prior: root.get("j_regressor_prior").unwrap_or_else(|| root.zeros("j_regressor_prior", &[24, 6890])),
            bs_type: additional_data["bs_type"].as_str().unwrap_or("lrotmin").to_string(),
            vs,
        };

        Ok(result)
    }

    pub fn log_tensor_sizes(&self) {
        info!("SMPL Data Contents:");
        info!("J_regressor_prior: Sparse tensor of shape {:?}", self.j_regressor_prior.size());
        info!("f: Dense tensor of shape {:?}", self.f.size());
        info!("J_regressor: Sparse tensor of shape {:?}", self.j_regressor.size());
        info!("kintree_table: Dense tensor of shape {:?}", self.kintree_table.size());
        info!("J: Dense tensor of shape {:?}", self.j.size());
        info!("weights_prior: Dense tensor of shape {:?}", self.weights_prior.size());
        info!("weights: Dense tensor of shape {:?}", self.weights.size());
        info!("posedirs: Dense tensor of shape {:?}", self.posedirs.size());
        info!("v_template: Dense tensor of shape {:?}", self.v_template.size());
        info!("shapedirs: Dense tensor of shape {:?}", self.shapedirs.size());
        info!("bs_style: String value: '{}'", self.bs_style);
        info!("bs_type: String value: '{}'", self.bs_type);
    }
}