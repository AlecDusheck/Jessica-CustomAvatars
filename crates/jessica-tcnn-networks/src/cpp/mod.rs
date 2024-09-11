use tch::{Tensor, Kind};
use torch_sys::*;

#[repr(C)]
pub struct Module {
    _private: [u8; 0],
}

#[repr(C)]
pub struct Context {
    _private: [u8; 0],
}

#[link(name = "rust_tcnn", kind = "static")]
extern "C" {
    fn c_create_encoder() -> *mut Module;
    fn c_create_color_net() -> *mut Module;
    fn c_module_free(module: *mut Module);

    // fn batch_size_granularity() -> u32;
    // fn default_loss_scale(precision: Precision) -> f32;
    // fn free_temporary_memory();
    // fn has_networks() -> bool;
    // fn preferred_precision() -> Precision;
    // 
    // 
    // fn module_fwd(module: *mut Module, input: *const C_tensor, params: *const C_tensor, output: *mut C_tensor, ctx: *mut Context);
    // fn module_bwd(module: *mut Module, ctx: *const Context, input: *const C_tensor, params: *const C_tensor, output: *const C_tensor, dL_doutput: *const C_tensor, dL_dinput: *mut C_tensor, dL_dparams: *mut C_tensor);
    // fn module_bwd_bwd_input(module: *mut Module, ctx: *const Context, input: *const C_tensor, params: *const C_tensor, dL_ddLdinput: *const C_tensor, dL_doutput: *const C_tensor, dL_ddLdoutput: *mut C_tensor, dL_dparams: *mut C_tensor, dL_dinput: *mut C_tensor);
    // fn module_initial_params(module: *mut Module, seed: usize) -> *mut C_tensor;
    // fn module_n_input_dims(module: *mut Module) -> u32;
    // fn module_n_params(module: *mut Module) -> u32;
    // fn module_param_precision(module: *mut Module) -> Precision;
    // fn module_n_output_dims(module: *mut Module) -> u32;
    // fn module_output_precision(module: *mut Module) -> Precision;
}

#[repr(C)]
pub enum Precision {
    Fp32,
    Fp16,
}

#[repr(C)]
pub enum LogSeverity {
    Debug,
    Info,
    Warning,
    Error,
    Success,
}

pub struct TcnnModule {
    pub inner: *mut Module,
}

impl TcnnModule {
    pub fn new_encoder() -> Self {
        let inner = unsafe { c_create_encoder() };
        Self { inner }
    }

    pub fn new_color_net() -> Self {
        let inner = unsafe { c_create_color_net() };
        Self { inner }
    }

    // pub fn forward(&self, input: &Tensor, params: &Tensor) -> (Tensor, Context) {
    //     let mut output = Tensor::zeros([input.size()[0], self.n_output_dims() as i64], (Kind::Float, input.device()));
    //     let mut ctx = Context { _private: [] };
    //     unsafe {
    //         module_fwd(self.inner, input.as_ptr(), params.as_ptr(), output.as_mut_ptr(), &mut ctx);
    //     }
    //     (output, ctx)
    // }
    // 
    // pub fn backward(&self, ctx: &Context, input: &Tensor, params: &Tensor, output: &Tensor, dL_doutput: &Tensor) -> (Tensor, Tensor) {
    //     let mut dL_dinput = Tensor::zeros_like(input);
    //     let mut dL_dparams = Tensor::zeros([self.n_params() as i64], (Kind::Float, input.device()));
    //     unsafe {
    //         module_bwd(self.inner, ctx, input.as_ptr(), params.as_ptr(), output.as_ptr(), dL_doutput.as_ptr(), dL_dinput.as_mut_ptr(), dL_dparams.as_mut_ptr());
    //     }
    //     (dL_dinput, dL_dparams)
    // }
    // 
    // pub fn backward_backward_input(&self, ctx: &Context, input: &Tensor, params: &Tensor, dL_ddLdinput: &Tensor, dL_doutput: &Tensor) -> (Tensor, Tensor, Tensor) {
    //     let mut dL_ddLdoutput = Tensor::zeros_like(dL_doutput);
    //     let mut dL_dparams = Tensor::zeros([self.n_params() as i64], (Kind::Float, input.device()));
    //     let mut dL_dinput = Tensor::zeros_like(input);
    //     unsafe {
    //         module_bwd_bwd_input(self.inner, ctx, input.as_ptr(), params.as_ptr(), dL_ddLdinput.as_ptr(), dL_doutput.as_ptr(), dL_ddLdoutput.as_mut_ptr(), dL_dparams.as_mut_ptr(), dL_dinput.as_mut_ptr());
    //     }
    //     (dL_ddLdoutput, dL_dparams, dL_dinput)
    // }
    // 
    // pub fn initial_params(&self, seed: usize) -> Tensor {
    //     unsafe {
    //         let ptr = module_initial_params(self.inner, seed);
    //         Tensor::from_ptr(ptr).to_kind(Kind::Float)
    //     }
    // }
    // 
    // pub fn n_input_dims(&self) -> u32 {
    //     unsafe { module_n_input_dims(self.inner) }
    // }
    // 
    // pub fn n_params(&self) -> u32 {
    //     unsafe { module_n_params(self.inner) }
    // }
    // 
    // pub fn param_precision(&self) -> Precision {
    //     unsafe { module_param_precision(self.inner) }
    // }
    // 
    // pub fn n_output_dims(&self) -> u32 {
    //     unsafe { module_n_output_dims(self.inner) }
    // }
    // 
    // pub fn output_precision(&self) -> Precision {
    //     unsafe { module_output_precision(self.inner) }
    // }
}

impl Drop for TcnnModule {
    fn drop(&mut self) {
        unsafe { c_module_free(self.inner) };
    }
}