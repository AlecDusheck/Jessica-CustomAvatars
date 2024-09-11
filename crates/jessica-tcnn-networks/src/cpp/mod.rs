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
    // Existing functions
    fn c_create_encoder() -> *mut Module;
    fn c_create_color_net() -> *mut Module;
    fn c_module_free(module: *mut Module);

    fn batch_size_granularity() -> u32;
    fn default_loss_scale(precision: Precision) -> f32;
    fn free_temporary_memory();
    fn has_networks() -> bool;
    fn preferred_precision() -> Precision;

    // Updated Module functions
    fn module_destroy(module: *mut Module);

    fn module_fwd(module: *mut Module, input: *const C_tensor, params: *const C_tensor) -> (*mut Context, *mut C_tensor);
    fn module_bwd(module: *mut Module, ctx: *mut Context, input: *const C_tensor, params: *const C_tensor, output: *const C_tensor, dL_doutput: *const C_tensor) -> (*mut C_tensor, *mut C_tensor);
    fn module_bwd_bwd_input(module: *mut Module, ctx: *mut Context, input: *const C_tensor, params: *const C_tensor, dL_ddLdinput: *const C_tensor, dL_doutput: *const C_tensor) -> (*mut C_tensor, *mut C_tensor, *mut C_tensor);
    fn module_initial_params(module: *mut Module, seed: usize) -> *mut C_tensor;

    fn module_n_input_dims(module: *const Module) -> u32;
    fn module_n_params(module: *const Module) -> u32;
    fn module_param_precision(module: *const Module) -> Precision;
    fn module_n_output_dims(module: *const Module) -> u32;
    fn module_output_precision(module: *const Module) -> Precision;
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

    pub fn forward(&self, input: &Tensor, params: &Tensor) -> (Context, Tensor) {
        unsafe {
            let (ctx_ptr, output_ptr) = module_fwd(self.inner, input.as_ptr(), params.as_ptr());
            // TODO: this is bad
            let ctx = Context { _private: [] }; // Create a new Context struct
            let output = Tensor::from_ptr(output_ptr); // Assuming Tensor::from_ptr exists and takes a *mut tch::CModule
            (ctx, output)
        }
    }

    pub fn backward(&self, ctx: &Context, input: &Tensor, params: &Tensor, output: &Tensor, dL_doutput: &Tensor) -> (Tensor, Tensor) {
        unsafe {
            let (dL_dinput_ptr, dL_dparams_ptr) = module_bwd(
                self.inner,
                ctx as *const _ as *mut Context,
                input.as_ptr() as *const C_tensor,
                params.as_ptr() as *const C_tensor,
                output.as_ptr() as *const C_tensor,
                dL_doutput.as_ptr() as *const C_tensor
            );
            let dL_dinput = Tensor::from_ptr(dL_dinput_ptr);
            let dL_dparams = Tensor::from_ptr(dL_dparams_ptr);
            (dL_dinput, dL_dparams)
        }
    }

    pub fn backward_backward_input(&self, ctx: &Context, input: &Tensor, params: &Tensor, dL_ddLdinput: &Tensor, dL_doutput: &Tensor) -> (Tensor, Tensor, Tensor) {
        unsafe {
            let (dL_ddLdoutput_ptr, dL_dparams_ptr, dL_dinput_ptr) = module_bwd_bwd_input(
                self.inner,
                ctx as *const _ as *mut Context,
                input.as_ptr(),
                params.as_ptr() as *const C_tensor,
                dL_ddLdinput.as_ptr() as *const C_tensor,
                dL_doutput.as_ptr() as *const C_tensor
            );
            let dL_ddLdoutput = Tensor::from_ptr(dL_ddLdoutput_ptr);
            let dL_dparams = Tensor::from_ptr(dL_dparams_ptr);
            let dL_dinput = Tensor::from_ptr(dL_dinput_ptr);
            (dL_ddLdoutput, dL_dparams, dL_dinput)
        }
    }

    pub fn initial_params(&self, seed: usize) -> Tensor {
        unsafe {
            let ptr = module_initial_params(self.inner, seed);
            Tensor::from_ptr(ptr).to_kind(Kind::Float)
        }
    }

    pub fn n_input_dims(&self) -> u32 {
        unsafe { module_n_input_dims(self.inner) }
    }

    pub fn n_params(&self) -> u32 {
        unsafe { module_n_params(self.inner) }
    }

    pub fn param_precision(&self) -> Precision {
        unsafe { module_param_precision(self.inner) }
    }

    pub fn n_output_dims(&self) -> u32 {
        unsafe { module_n_output_dims(self.inner) }
    }

    pub fn output_precision(&self) -> Precision {
        unsafe { module_output_precision(self.inner) }
    }
}

impl Drop for TcnnModule {
    fn drop(&mut self) {
        unsafe { c_module_free(self.inner) };
    }
}