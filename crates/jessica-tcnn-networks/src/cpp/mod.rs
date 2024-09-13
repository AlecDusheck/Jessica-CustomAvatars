use tch::{Device, Kind, Tensor};
use torch_sys::*;

#[repr(C)]
pub struct WrappedModule {
    _private: *mut std::ffi::c_void,
}

#[repr(C)]
pub struct Context {
    _private: *mut std::ffi::c_void,
}

#[link(name = "rust_tcnn", kind = "static")]
extern "C" {
    // Existing functions
    fn c_create_encoder() -> *mut WrappedModule;
    fn c_create_color_net() -> *mut WrappedModule;
    fn c_module_free(module: *mut WrappedModule);

    fn c_module_fwd(module: *mut WrappedModule, input: *const C_tensor, params: *const C_tensor, output: *mut C_tensor) -> *mut std::ffi::c_void;
    fn c_module_bwd(module: *mut WrappedModule, ctx: *mut std::ffi::c_void, input: *const C_tensor, params: *const C_tensor, output: *const C_tensor, dL_doutput: *const C_tensor, dL_dinput: *mut C_tensor, dL_dparams: *mut C_tensor);
    fn c_module_bwd_bwd_input(module: *mut WrappedModule, ctx: *mut std::ffi::c_void, input: *const C_tensor, params: *const C_tensor, dL_ddLdinput: *const C_tensor, dL_doutput: *const C_tensor, dL_ddLdoutput: *mut C_tensor, dL_dparams: *mut C_tensor, dL_dinput: *mut C_tensor);
    fn c_module_initial_params(module: *mut WrappedModule, seed: usize, output: *const C_tensor);

    fn c_module_n_input_dims(module: *const WrappedModule) -> u32;
    fn c_module_n_params(module: *const WrappedModule) -> u32;
    fn c_module_param_precision(module: *const WrappedModule) -> Precision;
    fn c_module_n_output_dims(module: *const WrappedModule) -> u32;
    fn c_module_output_precision(module: *const WrappedModule) -> Precision;
}

#[repr(C)]
#[derive(Debug)]
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
    // The inner C++ module (pointer)
    pub inner: *mut WrappedModule,
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
        let output_type: Kind = match self.param_precision() {
            Precision::Fp32 => Kind::Float,
            Precision::Fp16 => Kind::Half,
        };

        let batch_size = input.size()[0];
        let output_dims = self.n_output_dims() as i64;
        let mut output = Tensor::zeros(&[batch_size, output_dims], (output_type, input.device()));

        let ctx_ptr = unsafe {
            c_module_fwd(self.inner, input.as_ptr(), params.as_ptr(), output.as_mut_ptr())
        };

        let ctx = Context { _private: ctx_ptr };

        (ctx, output)
    }

    pub fn backward(&self, ctx: &Context, input: &Tensor, params: &Tensor, output: &Tensor, dL_doutput: &Tensor) -> (Tensor, Tensor) {
        let mut dL_dinput = Tensor::zeros_like(input);

        let dparams_type: Kind = match self.param_precision() {
            Precision::Fp32 => Kind::Float,
            Precision::Fp16 => Kind::Half,
        };
        
        let mut dL_dparams = Tensor::zeros(&[self.n_params() as i64], (dparams_type, input.device()));

        unsafe {
            c_module_bwd(
                self.inner,
                ctx._private,
                input.as_ptr(),
                params.as_ptr(),
                output.as_ptr(),
                dL_doutput.as_ptr(),
                dL_dinput.as_mut_ptr(),
                dL_dparams.as_mut_ptr(),
            );
        }

        (dL_dinput, dL_dparams)
    }

    // TODO: C++ exception caught in bwd_bwd_input: DifferentiableObject::backward_backward_input_impl: not implemented error
    pub fn backward_backward_input(&self, ctx: &Context, input: &Tensor, params: &Tensor, dL_ddLdinput: &Tensor, dL_doutput: &Tensor) -> (Tensor, Tensor, Tensor) {
        let mut dL_ddLdoutput = Tensor::zeros_like(dL_doutput);

        let dparams_type: Kind = match self.param_precision() {
            Precision::Fp32 => Kind::Float,
            Precision::Fp16 => Kind::Half,
        };

        let mut dL_dparams = Tensor::zeros(&[self.n_params() as i64], (dparams_type, input.device()));
        let mut dL_dinput = Tensor::zeros_like(input);

        unsafe {
            c_module_bwd_bwd_input(
                self.inner,
                ctx._private,
                input.as_ptr(),
                params.as_ptr(),
                dL_ddLdinput.as_ptr(),
                dL_doutput.as_ptr(),
                dL_ddLdoutput.as_mut_ptr(),
                dL_dparams.as_mut_ptr(),
                dL_dinput.as_mut_ptr(),
            );
        }

        (dL_ddLdoutput, dL_dparams, dL_dinput)
    }

    pub fn initial_params(&self, seed: usize) -> Tensor {
        let device = Device::Cuda(0);
        let num_params = self.n_params() as i64;
        let mut output = Tensor::zeros(&[num_params], (Kind::Float, device));
        
        unsafe {
            c_module_initial_params(self.inner, seed, output.as_mut_ptr());
        }

        output
    }

    pub fn n_input_dims(&self) -> u32 {
        unsafe { c_module_n_input_dims(self.inner) }
    }

    pub fn n_params(&self) -> u32 {
        unsafe { c_module_n_params(self.inner) }
    }

    pub fn param_precision(&self) -> Precision {
        unsafe { c_module_param_precision(self.inner) }
    }

    pub fn n_output_dims(&self) -> u32 {
        unsafe { c_module_n_output_dims(self.inner) }
    }

    pub fn output_precision(&self) -> Precision {
        unsafe { c_module_output_precision(self.inner) }
    }
}

impl Drop for TcnnModule {
    fn drop(&mut self) {
        unsafe { c_module_free(self.inner) };
    }
}
