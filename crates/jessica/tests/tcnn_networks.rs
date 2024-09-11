use tch::{Device, Kind, Tensor};
use jessica_tcnn_networks::cpp::{Precision, TcnnModule};

#[test]
fn test_new_encoder() {
    let encoder = TcnnModule::new_encoder();
    assert!(!encoder.inner.is_null());
}

#[test]
fn test_new_color_net() {
    let color_net = TcnnModule::new_color_net();
    // d
    assert!(!color_net.inner.is_null());
}

// #[test]
// fn test_forward() {
//     let module = TcnnModule::new_encoder();
//     let device = Device::Cuda(0);
//     let batch_size = 2;
//     let input = Tensor::rand(&[batch_size, module.n_input_dims() as i64], (Kind::Float, device));
//     let params = Tensor::rand(&[module.n_params() as i64], (Kind::Float, device));
//
//     let (output, ctx) = module.forward(&input, &params);
//     assert_eq!(output.size(), &[batch_size, module.n_output_dims() as i64]);
//     assert_eq!(output.kind(), Kind::Float);
//     assert_eq!(output.device(), device);
// }
//
// #[test]
// fn test_backward() {
//     let module = TcnnModule::new_encoder();
//     let device = Device::Cuda(0);
//     let batch_size = 2;
//     let input = Tensor::rand(&[batch_size, module.n_input_dims() as i64], (Kind::Float, device));
//     let params = Tensor::rand(&[module.n_params() as i64], (Kind::Float, device));
//
//     let (output, ctx) = module.forward(&input, &params);
//     let dL_doutput = Tensor::ones_like(&output);
//     let (dL_dinput, dL_dparams) = module.backward(&ctx, &input, &params, &output, &dL_doutput);
//
//     assert_eq!(dL_dinput.size(), input.size());
//     assert_eq!(dL_dinput.kind(), Kind::Float);
//     assert_eq!(dL_dinput.device(), device);
//
//     assert_eq!(dL_dparams.size(), &[module.n_params() as i64]);
//     assert_eq!(dL_dparams.kind(), Kind::Float);
//     assert_eq!(dL_dparams.device(), device);
// }
//
// #[test]
// fn test_backward_backward_input() {
//     let module = TcnnModule::new_encoder();
//     let device = Device::Cuda(0);
//     let batch_size = 2;
//     let input = Tensor::rand(&[batch_size, module.n_input_dims() as i64], (Kind::Float, device));
//     let params = Tensor::rand(&[module.n_params() as i64], (Kind::Float, device));
//
//     let (output, ctx) = module.forward(&input, &params);
//     let dL_ddLdinput = Tensor::ones_like(&input);
//     let dL_doutput = Tensor::ones_like(&output);
//     let (dL_ddLdoutput, dL_dparams, dL_dinput) = module.backward_backward_input(&ctx, &input, &params, &dL_ddLdinput, &dL_doutput);
//
//     assert_eq!(dL_ddLdoutput.size(), output.size());
//     assert_eq!(dL_ddLdoutput.kind(), Kind::Float);
//     assert_eq!(dL_ddLdoutput.device(), device);
//
//     assert_eq!(dL_dparams.size(), &[module.n_params() as i64]);
//     assert_eq!(dL_dparams.kind(), Kind::Float);
//     assert_eq!(dL_dparams.device(), device);
//
//     assert_eq!(dL_dinput.size(), input.size());
//     assert_eq!(dL_dinput.kind(), Kind::Float);
//     assert_eq!(dL_dinput.device(), device);
// }
//
// #[test]
// fn test_initial_params() {
//     let module = TcnnModule::new_encoder();
//     let seed = 42;
//     let initial_params = module.initial_params(seed);
//
//     assert_eq!(initial_params.size(), &[module.n_params() as i64]);
//     assert_eq!(initial_params.kind(), Kind::Float);
// }
//
// #[test]
// fn test_module_properties() {
//     let module = TcnnModule::new_encoder();
//
//     assert_eq!(module.n_input_dims(), 3);
//     assert!(module.n_params() > 0);
//     assert!(matches!(module.param_precision(), Precision::Fp32));
//     assert_eq!(module.n_output_dims(), 16);
//     assert!(matches!(module.output_precision(), Precision::Fp32));
// }