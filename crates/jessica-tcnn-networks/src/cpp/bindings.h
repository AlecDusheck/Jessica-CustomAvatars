#ifndef TORCH_BINDINGS_H
#define TORCH_BINDINGS_H

#include <stdint.h>
#include <json/json.hpp>
#include <tiny-cuda-nn/cpp_api.h>

// Forward declare the Module class from bindings.cu
class Module;

enum class Precision {
    Fp32,
    Fp16
};

namespace tcnn { namespace cpp {

Module* create_network_with_input_encoding(uint32_t n_input_dims, uint32_t n_output_dims, const nlohmann::json& encoding, const nlohmann::json& network);
Module* create_network(uint32_t n_input_dims, uint32_t n_output_dims, const nlohmann::json& network);
Module* create_encoding(uint32_t n_input_dims, const nlohmann::json& encoding, Precision requested_precision);

}}

extern "C" {
Module* module_create(tcnn::cpp::Module* internal_module);
void module_destroy(Module* module);

std::tuple<tcnn::cpp::Context*, torch::Tensor> module_fwd(Module* module, torch::Tensor input, torch::Tensor params);
std::tuple<torch::Tensor, torch::Tensor> module_bwd(Module* module, tcnn::cpp::Context* ctx, torch::Tensor input, torch::Tensor params, torch::Tensor output, torch::Tensor dL_doutput);
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> module_bwd_bwd_input(Module* module, tcnn::cpp::Context* ctx, torch::Tensor input, torch::Tensor params, torch::Tensor dL_ddLdinput, torch::Tensor dL_doutput);
torch::Tensor module_initial_params(Module* module, size_t seed);

uint32_t module_n_input_dims(const Module* module);
uint32_t module_n_params(const Module* module);
tcnn::cpp::Precision module_param_precision(const Module* module);
c10::ScalarType module_c10_param_precision(const Module* module);
uint32_t module_n_output_dims(const Module* module);
tcnn::cpp::Precision module_output_precision(const Module* module);
c10::ScalarType module_c10_output_precision(const Module* module);
const char* module_hyperparams(const Module* module);
const char* module_name(const Module* module);

Module* create_network_with_input_encoding(uint32_t n_input_dims, uint32_t n_output_dims, const nlohmann::json& encoding, const nlohmann::json& network);
Module* create_network(uint32_t n_input_dims, uint32_t n_output_dims, const nlohmann::json& network);
Module* create_encoding(uint32_t n_input_dims, const nlohmann::json& encoding, tcnn::cpp::Precision requested_precision);

}  // extern "C"

#endif // TORCH_BINDINGS_H