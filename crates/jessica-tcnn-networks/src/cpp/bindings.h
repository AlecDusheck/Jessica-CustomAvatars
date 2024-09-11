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

extern "C" Module* create_network_with_input_encoding(uint32_t n_input_dims, uint32_t n_output_dims, const nlohmann::json& encoding, const nlohmann::json& network);
extern "C" Module* create_network(uint32_t n_input_dims, uint32_t n_output_dims, const nlohmann::json& network);
extern "C" Module* create_encoding(uint32_t n_input_dims, const nlohmann::json& encoding, tcnn::cpp::Precision requested_precision);

#endif // TORCH_BINDINGS_H