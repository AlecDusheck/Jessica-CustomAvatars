#pragma once

#include <torch/extension.h>
#include <tiny-cuda-nn/cpp_api.h>
#include <json/json.hpp>

class WrappedModule {
public:
    WrappedModule(tcnn::cpp::Module* module);

    tcnn::cpp::Context fwd(torch::Tensor input, torch::Tensor params, torch::Tensor output);
    void bwd(const tcnn::cpp::Context& ctx, torch::Tensor input, torch::Tensor params, torch::Tensor output, torch::Tensor dL_doutput, torch::Tensor dL_dinput, torch::Tensor dL_dparams);
    void bwd_bwd_input(const tcnn::cpp::Context& ctx, torch::Tensor input, torch::Tensor params, torch::Tensor dL_ddLdinput, torch::Tensor dL_doutput, torch::Tensor dL_ddLdoutput, torch::Tensor dL_dparams, torch::Tensor dL_dinput);
    void initial_params(size_t seed, torch::Tensor output);

    uint32_t n_input_dims() const;
    uint32_t n_params() const;
    tcnn::cpp::Precision param_precision() const;
    c10::ScalarType c10_param_precision() const;
    uint32_t n_output_dims() const;
    tcnn::cpp::Precision output_precision() const; 
    c10::ScalarType c10_output_precision() const;

    nlohmann::json hyperparams() const;
    std::string name() const;

private:
    std::unique_ptr<tcnn::cpp::Module> m_module;
};

c10::ScalarType torch_type(tcnn::cpp::Precision precision);
void* void_data_ptr(torch::Tensor& tensor);