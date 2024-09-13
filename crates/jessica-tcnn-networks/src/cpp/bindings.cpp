/*
 * Copyright (c) 2020-2023, NVIDIA CORPORATION.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without modification, are permitted
 * provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright notice, this list of
 *       conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright notice, this list of
 *       conditions and the following disclaimer in the documentation and/or other materials
 *       provided with the distribution.
 *     * Neither the name of the NVIDIA CORPORATION nor the names of its contributors may be used
 *       to endorse or promote products derived from this software without specific prior written
 *       permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR
 * IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND
 * FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 * BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
 * OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
 * STRICT LIABILITY, OR TOR (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

/** @file   torch_bindings.cu
 *  @author Thomas Müller, Jacob Munkberg, Jon Hasselgren, Or Perel, NVIDIA
 */

#include <torch/extension.h>

#include <ATen/cuda/CUDAUtils.h>
#include <c10/cuda/CUDAGuard.h>

#ifdef snprintf
#undef snprintf
#endif

#include <json/json.hpp>
#include <tiny-cuda-nn/cpp_api.h>
#include "include/bindings.h"

#define STRINGIFY(x) #x
#define STR(x) STRINGIFY(x)
#define FILE_LINE __FILE__ ":" STR(__LINE__)
#define CHECK_THROW(x) \
	do { if (!(x)) throw std::runtime_error(std::string(FILE_LINE " check failed " #x)); } while(0)

c10::ScalarType torch_type(tcnn::cpp::Precision precision) {
	switch (precision) {
		case tcnn::cpp::Precision::Fp32: return torch::kFloat32;
		case tcnn::cpp::Precision::Fp16: return torch::kHalf;
		default:
		    throw std::runtime_error{"Unknown precision torch->type"};
	}
}

void* void_data_ptr(torch::Tensor& tensor) {
	switch (tensor.scalar_type()) {
		case torch::kFloat32: return tensor.data_ptr<float>();
		case torch::kHalf: return tensor.data_ptr<torch::Half>();
		default: throw std::runtime_error{"Unknown precision torch->void"};
	}
}

#define CHECK_INPUT(x) CHECK_THROW(x.device().is_cuda()); CHECK_THROW(x.is_contiguous())


WrappedModule::WrappedModule(tcnn::cpp::Module* module) : m_module{module} {}

tcnn::cpp::Context WrappedModule::fwd(torch::Tensor input, torch::Tensor params, torch::Tensor output) {
    CHECK_INPUT(input);
    CHECK_INPUT(params);
    CHECK_INPUT(output);

    // Types
    CHECK_THROW(input.scalar_type() == torch::kFloat32);
    CHECK_THROW(params.scalar_type() == c10_param_precision());
    CHECK_THROW(output.scalar_type() == c10_output_precision());

    // Sizes
    CHECK_THROW(input.size(1) == n_input_dims());
    CHECK_THROW(params.size(0) == n_params());
    CHECK_THROW(output.size(0) == input.size(0));
    CHECK_THROW(output.size(1) == n_output_dims());

    // Device
    at::Device device = input.device();
    CHECK_THROW(device == params.device());
    CHECK_THROW(device == output.device());

    const at::cuda::CUDAGuard device_guard{device};
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    uint32_t batch_size = input.size(0);

    tcnn::cpp::Context ctx;
    if (!input.requires_grad() && !params.requires_grad()) {
        m_module->inference(stream, batch_size, input.data_ptr<float>(), void_data_ptr(output), void_data_ptr(params));
    } else {
        ctx = m_module->forward(stream, batch_size, input.data_ptr<float>(), void_data_ptr(output), void_data_ptr(params), input.requires_grad());
    }

    return std::move(ctx);
}


void WrappedModule::bwd(const tcnn::cpp::Context& ctx, torch::Tensor input, torch::Tensor params, torch::Tensor output, torch::Tensor dL_doutput, torch::Tensor dL_dinput, torch::Tensor dL_dparams) {
    if (!ctx.ctx) {
        throw std::runtime_error{"WrappedModule::bwd: called with invalid context. fwd likely (mistakenly) ran in inference mode."};
    }

    CHECK_INPUT(input);
    CHECK_INPUT(params);
    CHECK_INPUT(output);
    CHECK_INPUT(dL_doutput);
    CHECK_INPUT(dL_dinput);
    CHECK_INPUT(dL_dparams);

    // Types
    CHECK_THROW(input.scalar_type() == torch::kFloat32);
    CHECK_THROW(params.scalar_type() == c10_param_precision());
    CHECK_THROW(output.scalar_type() == c10_output_precision());
    CHECK_THROW(dL_doutput.scalar_type() == c10_output_precision());
    CHECK_THROW(dL_dinput.scalar_type() == torch::kFloat32);
    CHECK_THROW(dL_dparams.scalar_type() == c10_param_precision());

    // Sizes
    CHECK_THROW(input.size(1) == n_input_dims());
    CHECK_THROW(output.size(1) == n_output_dims());
    CHECK_THROW(params.size(0) == n_params());
    CHECK_THROW(output.size(0) == input.size(0));
    CHECK_THROW(dL_doutput.size(0) == input.size(0));
    CHECK_THROW(dL_dinput.size(0) == input.size(0));
    CHECK_THROW(dL_dinput.size(1) == n_input_dims());
    CHECK_THROW(dL_dparams.size(0) == n_params());

    // Device
    at::Device device = input.device();
    CHECK_THROW(device == params.device());
    CHECK_THROW(device == output.device());
    CHECK_THROW(device == dL_doutput.device());
    CHECK_THROW(device == dL_dinput.device());
    CHECK_THROW(device == dL_dparams.device());

    const at::cuda::CUDAGuard device_guard{device};
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    uint32_t batch_size = input.size(0);

    if (input.requires_grad() || params.requires_grad()) {
        m_module->backward(
            stream,
            ctx,
            batch_size,
            input.requires_grad() ? dL_dinput.data_ptr<float>() : nullptr,
            void_data_ptr(dL_doutput),
            params.requires_grad() ? void_data_ptr(dL_dparams) : nullptr,
            input.data_ptr<float>(),
            void_data_ptr(output),
            void_data_ptr(params)
        );
    }
}

void WrappedModule::bwd_bwd_input(const tcnn::cpp::Context& ctx, torch::Tensor input, torch::Tensor params, torch::Tensor dL_ddLdinput, torch::Tensor dL_doutput, torch::Tensor dL_ddLdoutput, torch::Tensor dL_dparams, torch::Tensor dL_dinput) {
    // from: dL_ddLdinput
    // to:   dL_ddLdoutput, dL_dparams, dL_dinput

    if (!ctx.ctx) {
        throw std::runtime_error{"WrappedModule::bwd_bwd_input: called with invalid context. fwd likely (mistakenly) ran in inference mode."};
    }

    CHECK_INPUT(input);
    CHECK_INPUT(params);
    CHECK_INPUT(dL_ddLdinput);
    CHECK_INPUT(dL_doutput);
    CHECK_INPUT(dL_ddLdoutput);
    CHECK_INPUT(dL_dparams);
    CHECK_INPUT(dL_dinput);

    // Types
    CHECK_THROW(input.scalar_type() == torch::kFloat32);
    CHECK_THROW(dL_ddLdinput.scalar_type() == torch::kFloat32);
    CHECK_THROW(params.scalar_type() == c10_param_precision());
    CHECK_THROW(dL_doutput.scalar_type() == c10_output_precision());
    CHECK_THROW(dL_ddLdoutput.scalar_type() == c10_output_precision());
    CHECK_THROW(dL_dparams.scalar_type() == c10_param_precision());
    CHECK_THROW(dL_dinput.scalar_type() == torch::kFloat32);

    // Sizes
    CHECK_THROW(input.size(1) == n_input_dims());
    CHECK_THROW(dL_doutput.size(1) == n_output_dims());
    CHECK_THROW(dL_ddLdinput.size(1) == n_input_dims());
    CHECK_THROW(params.size(0) == n_params());
    CHECK_THROW(dL_doutput.size(0) == input.size(0));
    CHECK_THROW(dL_ddLdinput.size(0) == input.size(0));
    CHECK_THROW(dL_ddLdoutput.size(0) == input.size(0));
    CHECK_THROW(dL_ddLdoutput.size(1) == n_output_dims());
    CHECK_THROW(dL_dparams.size(0) == n_params());
    CHECK_THROW(dL_dinput.size(0) == input.size(0));
    CHECK_THROW(dL_dinput.size(1) == n_input_dims());

    // Device
    at::Device device = input.device();
    CHECK_THROW(device == params.device());
    CHECK_THROW(device == dL_ddLdinput.device());
    CHECK_THROW(device == dL_doutput.device());
    CHECK_THROW(device == dL_ddLdoutput.device());
    CHECK_THROW(device == dL_dparams.device());
    CHECK_THROW(device == dL_dinput.device());

    const at::cuda::CUDAGuard device_guard{device};
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    uint32_t batch_size = input.size(0);

    if (dL_doutput.requires_grad() || params.requires_grad() || input.requires_grad()) {
        m_module->backward_backward_input(
            stream,
            ctx,
            batch_size,
            dL_ddLdinput.data_ptr<float>(),
            input.data_ptr<float>(),
            (params.requires_grad() || input.requires_grad()) ? void_data_ptr(dL_doutput) : nullptr,
            params.requires_grad() ? void_data_ptr(dL_dparams) : nullptr,
            dL_doutput.requires_grad() ? void_data_ptr(dL_ddLdoutput) : nullptr,
            input.requires_grad() ? dL_dinput.data_ptr<float>() : nullptr,
            void_data_ptr(params)
        );
    }
}

void WrappedModule::initial_params(size_t seed, torch::Tensor output) {
    // Validate the output tensor
    if (output.numel() != n_params()) {
        throw std::runtime_error("WrappedModule::initial_params: output tensor has incorrect number of elements");
    }

    if (output.device().type() != torch::kCUDA) {
        throw std::runtime_error("WrappedModule::initial_params: output tensor must be on CUDA device");
    }
    if (output.dtype() != torch::kFloat32) {
        throw std::runtime_error("WrappedModule::initial_params: output tensor must have dtype float32");
    }

    m_module->initialize_params(seed, output.data_ptr<float>());
}

uint32_t WrappedModule::n_input_dims() const { return m_module->n_input_dims(); }

uint32_t WrappedModule::n_params() const { return (uint32_t)m_module->n_params(); }
tcnn::cpp::Precision WrappedModule::param_precision() const { return m_module->param_precision(); }
c10::ScalarType WrappedModule::c10_param_precision() const { return torch_type(param_precision()); }

uint32_t WrappedModule::n_output_dims() const { return m_module->n_output_dims(); }
tcnn::cpp::Precision WrappedModule::output_precision() const { return m_module->output_precision(); }
c10::ScalarType WrappedModule::c10_output_precision() const { return torch_type(output_precision()); }

nlohmann::json WrappedModule::hyperparams() const { return m_module->hyperparams(); }
std::string WrappedModule::name() const { return m_module->name(); }