#include <torch/extension.h>
#include "include/bindings.h"

extern "C" {
WrappedModule* module_create(tcnn::cpp::Module* internal_module) {
    try {
        return new WrappedModule(internal_module);
    } catch (const std::exception& e) {
        std::cerr << "C++ exception caught: " << e.what() << std::endl;
        return nullptr;
    }
}

WrappedModule* create_network_with_input_encoding(uint32_t n_input_dims, uint32_t n_output_dims, const nlohmann::json& encoding, const nlohmann::json& network) {
    try {
        return module_create(tcnn::cpp::create_network_with_input_encoding(n_input_dims, n_output_dims, encoding, network));
    } catch (const std::exception& e) {
        std::cerr << "C++ exception caught: " << e.what() << std::endl;
        return nullptr;
    }
}

WrappedModule* create_network(uint32_t n_input_dims, uint32_t n_output_dims, const nlohmann::json& network) {
    try {
        return module_create(tcnn::cpp::create_network(n_input_dims, n_output_dims, network));
    } catch (const std::exception& e) {
        std::cerr << "C++ exception caught: " << e.what() << std::endl;
        return nullptr;
    }
}

WrappedModule* create_encoding(uint32_t n_input_dims, const nlohmann::json& encoding, tcnn::cpp::Precision requested_precision) {
    try {
        return module_create(tcnn::cpp::create_encoding(n_input_dims, encoding, requested_precision));
    } catch (const std::exception& e) {
        std::cerr << "C++ exception caught: " << e.what() << std::endl;
        return nullptr;
    }
}

void module_destroy(WrappedModule* module) {
    try {
        delete module;
    } catch (const std::exception& e) {
        std::cerr << "C++ exception caught: " << e.what() << std::endl;
    }
}

// Wrapper for fwd
tcnn::cpp::Context* c_module_fwd(WrappedModule* module, at::Tensor& input, at::Tensor& params, at::Tensor& output) {
    try {
        auto ctx = module->fwd(input, params, output);
        return new tcnn::cpp::Context(std::move(ctx));
    } catch (const std::exception& e) {
        std::cerr << "C++ exception caught in fwd: " << e.what() << std::endl;
        return nullptr;
    }
}

// Wrapper for bwd
void c_module_bwd(WrappedModule* module, tcnn::cpp::Context* ctx, torch::Tensor input, torch::Tensor params, torch::Tensor output, torch::Tensor dL_doutput, torch::Tensor dL_dinput, torch::Tensor dL_dparams) {
    try {
        module->bwd(*ctx, input, params, output, dL_doutput, dL_dinput, dL_dparams);
    } catch (const std::exception& e) {
        std::cerr << "C++ exception caught in bwd: " << e.what() << std::endl;
    }
}

// Wrapper for bwd_bwd_input
void c_module_bwd_bwd_input(WrappedModule* module, tcnn::cpp::Context* ctx, torch::Tensor input, torch::Tensor params, torch::Tensor dL_ddLdinput, torch::Tensor dL_doutput, torch::Tensor dL_ddLdoutput, torch::Tensor dL_dparams, torch::Tensor dL_dinput) {
    try {
        module->bwd_bwd_input(*ctx, input, params, dL_ddLdinput, dL_doutput, dL_ddLdoutput, dL_dparams, dL_dinput);
    } catch (const std::exception& e) {
        std::cerr << "C++ exception caught in bwd_bwd_input: " << e.what() << std::endl;
    }
}

// Wrapper for initial_params
void c_module_initial_params(WrappedModule* module, size_t seed, torch::Tensor output) {
    try {
        module->initial_params(seed, output);
    } catch (const std::exception& e) {
        std::cerr << "C++ exception caught: " << e.what() << std::endl;
    }
}

// Wrappers for other methods
uint32_t c_module_n_input_dims(const WrappedModule* module) {
    try {
        return module->n_input_dims();
    } catch (const std::exception& e) {
        std::cerr << "C++ exception caught: " << e.what() << std::endl;
        return 0;
    }
}

uint32_t c_module_n_params(const WrappedModule* module) {
    try {
        return module->n_params();
    } catch (const std::exception& e) {
        std::cerr << "C++ exception caught: " << e.what() << std::endl;
        return 0;
    }
}

tcnn::cpp::Precision c_module_param_precision(const WrappedModule* module) {
    try {
        return module->param_precision();
    } catch (const std::exception& e) {
        std::cerr << "C++ exception caught: " << e.what() << std::endl;
        return tcnn::cpp::Precision::Fp32;
    }
}

c10::ScalarType c_module_c10_param_precision(const WrappedModule* module) {
    try {
        return module->c10_param_precision();
    } catch (const std::exception& e) {
        std::cerr << "C++ exception caught: " << e.what() << std::endl;
        return torch::kFloat32;
    }
}

uint32_t c_module_n_output_dims(const WrappedModule* module) {
    try {
        return module->n_output_dims();
    } catch (const std::exception& e) {
        std::cerr << "C++ exception caught: " << e.what() << std::endl;
        return 0;
    }
}

tcnn::cpp::Precision c_module_output_precision(const WrappedModule* module) {
    try {
        return module->output_precision();
    } catch (const std::exception& e) {
        std::cerr << "C++ exception caught: " << e.what() << std::endl;
        return tcnn::cpp::Precision::Fp32;
    }
}

c10::ScalarType c_module_c10_output_precision(const WrappedModule* module) {
    try {
        return module->c10_output_precision();
    } catch (const std::exception& e) {
        std::cerr << "C++ exception caught: " << e.what() << std::endl;
        return torch::kFloat32;
    }
}

const char* c_module_hyperparams(const WrappedModule* module) {
    try {
        static std::string hyperparams_str;
        hyperparams_str = module->hyperparams().dump();
        return hyperparams_str.c_str();
    } catch (const std::exception& e) {
        std::cerr << "C++ exception caught: " << e.what() << std::endl;
        return nullptr;
    }
}

const char* c_module_name(const WrappedModule* module) {
    try {
        return module->name().c_str();
    } catch (const std::exception& e) {
        std::cerr << "C++ exception caught: " << e.what() << std::endl;
        return nullptr;
    }
}

}

extern "C" WrappedModule* c_create_encoder() {
    try {
        nlohmann::json encoding_config = {
            {"otype", "HashGrid"},
            {"n_levels", 16},
            {"n_features_per_level", 2},
            {"log2_hashmap_size", 19},
            {"base_resolution", 16},
            {"per_level_scale", 1.5}
        };

        nlohmann::json network_config = {
            {"otype", "FullyFusedMLP"},
            {"activation", "ReLU"},
            {"output_activation", "None"},
            {"n_neurons", 64},
            {"n_hidden_layers", 1}
        };

        return create_network_with_input_encoding(3, 16, encoding_config, network_config);
    } catch (const std::exception& e) {
        std::cerr << "C++ exception caught: " << e.what() << std::endl;
        return nullptr;
    }
}

extern "C" WrappedModule* c_create_color_net() {
    try {
        nlohmann::json network_config = {
            {"otype", "FullyFusedMLP"},
            {"activation", "ReLU"},
            {"output_activation", "Sigmoid"},
            {"n_neurons", 64},
            {"n_hidden_layers", 2}
        };

        return create_network(15, 3, network_config);
    } catch (const std::exception& e) {
        std::cerr << "C++ exception caught: " << e.what() << std::endl;
        return nullptr;
    }
}

extern "C" void c_module_free(WrappedModule* module) {
    delete module;
}