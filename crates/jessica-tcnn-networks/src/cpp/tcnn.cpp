#include <torch/extension.h>
#include "bindings.h"

extern "C" Module* c_create_encoder() {
    std::cout << "Hello";
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

extern "C" Module* c_create_color_net() {
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

extern "C" void c_module_free(Module* module) {
    delete module;
}