#pragma once

namespace pybind11 {
    class module {};
    template <typename T> class class_ {};
}

#define PYBIND11_MODULE(name, variable) void pybind11_init_##name()

namespace py = pybind11;