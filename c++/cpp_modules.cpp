#include <pybind11/pybind11.h>
#include <pybind11/complex.h>
#include <pybind11/eigen.h>

#include <pybind11/stl.h>
#include <pybind11/functional.h>
#include <pybind11/chrono.h>

#include "occupation.hpp" 
#include "exchange.hpp" 

namespace py = pybind11;

PYBIND11_MODULE(cpp_modules, m) {
    m.def("calc_occupation", &calc_occupation, "Calculate occupation matrices");
    m.def("calc_exchange", &calc_exchange, "Calculate exchange interactions");
}

