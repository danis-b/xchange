cmake_minimum_required(VERSION 3.12)
project(cpp_module)

# Set the path to Pybind11Config.cmake
set(PYBIND11_DIR "/Users/danis/local/miniconda3/share/cmake/pybind11")

# Find Python and Pybind11
find_package(Python COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 CONFIG REQUIRED)

# Find Eigen (replace with your Eigen installation path)
set(EIGEN_INCLUDE_DIR "/Users/danis/local/miniconda3/include/eigen3")
find_package(Eigen3 REQUIRED)

# Add OpenMP support
find_package(OpenMP REQUIRED)

add_library(cpp_modules MODULE cpp_modules.cpp)

# Set OpenMP compiler flags
target_compile_options(cpp_modules PRIVATE ${OpenMP_CXX_FLAGS})

# Specify the OpenMP library from Homebrew
target_link_libraries(cpp_modules PRIVATE Eigen3::Eigen pybind11::module OpenMP::OpenMP_CXX -L/opt/homebrew/opt/libomp/lib -lomp)

set_target_properties(cpp_modules PROPERTIES PREFIX "${PYTHON_MODULE_PREFIX}"
                                         SUFFIX "${PYTHON_MODULE_EXTENSION}")

