cmake_minimum_required(VERSION 3.12)
project(cpp_module)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native")


# Set the path to Pybind11Config.cmake
set(PYBIND11_DIR "/Users/danis/local/miniconda3/share/cmake/pybind11")

# Find Python and Pybind11
find_package(Python COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 CONFIG REQUIRED)

# Find Eigen (replace with your Eigen installation path)
set(EIGEN_INCLUDE_DIR "/Users/danis/local/miniconda3/include/eigen3")
find_package(Eigen3 REQUIRED)

# Find OpenMP package
find_package(OpenMP REQUIRED)

# Add OpenMP flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")

add_library(cpp_modules MODULE cpp_modules.cpp)

target_link_libraries(cpp_modules PRIVATE Eigen3::Eigen pybind11::module PUBLIC OpenMP::OpenMP_CXX)

set_target_properties(cpp_modules PROPERTIES PREFIX "${PYTHON_MODULE_PREFIX}"
                                         SUFFIX "${PYTHON_MODULE_EXTENSION}")
