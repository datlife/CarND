cmake_minimum_required(VERSION 3.7)
project(Term2_P1_Extended_Kalman_Filter)

set(CMAKE_CXX_STANDARD 11)
set(EIGEN_LIBS ../Eigen)
include_directories(include)

file(GLOB src_files "src/*.cpp")
file(GLOB header_files "include/*.h")

add_executable(Term2_P1_Extended_Kalman_Filter ${src_files} ${header_files} src/measurement_package.cpp)
target_include_directories(Term2_P1_Extended_Kalman_Filter PUBLIC ${EIGEN_LIBS})
