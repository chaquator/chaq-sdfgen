cmake_minimum_required(VERSION 3.15)

add_executable(chaq_sdfgen_opencl main.cpp)

set_target_properties(
  chaq_sdfgen_opencl
  PROPERTIES
  CXX_STANDARD 17
  CXX_STANDARD_REQUIRED ON
  CXX_EXTENSIONS OFF
)

if(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
  target_compile_options(chaq_sdfgen_opencl PRIVATE /W4 /Wx)
else()
  target_compile_options(chaq_sdfgen_opencl PRIVATE -Wall -Wextra -Wpedantic)
endif()

set(SPDLOG_FMT_EXTERNAL)

find_package(OpenCL REQUIRED)
find_package(fmt CONFIG REQUIRED)
find_package(spdlog CONFIG REQUIRED)
find_package(argparse CONFIG REQUIRED)

target_link_libraries(chaq_sdfgen_opencl PRIVATE OpenCL::OpenCL spdlog::spdlog fmt::fmt argparse::argparse)
target_include_directories(chaq_sdfgen_opencl PRIVATE ${CMAKE_SOURCE_DIR}/include)
