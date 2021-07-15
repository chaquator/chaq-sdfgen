cmake_minimum_required(VERSION 3.15)

# adapted from https://beesbuzz.biz/code/4399-Embedding-binary-resources-with-CMake-and-C-11
function(add_resources out_var)
  set(result)
  foreach(in_f ${ARGN})
    file(RELATIVE_PATH src_f ${CMAKE_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/${in_f})
    set(out_f "${PROJECT_BINARY_DIR}/${in_f}.o")
    add_custom_command(
      OUTPUT ${out_f}
      COMMAND ld -r -b binary -o ${out_f} ${src_f}
      DEPENDS ${in_f}
      WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
      VERBATIM
    )
    list(APPEND result ${out_f})
  endforeach()
  set(${out_var} "${result}" PARENT_SCOPE)
endfunction()

add_resources(CL_TEXT sdf.cl)
add_executable(chaq_sdfgen_opencl main.cpp ${CL_TEXT})

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
  target_compile_options(chaq_sdfgen_opencl PRIVATE -Wall -Wextra -Wpedantic -flto)
endif()

set(SPDLOG_FMT_EXTERNAL)

find_package(OpenCL REQUIRED)
find_package(fmt CONFIG REQUIRED)
find_package(spdlog CONFIG REQUIRED)
find_package(argparse CONFIG REQUIRED)

target_link_libraries(chaq_sdfgen_opencl PRIVATE OpenCL::OpenCL spdlog::spdlog fmt::fmt argparse::argparse)
target_include_directories(chaq_sdfgen_opencl PRIVATE ${CMAKE_SOURCE_DIR}/include)
