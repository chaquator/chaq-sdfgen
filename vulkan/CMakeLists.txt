cmake_minimum_required(VERSION 3.15)

add_executable(chaq_sdfgen_vk main.cpp filetype.cpp vma_impl.cpp vulkan_ctx.cpp)

set_target_properties(
  chaq_sdfgen_vk
  PROPERTIES
  CXX_STANDARD 20
  CXX_STANDARD_REQUIRED ON
  CXX_EXTENSIONS OFF
)

if(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
  target_compile_options(chaq_sdfgen_vk PRIVATE /W4 /Wx)
else()
  target_compile_options(chaq_sdfgen_vk PRIVATE -Wall -Wextra -Wpedantic)
endif()

set(SPDLOG_FMT_EXTERNAL)

find_package(Vulkan REQUIRED)
find_package(fmt CONFIG REQUIRED)
find_package(spdlog CONFIG REQUIRED)
find_package(argparse CONFIG REQUIRED)
find_package(unofficial-vulkan-memory-allocator CONFIG REQUIRED)

find_path(VULKAN_HPP_INCLUDE_DIRS "vulkan/vulkan.hpp")

target_link_libraries(chaq_sdfgen_vk PRIVATE
    Vulkan::Vulkan
    spdlog::spdlog
    fmt::fmt
    argparse::argparse
    unofficial::vulkan-memory-allocator::vulkan-memory-allocator
)
target_include_directories(chaq_sdfgen_vk PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_include_directories(chaq_sdfgen_vk PRIVATE ${Vulkan_INCLUDE_DIRS})
target_include_directories(chaq_sdfgen_vk PRIVATE ${VULKAN_HPP_INCLUDE_DIRS})
