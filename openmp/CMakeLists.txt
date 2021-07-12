cmake_minimum_required(VERSION 3.15)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_BUILD_TYPE Release)

find_package(OpenMP)

if(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
  add_compile_options(/W4 /WX)
else()
  add_compile_options(-Wall -Wextra -Wpedantic -flto)
endif()

if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  link_libraries(m)
endif()

add_executable(chaq_sdfgen sdfgen.c df.c)
if(OpenMP_FOUND)
  target_link_libraries(chaq_sdfgen PRIVATE OpenMP::OpenMP_C)
endif()

target_include_directories(chaq_sdfgen PRIVATE ${CMAKE_SOURCE_DIR}/include)
