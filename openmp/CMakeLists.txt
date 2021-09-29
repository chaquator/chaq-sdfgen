cmake_minimum_required(VERSION 3.15)

find_package(OpenMP)

add_executable(chaq_sdfgen sdfgen.c df.c)

set_target_properties(
  chaq_sdfgen PROPERTIES
  C_STANDARD 11
  C_STANDARD_REQUIRED ON
  C_EXTENSIONS OFF
)

if(OpenMP_FOUND)
  target_link_libraries(chaq_sdfgen PRIVATE OpenMP::OpenMP_C)
endif()

if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  target_link_libraries(chaq_sdfgen PRIVATE m)
endif()

if(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
  target_compile_options(chaq_sdfgen PRIVATE /W4 /WX)
else()
  target_compile_options(chaq_sdfgen PRIVATE -Wall -Wextra -Wpedantic -flto)
endif()

target_include_directories(chaq_sdfgen PRIVATE ${CMAKE_SOURCE_DIR}/include)
