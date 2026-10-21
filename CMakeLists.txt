cmake_minimum_required(VERSION 3.26)
project(VKAI CXX)

if(ANDROID AND ANDROID_NATIVE_API_LEVEL LESS 28)
  message(FATAL_ERROR "VKAI requires Android API 28 or newer for Vulkan 1.1 entry points")
endif()

enable_testing()

set(VKAI_TEST_BINARY_DIR "${CMAKE_BINARY_DIR}" CACHE STRING "Runtime generated-data root for tests")

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_subdirectory(external)

# Keep project warnings away from third-party targets in external/.
add_compile_options(
  -Wall
  -Wextra
  -Wpedantic
  -Wunused-variable
  -Wunused-function
  -Wshadow
  -Werror
  -O3
)

# Vulkan's designated struct initializers intentionally zero omitted fields.
# Recent NDK Clang versions diagnose these separately from ordinary aggregates.
include(CheckCXXCompilerFlag)
check_cxx_compiler_flag("-Wno-missing-designated-field-initializers" VKAI_HAS_DESIGNATED_WARNING)
if(VKAI_HAS_DESIGNATED_WARNING)
  add_compile_options(-Wno-missing-designated-field-initializers)
endif()

flatbuffers_generate_headers(
  TARGET vkai_flatbuffers
  SCHEMAS schemas/model.fbs
)

flatbuffers_generate_binary_files(
  TARGET vkai_weights
  SCHEMA schemas/model.fbs
  JSON_FILES assets/weights.json
  OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR}
)

add_custom_target(vkai_weights_target DEPENDS "${CMAKE_CURRENT_BINARY_DIR}/weights.bin")

# add_definitions(-DPIPELINE_CACHE_DIR="${PIPELINE_CACHE_DIR}")
add_definitions(-DPIPELINE_CACHE_DIR="")

add_subdirectory(vkai)
add_subdirectory(tasks)
add_subdirectory(tests)
