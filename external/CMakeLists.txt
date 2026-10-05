if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/flatbuffers/CMakeLists.txt")
  message(FATAL_ERROR "Missing dependencies. Run: git submodule update --init")
endif()

# Build only the FlatBuffers compiler and header-only C++ runtime.
set(FLATBUFFERS_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(FLATBUFFERS_BUILD_FLATLIB OFF CACHE BOOL "" FORCE)
set(FLATBUFFERS_BUILD_FLATC ON CACHE BOOL "" FORCE)
set(FLATBUFFERS_INSTALL OFF CACHE BOOL "" FORCE)
add_subdirectory(flatbuffers EXCLUDE_FROM_ALL)

# ONNX serializes ModelProto with Protocol Buffers. Build the bundled runtime
# once and let the official ONNX target generate its protobuf bindings from the
# ONNX schema.
set(protobuf_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(protobuf_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(protobuf_INSTALL OFF CACHE BOOL "" FORCE)
add_subdirectory(protobuf EXCLUDE_FROM_ALL)

set(ONNX_BUILD_PYTHON OFF CACHE BOOL "" FORCE)
set(ONNX_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(ONNX_GEN_PB_TYPE_STUBS OFF CACHE BOOL "" FORCE)
set(ONNX_INSTALL OFF CACHE BOOL "" FORCE)
set(ONNX_ML OFF CACHE BOOL "" FORCE)
add_subdirectory(onnx EXCLUDE_FROM_ALL)

set(BUILD_VARIANT core)
add_subdirectory(CORE)
