if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/flatbuffers/CMakeLists.txt")
  message(FATAL_ERROR "Missing dependencies. Run: git submodule update --init")
endif()

# Build only the FlatBuffers compiler and header-only C++ runtime.
if(CMAKE_CROSSCOMPILING)
  set(VKAI_HOST_BUILD_DIR "${CMAKE_SOURCE_DIR}/build" CACHE PATH
      "Native build containing flatc and protoc for cross-compilation")
  set(FLATBUFFERS_FLATC_EXECUTABLE "${VKAI_HOST_BUILD_DIR}/external/flatbuffers/flatc"
      CACHE FILEPATH "Host flatc")
  set(ONNX_CUSTOM_PROTOC_EXECUTABLE "${VKAI_HOST_BUILD_DIR}/external/protobuf/protoc"
      CACHE FILEPATH "Host protoc")
  foreach(tool IN ITEMS FLATBUFFERS_FLATC_EXECUTABLE ONNX_CUSTOM_PROTOC_EXECUTABLE)
    if(NOT EXISTS "${${tool}}")
      message(FATAL_ERROR "Missing host code generator ${${tool}}. Build natively first.")
    endif()
  endforeach()
  set(FLATBUFFERS_BUILD_FLATC OFF CACHE BOOL "" FORCE)
  set(protobuf_BUILD_PROTOC_BINARIES OFF CACHE BOOL "" FORCE)
else()
  set(FLATBUFFERS_BUILD_FLATC ON CACHE BOOL "" FORCE)
endif()
set(FLATBUFFERS_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(FLATBUFFERS_BUILD_FLATLIB OFF CACHE BOOL "" FORCE)
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
# VKAI needs the Vulkan compute module only. Keep CORE's unrelated modules out
# of a fresh configure unless explicitly requested by the caller.
foreach(module IN ITEMS OPENCL MAT TIMER TRACE OPENGL METAL EGL IO THREADPOOL TESTS EXAMPLES)
  set(ENABLE_${module} OFF CACHE BOOL "Build CORE ${module}")
endforeach()
add_subdirectory(CORE)
