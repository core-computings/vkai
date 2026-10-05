file(GLOB test_SRC CONFIGURE_DEPENDS "*.cpp")
add_executable(vkai_unittests ${test_SRC})

target_link_libraries(vkai_unittests PRIVATE vkai gtest_main vkai_flatbuffers onnx)

# ONNX's public protobuf headers contain intentionally unused parameters. Keep
# the exception local to the raw ONNX inspection test.
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    set_source_files_properties(ONNXInfoTest.cpp PROPERTIES
        COMPILE_OPTIONS "-Wno-unused-parameter")
endif()

target_include_directories(vkai_unittests
    PRIVATE
        "${CMAKE_SOURCE_DIR}/external/flatbuffers/include"
)

add_dependencies(vkai_unittests GENERATE_vkai_flatbuffers vkai_weights_target)

target_compile_definitions(vkai_unittests
    PRIVATE
        VKAI_SOURCE_DIR="${CMAKE_SOURCE_DIR}"
        VKAI_BINARY_DIR="${CMAKE_BINARY_DIR}")
