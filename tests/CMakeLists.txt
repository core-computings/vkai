file(GLOB test_SRC CONFIGURE_DEPENDS "*.cpp")
add_executable(vkai_unittests ${test_SRC})

target_link_libraries(vkai_unittests PRIVATE vkai gtest_main vkai_flatbuffers onnx)

# ONNX's public protobuf headers contain intentionally unused parameters. Keep
# the exception local to tests that directly include those headers.
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    set_source_files_properties(
        ONNXInfoTest.cpp ONNXExecutionTest.cpp ONNXLoaderTest.cpp EngineTest.cpp PROPERTIES
        COMPILE_OPTIONS "-Wno-unused-parameter")
endif()

target_include_directories(vkai_unittests
    PRIVATE
        "${CMAKE_SOURCE_DIR}/external/flatbuffers/include"
)

add_dependencies(vkai_unittests GENERATE_vkai_flatbuffers vkai_weights_target)

target_compile_definitions(vkai_unittests
    PRIVATE
        VKAI_BINARY_DIR="${VKAI_TEST_BINARY_DIR}")

add_test(NAME vkai_unittests COMMAND vkai_unittests)
