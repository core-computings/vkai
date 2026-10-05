#compile vulkan shaders
include("${CMAKE_SOURCE_DIR}/external/CORE/cmake/vulkan_shaders.cmake")
add_vulkan_shader_target(vkai_shaders
    SOURCE_DIR ${CMAKE_CURRENT_LIST_DIR}
    OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR}/shaders)

file(GLOB_RECURSE vkai_SRC CONFIGURE_DEPENDS "src/*.cpp")

add_library(vkai ${vkai_SRC})

# ONNX's public protobuf headers contain intentionally unused parameters. Keep
# the exception limited to the translation unit that imports those headers.
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    set_source_files_properties(src/utils/ONNXLoader.cpp PROPERTIES
        COMPILE_OPTIONS "-Wno-unused-parameter")
endif()

add_dependencies(vkai vkai_shaders)

target_include_directories(vkai
    PUBLIC
        "${CMAKE_CURRENT_SOURCE_DIR}/include"
    PRIVATE
        "${CMAKE_CURRENT_BINARY_DIR}/shaders"
        "${CMAKE_SOURCE_DIR}/external/flatbuffers/include"
)

target_link_libraries(vkai
    PUBLIC
        vulkan
    PRIVATE
        vkai_flatbuffers
        onnx
)

add_dependencies(vkai GENERATE_vkai_flatbuffers)
