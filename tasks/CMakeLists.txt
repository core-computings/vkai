# A portable command-line runner; no Python or image libraries on the device.
add_executable(vkai_infer tools/Infer.cpp)
target_link_libraries(vkai_infer PRIVATE vkai)

add_executable(vkai_benchmark tools/Benchmark.cpp)
target_link_libraries(vkai_benchmark PRIVATE vkai)

add_executable(vkai_detect tools/Detect.cpp)
target_link_libraries(vkai_detect PRIVATE vkai)
