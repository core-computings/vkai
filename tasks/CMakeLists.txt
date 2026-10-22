# Each network has its own portable C++ test entry point, including GPU timing.
add_executable(vkai_resnet18_test ResNet18Test.cpp)
add_executable(vkai_unet_test UNetTest.cpp)
add_executable(vkai_mobilenet_v2_test MobileNetV2Test.cpp)
add_executable(vkai_yolov8n_test YOLOv8nTest.cpp)
add_executable(vkai_dncnn_test DnCNNTest.cpp)
add_executable(vkai_lraspp_test LRASPPTest.cpp)
foreach(model IN ITEMS yolov8n dncnn lraspp)
  target_include_directories(vkai_${model}_test SYSTEM PRIVATE
    "${CMAKE_SOURCE_DIR}/external/CORE/external/stb")
endforeach()

foreach(model IN ITEMS resnet18 unet mobilenet_v2 yolov8n dncnn lraspp)
  target_link_libraries(vkai_${model}_test PRIVATE vkai)
endforeach()

if(UNIX)
  add_executable(vkai_memory_benchmark MemoryBenchmark.cpp)
  target_link_libraries(vkai_memory_benchmark PRIVATE vkai ${CMAKE_DL_LIBS})
endif()
