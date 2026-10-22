# Each network has its own portable C++ test entry point, including GPU timing.
add_executable(vkai_resnet18_test ResNet18Test.cpp)
add_executable(vkai_unet_test UNetTest.cpp)
add_executable(vkai_mobilenet_v2_test MobileNetV2Test.cpp)
add_executable(vkai_yolov8n_test YOLOv8nTest.cpp)
target_include_directories(vkai_yolov8n_test SYSTEM PRIVATE
  "${CMAKE_SOURCE_DIR}/external/CORE/external/stb")

foreach(model IN ITEMS resnet18 unet mobilenet_v2 yolov8n)
  target_link_libraries(vkai_${model}_test PRIVATE vkai)
endforeach()
