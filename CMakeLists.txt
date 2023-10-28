# Minimum CMake required
cmake_minimum_required(VERSION 3.26)

# Project for executing SN-Net onnx model with C++
project(snnet-onnx)

# Source files
set(SOURCE_FILES
    src/main.cpp
    src/stitch_config.cpp
    src/image_loader.cpp
    # src/test-vitlayers.cpp
    # src/test-stitchlayers.cpp
    # src/test-resnet50v2.cpp
)

# Needed for Java
set(CMAKE_C_STANDARD 99)

# Generating exe file named "snnet-onnx"
add_executable(snnet-onnx ${SOURCE_FILES})

# find_package(OpenCV REQUIRED)

# Include onnx header files
include_directories(
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/include/common
    ${PROJECT_SOURCE_DIR}/include/opencv2
    ${PROJECT_SOURCE_DIR}/include/snnet
    ${PROJECT_SOURCE_DIR}/include/onnxruntime/core/session
)

# Link directories
link_directories(${PROJECT_SOURCE_DIR}/lib/onnxruntime)
link_directories(${PROJECT_SOURCE_DIR}/lib/opencv)

# Link ONNX Runtime
target_link_libraries(snnet-onnx PRIVATE 
    ${PROJECT_SOURCE_DIR}/lib/onnxruntime/libonnxruntime.so
    ${PROJECT_SOURCE_DIR}/lib/opencv/libopencv_core.so
    ${PROJECT_SOURCE_DIR}/lib/opencv/libopencv_imgproc.so
    ${PROJECT_SOURCE_DIR}/lib/opencv/libopencv_highgui.so
    ${PROJECT_SOURCE_DIR}/lib/opencv/libopencv_imgcodecs.so
)
