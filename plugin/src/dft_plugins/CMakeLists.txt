cmake_minimum_required(VERSION 3.20)

set(PROJECT_NAME trt_dft_plugins)
project(${PROJECT_NAME} LANGUAGES CXX)

find_package(CUDAToolkit REQUIRED)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(MSVC)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /W3 /O2")
else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -O3 -fPIC")
endif()

file(GLOB ${PROJECT_NAME}_sources ./*.cpp)

add_library(${PROJECT_NAME} SHARED ${${PROJECT_NAME}_sources})

# Locate TensorRT headers and libraries
if(DEFINED ENV{TENSORRT_DIR})
    set(TRT_ROOT "$ENV{TENSORRT_DIR}")
elseif(DEFINED ENV{TENSORRT_PATH})
    set(TRT_ROOT "$ENV{TENSORRT_PATH}")
elseif(EXISTS "G:/TensorRT")
    set(TRT_ROOT "G:/TensorRT")
endif()

if(TRT_ROOT)
    target_include_directories(${PROJECT_NAME} PRIVATE "${TRT_ROOT}/include")
    target_link_directories(${PROJECT_NAME} PRIVATE "${TRT_ROOT}/lib")
endif()

# Find nvinfer library
find_library(NVINFER_LIB NAMES nvinfer_10 nvinfer
    HINTS ${TRT_ROOT}/lib
    REQUIRED
)

target_link_libraries(${PROJECT_NAME} PRIVATE
    CUDA::cufft
    CUDA::cublas
    ${NVINFER_LIB}
)

if(WIN32)
    set_target_properties(${PROJECT_NAME} PROPERTIES
        OUTPUT_NAME "trt_dft_plugins"
        PREFIX ""
    )
else()
    set_target_properties(${PROJECT_NAME} PROPERTIES
        OUTPUT_NAME "trt_dft_plugins"
    )
endif()
