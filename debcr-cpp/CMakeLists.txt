cmake_minimum_required(VERSION 3.20)
project(debcr-cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(onnxruntime CONFIG QUIET)

add_executable(debcr-cpp src/main.cpp)

if(onnxruntime_FOUND)
  target_link_libraries(debcr-cpp PRIVATE onnxruntime::onnxruntime)
else()
  if(NOT DEFINED ENV{ONNXRUNTIME_ROOT})
    message(FATAL_ERROR "Set ONNXRUNTIME_ROOT or provide an onnxruntime CMake package.")
  endif()
  target_include_directories(debcr-cpp PRIVATE "$ENV{ONNXRUNTIME_ROOT}/include")
  if(WIN32)
    target_link_libraries(debcr-cpp PRIVATE "$ENV{ONNXRUNTIME_ROOT}/lib/onnxruntime.lib")
  else()
    target_link_libraries(debcr-cpp PRIVATE "$ENV{ONNXRUNTIME_ROOT}/lib/libonnxruntime.so")
  endif()
endif()
