cmake_minimum_required(VERSION 3.18)
project(rusttorch_native_probes LANGUAGES CXX)

find_package(Torch 2.13.0 EXACT REQUIRED)

add_executable(runtime_probe runtime_probe.cpp)
target_compile_features(runtime_probe PRIVATE cxx_std_20)
target_link_libraries(runtime_probe PRIVATE ${TORCH_LIBRARIES})

enable_testing()
add_test(NAME native_cpu COMMAND runtime_probe)

add_executable(stable_abi_probe stable_abi_probe.cpp)
target_compile_features(stable_abi_probe PRIVATE cxx_std_20)
target_link_libraries(stable_abi_probe PRIVATE ${TORCH_LIBRARIES})
add_test(NAME stable_abi_metadata COMMAND stable_abi_probe)

set(STREAM_DEVICE "" CACHE STRING "An available accelerator, e.g. mps:0 or cuda:0")
if(STREAM_DEVICE)
  add_test(NAME native_stream COMMAND runtime_probe "${STREAM_DEVICE}")
endif()

set(AOTI_PACKAGE "" CACHE FILEPATH "Locally exported x.square() CPU model package")
if(AOTI_PACKAGE)
  add_executable(aoti_probe aoti_probe.cpp)
  target_compile_features(aoti_probe PRIVATE cxx_std_20)
  target_link_libraries(aoti_probe PRIVATE ${TORCH_LIBRARIES})
  add_test(NAME compiled_model COMMAND aoti_probe "${AOTI_PACKAGE}")
endif()

set(GLOO_INCLUDE_DIR "" CACHE PATH "Root of the Gloo headers matching this LibTorch build")
if(GLOO_INCLUDE_DIR)
  find_package(Python3 COMPONENTS Interpreter REQUIRED)
  add_executable(gloo_probe gloo_probe.cpp)
  target_compile_features(gloo_probe PRIVATE cxx_std_20)
  target_compile_definitions(gloo_probe PRIVATE USE_C10D_GLOO)
  target_include_directories(gloo_probe PRIVATE "${GLOO_INCLUDE_DIR}")
  target_link_libraries(gloo_probe PRIVATE ${TORCH_LIBRARIES})
  add_test(NAME native_gloo COMMAND "${Python3_EXECUTABLE}"
    "${CMAKE_CURRENT_SOURCE_DIR}/run_two_ranks.py" "$<TARGET_FILE:gloo_probe>")
endif()
