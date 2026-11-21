# Puddle - the root build: strata-poly (CUDA or HIP), the libraries it links and its tests.  metal/ is its own
# project (macOS).  setup.py configures it and builds the strata-poly target.
cmake_minimum_required(VERSION 3.24)
project(strata VERSION 0.1.45 LANGUAGES CXX)
add_compile_definitions(STRATA_VERSION="${PROJECT_VERSION}")
# issue #31: release builds keep a PDB beside the exe (not shipped in it), so a minidump from a stalled engine can
# be read back to the exact source line
if(MSVC AND CMAKE_BUILD_TYPE STREQUAL "Release")
  add_compile_options($<$<COMPILE_LANGUAGE:CXX>:/Zi>)
  add_link_options(/DEBUG /OPT:REF /OPT:ICF)
endif()

# An engine for other PCs: ggml's CPU backend for any AVX2 CPU instead of this machine's, and the MSVC runtime linked
# in, so no Visual C++ redistributable is needed.
option(STRATA_PORTABLE "build for other PCs: AVX2 baseline, static MSVC runtime" OFF)
# EXPERIMENTAL, opt-in: an engine for CPUs without AVX2 (#394, #595, #623), compiled on that PC.  ggml-cpu, which then
# computes every CPU expert row, is built for this floor instead of AVX2 or the build host: "avx" (Sandy/Ivy Bridge:
# AVX, no FMA/F16C/AVX2) or "none" (SSE4.2 + POPCNT: Nehalem/Westmere).  Empty (the default) changes nothing.
set(STRATA_ISA_FLOOR "" CACHE STRING "experimental older-CPU build: empty (default, AVX2 floor), avx, or none (SSE4.2)")
set_property(CACHE STRATA_ISA_FLOOR PROPERTY STRINGS "" avx none)
if(NOT STRATA_ISA_FLOOR STREQUAL "" AND NOT STRATA_ISA_FLOOR STREQUAL "avx" AND NOT STRATA_ISA_FLOOR STREQUAL "none")
  message(FATAL_ERROR "STRATA_ISA_FLOOR must be empty, avx or none (got '${STRATA_ISA_FLOOR}')")
endif()
if(STRATA_ISA_FLOOR AND STRATA_PORTABLE)
  message(FATAL_ERROR "STRATA_ISA_FLOOR (an older-CPU build) and STRATA_PORTABLE (the AVX2 release build) exclude each other")
endif()
if(STRATA_PORTABLE AND MSVC)
  set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded")
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

option(STRATA_ENABLE_CUDA "Build the CUDA targets (NVIDIA)" OFF)
option(STRATA_ENABLE_HIP "Build the HIP targets for wave32 AMD GPUs (gfx1100, gfx1201; gfx1012/gfx1030 unvalidated)" OFF)
if(STRATA_ENABLE_CUDA AND STRATA_ENABLE_HIP)
  message(FATAL_ERROR "STRATA_ENABLE_CUDA and STRATA_ENABLE_HIP are separate backends; enable only one")
endif()
if(STRATA_ENABLE_HIP)
  include(cmake/hip_backend.cmake)
endif()
option(STRATA_BUILD_TESTS "Build the test suite" OFF)
option(STRATA_BUILD_CONVERSATION_TESTS "Build standalone conversation cache tests" OFF)
option(STRATA_WERROR "Treat compiler warnings as errors" OFF)
# ggml from llama.cpp (the CPU experts' dot products, the prompt path's MMQ kernels): strata-poly needs it
option(STRATA_NATIVE_EXPERTS "Native (IQ) expert support through ggml-cpu" ON)

if(STRATA_BUILD_CONVERSATION_TESTS)
  enable_testing()
  add_executable(conversation_memory_test src/core/conversation_memory_test.cpp src/core/conversation_memory.cpp)
  target_include_directories(conversation_memory_test PRIVATE include)
  add_test(NAME conversation_memory_test COMMAND conversation_memory_test)
  add_executable(glm_conversations_test src/poly/conversations_test.cpp src/poly/spill.cpp)
  add_test(NAME glm_conversations_test COMMAND glm_conversations_test)
  add_executable(poly_guesses_test src/poly/guesses_test.cpp src/spec/draft_policy.cpp)
  target_include_directories(poly_guesses_test PRIVATE include)
  add_test(NAME poly_guesses_test COMMAND poly_guesses_test)
endif()

if(STRATA_ENABLE_CUDA)
  # #585: CMake links CUDA code with the STATIC runtime by default, and the targets below also link CUDA::cudart (the
  # DLL's import library).  MSVC with a CUDA 12.x toolkit - the experimental Pascal/Volta build - then fails with
  # LNK1169 (every cudart symbol defined twice); GNU ld lets it pass.  That build links the shared runtime alone.  A
  # CUDA 13 build without STRATA_EXPERIMENTAL_SM60 is unchanged; -DCMAKE_CUDA_RUNTIME_LIBRARY overrides.
  if(MSVC AND STRATA_EXPERIMENTAL_SM60 AND NOT DEFINED CMAKE_CUDA_RUNTIME_LIBRARY)
    set(CMAKE_CUDA_RUNTIME_LIBRARY Shared)
  endif()
  enable_language(CUDA)
  # `enable_language(CUDA)` gives the .cu compile rules but NOT the imported targets, so `CUDA::cudart` has to
  # be brought in explicitly for the host-side tests that call cudaMalloc themselves.
  find_package(CUDAToolkit REQUIRED)
  option(STRATA_EXPERIMENTAL_SM60 "community build for GPUs below compute capability 7.5: Pascal sm_60, Volta sm_70 (#236)" OFF)
  if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 120 CACHE STRING "CUDA architectures" FORCE)
  endif()
  foreach(_a IN LISTS CMAKE_CUDA_ARCHITECTURES)
    string(REGEX REPLACE "-.*$" "" _base "${_a}")     # 120-real / 120-virtual are still 120
    # RTX 20 (75), 30 (86), 40 (89) and 50 (120) build; pre-sm_75 is refused below.
    if(_base STREQUAL "native" OR _base STREQUAL "all" OR _base STREQUAL "all-major")
      continue()
    endif()
    # STRATA_EXPERIMENTAL_SM60: a Pascal (sm_60, GP100-class) or Volta (sm_70, V100 / Titan V, #236) build.
    # These sit below every architecture the kernels were written for, so `__dp4a` (6.1+) is replaced below its
    # arch, bit-exact, in include/strata/kernels/dp4a.hpp.  Opt-in; RTX 20/30/40/50 builds do not need it.  It needs
    # a CUDA 12.x toolkit: CUDA 13 dropped sm_60 and sm_70 (nvcc fails CMake's compiler test with "compute_70").
    if(_base LESS 75 AND NOT STRATA_EXPERIMENTAL_SM60)
      message(FATAL_ERROR
        "Strata needs an NVIDIA GPU of compute capability 7.5 or newer (RTX 20 / 30 / 40 / 50 series), but "
        "CMAKE_CUDA_ARCHITECTURES is '${CMAKE_CUDA_ARCHITECTURES}'.  (A Pascal sm_60 or Volta sm_70 build is "
        "-DSTRATA_EXPERIMENTAL_SM60=ON; upstream does not support it.)")
    endif()
    if(_base LESS 60)
      message(FATAL_ERROR "CMAKE_CUDA_ARCHITECTURES '${CMAKE_CUDA_ARCHITECTURES}': even the experimental build "
                          "needs compute capability 6.0 (sm_60) or newer")
    endif()
    # #220 / #224: RTX 50 engines built with CUDA 12.8 crashed in the prompt path on Linux (13.0 and newer did not)
    if(_base GREATER_EQUAL 120 AND CMAKE_CUDA_COMPILER_VERSION VERSION_LESS 13.0)
      message(WARNING "Building for sm_${_base} (RTX 50) with CUDA ${CMAKE_CUDA_COMPILER_VERSION}: engines built with "
                      "CUDA 12.8 crashed on long prompts on these cards (issues #220, #224). Use CUDA 13.0 or newer.")
    endif()
  endforeach()
  if(STRATA_EXPERIMENTAL_SM60)
    # every target sees it: the Pascal / Volta code paths are compiled into this build only, so an sm_75+ build keeps
    # exactly the code it had
    add_compile_definitions(STRATA_EXPERIMENTAL_SM60=1)
  endif()
  message(STATUS "Strata: CUDA enabled, arch ${CMAKE_CUDA_ARCHITECTURES}")
  set(_strata_gpu_runtime_target CUDA::cudart)
  set(_strata_gpu_blas_target CUDA::cublas)
  set(_strata_gpu_include_directories ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
elseif(STRATA_ENABLE_HIP)
  set(_strata_gpu_runtime_target strata_hip_runtime)
  set(_strata_gpu_blas_target ${STRATA_HIP_BLAS_TARGETS})
  set(_strata_gpu_include_directories ${STRATA_HIP_COMPAT_INCLUDE_DIR})
endif()

# Warnings live on an INTERFACE target the project's own code links, NOT in add_compile_options: applied
# globally they would also hit FetchContent'd Catch2, whose -W4 warnings are not this project's to fix and
# whose /WX failures would be indistinguishable from our own.
#
# Every flag is guarded by COMPILE_LANGUAGE:CXX.  Without that guard nvcc receives bare `/W4 /utf-8` and dies
# with "A single input file is required for a non-link phase" - it parses them as inputs rather than host
# flags.  Host flags for CUDA must go through -Xcompiler, which is what COMPILE_LANGUAGE:CUDA does below.
add_library(strata_warnings INTERFACE)
target_compile_options(strata_warnings INTERFACE
  $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CXX_COMPILER_ID:MSVC>>:/W4>
  $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CXX_COMPILER_ID:MSVC>>:/utf-8>
  $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<NOT:$<CXX_COMPILER_ID:MSVC>>>:-Wall>
  $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<NOT:$<CXX_COMPILER_ID:MSVC>>>:-Wextra>
  $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CXX_COMPILER_ID:MSVC>>:-Xcompiler=/W4>
  $<$<COMPILE_LANGUAGE:HIP>:-Wall>
  $<$<COMPILE_LANGUAGE:HIP>:-Wextra>)
if(STRATA_WERROR)
  target_compile_options(strata_warnings INTERFACE
    $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CXX_COMPILER_ID:MSVC>>:/WX>
    $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<NOT:$<CXX_COMPILER_ID:MSVC>>>:-Werror>
    $<$<COMPILE_LANGUAGE:HIP>:-Werror>)
endif()

# ---- strata_artifact -------------------------------------------------------
# Header-only: the GGUF reader (include/strata/artifact) holds no state worth hiding.
add_library(strata_artifact INTERFACE)
target_include_directories(strata_artifact INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(strata_artifact INTERFACE strata_warnings)

if(STRATA_BUILD_TESTS)
  # the reader refuses a duplicate tensor name at open, naming it (a synthetic GGUF: no model, no GPU)
  add_executable(gguf_reader_test src/artifact/gguf_reader_test.cpp)
  target_link_libraries(gguf_reader_test PRIVATE strata_artifact)
  add_test(NAME gguf_reader_test COMMAND gguf_reader_test)
endif()

# ---- strata_core: the GPU's memory budget as Windows reports it (src/platform/memory.cpp) ----
# Only built when a GPU backend is on.  The selected runtime is PUBLIC: strata_kernels and its users take it from here.
if(STRATA_ENABLE_CUDA OR STRATA_ENABLE_HIP)
  add_library(strata_core STATIC src/platform/memory.cpp)
  target_include_directories(strata_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
  target_link_libraries(strata_core PUBLIC strata_warnings ${_strata_gpu_runtime_target})

  # ---- strata_kernels: the decode matrix-vector products (native_mmvq.cu, and iq_kernels.cu for the formats it
  # hands over) ----
  add_library(strata_kernels STATIC src/kernels/cuda/native_mmvq.cu src/kernels/cuda/iq_kernels.cu)
  target_link_libraries(strata_kernels PUBLIC strata_core strata_artifact)
  # llama.cpp's block layouts and codebook grids for the i-quant formats (MIT, third_party/ggml)
  target_include_directories(strata_kernels PRIVATE third_party/ggml)
  # Match the pinned llama.cpp CUDA arithmetic contract for this adapter only.
  if(STRATA_ENABLE_CUDA)
    set_source_files_properties(src/kernels/cuda/native_mmvq.cu
      PROPERTIES COMPILE_OPTIONS "--use_fast_math")
  endif()

  # available_host_bytes(): MemAvailable, cgroup limits and the operator's --memory-limit-mib cap on fixture directories
  # - no GPU, no CUDA
  if(STRATA_BUILD_TESTS AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/core/memory_guard_test.cpp")
    add_executable(memory_guard_test tests/core/memory_guard_test.cpp src/core/conversation_memory.cpp)
    target_include_directories(memory_guard_test PRIVATE include)
    add_test(NAME memory_guard_test COMMAND memory_guard_test)
    set_tests_properties(memory_guard_test PROPERTIES TIMEOUT 60)
  endif()
endif()

# The draft machinery's host side (suffix drafter, draft policy, controller): no GPU.
add_library(strata_spec STATIC src/spec/suffix_drafter.cpp src/spec/controller.cpp src/spec/draft_policy.cpp
            src/spec/draft_source.cpp)
target_include_directories(strata_spec PUBLIC include)
if(STRATA_BUILD_TESTS)
  add_executable(suffix_drafter_test src/spec/suffix_drafter_test.cpp)
  target_link_libraries(suffix_drafter_test PRIVATE strata_spec)
  add_test(NAME suffix_drafter_test COMMAND suffix_drafter_test)
  add_executable(controller_test src/spec/controller_test.cpp)
  target_link_libraries(controller_test PRIVATE strata_spec)
  add_test(NAME controller_test COMMAND controller_test)
  add_executable(draft_policy_test src/spec/draft_policy_test.cpp)
  target_link_libraries(draft_policy_test PRIVATE strata_spec)
  add_test(NAME draft_policy_test COMMAND draft_policy_test)
endif()

find_package(Threads REQUIRED)

# ---- ggml from llama.cpp: the CPU experts and the GGUF block formats ----
#
# The CPU computes a native expert with ggml-cpu's own dot products (`ggml_get_type_traits_cpu`), so an IQ
# weight means exactly what it means in llama.cpp.  ggml is built static from a llama.cpp checkout: a local one
# with -DSTRATA_GGML_DIR=..., otherwise the pinned upstream commit (third_party/ggml/VERSION.txt).
set(STRATA_GGML_DIR "" CACHE PATH "llama.cpp checkout to take ggml from (empty: fetch the pinned commit)")
if(STRATA_NATIVE_EXPERTS)
  enable_language(C)
  set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
  set(GGML_STATIC ON CACHE BOOL "" FORCE)
  set(GGML_CUDA OFF CACHE BOOL "" FORCE)
  set(GGML_OPENMP OFF CACHE BOOL "" FORCE)
  set(GGML_BACKEND_DL OFF CACHE BOOL "" FORCE)
  set(GGML_BUILD_TESTS OFF CACHE BOOL "" FORCE)
  set(GGML_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
  if(STRATA_PORTABLE)
    set(GGML_NATIVE OFF CACHE BOOL "" FORCE)
    foreach(_f GGML_AVX GGML_AVX2 GGML_FMA GGML_F16C GGML_BMI2)
      set(${_f} ON CACHE BOOL "" FORCE)
    endforeach()
  elseif(STRATA_ISA_FLOOR)
    # the experimental older-CPU build (STRATA_ISA_FLOOR above): one ggml-cpu for the floor, ggml's "sandybridge"
    # feature set (SSE4.2 + AVX) or SSE4.2 alone - not the build host's, which may be newer than the PC it runs on.
    # From the Strata_Dirigo fork (rwkeyes).  ggml's defaults turn the rest on when GGML_NATIVE is off: all off here.
    set(GGML_NATIVE OFF CACHE BOOL "" FORCE)
    set(GGML_CPU_ALL_VARIANTS OFF CACHE BOOL "" FORCE)
    set(GGML_SSE42 ON CACHE BOOL "" FORCE)
    if(STRATA_ISA_FLOOR STREQUAL "avx")
      set(GGML_AVX ON CACHE BOOL "" FORCE)
    else()
      set(GGML_AVX OFF CACHE BOOL "" FORCE)
    endif()
    foreach(_f GGML_AVX2 GGML_FMA GGML_F16C GGML_BMI2 GGML_AVX_VNNI GGML_AVX512 GGML_AVX512_VBMI GGML_AVX512_VNNI
               GGML_AVX512_BF16 GGML_AMX_TILE GGML_AMX_INT8 GGML_AMX_BF16)
      set(${_f} OFF CACHE BOOL "" FORCE)
    endforeach()
  else()
    set(GGML_NATIVE ON CACHE BOOL "" FORCE)
  endif()
  if(STRATA_GGML_DIR)
    add_subdirectory(${STRATA_GGML_DIR}/ggml ${CMAKE_BINARY_DIR}/ggml EXCLUDE_FROM_ALL)
  else()
    include(FetchContent)
    FetchContent_Declare(strata_llamacpp
      GIT_REPOSITORY https://github.com/ggml-org/llama.cpp.git
      GIT_TAG 3cf03257f219afbe7334045ff7c6a06ac68c627d
      GIT_SHALLOW FALSE)
    FetchContent_GetProperties(strata_llamacpp)
    if(NOT strata_llamacpp_POPULATED)
      FetchContent_Populate(strata_llamacpp)
    endif()
    add_subdirectory(${strata_llamacpp_SOURCE_DIR}/ggml ${CMAKE_BINARY_DIR}/ggml EXCLUDE_FROM_ALL)
  endif()
endif()

# The opt-in lifecycle is a host-only test, including on non-CUDA builds.
if(STRATA_BUILD_TESTS)
  add_executable(poly_hip_value_split_policy_test src/poly/hip_value_split_policy_test.cpp)
  add_test(NAME poly_hip_value_split_policy_test COMMAND poly_hip_value_split_policy_test)
  add_executable(poly_gdn_mma_control_test src/poly/gdn_mma_control_test.cpp)
  add_test(NAME poly_gdn_mma_control_test COMMAND poly_gdn_mma_control_test)
  add_executable(poly_stream_stage_test src/poly/stage_copy_test.cpp)
  add_test(NAME poly_stream_stage_test COMMAND poly_stream_stage_test)
endif()

# The prepared-input GDN recurrence of qwen35moe's linear-attention layers (CUDA). Keep FP32/BF16 hi+lo math
# separate from the MMQ targets' -use_fast_math compilation options.
if(STRATA_NATIVE_EXPERTS AND STRATA_ENABLE_CUDA)
  if(STRATA_GGML_DIR)
    set(_strata_gdn_ggml ${STRATA_GGML_DIR}/ggml)
  else()
    set(_strata_gdn_ggml ${strata_llamacpp_SOURCE_DIR}/ggml)
  endif()
  add_library(strata_gdn_mma STATIC src/kernels/cuda/gdn_mma.cu)
  target_include_directories(strata_gdn_mma PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include
    PRIVATE ${_strata_gdn_ggml}/include ${_strata_gdn_ggml}/src ${_strata_gdn_ggml}/src/ggml-cuda)
  target_link_libraries(strata_gdn_mma PRIVATE ggml-base ${_strata_gpu_runtime_target})
  if(STRATA_BUILD_TESTS)
    add_executable(poly_gdn_mma_policy_test src/kernels/gdn_mma_policy_test.cpp)
    target_include_directories(poly_gdn_mma_policy_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
    add_test(NAME poly_gdn_mma_policy_test COMMAND poly_gdn_mma_policy_test)
    add_executable(gdn_mma_test src/kernels/gdn_mma_test.cu)
    target_link_libraries(gdn_mma_test PRIVATE strata_gdn_mma ${_strata_gpu_runtime_target})
    add_test(NAME gdn_mma_test COMMAND gdn_mma_test)
    set_tests_properties(gdn_mma_test PROPERTIES SKIP_RETURN_CODE 77)
  endif()
endif()

# ---- strata-poly: the engine (qwen35moe, gemma4, laguna), read from the model's GGUF ----
#
# It takes the decode products from strata_kernels, the prompt products from llama.cpp's MMQ kernels
# (strata_poly_mmq) and the CPU experts' dot products from ggml-cpu.  On HIP it is the same CUDA-shaped code through
# the hip_compat headers (the wave32 RDNA backend).
if(STRATA_NATIVE_EXPERTS AND (STRATA_ENABLE_CUDA OR STRATA_ENABLE_HIP))
  # The prompt path's quantized products: llama.cpp's MMQ kernels (ggml-cuda, MIT) for the types the model's
  # GGUFs keep their matrices in (the list in src/poly/mmq.cu), from the llama.cpp checkout ggml comes from.
  if(STRATA_GGML_DIR)
    set(_strata_gg ${STRATA_GGML_DIR}/ggml)
  else()
    set(_strata_gg ${strata_llamacpp_SOURCE_DIR}/ggml)
  endif()
  set(_strata_mmq ${_strata_gg}/src/ggml-cuda)
  set(_strata_poly_mmq_instances "")
  foreach(_t q4_0 q4_1 q5_0 q5_1 q8_0 q2_k q3_k q4_k q5_k q6_k iq1_s iq2_xxs iq2_xs iq2_s iq3_xxs iq3_s iq4_nl
             iq4_xs)
    list(APPEND _strata_poly_mmq_instances ${_strata_mmq}/template-instances/mmq-instance-${_t}.cu)
  endforeach()
  # MXFP4 and NVFP4: on RTX 50 llama.cpp's tiles use the FP4 MMA with block scales, which exists only in the
  # arch-specific target, so on CUDA their instances alone are built with 12x -> 12xa.  A
  # Blackwell other than 12.0 has no image of them, and Mmq::fits() leaves MXFP4 out there.  NVFP4 takes this
  # instance only on a 12.0 card with STRATA_PREFILL_NVFP4=w4a4; otherwise its tiles are mmq_nvfp4.cu's int8 ones,
  # built for the architectures as given.  On HIP there is no NVFP4 prompt product.
  set(_strata_poly_mxfp4 ${_strata_mmq}/template-instances/mmq-instance-mxfp4.cu)
  if(STRATA_ENABLE_HIP)
    list(APPEND _strata_poly_mmq_instances ${_strata_poly_mxfp4})
  else()
    list(APPEND _strata_poly_mxfp4 ${_strata_mmq}/template-instances/mmq-instance-nvfp4.cu)
    list(APPEND _strata_poly_mmq_instances src/poly/mmq_nvfp4.cu)
    set(_strata_poly_fp4_archs ${CMAKE_CUDA_ARCHITECTURES})
    list(TRANSFORM _strata_poly_fp4_archs REPLACE "^(12[0-9])$" "\\1a")
    list(TRANSFORM _strata_poly_fp4_archs REPLACE "^(12[0-9])-(real|virtual)$" "\\1a-\\2")
    add_library(strata_poly_mmq_mxfp4 OBJECT ${_strata_poly_mxfp4})
    set_target_properties(strata_poly_mmq_mxfp4 PROPERTIES CUDA_ARCHITECTURES "${_strata_poly_fp4_archs}")
    target_include_directories(strata_poly_mmq_mxfp4 PRIVATE ${_strata_gg}/include ${_strata_gg}/src ${_strata_mmq})
    target_compile_options(strata_poly_mmq_mxfp4 PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda -use_fast_math>)
    if(MSVC)
      target_compile_options(strata_poly_mmq_mxfp4 PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/bigobj>)
    endif()
    list(APPEND _strata_poly_mmq_instances $<TARGET_OBJECTS:strata_poly_mmq_mxfp4>)
  endif()
  if(STRATA_ENABLE_HIP)
    # CMake does not infer HIP from the .cu suffixes outside src/ (and drops a .cu source it has no language
    # for), and ggml's MMQ sources take its native HIP path (vendors/hip.h) instead of the hip_compat headers,
    # as strata_mmq's does.
    set_source_files_properties(${_strata_mmq}/quantize.cu ${_strata_poly_mmq_instances} PROPERTIES LANGUAGE HIP)
  endif()
  add_library(strata_poly_mmq STATIC src/poly/mmq.cu src/poly/fattn.cu src/prefill/ggml_cuda_host.cu ${_strata_mmq}/quantize.cu
              ${_strata_poly_mmq_instances})
  target_include_directories(strata_poly_mmq PRIVATE ${_strata_gg}/include ${_strata_gg}/src ${_strata_mmq}
                                                    ${CMAKE_CURRENT_SOURCE_DIR}/include)
  target_compile_options(strata_poly_mmq PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda -use_fast_math>)
  if(MSVC)
    target_compile_options(strata_poly_mmq PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/bigobj>)
  endif()
  if(STRATA_ENABLE_HIP)
    target_compile_definitions(strata_poly_mmq PRIVATE GGML_USE_HIP=1)
    target_link_libraries(strata_poly_mmq PUBLIC ggml-base hip::host)
  else()
    target_compile_definitions(strata_poly_mmq PRIVATE STRATA_POLY_NVFP4=1)
    target_link_libraries(strata_poly_mmq PUBLIC ggml-base ${_strata_gpu_runtime_target})
  endif()

  if(STRATA_ENABLE_CUDA)
    target_compile_definitions(strata_poly_mmq PUBLIC STRATA_POLY_GDN_MMA=1)
    target_link_libraries(strata_poly_mmq PUBLIC strata_gdn_mma)

    add_executable(poly_gdn_mma_test src/poly/gdn_mma_test.cu src/poly/kernels.cu)
    target_include_directories(poly_gdn_mma_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
    target_link_libraries(poly_gdn_mma_test PRIVATE strata_poly_mmq ${_strata_gpu_blas_target} ${_strata_gpu_runtime_target})
    add_test(NAME poly_gdn_mma_test COMMAND poly_gdn_mma_test)
    set_tests_properties(poly_gdn_mma_test PROPERTIES SKIP_RETURN_CODE 77)
    # Optional model/state check; run by hand with an MTP GGUF and frozen prompt IDs.
    add_executable(poly_gdn_mma_state_test src/poly/gdn_mma_state_test.cpp src/poly/engine.cpp
                   src/poly/qwen35.cpp src/poly/gemma4.cpp src/poly/laguna.cpp src/poly/cpu_experts.cpp
                   src/poly/kernels.cu src/core/conversation_memory.cpp)
    target_include_directories(poly_gdn_mma_state_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include ${_strata_gpu_include_directories})
    target_link_libraries(poly_gdn_mma_state_test PRIVATE strata_poly_mmq strata_kernels strata_spec ggml-cpu ggml-base
                         ${_strata_gpu_blas_target} ${_strata_gpu_runtime_target} CUDA::cuda_driver Threads::Threads)
  endif()

  add_executable(strata-poly src/poly/main.cpp src/poly/engine.cpp src/poly/qwen35.cpp src/poly/gemma4.cpp
                            src/poly/laguna.cpp src/poly/cpu_experts.cpp src/poly/kernels.cu src/poly/spill.cpp
                            src/core/conversation_memory.cpp)
  target_include_directories(strata-poly PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include ${_strata_gpu_include_directories})
  # the driver's virtual memory calls: the attention caches grow with the conversation; on HIP they are HIP's
  # memory management calls, which the HIP runtime itself provides (hip_compat/cuda.h)
  if(STRATA_ENABLE_HIP)
    set(_strata_poly_driver "")
  else()
    set(_strata_poly_driver CUDA::cuda_driver)
  endif()
  target_link_libraries(strata-poly PRIVATE strata_poly_mmq strata_kernels strata_spec ggml-cpu ggml-base
                                           ${_strata_gpu_blas_target} ${_strata_gpu_runtime_target}
                                           ${_strata_poly_driver}
                                           Threads::Threads)
  # needs the model file, so it is run by hand: poly_expert_parity MODEL.gguf
  add_executable(poly_expert_parity src/poly/expert_parity.cpp src/poly/cpu_experts.cpp src/poly/kernels.cu)
  target_include_directories(poly_expert_parity PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include
                                                       ${_strata_gpu_include_directories})
  target_link_libraries(poly_expert_parity PRIVATE strata_poly_mmq strata_kernels ggml-cpu ggml-base
                                                  ${_strata_gpu_blas_target} ${_strata_gpu_runtime_target}
                                                  Threads::Threads)
  # a read with the draft block repeats bit for bit; needs the model file, so it is run by hand:
  # poly_draft_race_test MODEL.gguf [RUNS] [LAYER_SPLIT]
  add_executable(poly_draft_race_test src/poly/draft_race_test.cpp src/poly/engine.cpp
                                      src/poly/qwen35.cpp src/poly/gemma4.cpp
                                      src/poly/laguna.cpp src/poly/cpu_experts.cpp src/poly/kernels.cu
                                      src/core/conversation_memory.cpp)
  target_include_directories(poly_draft_race_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include
                                                         ${_strata_gpu_include_directories})
  target_link_libraries(poly_draft_race_test PRIVATE strata_poly_mmq strata_kernels strata_spec ggml-cpu ggml-base
                                                    ${_strata_gpu_blas_target} ${_strata_gpu_runtime_target}
                                                    ${_strata_poly_driver} Threads::Threads)
  # a prompt batch's routing on the card against the hosts' routing; needs a card, so it is run by hand
  add_executable(poly_route_test src/poly/route_test.cu src/poly/kernels.cu)
  target_include_directories(poly_route_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include
                                                    ${_strata_gpu_include_directories})
  target_link_libraries(poly_route_test PRIVATE ${_strata_gpu_blas_target} ${_strata_gpu_runtime_target})
  # the fused residual sum, RMS norm and Q8_1 quantization against the three launches; needs a card, run by hand
  add_executable(poly_norm_q8_test src/poly/norm_q8_test.cu src/poly/kernels.cu)
  target_include_directories(poly_norm_q8_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include
                                                      ${_strata_gpu_include_directories})
  target_link_libraries(poly_norm_q8_test PRIVATE strata_kernels ${_strata_gpu_blas_target}
                                                  ${_strata_gpu_runtime_target})
  # a prompt batch's products with few outputs against the same sums on the host; needs a card, run by hand
  add_executable(poly_gemm_small_test src/poly/gemm_small_test.cu src/poly/kernels.cu)
  target_include_directories(poly_gemm_small_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include
                                                         ${_strata_gpu_include_directories})
  target_link_libraries(poly_gemm_small_test PRIVATE ${_strata_gpu_blas_target} ${_strata_gpu_runtime_target})
  # attend()'s blocks of rows against its other kernels and itself; needs a card, so it is run by hand
  add_executable(poly_attend_test src/poly/attend_test.cu src/poly/kernels.cu)
  target_include_directories(poly_attend_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include
                                                     ${_strata_gpu_include_directories})
  target_link_libraries(poly_attend_test PRIVATE ${_strata_gpu_blas_target} ${_strata_gpu_runtime_target})
  # attend_group() for Laguna's full layers (heads of 128, 6 a group, fp16 and q8_0) against attend(), run by hand
  add_executable(poly_laguna_attend_test src/poly/laguna_attend_test.cu src/poly/kernels.cu)
  target_include_directories(poly_laguna_attend_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include
                                                            ${_strata_gpu_include_directories})
  target_link_libraries(poly_laguna_attend_test PRIVATE ${_strata_gpu_blas_target} ${_strata_gpu_runtime_target})
  add_executable(poly_hip_value_split_test src/poly/hip_value_split_test.cu src/poly/kernels.cu)
  target_include_directories(poly_hip_value_split_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include
                            ${_strata_gpu_include_directories})
  target_link_libraries(poly_hip_value_split_test PRIVATE ${_strata_gpu_blas_target} ${_strata_gpu_runtime_target})
  add_test(NAME poly_hip_value_split_test COMMAND poly_hip_value_split_test)
  set_tests_properties(poly_hip_value_split_test PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 240)
  # STRATA_HIP_SPLIT_ATTEND (heads of 256 over a q8_0 cache): against an FP64 reference and the old kernels, the position on the card, the
  # fallback and a graph replay; needs a card with 32-lane waves (77 = skipped)
  add_executable(poly_hip_split_attend_test src/poly/hip_split_attend_test.cu src/poly/kernels.cu)
  target_include_directories(poly_hip_split_attend_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include
                            ${_strata_gpu_include_directories})
  target_link_libraries(poly_hip_split_attend_test PRIVATE ${_strata_gpu_blas_target} ${_strata_gpu_runtime_target})
  add_test(NAME poly_hip_split_attend_test COMMAND poly_hip_split_attend_test)
  set_tests_properties(poly_hip_split_attend_test PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 600)
  # attend_batch() against attend() with Flint's heads, sinks and window ring; needs a card
  add_executable(poly_sink_attend_test src/poly/sink_attend_test.cu src/poly/kernels.cu)
  target_include_directories(poly_sink_attend_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include
                                                          ${_strata_gpu_include_directories})
  target_link_libraries(poly_sink_attend_test PRIVATE ${_strata_gpu_blas_target} ${_strata_gpu_runtime_target})
  # STRATA_HIP_PREFILL_FP32 (attend_batch() on FP32 inputs over a q8_0 cache): against an FP64 reference and the default path, and the
  # fallbacks; needs a card (77 = skipped)
  add_executable(poly_hip_prefill_fp32_test src/poly/hip_prefill_fp32_test.cu src/poly/kernels.cu)
  target_include_directories(poly_hip_prefill_fp32_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include
                            ${_strata_gpu_include_directories})
  target_link_libraries(poly_hip_prefill_fp32_test PRIVATE ${_strata_gpu_blas_target} ${_strata_gpu_runtime_target})
  add_test(NAME poly_hip_prefill_fp32_test COMMAND poly_hip_prefill_fp32_test)
  set_tests_properties(poly_hip_prefill_fp32_test PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 600)
  # fattn_q8() and fattn_f16() (llama.cpp's flash-attention kernel) against attend_batch(), and with `dec` fdec_q8()
  # against attend(); needs a card, run by hand
  add_executable(poly_fattn_test src/poly/fattn_test.cu src/poly/kernels.cu)
  target_include_directories(poly_fattn_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include ${CMAKE_CURRENT_SOURCE_DIR}/src/poly
                                                    ${_strata_gpu_include_directories})
  target_link_libraries(poly_fattn_test PRIVATE strata_poly_mmq ${_strata_gpu_blas_target} ${_strata_gpu_runtime_target})
  # STRATA_POLY_STAGED_TILES: the staged experts in size classes, each a launch of its own tile width, against one
  # launch, bit for bit; needs a card
  add_executable(poly_mmq_tiles_test src/poly/mmq_tiles_test.cu)
  target_include_directories(poly_mmq_tiles_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include ${CMAKE_CURRENT_SOURCE_DIR}/src/poly
                                                        ${_strata_gpu_include_directories})
  target_link_libraries(poly_mmq_tiles_test PRIVATE strata_poly_mmq ${_strata_gpu_blas_target} ${_strata_gpu_runtime_target})
  add_test(NAME poly_mmq_tiles_test COMMAND poly_mmq_tiles_test)
  set_tests_properties(poly_mmq_tiles_test PROPERTIES SKIP_RETURN_CODE 77)
  # the pool of the CPU experts, without a model
  add_executable(poly_workers_test src/poly/workers_test.cpp src/poly/cpu_experts.cpp)
  target_include_directories(poly_workers_test PRIVATE ${_strata_gg}/include)
  target_link_libraries(poly_workers_test PRIVATE ggml-cpu ggml-base Threads::Threads)
  add_test(NAME poly_workers_test COMMAND poly_workers_test)
  # the card's MXFP4 product against ggml's dequantizer on random blocks, without a model; needs a card, run by hand
  add_executable(poly_mxfp4_test src/poly/mxfp4_test.cu)
  target_include_directories(poly_mxfp4_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include ${_strata_gg}/include
                                                    ${_strata_gpu_include_directories})
  target_link_libraries(poly_mxfp4_test PRIVATE strata_kernels ggml-base ${_strata_gpu_runtime_target})
  # the Q8_0 KV cache kernels (store_q8 / store_q8_at) against a host reference (GPU, synthetic, no model)
  if(STRATA_BUILD_TESTS)
    add_executable(poly_kv_q8_test src/poly/kv_q8_test.cpp src/poly/kernels.cu)
    target_include_directories(poly_kv_q8_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include
                                                       ${_strata_gpu_include_directories})
    target_link_libraries(poly_kv_q8_test PRIVATE ${_strata_gpu_blas_target} ${_strata_gpu_runtime_target}
                                                  Threads::Threads)
    add_test(NAME poly_kv_q8_test COMMAND poly_kv_q8_test)
  endif()
  # --kv int8/int8k's kernels (store_int8 / attend_int8) against ggml's own quantizer and against attend() over the
  # same numbers in fp16.  The reference is ggml-cpu's quantize_row_q8_0_ref, hence the ggml links; under HIP the
  # mma tier's probes are behind STRATA_USE_HIP (the dp4a tier is what runs there anyway).
  if(STRATA_BUILD_TESTS)
    add_executable(poly_kv_int8_test src/poly/kv_int8_test.cpp src/poly/kernels.cu)
    target_include_directories(poly_kv_int8_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include
                                                      ${_strata_gpu_include_directories} ${_strata_gg}/include)
    target_link_libraries(poly_kv_int8_test PRIVATE ${_strata_gpu_blas_target} ${_strata_gpu_runtime_target}
                                                  ggml-cpu ggml-base Threads::Threads)
    add_test(NAME poly_kv_int8_test COMMAND poly_kv_int8_test)
  endif()
  # the Mamba-2 mixer's kernels (Nemotron-H) against a host copy of llama.cpp's arithmetic (GPU, synthetic, no model)
  if(STRATA_BUILD_TESTS)
    add_executable(poly_mamba2_test src/poly/mamba2_test.cu src/poly/mamba2.cu)
    target_include_directories(poly_mamba2_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include
                                                      ${_strata_gpu_include_directories})
    target_link_libraries(poly_mamba2_test PRIVATE ${_strata_gpu_runtime_target})
    add_test(NAME poly_mamba2_test COMMAND poly_mamba2_test)
    set_tests_properties(poly_mamba2_test PROPERTIES SKIP_RETURN_CODE 77)
  endif()
endif()

if(STRATA_BUILD_TESTS)
  enable_testing()
endif()

