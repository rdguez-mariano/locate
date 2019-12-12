project(DeepAsift)
CMAKE_MINIMUM_REQUIRED(VERSION 2.6)

set(CMAKE_BUILD_TYPE Release)

set(CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake" ${CMAKE_MODULE_PATH})

# AVX does not do much benefit at the moment, but breaks compilation on some platforms.
# Disabled for now until AVX becomes important enough to reconsider.
SET(EMBREE_MAX_ISA "SSE4.2" CACHE STRING "Selects highest ISA to support.")
set(USE_AVX FALSE CACHE BOOL "Use AVX.")
set(USE_AVX2 FALSE CACHE BOOL "Use AVX2.")


include(OptimizeForArchitecture)
OptimizeForArchitecture()

if (USE_AVX)
    message(STATUS "Compiling with AVX support")
    set(__AVX__ 1)
elseif (USE_SSE4_2)
    message(STATUS "Compiling with SSE4.2 support")
elseif (USE_SSSE3)
    message(STATUS "Compiling with SSE3 support")
else()
    message(FATAL_ERROR "The target machine does not support SSE3. At least SSE3 is required")
endif()

foreach(flag ${Vc_ARCHITECTURE_FLAGS})
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${flag}")
endforeach()
message("Optimized Flags for this machine: ${CMAKE_CXX_FLAGS}")

FIND_PACKAGE(OpenMP)
if (OPENMP_FOUND)
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif (OPENMP_FOUND)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -fPIC")

ADD_SUBDIRECTORY(cpp_libs)
ADD_SUBDIRECTORY(cpp_libs/libNumerics)
ADD_SUBDIRECTORY(cpp_libs/libOrsa)
ADD_SUBDIRECTORY(cpp_libs/vl)


# ACTIVATE USAC - needs libconfig and lapack
set(USAC ON)
set(VLFEAT ON)

set(src
    libDA.cpp

    cpp_libs/ACmatch.cpp

    #ORSA
    cpp_libs/libOrsa/orsa_fundamental.cpp cpp_libs/libOrsa/conditioning.cpp
    cpp_libs/libOrsa/orsa_model.cpp cpp_libs/libOrsa/fundamental_model.cpp
    cpp_libs/libOrsa/homography_model.hpp cpp_libs/libOrsa/homography_model.cpp
    cpp_libs/libOrsa/orsa_homography.cpp cpp_libs/libOrsa/orsa_homography.hpp   
)

if (VLFEAT)
    SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -D VL_DISABLE_AVX=1")
    message("************* with VLFEAT *************")
    set(src
        ${src}

        #affine shape adaptation
        cpp_libs/vl/generic.c
        cpp_libs/vl/covdet.c cpp_libs/vl/host.c cpp_libs/vl/random.c cpp_libs/vl/stringop.c
        cpp_libs/vl/scalespace.c cpp_libs/vl/imopv.c cpp_libs/vl/imopv_sse2.c 
        cpp_libs/vl/mathop.c cpp_libs/vl/mathop_sse2.c cpp_libs/vl/mathop_avx.c
        cpp_libs/vl/sift.c 
        )
    include_directories(./cpp_libs/vl)
endif()

if (USAC)
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D _USAC=1")
    message("************* with USAC *************")
    set(src
        ${src}

        #USAC
        cpp_libs/libUSAC/config/ConfigFileReader.cpp
        cpp_libs/libUSAC/config/ConfigParams.cpp
        cpp_libs/libUSAC/config/ConfigParamsFundmatrix.cpp
        cpp_libs/libUSAC/config/ConfigParamsHomog.cpp
        cpp_libs/libUSAC/utils/FundmatrixFunctions.cpp
        cpp_libs/libUSAC/utils/HomographyFunctions.cpp
        cpp_libs/libUSAC/utils/MathFunctions.cpp
        )
    include_directories(cpp_libs/libUSAC/config cpp_libs/libUSAC/estimators cpp_libs/libUSAC/utils)
endif()

include_directories(. ./cpp_libs/libOrsa)
add_library(DA MODULE ${src})
TARGET_LINK_LIBRARIES(DA Numerics)
if (USAC)
    TARGET_LINK_LIBRARIES(DA config++ lapack)
endif()
