#
# Stanford University
#
# This file is part of Hierarchical Interpolative Factorization for
# Differential Equation 3D version (HIFDE3D) and is
# under the GPLv3 License, which can be found in the LICENSE file in the root
# directory, or at http://opensource.org/licenses/GPL-3.0
#
cmake_minimum_required(VERSION 2.8.5) # for the new FindMPI module
project(HIFDE3D)

set(HIFDE3D_VERSION_MAJOR 0)
set(HIFDE3D_VERSION_MINOR 2)

option(RELEASE "Avoid unnecessary assertions." ON)

option(BUILD_TESTS "Build the test drivers" ON)
option(AVOID_COMPLEX_MPI "Avoid complex MPI routines for robustness" ON)

mark_as_advanced(AVOID_COMPLEX_MPI)

if(RELEASE)
    set(CXX_FLAGS "-O3" CACHE STRING "CXX flags")
else()
    set(CXX_FLAGS "-g -Wall" CACHE STRING "CXX flags")
endif()

find_package(MPI)
if(NOT MPI_CXX_FOUND)
  message(FATAL_ERROR "MPI C++ compiler was not found and is required")
endif()
include_directories(${MPI_CXX_INCLUDE_PATH})
set(EXTRA_FLAGS "${MPI_CXX_COMPILE_FLAGS}")

# Now append the extra flags
set(CMAKE_CXX_FLAGS "${CXX_FLAGS} ${EXTRA_FLAGS}")

if(NOT MATH_LIBS)
  set(MATH_DESC "BLAS/LAPACK link flags")
  # Look for default BLAS/LAPACK
  set(REFERENCE_REQUIRED LAPACK BLAS)
  find_library(BLAS_LIB
               NAMES blas blas.a blas.lib
               PATHS ${REFERENCE_ROOT})
  find_library(LAPACK_LIB
               NAMES lapack lapack.a lapack.lib
                     reflapack reflapack.a reflapack.lib
               PATHS ${REFERENCE_ROOT})
  set(REFERENCE_FOUND ON)
  set(MATH_LIBS "")
  foreach(NAME ${REFERENCE_REQUIRED})
    if( ${NAME}_LIB )
      message(STATUS "Found ${NAME}_LIB: ${${NAME}_LIB}")
      list(APPEND MATH_LIBS ${${NAME}_LIB})
    else()
      message(STATUS "Could not find ${NAME}_LIB")
      set(REFERENCE_FOUND OFF)
    endif()
  endforeach()
  message(STATUS "REFERENCE_FOUND=${REFERENCE_FOUND}")
  if(REFERENCE_FOUND)
    message(STATUS "WARNING: Using reference BLAS/LAPACK.")
    message(STATUS "MATH_LIBS=${MATH_LIBS}")
  else()
    set(MATH_LIBS "" CACHE STRING ${MATH_DESC})
    message(FATAL_ERROR
      "Could not find BLAS/LAPACK libs. Please provide the root directory of MKL with -DMKL_ROOT, the directory of reference implementations with -DREFERENCE_ROOT, or manually specify all math libraries with -DMATH_LIBS. There are numerous idiosyncratic library dependencies for BLAS/LAPACK, so you will almost certainly need to manually specify -DMATH_LIBS.")
  endif()
  # Append the standard math libraries to the link list.
  list(APPEND MATH_LIBS m)
endif()

# Attempt to detect the BLAS and LAPACK underscore conventions.
# We currently only handle whether or not there is an underscore appended.
include(CheckFunctionExists)
set(CMAKE_REQUIRED_LIBRARIES ${MATH_LIBS})
check_function_exists(daxpy  HAVE_DAXPY)
check_function_exists(daxpy_ HAVE_DAXPY_POST)
if(HAVE_DAXPY)
  set(BLAS_POST FALSE)
  set(BLAS_DEFS "")
elseif(HAVE_DAXPY_POST)
  set(BLAS_POST TRUE)
  set(BLAS_DEFS "-DBLAS_POST")
else()
  message(FATAL_ERROR "Could not determine BLAS format.")
endif()
check_function_exists(dpotrf  HAVE_DPOTRF)
check_function_exists(dpotrf_ HAVE_DPOTRF_POST)
if(HAVE_DPOTRF)
  set(LAPACK_POST FALSE)
  set(LAPACK_DEFS "")
elseif(HAVE_DPOTRF_POST)
  set(LAPACK_POST TRUE)
  set(LAPACK_DEFS "-DLAPACK_POST")
else()
  message(FATAL_ERROR "Could not determine LAPACK format.")
endif()

# Look for MPI_Reduce_scatter_block (and MPI_Reduce_scatter as sanity check)
set(CMAKE_REQUIRED_FLAGS "${MPI_C_COMPILE_FLAGS} ${MPI_C_LINK_FLAGS}")
set(CMAKE_REQUIRED_INCLUDES ${MPI_C_INCLUDE_PATH})
set(CMAKE_REQUIRED_LIBRARIES ${MPI_C_LIBRARIES})
check_function_exists(MPI_Reduce_scatter       HAVE_MPI_REDUCE_SCATTER)
if(NOT HAVE_MPI_REDUCE_SCATTER)
  message(FATAL_ERROR "Could not find MPI_Reduce_scatter")
endif()
# Check for MPI_IN_PLACE (essentially MPI2 support)
include(CheckCSourceCompiles)
set(MPI_IN_PLACE_CODE
    "#include \"mpi.h\"
     int main( int argc, char* argv[] )
     {
         MPI_Init( &argc, &argv );
         float a;
         MPI_Allreduce
         ( MPI_IN_PLACE, &a, 1, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD );
         MPI_Finalize();
         return 0;
     }")
check_c_source_compiles("${MPI_IN_PLACE_CODE}" HAVE_MPI_IN_PLACE)
if(NOT HAVE_MPI_IN_PLACE)
  message(FATAL_ERROR "Could not find MPI_IN_PLACE")
endif()

# Look for restrict support
include(CheckCXXSourceCompiles)
set(RESTRICT_CODE "int main(void) { int* RESTRICT a; return 0; }")
set(CMAKE_REQUIRED_DEFINITIONS "-DRESTRICT=__restrict__")
check_cxx_source_compiles("${RESTRICT_CODE}" HAVE___restrict__)
set(CMAKE_REQUIRED_DEFINITIONS "-DRESTRICT=__restrict")
check_cxx_source_compiles("${RESTRICT_CODE}" HAVE___restrict)
set(CMAKE_REQUIRED_DEFINITIONS "-DRESTRICT=restrict")
check_cxx_source_compiles("${RESTRICT_CODE}" HAVE_restrict)
if(HAVE___restrict__)
  set(RESTRICT "__restrict__")
  message(STATUS "Using __restrict__ keyword.")
elseif(HAVE___restrict)
  set(RESTRICT "__restrict")
  message(STATUS "Using __restrict keyword.")
elseif(HAVE_restrict)
  set(RESTRICT "restrict")
  message(STATUS "Using restrict keyword.")
else()
  set(RESTRICT "")
  message(STATUS "Could not find a restrict keyword.")
endif()

# Store all of our system-specific definitions
configure_file(${PROJECT_SOURCE_DIR}/cmake/config.h.cmake
               ${PROJECT_BINARY_DIR}/include/hifde3d/config.h)
install(FILES ${PROJECT_BINARY_DIR}/include/hifde3d/config.h
        DESTINATION include/hifde3d)

# Grab all of the .c, .cpp, .h, and .hpp HIFDE3D files
file(GLOB_RECURSE HIFDE3D_CPP RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
     "src/*.c" "src/*.cpp" "src/*-incl.hpp")
file(GLOB_RECURSE HIFDE3D_HEADERS RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
     "include/*.h" "include/*.hpp")
set(HIFDE3D_SRC "${HIFDE3D_CPP};${HIFDE3D_HEADERS}")

# Build the HIFDE3D source
add_library(hifde3d STATIC ${HIFDE3D_SRC})
install(TARGETS hifde3d DESTINATION lib)

# Copy the headers into the build directory
set(PREPARED_HEADERS)
foreach(HEADER ${HIFDE3D_HEADERS})
  add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${HEADER}
    COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/${HEADER}
            ${CMAKE_CURRENT_BINARY_DIR}/${HEADER}
    DEPENDS "${PROJECT_SOURCE_DIR}/${HEADER}")
  list(APPEND PREPARED_HEADERS ${PROJECT_BINARY_DIR}/${HEADER})

  get_filename_component(HEADER_PATH ${HEADER} PATH)
  install(FILES ${CMAKE_CURRENT_BINARY_DIR}/${HEADER} DESTINATION ${HEADER_PATH})
endforeach()
add_custom_target(prepare_hifde3d_headers DEPENDS ${PREPARED_HEADERS})
add_dependencies(hifde3d prepare_hifde3d_headers)

# Make sure the HIFDE3D headers can be found
include_directories("${PROJECT_BINARY_DIR}/include")

# Create a dummy library in order to be able to force the math libraries
# to be linked last
add_library(cmake-dummy-lib STATIC cmake/CMakeDummyFunction.cpp)
list(APPEND EXTERNAL_LIBS "${MATH_LIBS} ${MPI_CXX_LIBRARIES}")
if(HAVE_QT5)
  list(APPEND EXTERNAL_LIBS "${Qt5Widgets_LIBRARIES}")
endif()
target_link_libraries(cmake-dummy-lib ${EXTERNAL_LIBS})

# Build the test drivers if necessary
set(MPI_LINK_FLAGS "${MPI_CXX_LINK_FLAGS}")
if(BUILD_TESTS)
  set(TEST_DIR ${PROJECT_SOURCE_DIR}/tests)
  set(TEST_TYPES core hmat_tools HIFDE3D)
  set(HIFDE3D_TESTS Apply_test Factor_test Sparse_test main)

  foreach(TYPE ${TEST_TYPES})
    set(OUTPUT_DIR "${PROJECT_BINARY_DIR}/bin/${TYPE}")
    foreach(TEST ${${TYPE}_TESTS})
      add_executable(${TYPE}-${TEST} ${TEST_DIR}/${TYPE}/${TEST}.cpp)
      target_link_libraries(${TYPE}-${TEST} hifde3d cmake-dummy-lib)
      set_target_properties(${TYPE}-${TEST}
                            PROPERTIES OUTPUT_NAME ${TEST}
                            RUNTIME_OUTPUT_DIRECTORY ${OUTPUT_DIR})
      if(MPI_LINK_FLAGS)
        set_target_properties(${TYPE}-${TEST} PROPERTIES
          LINK_FLAGS ${MPI_LINK_FLAGS})
      endif()
      install(TARGETS ${TYPE}-${TEST} DESTINATION bin/${TYPE})
    endforeach()
  endforeach()
endif()
