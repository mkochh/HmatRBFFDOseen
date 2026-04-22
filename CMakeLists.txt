cmake_minimum_required(VERSION 3.25)

project("MedusaH2LibExtended")

# Set build type to release, if not specified.
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Debug)
endif()

# Add math library to linked libraries
set(LINK_LIBRARIES m)

# Add definition for H2Lib quick exit at rank 0.
add_definitions(-DHARITH_RKMATRIX_QUICK_EXIT)

# Add BLAS library to linked libraries
set(BLA_VENDOR OpenBLAS)
find_package(BLAS)
if(BLAS_FOUND)
  message("-- " ${BLAS_LIBRARIES})
  add_definitions(-DUSE_BLAS)
  set(LINK_LIBRARIES ${LINK_LIBRARIES} ${BLAS_LIBRARIES})
else()
  message("-- BLAS not found")
endif(BLAS_FOUND)

if(USE_OPENMP)
find_package(OpenMP)
if (OPENMP_FOUND)
  message("-- OpenMP found")
  add_definitions(-DUSE_OPENMP)
  set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
  set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
  set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
else()
  message("-- OpenMP not found")
endif()
endif()

if(USE_CAIRO)
 add_definitions(-DUSE_CAIRO)

 find_library(CAIRO_LIB cairo)

 find_path(CAIRO_INCLUDE_DIR cairo/cairo.h)

 message("-- Cairo found")

 include_directories(${CAIRO_INCLUDE_DIR}/cairo)
 set(LINK_LIBRARIES ${LINK_LIBRARIES} ${CAIRO_LIB})
endif()

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
find_package(METIS REQUIRED)
include_directories(${METIS_INCLUDES})
set(LINK_LIBRARIES ${LINK_LIBRARIES} ${METIS_LIBRARIES})

if(USE_UMFPACK)
  find_package(UMFPACK REQUIRED)
  include_directories(${UMFPACK_INCLUDES})
  set(LINK_LIBRARIES ${LINK_LIBRARIES} ${UMFPACK_LIBRARIES})
endif()

# add spectra to linked libraries for eigenvalue computation
add_library(spectra INTERFACE)
target_include_directories(spectra INTERFACE ${CMAKE_SOURCE_DIR}/spectra/include)
set(LINK_LIBRARIES ${LINK_LIBRARIES} spectra)

# Build extended H2Lib
add_subdirectory(H2Lib)
include_directories(H2Lib)
link_directories(H2Lib)

# Add H2Lib to linked libraries
set(LINK_LIBRARIES ${LINK_LIBRARIES} h2lib)

add_subdirectory(${CMAKE_SOURCE_DIR}/medusa medusa)
include_directories(${CMAKE_SOURCE_DIR}/medusa/include)
set(LINK_LIBRARIES ${LINK_LIBRARIES} medusa)

set(CGAL_DO_NOT_WARN_ABOUT_CMAKE_BUILD_TYPE TRUE)
find_package(CGAL REQUIRED)
if(CGAL_FOUND)
  message("-- Using CGAL Version ${CGAL_VERSION}")
  if(CGAL_VERSION_MAJOR LESS 6)
    add_definitions(-DUSE_CGAL_LESS_THAN_6)
  else()
    add_definitions(-DUSE_CGAL6)
  endif()
endif()

# Build Extensions
# Set flags for debugging or optimization.
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 -g -O1 -Wall -Wextra -Wno-unused-function -Wno-maybe-uninitialized -Wno-deprecated-copy -Wno-deprecated-declarations -Wno-overloaded-virtual")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -g -O1 -Wall -Wextra -Wno-unused-function -Wno-maybe-uninitialized -Wno-deprecated-copy -Wno-deprecated-declarations -Wno-overloaded-virtual")

set(extension_headers)
set(extension_sources)
add_subdirectory(Extensions)
include_directories(Extensions)

add_library(h2libext ${extension_sources})

foreach(lib ${LINK_LIBRARIES})
  target_link_libraries(h2libext ${lib})
endforeach(lib ${LINK_LIBRARIES})

set(LINK_LIBRARIES ${LINK_LIBRARIES} h2libext)

# Build Tests
add_subdirectory(Tests)
