cmake_minimum_required(VERSION 2.8.9)
project(PERDIF)

find_package(OpenMP REQUIRED)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()
find_package(MKL REQUIRED)
include_directories(${MKL_INCLUDE_DIRS})
link_directories(${MKL_LIBRARIES})
target_link_libraries(<module>
mkl_intel_lp64
mkl_sequential
mkl_core
)
#Bring the headers, such as Student.h into the project
include_directories(include)
include(GNUInstallDirs)
#Can manually add the sources using the set command as follows:

#However, the file(GLOB...) allows for wildcard additions:
file(GLOB SOURCES "src/*.c")

set(CMAKE_C_FLAGS "-O -lblas -O3 -lpthread -lm -Wall -g -ffast-math")

add_executable(PERDIF ${SOURCES})