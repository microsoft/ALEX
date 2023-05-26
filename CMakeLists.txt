cmake_minimum_required(VERSION 3.12)
project(alex)

set(CMAKE_CXX_STANDARD 14)
set(MKL_LINK_DIRECTORY "/opt/intel/mkl/lib/intel64")
set(MKL_INCLUDE_DIRECTORY "/opt/intel/mkl/include")

# Define the macro ‘DEBUG' in the debug mode
if(CMAKE_BUILD_TYPE STREQUAL Debug)        
    ADD_DEFINITIONS(-DDEBUG)               
endif()

if(MSVC)
    set(CMAKE_CXX_FLAGS "/O2 /arch:AVX2 /W1 /EHsc")
elseif (CMAKE_CXX_COMPILER_ID STREQUAL "Intel")
    set(CMAKE_CXX_FLAGS "-O3 -xHost")
else()
    # clang and gcc
    set(CMAKE_CXX_FLAGS "-O3 -march=native -Wall -Wextra")
endif()

link_directories(${MKL_LINK_DIRECTORY})

include_directories(${MKL_INCLUDE_DIRECTORY})
include_directories(src/core)

#add_executable(example src/examples/main.cpp`)
#target_link_libraries(example PRIVATE mkl_rt -lpthread)

add_executable(benchmark src/benchmark/main.cpp)
target_link_libraries(benchmark PRIVATE mkl_rt -lpthread)

#set(DOCTEST_DOWNLOAD_DIR ${CMAKE_CURRENT_BINARY_DIR}/doctest)
#file(DOWNLOAD
#    https://raw.githubusercontent.com/onqtam/doctest/2.4.6/doctest/doctest.h
#    ${DOCTEST_DOWNLOAD_DIR}/doctest.h
#    EXPECTED_HASH SHA1=40728d2bed7f074e80cb095844820114e7d16ce0
#)

#add_executable(test_alex test/unittest_main.cpp)
#target_include_directories(test_alex PRIVATE ${DOCTEST_DOWNLOAD_DIR})
#target_link_libraries(test_alex PRIVATE mkl_rt -lpthread)

#enable_testing()
#add_test(test_alex test_alex)