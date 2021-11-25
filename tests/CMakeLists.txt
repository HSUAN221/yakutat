cmake_minimum_required(VERSION 3.12) # version can be different

if(${CMAKE_VERSION} VERSION_LESS 3.12)
    cmake_policy(VERSION ${CMAKE_MAJOR_VERSION}.${CMAKE_MINOR_VERSION})
endif()


project(yakutat_gtest) #name of your project

set(CMAKE_CXX_COMPILER "mpicxx")
set(CMAKE_CXX_FLAGS   "-std=c++17 -fopenmp -O3 -Wall -march=native")             


include_directories(..) 

include(GoogleTest) # this is so we can #include <gtest/gtest.h>


find_package(GTest REQUIRED CONFIG)
find_package(OpenMP REQUIRED)

set(class_name "mpi_SparseMatrixELL")
add_executable(${class_name}_test ${class_name}_test.cpp) # add executable
target_link_libraries(${class_name}_test GTest::gtest GTest::gtest_main GTest::gmock GTest::gmock_main Threads::Threads)
gtest_discover_tests(${class_name}_test)
target_link_libraries(${class_name}_test OpenMP::OpenMP_CXX)

set(class_name "mpi_cg")
add_executable(${class_name}_test ${class_name}_test.cpp) # add executable
target_link_libraries(${class_name}_test GTest::gtest GTest::gtest_main GTest::gmock GTest::gmock_main Threads::Threads)
gtest_discover_tests(${class_name}_test)
target_link_libraries(${class_name}_test OpenMP::OpenMP_CXX)

set(class_name "mpi_bicgstab")
add_executable(${class_name}_test ${class_name}_test.cpp) # add executable
target_link_libraries(${class_name}_test GTest::gtest GTest::gtest_main GTest::gmock GTest::gmock_main Threads::Threads)
gtest_discover_tests(${class_name}_test)
target_link_libraries(${class_name}_test OpenMP::OpenMP_CXX)

set(class_name "cg")
add_executable(${class_name}_test ${class_name}_test.cpp) # add executable
target_link_libraries(${class_name}_test GTest::gtest GTest::gtest_main GTest::gmock GTest::gmock_main Threads::Threads)
gtest_discover_tests(${class_name}_test)
target_link_libraries(${class_name}_test OpenMP::OpenMP_CXX)

set(class_name "bicgstab")
add_executable(${class_name}_test ${class_name}_test.cpp) # add executable
target_link_libraries(${class_name}_test GTest::gtest GTest::gtest_main GTest::gmock GTest::gmock_main Threads::Threads)
gtest_discover_tests(${class_name}_test)
target_link_libraries(${class_name}_test OpenMP::OpenMP_CXX)

set(class_name "mpi_DynamicMatrix")
add_executable(${class_name}_test ${class_name}_test.cpp) # add executable
target_link_libraries(${class_name}_test GTest::gtest GTest::gtest_main GTest::gmock GTest::gmock_main Threads::Threads)
gtest_discover_tests(${class_name}_test)
target_link_libraries(${class_name}_test OpenMP::OpenMP_CXX)