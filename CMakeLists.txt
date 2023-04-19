cmake_minimum_required(VERSION 3.0.0)
project(linear-VEC-GP VERSION 0.1.0 LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set_property(GLOBAL PROPERTY CTEST_TARGETS_ADDED 1)
include(CTest)
enable_testing()

if(WIN32)
    message(FATAL_ERROR "Not yet supported on windows. Only Unix is supported")
endif(WIN32)

if(UNIX)
    # set(CXX_STANDARD 20)
    # set_property(TARGET PROPERTY CUDA_STANDARD 20)
    set(CMAKE_CUDA_ARCHITECTURES 75)
    find_package(Threads)
    # find_package(Python COMPONENTS Interpreter Development NumPy)
    add_executable(LGP-VEC LGP-VEC.cpp)
    add_executable(some-tests some-tests.cpp)
    # target_link_libraries(eigen-test Threads::Threads)
    target_link_libraries(LGP-VEC Threads::Threads)
endif(UNIX)


set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)
