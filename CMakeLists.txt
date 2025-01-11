# Root CMakeLists.txt
cmake_minimum_required(VERSION 3.14)
project(blitzmat)

# Set up Google Test
include(FetchContent)
FetchContent_Declare(
    googletest
    URL https://github.com/google/googletest/archive/refs/tags/v1.13.0.zip
)
FetchContent_MakeAvailable(googletest)

# Add the tests directory
add_subdirectory(tests)