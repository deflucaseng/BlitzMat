# tests/CMakeLists.txt
add_executable(test_operations
    test_operations.cpp           # Your test file
    ../../src/cpp/core/kernel_manager.cpp
	../../src/cpp/core/operation_manager.cpp         # The actual implementation
)

# Add the directory containing header files
target_include_directories(calculator_tests
    PRIVATE
        ../../src/cpp/core/include/pch.hpp
)

# Link with Google Test
target_link_libraries(test_operations
    PRIVATE
        gtest_main
        gmock_main
)

# Enable testing
include(GoogleTest)
gtest_discover_tests(test_operations)