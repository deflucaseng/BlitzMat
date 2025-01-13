# Compiler and flags
CXX = g++
CXXFLAGS = -std=c++14 -Wall -g
TEST_FLAGS = -pthread -lOpenCL -lgtest_main -lgtest -lgmock_main -lgmock

# Directories
SRC_DIR = src/cpp/core
TEST_DIR = tests/cpp
OBJ_DIR = obj
BIN_DIR = bin

# Source and test files
SRC_FILES = $(wildcard $(SRC_DIR)/*.cpp)
TEST_FILES = $(wildcard $(TEST_DIR)/*.cpp)

# Object files
SRC_OBJ = $(SRC_FILES:$(SRC_DIR)/%.cpp=$(OBJ_DIR)/%.o)
TEST_OBJ = $(TEST_FILES:$(TEST_DIR)/%.cpp=$(OBJ_DIR)/%.o)

# Final test executable
TEST_EXEC = $(BIN_DIR)/run_tests

# Default target
all: $(TEST_EXEC)

# Create necessary directories
$(OBJ_DIR):
	mkdir -p $(OBJ_DIR)

$(BIN_DIR):
	mkdir -p $(BIN_DIR)

# Compile source files
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp | $(OBJ_DIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Compile test files
$(OBJ_DIR)/%.o: $(TEST_DIR)/%.cpp | $(OBJ_DIR)
	$(CXX) $(CXXFLAGS) -I$(SRC_DIR) -c $< -o $@

# Link everything into test executable
$(TEST_EXEC): $(SRC_OBJ) $(TEST_OBJ) | $(BIN_DIR)
	$(CXX) $(CXXFLAGS) $^ $(TEST_FLAGS) -o $@

# Run the tests
.PHONY: test
test: $(TEST_EXEC)
	./$(TEST_EXEC)

# Clean build files
.PHONY: clean
clean:
	rm -rf $(OBJ_DIR) $(BIN_DIR)


