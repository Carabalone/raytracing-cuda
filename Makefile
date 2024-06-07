# Compiler and flags
NVCC := /usr/local/cuda/bin/nvcc
NVCC_FLAGS := -arch=sm_50 -Xcompiler -Wall -O3

# Directories
SRC_DIR := src
BUILD_DIR := build
OUTPUT_DIR := output

# Target executable
TARGET := $(OUTPUT_DIR)/raytracer

# Find all CUDA source files
SRCS := $(wildcard $(SRC_DIR)/*.cu)

# Convert the source file list into a list of object files in the build directory
OBJS := $(SRCS:$(SRC_DIR)/%.cu=$(BUILD_DIR)/%.o)

# Default target
all: $(TARGET)

# Rule to build the target executable
$(TARGET): $(OBJS)
	@mkdir -p $(OUTPUT_DIR)
	$(NVCC) $(NVCC_FLAGS) -o $@ $^

# Rule to compile source files into object files
$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cu
	@mkdir -p $(BUILD_DIR)
	$(NVCC) $(NVCC_FLAGS) -c -o $@ $<

run: $(TARGET)
	@make
	$(TARGET)
	@convert output/output.ppm output/output.png
	@xdg-open output/output.png

# Clean up build and output directories
clean:
	rm -rf $(BUILD_DIR) $(OUTPUT_DIR)

# Phony targets
.PHONY: all clean

