#!/bin/bash
set -e  # Exit on error

# Navigate to repository root from scripts directory
cd "$(dirname "$0")/.."

# Create build directory if it doesn't exist
mkdir -p build
cd build

# Configure and build
cmake ..
make -j4

# Run basic test to verify build
echo "Running basic verification test..."
./part_a/vecadd00 100
echo "Build and verification completed successfully"