#!/bin/bash
set -e  # Exit on error

# Navigate to repository root from scripts directory
cd "$(dirname "$0")/.."

# Ensure we have a build
if [ ! -d "build" ]; then
    ./scripts/build.sh
fi

cd build

# Quick verification tests
echo "Running basic verification tests..."

# Part A - Small test
echo "Testing Part A..."
./part_a/vecadd00 100
./part_a/vecadd01 100

# Part B - Quick test
echo "Testing Part B..."
./part_b/q1 1
./part_b/q2 1
./part_b/q3 1

# Part C - Basic test
echo "Testing Part C..."
./part_c/c1
./part_c/c2
./part_c/c3

echo "All verification tests passed"