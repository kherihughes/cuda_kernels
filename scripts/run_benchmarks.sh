#!/bin/bash
set -e  # Exit on error

# Navigate to repository root from scripts directory
cd "$(dirname "$0")/.."

# Ensure we have a build
if [ ! -d "build" ]; then
    ./scripts/build.sh
fi

cd build

# Part A benchmarks
echo "Running Part A benchmarks..."
for size in 500 1000 2000; do
    echo "Testing vector addition size $size"
    ./part_a/vecadd00 $size
    ./part_a/vecadd01 $size
done

# Part B benchmarks
echo "Running Part B benchmarks..."
for k in 1 5 10 50 100; do
    echo "Testing with K=$k million elements"
    ./part_b/q1 $k
    ./part_b/q2 $k
    ./part_b/q3 $k
done

# Part C benchmarks
echo "Running Part C benchmarks..."
./part_c/c1
./part_c/c2
./part_c/c3

echo "All benchmarks completed"