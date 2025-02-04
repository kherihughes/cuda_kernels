// q1.cpp: CPU-based vector addition for performance comparison with GPU implementations

#include <iostream>
#include <cstdlib>
#include <chrono>

// Simple CPU function for vector addition
void add_arrays(const float *a, const float *b, float *c, int size) {
    for (int i = 0; i < size; ++i) {
        c[i] = a[i] + b[i];
    }
}

int main(int argc, char *argv[]) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <K value (in millions)>" << std::endl;
        return 1;
    }

    int K = std::atoi(argv[1]);
    int size = K * 1000000;

    // Allocate memory for vectors
    float *a = (float *)malloc(size * sizeof(float));
    float *b = (float *)malloc(size * sizeof(float));
    float *c = (float *)malloc(size * sizeof(float));

    if (a == NULL || b == NULL || c == NULL) {
        std::cerr << "Memory allocation failed!" << std::endl;
        return 1;
    }

    // Initialize vectors
    for (int i = 0; i < size; ++i) {
        a[i] = 1.0f;
        b[i] = 2.0f;
    }

    // Measure execution time for vector addition
    auto start = std::chrono::high_resolution_clock::now();
    add_arrays(a, b, c, size);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;

    std::cout << "Time taken for K = " << K << " million elements: " << duration.count() << " seconds" << std::endl;

    // Free allocated memory
    free(a);
    free(b);
    free(c);

    return 0;
}
