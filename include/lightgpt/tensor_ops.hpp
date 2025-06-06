#pragma once

#include <vector>
#include <cstdint>
#include <cmath>
#include <memory>
#include <string>

namespace lightgpt {

// Slice helper for tensor operations
struct Slice {
    int64_t start = 0;
    int64_t end = -1; // -1 means until the end
    int64_t step = 1;

    Slice() = default;
    Slice(int64_t s, int64_t e = -1, int64_t st = 1)
        : start(s), end(e), step(st) {}
};

// Tensor operations that are architecture-specific
namespace ops {

// Softmax implementation
std::vector<float> softmax(const float* input, size_t size);

// Layer normalization
void layer_norm(
    const float* input, float* output, size_t size,
    const float* gamma, const float* beta, float eps);

// Matrix multiplication (naive implementation for now)
void matmul(
    const float* a, const float* b, float* out,
    int m, int n, int k);

// GELU activation
void gelu(const float* input, float* output, size_t size);

} // namespace ops

} // namespace lightgpt
