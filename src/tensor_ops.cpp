#include "lightgpt/tensor_ops.hpp"
#include "lightgpt/tensor.hpp"
#include <cmath>
#include <algorithm>
#include <numeric>
#include <stdexcept>

namespace lightgpt {
namespace ops {

std::vector<float> softmax(const float* input, size_t size) {
    std::vector<float> result(size);
    float max_val = *std::max_element(input, input + size);
    float sum = 0.0f;
    
    // Compute exponentials and sum
    for (size_t i = 0; i < size; ++i) {
        result[i] = std::exp(input[i] - max_val);
        sum += result[i];
    }
    
    // Normalize
    for (size_t i = 0; i < size; ++i) {
        result[i] /= sum;
    }
    
    return result;
}

void layer_norm(const float* input, float* output, size_t size, 
               const float* gamma, const float* beta, float eps) {
    // Compute mean
    float mean = 0.0f;
    for (size_t i = 0; i < size; ++i) {
        mean += input[i];
    }
    mean /= size;
    
    // Compute variance
    float variance = 0.0f;
    for (size_t i = 0; i < size; ++i) {
        float diff = input[i] - mean;
        variance += diff * diff;
    }
    variance = variance / size + eps;
    float inv_std = 1.0f / std::sqrt(variance);
    
    // Normalize and apply affine transform
    for (size_t i = 0; i < size; ++i) {
        output[i] = (input[i] - mean) * inv_std * gamma[i] + beta[i];
    }
}

void matmul(const float* a, const float* b, float* out, int m, int n, int k) {
    // Simple implementation of matrix multiplication: C = A @ B
    // where A is m×k, B is k×n, and C is m×n
    
    // Initialize output to zero
    std::memset(out, 0, m * n * sizeof(float));
    
    // Perform matrix multiplication
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            float sum = 0.0f;
            for (int l = 0; l < k; ++l) {
                sum += a[i * k + l] * b[l * n + j];
            }
            out[i * n + j] = sum;
        }
    }
}

void gelu(const float* input, float* output, size_t size) {
    // GELU activation function: 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
    constexpr float sqrt_2_over_pi = 0.7978845608028654f;
    constexpr float coef = 0.044715f;
    
    for (size_t i = 0; i < size; ++i) {
        float x = input[i];
        float x_cubed = x * x * x;
        float inner = sqrt_2_over_pi * (x + coef * x_cubed);
        output[i] = 0.5f * x * (1.0f + std::tanh(inner));
    }
}

// Implementation of free functions
Tensor add(const Tensor& a, const Tensor& b) {
    if (a.shape() != b.shape()) {
        throw std::runtime_error("Shapes must match for addition");
    }
    
    Tensor result(a.shape(), a.dtype());
    size_t n = a.numel();
    
    if (a.dtype() == DType::F32) {
        const float* a_data = a.data<float>();
        const float* b_data = b.data<float>();
        float* out_data = result.data<float>();
        
        for (size_t i = 0; i < n; ++i) {
            out_data[i] = a_data[i] + b_data[i];
        }
    } else {
        throw std::runtime_error("Unsupported data type for add operation");
    }
    
    return result;
}

Tensor matmul(const Tensor& a, const Tensor& b) {
    if (a.dim() != 2 || b.dim() != 2) {
        throw std::runtime_error("Only 2D tensors are supported for matrix multiplication");
    }
    if (a.size(-1) != b.size(0)) {
        throw std::runtime_error("Incompatible dimensions for matrix multiplication");
    }
    
    int m = a.size(0);
    int k = a.size(1);
    int n = b.size(1);
    
    Tensor result({m, n}, a.dtype());
    
    if (a.dtype() == DType::F32) {
        const float* a_data = a.data<float>();
        const float* b_data = b.data<float>();
        float* out_data = result.data<float>();
        
        // Naive matrix multiplication
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < n; ++j) {
                float sum = 0.0f;
                for (int l = 0; l < k; ++l) {
                    sum += a_data[i * k + l] * b_data[l * n + j];
                }
                out_data[i * n + j] = sum;
            }
        }
    } else {
        throw std::runtime_error("Unsupported data type for matrix multiplication");
    }
    
    return result;
}

Tensor softmax(const Tensor& input, int dim) {
    if (dim < 0) dim += input.dim();
    if (dim < 0 || dim >= input.dim()) {
        throw std::out_of_range("Dimension out of range for softmax");
    }
    
    // For now, only support 2D tensors and dim=1
    if (input.dim() != 2 || dim != 1) {
        throw std::runtime_error("Only 2D tensors with dim=1 are supported for softmax");
    }
    
    int batch_size = input.size(0);
    int num_classes = input.size(1);
    
    Tensor result(input.shape(), input.dtype());
    
    if (input.dtype() == DType::F32) {
        const float* input_data = input.data<float>();
        float* output_data = result.data<float>();
        
        for (int i = 0; i < batch_size; ++i) {
            const float* row = input_data + i * num_classes;
            float* out_row = output_data + i * num_classes;
            
            // Find max for numerical stability
            float max_val = *std::max_element(row, row + num_classes);
            
            // Compute exponentials and sum
            float sum = 0.0f;
            for (int j = 0; j < num_classes; ++j) {
                out_row[j] = std::exp(row[j] - max_val);
                sum += out_row[j];
            }
            
            // Normalize
            for (int j = 0; j < num_classes; ++j) {
                out_row[j] /= sum;
            }
        }
    } else {
        throw std::runtime_error("Unsupported data type for softmax");
    }
    
    return result;
}

Tensor layer_norm(const Tensor& input, const Tensor& weight, const Tensor& bias, float eps) {
    if (input.dim() < 1) {
        throw std::runtime_error("Input must be at least 1D");
    }
    
    // The last dimension is normalized
    int64_t normalized_dim = input.dim() - 1;
    int64_t normalized_size = input.size(normalized_dim);
    
    if (weight.numel() != static_cast<size_t>(normalized_size) || bias.numel() != static_cast<size_t>(normalized_size)) {
        throw std::runtime_error("Weight and bias must have the same size as the last dimension of input");
    }
    
    Tensor result(input.shape(), input.dtype());
    
    if (input.dtype() == DType::F32) {
        const float* input_data = input.data<float>();
        const float* weight_data = weight.data<float>();
        const float* bias_data = bias.data<float>();
        float* output_data = result.data<float>();
        
        // Calculate total number of elements per normalization group
        int64_t num_groups = input.numel() / normalized_size;
        
        for (int64_t i = 0; i < num_groups; ++i) {
            const float* group_input = input_data + i * normalized_size;
            float* group_output = output_data + i * normalized_size;
            
            // Compute mean
            float mean = 0.0f;
            for (int64_t j = 0; j < normalized_size; ++j) {
                mean += group_input[j];
            }
            mean /= normalized_size;
            
            // Compute variance
            float variance = 0.0f;
            for (int64_t j = 0; j < normalized_size; ++j) {
                float diff = group_input[j] - mean;
                variance += diff * diff;
            }
            variance = variance / normalized_size + eps;
            float inv_std = 1.0f / std::sqrt(variance);
            
            // Normalize and apply affine transform
            for (int64_t j = 0; j < normalized_size; ++j) {
                group_output[j] = (group_input[j] - mean) * inv_std * weight_data[j] + bias_data[j];
            }
        }
    } else {
        throw std::runtime_error("Unsupported data type for layer_norm");
    }
    
    return result;
}

} // namespace ops
} // namespace lightgpt
