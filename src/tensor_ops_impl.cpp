#include "lightgpt/tensor.hpp"
#include "lightgpt/tensor_ops.hpp"
#include <stdexcept>

namespace lightgpt {

// Forward declarations from the ops namespace
namespace ops {
Tensor add(const Tensor& a, const Tensor& b);
Tensor matmul(const Tensor& a, const Tensor& b);
Tensor softmax(const Tensor& input, int dim);
Tensor layer_norm(const Tensor& input, const Tensor& weight, const Tensor& bias, float eps);
} // namespace ops

// Tensor operations implementation
Tensor add(const Tensor& a, const Tensor& b) {
    return ops::add(a, b);
}

Tensor matmul(const Tensor& a, const Tensor& b) {
    return ops::matmul(a, b);
}

Tensor softmax(const Tensor& t, int dim) {
    return ops::softmax(t, dim);
}

Tensor layer_norm(const Tensor& input, const Tensor& weight, const Tensor& bias, float eps) {
    return ops::layer_norm(input, weight, bias, eps);
}

// Implement other required operations
Tensor sub(const Tensor& a, const Tensor& b) {
    // Implement subtraction using add and negate
    return a + (-b);
}

Tensor mul(const Tensor& a, const Tensor& b) {
    // For now, implement using a simple loop
    // In a real implementation, this would be optimized
    if (a.shape() != b.shape()) {
        throw std::runtime_error("Shape mismatch in tensor multiplication");
    }
    
    Tensor result(a.shape(), a.dtype());
    size_t n = a.numel();
    
    if (a.dtype() == DType::F32) {
        const float* a_data = a.data<float>();
        const float* b_data = b.data<float>();
        float* out_data = result.data<float>();
        
        for (size_t i = 0; i < n; ++i) {
            out_data[i] = a_data[i] * b_data[i];
        }
    } else {
        throw std::runtime_error("Unsupported data type for multiplication");
    }
    
    return result;
}

Tensor div(const Tensor& a, const Tensor& b) {
    // For now, implement using a simple loop
    // In a real implementation, this would be optimized
    if (a.shape() != b.shape()) {
        throw std::runtime_error("Shape mismatch in tensor division");
    }
    
    Tensor result(a.shape(), a.dtype());
    size_t n = a.numel();
    
    if (a.dtype() == DType::F32) {
        const float* a_data = a.data<float>();
        const float* b_data = b.data<float>();
        float* out_data = result.data<float>();
        
        for (size_t i = 0; i < n; ++i) {
            if (b_data[i] == 0.0f) {
                throw std::runtime_error("Division by zero");
            }
            out_data[i] = a_data[i] / b_data[i];
        }
    } else {
        throw std::runtime_error("Unsupported data type for division");
    }
    
    return result;
}

Tensor sqrt(const Tensor& t) {
    Tensor result(t.shape(), t.dtype());
    size_t n = t.numel();
    
    if (t.dtype() == DType::F32) {
        const float* t_data = t.data<float>();
        float* out_data = result.data<float>();
        
        for (size_t i = 0; i < n; ++i) {
            out_data[i] = std::sqrt(t_data[i]);
        }
    } else {
        throw std::runtime_error("Unsupported data type for sqrt");
    }
    
    return result;
}

Tensor exp(const Tensor& t) {
    Tensor result(t.shape(), t.dtype());
    size_t n = t.numel();
    
    if (t.dtype() == DType::F32) {
        const float* t_data = t.data<float>();
        float* out_data = result.data<float>();
        
        for (size_t i = 0; i < n; ++i) {
            out_data[i] = std::exp(t_data[i]);
        }
    } else {
        throw std::runtime_error("Unsupported data type for exp");
    }
    
    return result;
}

Tensor log_softmax(const Tensor& t, int /*dim*/) {
    // For now, implement a simple CPU version
    // This is a simplified version - in a real implementation, we'd handle batching and other dimensions properly
    
    // Get the shape and check if it's 1D or 2D
    auto shape = t.shape();
    if (shape.size() != 1 && shape.size() != 2) {
        throw std::runtime_error("log_softmax only supports 1D or 2D tensors");
    }
    
    // For now, only implement the 1D case
    if (shape.size() == 1) {
        size_t size = shape[0];
        Tensor result(shape, t.dtype());
        
        if (t.dtype() == DType::F32) {
            const float* t_data = t.data<float>();
            float* out_data = result.data<float>();
            
            // Find max for numerical stability
            float max_val = t_data[0];
            for (size_t i = 1; i < size; ++i) {
                if (t_data[i] > max_val) {
                    max_val = t_data[i];
                }
            }
            
            // Compute sum(exp(x - max_val))
            float sum_exp = 0.0f;
            for (size_t i = 0; i < size; ++i) {
                sum_exp += std::exp(t_data[i] - max_val);
            }
            
            // Compute log_softmax
            float log_sum_exp = std::log(sum_exp);
            for (size_t i = 0; i < size; ++i) {
                out_data[i] = t_data[i] - max_val - log_sum_exp;
            }
        } else {
            throw std::runtime_error("Unsupported data type for log_softmax");
        }
        
        return result;
    } else {
        // 2D case - for now, just throw an error
        throw std::runtime_error("2D log_softmax not implemented yet");
    }
}

} // namespace lightgpt
