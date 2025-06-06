#pragma once

#include <vector>
#include <memory>
#include <string>
#include <cstdint>
#include <stdexcept>
#include <functional>
#include <sstream>
#include <initializer_list>
#include <algorithm>
#include <limits>

namespace lightgpt {

// Supported data types
enum class DType {
    F32,
    F16,
    I8,
    I32,
    BOOL
};

// Helper function to get size of data types
constexpr size_t dtype_size(DType dtype) {
    switch (dtype) {
        case DType::F32: return 4;
        case DType::F16: return 2;
        case DType::I8:  return 1;
        case DType::I32: return 4;
        case DType::BOOL: return 1;
        default: return 0;
    }
}

// Tensor class for efficient numerical computations
class Tensor {
public:
    // Constructors
    Tensor();
    Tensor(const std::vector<int64_t>& shape, DType dtype = DType::F32);
    Tensor(const std::vector<int64_t>& shape, DType dtype, void* data, bool copy = true);
    Tensor(const Tensor& other);
    Tensor(Tensor&& other) noexcept;
    ~Tensor();

    // Assignment operators
    Tensor& operator=(const Tensor& other);
    Tensor& operator=(Tensor&& other) noexcept;

    // Tensor operations
    Tensor matmul(const Tensor& other) const;
    Tensor operator*(const Tensor& other) const;
    Tensor operator+(const Tensor& other) const;
    Tensor operator-() const;
    
    // Element-wise operations
    Tensor add(const Tensor& other) const;
    Tensor sub(const Tensor& other) const;
    Tensor mul(const Tensor& other) const;
    Tensor div(const Tensor& other) const;
    
    // Unary operations
    Tensor sqrt() const;
    Tensor exp() const;
    Tensor log_softmax(int dim = -1) const;
    Tensor softmax(int dim) const;
    Tensor layer_norm(const Tensor& weight, const Tensor& bias, float eps) const;
    
    // Shape manipulation
    Tensor view(const std::vector<int64_t>& new_shape) const;
    Tensor permute(const std::vector<size_t>& dims) const;
    Tensor transpose(int64_t dim0, int64_t dim1) const;
    Tensor t() const { return transpose(-2, -1); }  // Matrix transpose (for 2D tensors)
    Tensor contiguous() const;
    
    // Accessors
    bool empty() const { return numel() == 0; }
    size_t itemsize() const { return dtype_size(dtype_); }
    
    template<typename T>
    T* data() { return static_cast<T*>(data_); }
    
    template<typename T>
    const T* data() const { return static_cast<const T*>(data_); }
    
    DType dtype() const { return dtype_; }
    
    // Shape and size utilities
    const std::vector<int64_t>& shape() const { return shape_; }
    const std::vector<int64_t>& strides() const { return strides_; }
    
    // Find the index of the maximum value along a dimension
    int64_t argmax(int64_t dim = -1) const {
        if (dim < 0) dim += shape_.size();
        if (dim < 0 || static_cast<size_t>(dim) >= shape_.size()) {
            throw std::out_of_range("Dimension out of range");
        }
        
        // For now, only support 1D tensors for simplicity
        if (shape_.size() != 1) {
            throw std::runtime_error("argmax only supports 1D tensors for now");
        }
        
        if (dtype_ != DType::F32) {
            throw std::runtime_error("argmax only supports F32 tensors for now");
        }
        
        const float* data = this->data<float>();
        auto it = std::max_element(data, data + numel());
        return std::distance(data, it);
    }
    
    // Access element at position
    template<typename T>
    T at(const std::vector<int64_t>& indices) const {
        if (indices.size() != shape_.size()) {
            throw std::runtime_error("Number of indices must match tensor rank");
        }
        size_t offset = 0;
        for (size_t i = 0; i < indices.size(); ++i) {
            if (indices[i] >= shape_[i]) {
                throw std::out_of_range("Index out of range");
            }
            offset += indices[i] * strides_[i];
        }
        return data<T>()[offset];
    }

    // Get size of a specific dimension
    int64_t size(int64_t dim) const {
        if (dim < 0) dim += shape_.size();
        if (dim < 0 || static_cast<size_t>(dim) >= shape_.size()) {
            throw std::out_of_range("Dimension out of range");
        }
        return shape_[dim];
    }
    
    size_t numel() const {
        if (shape_.empty()) return 0;
        size_t n = 1;
        for (auto d : shape_) n *= d;
        return n;
    }
    
    size_t nbytes() const {
        size_t element_size = 0;
        switch (dtype_) {
            case DType::F32: element_size = sizeof(float); break;
            case DType::F16: element_size = sizeof(uint16_t); break;
            case DType::I8:  element_size = sizeof(int8_t); break;
            case DType::I32: element_size = sizeof(int32_t); break;
            case DType::BOOL: element_size = sizeof(bool); break;
        }
        return numel() * element_size;
    }
    
    int64_t dim() const { return static_cast<int64_t>(shape_.size()); }
    

    
    // Check if tensor is defined (has data)
    bool defined() const { return data_ != nullptr; }
    
    // Check if the tensor is contiguous in memory (C-contiguous)
    bool is_contiguous() const {
        if (strides_.empty() || shape_.empty()) {
            return true;  // Scalar or empty tensor is considered contiguous
        }
        
        int64_t stride = 1;
        // Check if strides are in decreasing order (C-contiguous)
        for (int i = static_cast<int>(shape_.size()) - 1; i >= 0; --i) {
            if (strides_[i] != stride) {
                return false;
            }
            if (i > 0) {
                stride *= shape_[i];
            }
        }
        return true;
    }
    
    // Memory management
    void* release();
    
    // Debugging
    std::string to_string() const;

private:
    void* data_ = nullptr;
    std::vector<int64_t> shape_;
    std::vector<int64_t> strides_;
    size_t numel_ = 0;
    DType dtype_ = DType::F32;
    bool owns_data_ = false;
    size_t offset_ = 0;
    
    // Helper functions
    void allocate_memory();
    void free_memory();
    void copy_from(const Tensor& other);
    void move_from(Tensor& other) noexcept;
    void check_shape_match(const Tensor& other, const std::string& op_name) const;
    void check_dtype_match(DType expected, const std::string& op_name) const;
    
    // Implementation details for operations
    template<typename T>
    void matmul_impl(const Tensor& a, const Tensor& b, Tensor& out) const;
    
    template<typename T>
    void add_impl(const Tensor& a, const Tensor& b, Tensor& out) const;
    
    template<typename T>
    void softmax_impl(Tensor& out, int dim) const;
};

// Utility functions
Tensor zeros(const std::vector<int64_t>& shape, DType dtype = DType::F32);
Tensor ones(const std::vector<int64_t>& shape, DType dtype = DType::F32);
Tensor arange(int64_t n, DType dtype = DType::F32);
Tensor full(const std::vector<int64_t>& shape, float value, DType dtype = DType::F32);

// Tensor creation from data
Tensor tensor(const std::vector<float>& data, const std::vector<int64_t>& shape = {});
Tensor tensor(const std::vector<int32_t>& data, const std::vector<int64_t>& shape = {});
Tensor tensor(const std::vector<int8_t>& data, const std::vector<int64_t>& shape = {});

// Tensor operations
Tensor matmul(const Tensor& a, const Tensor& b);
Tensor add(const Tensor& a, const Tensor& b);
Tensor sub(const Tensor& a, const Tensor& b);
Tensor mul(const Tensor& a, const Tensor& b);
Tensor div(const Tensor& a, const Tensor& b);
Tensor sqrt(const Tensor& t);
Tensor exp(const Tensor& t);
Tensor softmax(const Tensor& t, int dim = -1);
Tensor log_softmax(const Tensor& t, int dim = -1);
Tensor layer_norm(const Tensor& input, const Tensor& weight, const Tensor& bias, float eps = 1e-5);

} // namespace lightgpt
