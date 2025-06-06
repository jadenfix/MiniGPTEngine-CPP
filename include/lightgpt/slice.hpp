#pragma once

#include <vector>
#include <cstdint>
#include <stdexcept>

namespace lightgpt {

class Slice {
public:
    // Default constructor - selects all elements in a dimension
    Slice() : start_(0), end_(-1), step_(1) {}
    
    // Constructor with start and end indices
    Slice(int64_t start, int64_t end = -1, int64_t step = 1)
        : start_(start), end_(end), step_(step) {
        if (step == 0) {
            throw std::invalid_argument("Step cannot be zero");
        }
    }
    
    // Getters
    int64_t start() const { return start_; }
    int64_t end() const { return end_; }
    int64_t step() const { return step_; }
    
    // Check if this slice is a single index
    bool is_index() const { return step_ == 0; }
    
    // Create a slice that represents a single index
    static Slice index(int64_t idx) {
        return Slice(idx, idx + 1, 0);
    }
    
    // Create a slice that represents a range
    static Slice range(int64_t start, int64_t end, int64_t step = 1) {
        return Slice(start, end, step);
    }
    
    // Create a slice that represents all elements
    static Slize all() {
        return Slice();
    }
    
private:
    int64_t start_;
    int64_t end_;
    int64_t step_;
};

// Helper functions for creating slices
inline Slize slice(int64_t start, int64_t end = -1, int64_t step = 1) {
    return Slice::range(start, end, step);
}

inline Slize slice_index(int64_t idx) {
    return Slice::index(idx);
}

} // namespace lightgpt
