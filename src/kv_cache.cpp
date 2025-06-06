#include "lightgpt/kv_cache.hpp"
#include "lightgpt/tensor_ops.hpp"
#include <stdexcept>
#include <cassert>

namespace lightgpt {

KVCache::KVCache(
    int64_t batch_size,
    int64_t num_heads,
    int64_t head_dim,
    int64_t max_seq_len,
    DType dtype
) : current_seq_len_(0),
    max_seq_len_(max_seq_len),
    batch_size_(batch_size),
    num_heads_(num_heads),
    head_dim_(head_dim) {
    
    if (batch_size <= 0 || num_heads <= 0 || head_dim <= 0 || max_seq_len <= 0) {
        throw std::invalid_argument("All cache dimensions must be positive");
    }
    
    // Initialize key and value caches
    std::vector<int64_t> cache_shape = {batch_size, num_heads, max_seq_len, head_dim};
    k_cache_ = zeros(cache_shape, dtype);
    v_cache_ = zeros(cache_shape, dtype);
}

std::pair<Tensor, Tensor> KVCache::get() const {
    // For now, return the entire cache
    // TODO: Implement proper slicing based on current_seq_len_
    return {k_cache_, v_cache_};
}

void KVCache::update(const Tensor& k, const Tensor& v, int64_t seq_len) {
    if (k.dim() != 4 || v.dim() != 4) {
        throw std::runtime_error("K and V must be 4D tensors [batch, num_heads, seq_len, head_dim]");
    }
    
    // Check input shapes
    if (k.dim() != 4 || v.dim() != 4 ||
        k.shape()[0] != batch_size_ || v.shape()[0] != batch_size_ ||
        k.shape()[1] != num_heads_ || v.shape()[1] != num_heads_ ||
        k.shape()[3] != head_dim_ || v.shape()[3] != head_dim_) {
        throw std::invalid_argument("Invalid input tensor shapes for KV cache update");
    }
    
    // Update the current sequence length
    current_seq_len_ = seq_len;
    
    // Update the cache at the current position
    // For now, we'll do a simple copy for the current sequence position
    // This assumes k and v are properly sized tensors for a single timestep
    assert(current_seq_len_ < max_seq_len_);
    
    // Get pointers to the data
    float* k_cache_data = k_cache_.data<float>();
    float* v_cache_data = v_cache_.data<float>();
    const float* k_data = k.data<float>();
    const float* v_data = v.data<float>();
    
    // Calculate the offset in the cache for this timestep
    size_t offset = current_seq_len_ * head_dim_;
    size_t timestep_size = head_dim_ * sizeof(float);
    
    // Copy the data for each batch and head
    for (int64_t b = 0; b < batch_size_; ++b) {
        for (int64_t h = 0; h < num_heads_; ++h) {
            // Calculate the source and destination offsets
            size_t cache_offset = ((b * num_heads_ + h) * max_seq_len_ * head_dim_) + offset;
            size_t src_offset = (b * num_heads_ + h) * head_dim_;
            
            // Copy the data
            std::memcpy(k_cache_data + cache_offset, k_data + src_offset, timestep_size);
            std::memcpy(v_cache_data + cache_offset, v_data + src_offset, timestep_size);
        }
    }
    
    // Increment sequence length
    current_seq_len_++;
}

void KVCache::clear() {
    current_seq_len_ = 0;
    // Reset the cache tensors
    std::vector<int64_t> cache_shape = {batch_size_, num_heads_, max_seq_len_, head_dim_};
    k_cache_ = zeros(cache_shape, k_cache_.dtype());
    v_cache_ = zeros(cache_shape, v_cache_.dtype());
}

} // namespace lightgpt
