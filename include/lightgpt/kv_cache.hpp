#pragma once

#include "tensor.hpp"
#include <vector>
#include <memory>

namespace lightgpt {

/**
 * @brief Key-Value cache for transformer attention layers
 */
class KVCache {
public:
    /**
     * @brief Initialize the KV cache
     */
    KVCache(
        int64_t batch_size,
        int64_t num_heads,
        int64_t head_dim,
        int64_t max_seq_len,
        DType dtype = DType::F32
    );
    
    ~KVCache() = default;
    
    // Disable copy
    KVCache(const KVCache&) = delete;
    KVCache& operator=(const KVCache&) = delete;
    
    // Allow move
    KVCache(KVCache&&) = default;
    KVCache& operator=(KVCache&&) = default;
    
    /**
     * @brief Get the key and value tensors
     * @return std::pair<Tensor, Tensor> Key and value tensors
     */
    std::pair<Tensor, Tensor> get() const;
    
    /**
     * @brief Update the cache with new key and value tensors
     * @param k New key tensor to append to cache
     * @param v New value tensor to append to cache
     * @param seq_len Length of the sequence being added
     */
    void update(const Tensor& k, const Tensor& v, int64_t seq_len);
    
    /**
     * @brief Clear the cache
     */
    void clear();
    
    /**
     * @brief Get the current sequence length in the cache
     */
    int64_t current_seq_len() const { return current_seq_len_; }
    
    /**
     * @brief Get the maximum sequence length the cache can hold
     */
    int64_t max_seq_len() const { return max_seq_len_; }

private:
    Tensor k_cache_;  // [batch, num_heads, max_seq_len, head_dim]
    Tensor v_cache_;  // [batch, num_heads, max_seq_len, head_dim]
    
    int64_t current_seq_len_ = 0;
    int64_t max_seq_len_ = 0;
    int64_t batch_size_ = 0;
    int64_t num_heads_ = 0;
    int64_t head_dim_ = 0;
};

} // namespace lightgpt
