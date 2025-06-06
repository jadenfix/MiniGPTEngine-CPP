#pragma once

#include "tensor.hpp"
#include "kv_cache.hpp"
#include <memory>
#include <vector>

namespace lightgpt {

// Forward declarations
class ModelLoader;

// Positional embedding type
enum class PositionalEmbeddingType {
    NONE,
    ABSOLUTE,
    ROTARY
};

// Attention configuration
struct AttentionConfig {
    int64_t head_dim;       // Dimension of each attention head
    int64_t num_heads;      // Number of attention heads
    int64_t num_kv_heads;   // Number of key/value heads (for grouped-query attention)
    float attn_dropout;     // Attention dropout probability
    bool is_causal;         // Whether to use causal masking
    PositionalEmbeddingType pos_emb_type; // Type of positional embedding
    float rope_theta;       // Base value for RoPE (if used)
};

// Feed-forward network configuration
struct FFNConfig {
    int64_t hidden_dim;     // Hidden dimension of the FFN
    std::string activation; // Activation function ("gelu", "relu", etc.)
    float dropout;          // Dropout probability
};

// Transformer block configuration
struct TransformerBlockConfig {
    int64_t dim;            // Dimension of the model
    AttentionConfig attn;    // Attention configuration
    FFNConfig ffn;          // FFN configuration
    float layer_norm_eps;    // Epsilon for layer normalization
    bool pre_norm;          // Whether to use pre-norm (true) or post-norm (false)
};

// Multi-head self-attention layer
class MultiHeadAttention {
public:
    MultiHeadAttention(const AttentionConfig& config);
    
    // Initialize weights from a model loader
    void load_weights(const ModelLoader& loader, const std::string& prefix);
    
    // Forward pass
    Tensor forward(const Tensor& x, const Tensor& mask = Tensor()) const;
    
    // Enable/disable gradient computation
    void train(bool mode = true) { training_ = mode; }
    bool is_training() const { return training_; }
    
private:
    AttentionConfig config_;
    
    // Model weights
    Tensor q_proj_weight_;
    Tensor k_proj_weight_;
    Tensor v_proj_weight_;
    Tensor out_proj_weight_;
    Tensor q_proj_bias_;
    Tensor k_proj_bias_;
    Tensor v_proj_bias_;
    Tensor out_proj_bias_;
    
    // KV cache for autoregressive generation
    std::unique_ptr<KVCache> kv_cache_;
    
    // Training mode
    bool training_ = false;
    
    // Helper methods
    Tensor apply_rotary_embeddings(const Tensor& x, int64_t seq_len) const;
    Tensor apply_attention(const Tensor& q, const Tensor& k, const Tensor& v, 
                          const Tensor& mask) const;
};

// Feed-forward network
class FeedForwardNetwork {
public:
    FeedForwardNetwork(const FFNConfig& config, int64_t dim);
    
    // Initialize weights from a model loader
    void load_weights(const ModelLoader& loader, const std::string& prefix);
    
    // Forward pass
    Tensor forward(const Tensor& x) const;
    
    // Enable/disable gradient computation
    void train(bool mode = true) { training_ = mode; }
    bool is_training() const { return training_; }
    
private:
    FFNConfig config_;
    [[maybe_unused]] int64_t dim_;  // Will be used for input dimension validation
    
    // Model weights
    Tensor fc1_weight_;
    Tensor fc1_bias_;
    Tensor fc2_weight_;
    Tensor fc2_bias_;
    
    // Training mode
    bool training_ = false;
};

// A single transformer block
class TransformerBlock {
public:
    TransformerBlock(const TransformerBlockConfig& config);
    
    // Initialize weights from a model loader
    void load_weights(const ModelLoader& loader, int layer_idx);
    
    // Forward pass
    Tensor forward(const Tensor& x, const Tensor& mask = Tensor()) const;
    
    // Enable/disable gradient computation
    void train(bool mode = true);
    bool is_training() const { return training_; }
    
private:
    TransformerBlockConfig config_;
    
    // Sub-modules
    std::unique_ptr<MultiHeadAttention> self_attn_;
    std::unique_ptr<FeedForwardNetwork> ffn_;
    
    // Layer normalization layers
    Tensor attn_norm_weight_;
    Tensor attn_norm_bias_;
    Tensor ffn_norm_weight_;
    Tensor ffn_norm_bias_;
    
    // Training mode
    bool training_ = false;
};

// Complete transformer model
class TransformerModel {
public:
    TransformerModel(
        int64_t vocab_size,
        int64_t max_seq_len,
        int64_t dim,
        int64_t num_layers,
        int64_t num_heads,
        int64_t num_kv_heads,
        int64_t ffn_hidden_dim,
        float layer_norm_eps = 1e-5,
        bool pre_norm = true,
        PositionalEmbeddingType pos_emb_type = PositionalEmbeddingType::ROTARY
    );
    
    // Initialize from a model loader
    static std::unique_ptr<TransformerModel> from_gguf(const ModelLoader& loader);
    
    // Forward pass
    Tensor forward(const Tensor& input_ids, const Tensor& attention_mask = Tensor()) const;
    
    // Generate tokens autoregressively
    std::vector<int64_t> generate(
        const std::vector<int64_t>& input_ids,
        int64_t max_length = 100,
        float temperature = 1.0f,
        int64_t top_k = 50,
        float top_p = 1.0f,
        bool do_sample = true
    ) const;
    
    // Enable/disable gradient computation
    void train(bool mode = true);
    bool is_training() const { return training_; }
    
private:
    // Model configuration
    int64_t vocab_size_;
    int64_t max_seq_len_;
    [[maybe_unused]] int64_t dim_;  // Will be used in future implementations
    int64_t num_layers_;
    PositionalEmbeddingType pos_emb_type_;
    
    // Embedding layers
    Tensor token_embeddings_;
    Tensor position_embeddings_;
    Tensor lm_head_weight_;
    
    // Transformer blocks
    std::vector<std::unique_ptr<TransformerBlock>> layers_;
    
    // Final layer norm
    Tensor final_norm_weight_;
    
    // Helper function to create causal mask
    Tensor create_causal_mask(int64_t seq_len) const;
    Tensor final_norm_bias_;
    
    // Training mode
    bool training_ = false;
    
    // KV cache for generation
    mutable std::vector<Tensor> k_cache_;
    mutable std::vector<Tensor> v_cache_;
    
    // Helper methods
    void create_position_embeddings();
    Tensor get_position_embeddings(int64_t seq_len) const;
};

} // namespace lightgpt
