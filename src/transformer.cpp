#include "lightgpt/transformer.hpp"
#include "lightgpt/tensor_ops.hpp"
#include "lightgpt/model_loader.hpp"
#include <cmath>
#include <stdexcept>
#include <iostream>

namespace lightgpt {

// Helper function to compute GELU activation
Tensor gelu(const Tensor& x) {
    // GELU approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    constexpr float sqrt_2_over_pi = 0.7978845608028654f;
    constexpr float coef = 0.044715f;
    
    // Create a temporary tensor for the result
    Tensor result(x.shape(), x.dtype());
    
    // Get raw pointers to the data
    const float* x_data = x.data<float>();
    float* result_data = result.data<float>();
    size_t n = x.numel();
    
    // Apply GELU element-wise
    for (size_t i = 0; i < n; ++i) {
        float val = x_data[i];
        float inner = sqrt_2_over_pi * (val + coef * val * val * val);
        result_data[i] = 0.5f * val * (1.0f + std::tanh(inner));
    }
    
    return result;
}

// MultiHeadAttention implementation
MultiHeadAttention::MultiHeadAttention(const AttentionConfig& config)
    : config_(config) {
    if (config_.head_dim <= 0) {
        throw std::invalid_argument("head_dim must be positive");
    }
    if (config_.num_heads <= 0) {
        throw std::invalid_argument("num_heads must be positive");
    }
    if (config_.num_kv_heads <= 0) {
        throw std::invalid_argument("num_kv_heads must be positive");
    }
    
    // Initialize KV cache
    kv_cache_ = std::make_unique<KVCache>(
        1,  // batch_size (will be set in forward pass)
        config_.num_kv_heads,
        config_.head_dim,
        2048,  // max_seq_len (configurable)
        DType::F32
    );
}

void MultiHeadAttention::load_weights(const ModelLoader& loader, const std::string& prefix) {
    q_proj_weight_ = loader.load_tensor(prefix + "q_proj.weight");
    k_proj_weight_ = loader.load_tensor(prefix + "k_proj.weight");
    v_proj_weight_ = loader.load_tensor(prefix + "v_proj.weight");
    out_proj_weight_ = loader.load_tensor(prefix + "out_proj.weight");
    
    // Optional biases
    try {
        q_proj_bias_ = loader.load_tensor(prefix + "q_proj.bias");
        k_proj_bias_ = loader.load_tensor(prefix + "k_proj.bias");
        v_proj_bias_ = loader.load_tensor(prefix + "v_proj.bias");
        out_proj_bias_ = loader.load_tensor(prefix + "out_proj.bias");
    } catch (const std::exception&) {
        // Biases are optional
    }
}

Tensor MultiHeadAttention::forward(const Tensor& x, const Tensor& mask) const {
    const int64_t batch_size = x.shape()[0];
    const int64_t seq_len = x.shape()[1];
    const int64_t head_dim = config_.head_dim;
    const int64_t num_heads = config_.num_heads;
    
    // Project Q, K, V
    Tensor q = matmul(x, q_proj_weight_);
    Tensor k = matmul(x, k_proj_weight_);
    Tensor v = matmul(x, v_proj_weight_);
    
    // Add biases if they exist
    if (q_proj_bias_.defined()) q = add(q, q_proj_bias_);
    if (k_proj_bias_.defined()) k = add(k, k_proj_bias_);
    if (v_proj_bias_.defined()) v = add(v, v_proj_bias_);
    
    // Reshape for multi-head attention: [batch, seq_len, num_heads, head_dim]
    q = q.view({batch_size, seq_len, num_heads, head_dim}).transpose(1, 2);
    k = k.view({batch_size, seq_len, config_.num_kv_heads, head_dim}).transpose(1, 2);
    v = v.view({batch_size, seq_len, config_.num_kv_heads, head_dim}).transpose(1, 2);
    
    // Update KV cache if in inference mode
    if (!training_) {
        kv_cache_->update(k, v, seq_len);
        auto [cached_k, cached_v] = kv_cache_->get();
        k = cached_k;
        v = cached_v;
    }
    
    // Scaled dot-product attention
    // Transpose last two dimensions of k for matrix multiplication
    Tensor k_t = k.transpose(-2, -1);
    Tensor scores = matmul(q, k_t);
    
    // Scale the scores
    float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
    const float* scores_data = scores.data<float>();
    float* scaled_scores_data = scores.data<float>();
    size_t num_elements = scores.numel();
    for (size_t i = 0; i < num_elements; ++i) {
        scaled_scores_data[i] = scores_data[i] * scale;
    }
    
    // Apply attention mask if provided
    if (mask.defined()) {
        scores = add(scores, mask);
    }
    
    // Apply attention weights and get output
    Tensor weights = softmax(scores, -1);
    Tensor output = matmul(weights, v);
    
    // Reshape back: [batch, num_heads, seq_len, head_dim] -> [batch, seq_len, hidden_dim]
    output = output.transpose(1, 2).contiguous()
                  .view({batch_size, seq_len, -1});
    
    // Final projection
    Tensor out_proj_weight_t = out_proj_weight_.transpose(-2, -1);
    output = matmul(output, out_proj_weight_t);
    if (out_proj_bias_.defined()) {
        output = output + out_proj_bias_;
    }
    
    return output;
}

// FeedForwardNetwork implementation
FeedForwardNetwork::FeedForwardNetwork(const FFNConfig& config, int64_t dim)
    : config_(config), dim_(dim) {
    if (dim <= 0) {
        throw std::invalid_argument("dim must be positive");
    }
}

void FeedForwardNetwork::load_weights(const ModelLoader& loader, const std::string& prefix) {
    fc1_weight_ = loader.load_tensor(prefix + "fc1.weight");
    fc2_weight_ = loader.load_tensor(prefix + "fc2.weight");
    
    try {
        fc1_bias_ = loader.load_tensor(prefix + "fc1.bias");
        fc2_bias_ = loader.load_tensor(prefix + "fc2.bias");
    } catch (const std::exception&) {
        // Biases are optional
    }
}

Tensor FeedForwardNetwork::forward(const Tensor& x) const {
    Tensor h = matmul(x, fc1_weight_.t());
    if (fc1_bias_.defined()) h = h + fc1_bias_;
    
    // Apply GELU activation (most common for LLMs)
    h = gelu(h);
    
    // Second linear layer
    h = matmul(h, fc2_weight_.t());
    if (fc2_bias_.defined()) {
        h = h + fc2_bias_;
    }
    
    return h;
}

// TransformerBlock implementation
TransformerBlock::TransformerBlock(const TransformerBlockConfig& config)
    : config_(config) {
    self_attn_ = std::make_unique<MultiHeadAttention>(config.attn);
    ffn_ = std::make_unique<FeedForwardNetwork>(config.ffn, config.dim);
}

void TransformerBlock::load_weights(const ModelLoader& loader, int layer_idx) {
    std::string prefix = "model.layers." + std::to_string(layer_idx) + ".";
    
    // Load attention weights
    self_attn_->load_weights(loader, prefix + "self_attn.");
    
    // Load MLP weights
    ffn_->load_weights(loader, prefix + "mlp.");
    
    // Load layer norm weights
    if (config_.pre_norm) {
        attn_norm_weight_ = loader.load_tensor(prefix + "input_layernorm.weight");
        attn_norm_bias_ = loader.load_tensor(prefix + "input_layernorm.bias");
        ffn_norm_weight_ = loader.load_tensor(prefix + "post_attention_layernorm.weight");
        ffn_norm_bias_ = loader.load_tensor(prefix + "post_attention_layernorm.bias");
    } else {
        // For post-norm, we might have different layer norm names
        attn_norm_weight_ = loader.load_tensor(prefix + "self_attn_layer_norm.weight");
        attn_norm_bias_ = loader.load_tensor(prefix + "self_attn_layer_norm.bias");
        ffn_norm_weight_ = loader.load_tensor(prefix + "final_layer_norm.weight");
        ffn_norm_bias_ = loader.load_tensor(prefix + "final_layer_norm.bias");
    }
}

Tensor TransformerBlock::forward(const Tensor& x, const Tensor& mask) const {
    Tensor h = x;
    
    if (config_.pre_norm) {
        // Pre-LayerNorm
        h = layer_norm(h, attn_norm_weight_, attn_norm_bias_, config_.layer_norm_eps);
        Tensor attn_output = self_attn_->forward(h, mask);
        
        // Residual connection
        h = x + attn_output;
        
        // FFN with residual
        Tensor ffn_norm = layer_norm(h, ffn_norm_weight_, ffn_norm_bias_, config_.layer_norm_eps);
        Tensor ffn_output = ffn_->forward(ffn_norm);
        return h + ffn_output;
    } else {
        // Post-LayerNorm (original Transformer)
        Tensor attn_output = self_attn_->forward(h, mask);
        h = h + attn_output;
        h = layer_norm(h, attn_norm_weight_, attn_norm_bias_, config_.layer_norm_eps);
        
        // FFN with residual
        Tensor ffn_output = ffn_->forward(h);
        h = h + ffn_output;
        return layer_norm(h, ffn_norm_weight_, ffn_norm_bias_, config_.layer_norm_eps);
    }
}

// TransformerModel implementation
TransformerModel::TransformerModel(
    int64_t vocab_size,
    int64_t max_seq_len,
    int64_t dim,
    int64_t num_layers,
    int64_t num_heads,
    int64_t num_kv_heads,
    int64_t ffn_hidden_dim,
    float layer_norm_eps,
    bool pre_norm,
    PositionalEmbeddingType pos_emb_type)
    : vocab_size_(vocab_size),
      max_seq_len_(max_seq_len),
      dim_(dim),
      num_layers_(num_layers),
      pos_emb_type_(pos_emb_type) {
    
    if (vocab_size <= 0 || max_seq_len <= 0 || dim <= 0 || 
        num_layers <= 0 || num_heads <= 0 || ffn_hidden_dim <= 0) {
        throw std::invalid_argument("All model dimensions must be positive");
    }
    
    // Store the dimension in a local variable to ensure it's used
    const int64_t model_dim = dim_;
    
    // Initialize the transformer blocks with proper configuration
    for (int i = 0; i < num_layers; ++i) {
        TransformerBlockConfig block_config;
        block_config.dim = model_dim;  // Use the local variable
        block_config.layer_norm_eps = layer_norm_eps;
        block_config.pre_norm = pre_norm;
        
        // Configure attention
        block_config.attn.head_dim = dim_ / num_heads;
        block_config.attn.num_heads = num_heads;
        block_config.attn.num_kv_heads = num_kv_heads > 0 ? num_kv_heads : num_heads;
        block_config.attn.attn_dropout = 0.0f;  // No dropout for now
        block_config.attn.is_causal = true;
        block_config.attn.pos_emb_type = pos_emb_type_;
        block_config.attn.rope_theta = 10000.0f;  // Default RoPE theta
        
        // Configure FFN
        block_config.ffn.hidden_dim = ffn_hidden_dim > 0 ? ffn_hidden_dim : 4 * dim_;
        block_config.ffn.activation = "gelu";
        block_config.ffn.dropout = 0.0f;  // No dropout for now
        
        layers_.push_back(std::make_unique<TransformerBlock>(block_config));
    }
    
    // Initialize embedding layers with the correct dimensions using dim_
    token_embeddings_ = Tensor({vocab_size, dim_}, DType::F32);
    position_embeddings_ = Tensor({max_seq_len, dim_}, DType::F32);
    lm_head_weight_ = Tensor({vocab_size, dim_}, DType::F32);
    final_norm_weight_ = Tensor({dim_}, DType::F32);
    final_norm_bias_ = Tensor({dim_}, DType::F32);
    
    // Initialize the position embeddings (simple example - could be improved with learned embeddings)
    float* pos_emb_ptr = position_embeddings_.data<float>();
    for (int64_t i = 0; i < max_seq_len_; ++i) {
        for (int64_t j = 0; j < dim_; ++j) {
            // Simple positional encoding (could be replaced with learned embeddings)
            pos_emb_ptr[i * dim_ + j] = static_cast<float>(i) / static_cast<float>(max_seq_len_);
        }
    }
    
    // Initialize transformer blocks
    for (int i = 0; i < num_layers; ++i) {
        TransformerBlockConfig block_config;
        block_config.dim = dim_;
        block_config.layer_norm_eps = layer_norm_eps;
        block_config.pre_norm = pre_norm;
        
        // Configure attention
        block_config.attn.head_dim = dim / num_heads;
        block_config.attn.num_heads = num_heads;
        block_config.attn.num_kv_heads = num_kv_heads;
        block_config.attn.attn_dropout = 0.0f;  // No dropout for now
        block_config.attn.is_causal = true;
        block_config.attn.pos_emb_type = pos_emb_type;
        
        // Configure FFN
        block_config.ffn.hidden_dim = ffn_hidden_dim;
        block_config.ffn.activation = "gelu";
        block_config.ffn.dropout = 0.0f;  // No dropout for now
        
        layers_.push_back(std::make_unique<TransformerBlock>(block_config));
    }
}

std::unique_ptr<TransformerModel> TransformerModel::from_gguf(const ModelLoader& loader) {
    // Get model configuration from loader
    const int64_t vocab_size = loader.get_vocab_size();
    const int64_t max_seq_len = loader.get_max_sequence_length();
    const int64_t hidden_dim = loader.get_embedding_dim();
    const int64_t num_layers = loader.get_num_layers();
    const int64_t num_heads = loader.get_num_heads();
    const int64_t num_kv_heads = loader.get_num_kv_heads();
    
    // Create model config
    auto model = std::make_unique<TransformerModel>(
        vocab_size,
        max_seq_len,
        hidden_dim,
        num_layers,
        num_heads,
        num_kv_heads > 0 ? num_kv_heads : num_heads,  // Fallback to num_heads if not specified
        hidden_dim * 4,  // FFN hidden dim
        1e-5f,  // layer norm eps
        true,   // pre_norm
        PositionalEmbeddingType::NONE  // Start simple
    );
    
    // Load embeddings
    model->token_embeddings_ = loader.load_tensor("model.embed_tokens.weight");
    model->final_norm_weight_ = loader.load_tensor("model.norm.weight");
    model->final_norm_bias_ = loader.load_tensor("model.norm.bias");
    
    // Load transformer blocks
    for (int i = 0; i < num_layers; i++) {
        model->layers_[i]->load_weights(loader, i);
    }
    
    return model;
}

Tensor TransformerModel::forward(const Tensor& input_ids, const Tensor& attention_mask) const {
    // Get batch size and sequence length
    const int64_t batch_size = input_ids.shape()[0];
    const int64_t seq_len = input_ids.shape()[1];
    
    // Get token embeddings
    // TODO: Implement index_select or find alternative approach
    // For now, create a simple embedding lookup
    std::vector<float> emb_data;
    const float* emb_ptr = token_embeddings_.data<float>();
    const int64_t* ids_ptr = input_ids.data<int64_t>();
    int64_t emb_dim = token_embeddings_.shape()[1];
    
    for (int64_t i = 0; i < batch_size * seq_len; ++i) {
        int64_t token_id = ids_ptr[i];
        if (token_id < 0 || token_id >= vocab_size_) {
            throw std::out_of_range("Token ID out of range");
        }
        const float* token_emb = emb_ptr + token_id * emb_dim;
        emb_data.insert(emb_data.end(), token_emb, token_emb + emb_dim);
    }
    
    Tensor x({batch_size, seq_len, emb_dim}, DType::F32, emb_data.data(), true);
    
    // Add positional embeddings if using absolute position embeddings
    if (pos_emb_type_ == PositionalEmbeddingType::ABSOLUTE) {
        // TODO: Implement position embeddings
        // For now, skip adding position embeddings
    }
    
    // Create attention mask if not provided
    Tensor mask = attention_mask;
    if (!mask.defined()) {
        mask = create_causal_mask(seq_len);
    }
    
    // Forward through transformer blocks
    for (const auto& layer : layers_) {
        x = layer->forward(x, mask);
    }
    
    // Apply final layer norm
    x = layer_norm(x, final_norm_weight_, final_norm_bias_, 1e-5);
    
    // Compute logits
    Tensor logits = matmul(x, lm_head_weight_.t());
    
    return logits;
}

// Helper function to create attention mask for causal language modeling
Tensor TransformerModel::create_causal_mask(int64_t seq_len) const {
    // Create a lower triangular matrix with -inf above the diagonal
    std::vector<float> mask_data(seq_len * seq_len, 0.0f);
    float* mask_ptr = mask_data.data();
    
    for (int64_t i = 0; i < seq_len; ++i) {
        for (int64_t j = i + 1; j < seq_len; ++j) {
            // Set to -inf to mask future positions
            mask_ptr[i * seq_len + j] = -std::numeric_limits<float>::infinity();
        }
    }
    
    return Tensor({seq_len, seq_len}, DType::F32, mask_data.data(), true);
}

// ... (rest of the code remains the same)

} // namespace lightgpt
