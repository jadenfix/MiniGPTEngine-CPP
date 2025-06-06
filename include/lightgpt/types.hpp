#pragma once

#include <cstdint>
#include <vector>
#include <string>

namespace lightgpt {

/**
 * @brief Supported data types for tensors
 */
enum class DType : uint32_t {
    F32 = 0,    // 32-bit float
    F16 = 1,    // 16-bit float  
    Q8_0 = 2,   // 8-bit quantized (block-wise)
    Q8_1 = 3,   // 8-bit quantized (block-wise + zero point)
    Q4_0 = 6,   // 4-bit quantized (block-wise)
    Q4_1 = 7,   // 4-bit quantized (block-wise + zero point)
};

/**
 * @brief Get size in bytes for a data type
 */
constexpr size_t dtype_size(DType dtype) {
    switch (dtype) {
        case DType::F32: return 4;
        case DType::F16: return 2;
        case DType::Q8_0: return 1; // Approximate
        case DType::Q8_1: return 1; // Approximate
        case DType::Q4_0: return 1; // Approximate (actually 0.5 bytes per element)
        case DType::Q4_1: return 1; // Approximate (actually 0.5 bytes per element)
        default: return 4;
    }
}

/**
 * @brief Convert DType enum to string
 */
inline std::string dtype_to_string(DType dtype) {
    switch (dtype) {
        case DType::F32: return "F32";
        case DType::F16: return "F16";
        case DType::Q8_0: return "Q8_0";
        case DType::Q8_1: return "Q8_1";
        case DType::Q4_0: return "Q4_0";
        case DType::Q4_1: return "Q4_1";
        default: return "UNKNOWN";
    }
}

/**
 * @brief Tensor shape type
 */
using Shape = std::vector<int64_t>;

/**
 * @brief Model configuration parameters
 */
struct ModelConfig {
    std::string architecture = "gpt2";
    size_t vocab_size = 50257;
    size_t max_position_embeddings = 1024;
    size_t hidden_size = 768;
    size_t num_layers = 12;
    size_t num_heads = 12;
    size_t num_kv_heads = 12;  // For multi-query attention
    size_t intermediate_size = 3072;
    float layer_norm_epsilon = 1e-5f;
    bool use_cache = true;
    
    // Position embedding type
    enum class PositionEmbedding {
        LEARNED,    // Standard learned position embeddings
        ROTARY,     // Rotary Position Embedding (RoPE)
        ALIBI       // Attention with Linear Biases
    } position_embedding_type = PositionEmbedding::LEARNED;
    
    // Activation function
    enum class Activation {
        GELU,
        RELU,
        SWISH,
        SILU
    } activation = Activation::GELU;
};

/**
 * @brief Attention configuration
 */
struct AttentionConfig {
    size_t head_dim;
    size_t num_heads;
    size_t num_kv_heads;
    bool use_bias = true;
    float attention_dropout = 0.0f;
    float rope_theta = 10000.0f;  // For RoPE
    int rope_scaling = 1;         // For RoPE scaling
};

/**
 * @brief Generation configuration
 */
struct GenerationConfig {
    size_t max_new_tokens = 100;
    float temperature = 1.0f;
    size_t top_k = 50;
    float top_p = 0.9f;
    bool do_sample = true;
    size_t repetition_penalty = 1.0f;
    std::vector<int32_t> stop_tokens;
    
    enum class SamplingStrategy {
        GREEDY,     // Always pick highest probability
        TOP_K,      // Sample from top-k tokens
        TOP_P,      // Nucleus sampling
        TEMPERATURE // Temperature sampling
    } strategy = SamplingStrategy::GREEDY;
};

/**
 * @brief GGUF file format constants
 */
namespace gguf {
    constexpr uint32_t MAGIC = 0x46554747; // "GGUF"
    constexpr uint32_t VERSION = 1;
    
    enum class ValueType : uint32_t {
        UINT8   = 0,
        INT8    = 1,
        UINT16  = 2,
        INT16   = 3,
        UINT32  = 4,
        INT32   = 5,
        FLOAT32 = 6,
        BOOL    = 7,
        STRING  = 8,
        ARRAY   = 9,
        UINT64  = 10,
        INT64   = 11,
        FLOAT64 = 12,
    };
}

} // namespace lightgpt 