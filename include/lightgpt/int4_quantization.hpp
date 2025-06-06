#pragma once

#include <vector>
#include <cstdint>
#include <cmath>
#include <algorithm>
#include <immintrin.h>
#include <memory>

namespace lightgpt {

// INT4 block-wise quantization for maximum compression with accuracy preservation
class INT4BlockQuantizer {
public:
    static constexpr size_t BLOCK_SIZE = 32;  // Optimal for SIMD operations
    static constexpr size_t PACKED_BLOCK_SIZE = BLOCK_SIZE / 2;  // 2 values per byte
    
    struct QuantizedBlock {
        uint8_t data[PACKED_BLOCK_SIZE];  // Packed 4-bit values
        float scale;                      // Block scaling factor
        float zero_point;                 // Block zero point for asymmetric quantization
        
        QuantizedBlock() : scale(1.0f), zero_point(0.0f) {
            std::fill(data, data + PACKED_BLOCK_SIZE, 0);
        }
    };
    
    struct QuantizedTensor {
        std::vector<QuantizedBlock> blocks;
        std::vector<size_t> shape;
        size_t total_elements;
        bool is_symmetric;
        
        QuantizedTensor() : total_elements(0), is_symmetric(true) {}
        
        size_t memory_usage() const {
            return blocks.size() * (sizeof(QuantizedBlock));
        }
        
        float compression_ratio() const {
            size_t original_size = total_elements * sizeof(float);
            return static_cast<float>(original_size) / memory_usage();
        }
    };

private:
    bool use_symmetric_quantization_;
    float outlier_threshold_;
    
public:
    INT4BlockQuantizer(bool symmetric = false, float outlier_threshold = 3.0f)
        : use_symmetric_quantization_(symmetric), outlier_threshold_(outlier_threshold) {}
    
    // Quantize tensor with block-wise scaling
    QuantizedTensor quantize(const std::vector<float>& data, const std::vector<size_t>& shape) {
        QuantizedTensor result;
        result.shape = shape;
        result.total_elements = data.size();
        result.is_symmetric = use_symmetric_quantization_;
        
        size_t num_blocks = (data.size() + BLOCK_SIZE - 1) / BLOCK_SIZE;
        result.blocks.resize(num_blocks);
        
        #pragma omp parallel for
        for (size_t block_idx = 0; block_idx < num_blocks; ++block_idx) {
            quantize_block(data, block_idx, result.blocks[block_idx]);
        }
        
        return result;
    }
    
    // Dequantize tensor
    std::vector<float> dequantize(const QuantizedTensor& quantized) {
        std::vector<float> result(quantized.total_elements);
        
        #pragma omp parallel for
        for (size_t block_idx = 0; block_idx < quantized.blocks.size(); ++block_idx) {
            dequantize_block(quantized.blocks[block_idx], block_idx, result);
        }
        
        return result;
    }
    
    // Fast matrix multiplication with quantized weights
    void quantized_matmul(const std::vector<float>& input, const QuantizedTensor& weights,
                         std::vector<float>& output, size_t rows, size_t cols, size_t inner_dim) {
        
        output.resize(rows * cols);
        std::fill(output.begin(), output.end(), 0.0f);
        
        #pragma omp parallel for collapse(2)
        for (size_t i = 0; i < rows; ++i) {
            for (size_t j = 0; j < cols; ++j) {
                float sum = 0.0f;
                
                // Process blocks of the inner dimension
                size_t num_blocks = (inner_dim + BLOCK_SIZE - 1) / BLOCK_SIZE;
                
                for (size_t block_idx = 0; block_idx < num_blocks; ++block_idx) {
                    size_t block_start = block_idx * BLOCK_SIZE;
                    size_t block_end = std::min(block_start + BLOCK_SIZE, inner_dim);
                    
                    const QuantizedBlock& weight_block = weights.blocks[j * num_blocks + block_idx];
                    
                    sum += compute_block_dot_product(&input[i * inner_dim + block_start],
                                                   weight_block, block_end - block_start);
                }
                
                output[i * cols + j] = sum;
            }
        }
    }
    
    // Optimized quantized convolution
    void quantized_conv1d(const std::vector<float>& input, const QuantizedTensor& kernel,
                         std::vector<float>& output, size_t seq_len, size_t in_channels,
                         size_t out_channels, size_t kernel_size, size_t stride = 1) {
        
        size_t out_len = (seq_len - kernel_size) / stride + 1;
        output.resize(out_len * out_channels);
        std::fill(output.begin(), output.end(), 0.0f);
        
        #pragma omp parallel for collapse(2)
        for (size_t out_pos = 0; out_pos < out_len; ++out_pos) {
            for (size_t out_ch = 0; out_ch < out_channels; ++out_ch) {
                float sum = 0.0f;
                
                for (size_t in_ch = 0; in_ch < in_channels; ++in_ch) {
                    for (size_t k = 0; k < kernel_size; ++k) {
                        size_t in_pos = out_pos * stride + k;
                        float input_val = input[in_pos * in_channels + in_ch];
                        
                        // Get quantized kernel weight
                        size_t kernel_idx = (out_ch * in_channels + in_ch) * kernel_size + k;
                        float weight = dequantize_single_value(kernel, kernel_idx);
                        
                        sum += input_val * weight;
                    }
                }
                
                output[out_pos * out_channels + out_ch] = sum;
            }
        }
    }

private:
    void quantize_block(const std::vector<float>& data, size_t block_idx, QuantizedBlock& block) {
        size_t start_idx = block_idx * BLOCK_SIZE;
        size_t end_idx = std::min(start_idx + BLOCK_SIZE, data.size());
        size_t block_size = end_idx - start_idx;
        
        if (block_size == 0) return;
        
        // Find min/max with outlier handling
        std::vector<float> block_data(data.begin() + start_idx, data.begin() + end_idx);
        handle_outliers(block_data);
        
        float min_val = *std::min_element(block_data.begin(), block_data.end());
        float max_val = *std::max_element(block_data.begin(), block_data.end());
        
        if (use_symmetric_quantization_) {
            quantize_symmetric_block(block_data, block, min_val, max_val);
        } else {
            quantize_asymmetric_block(block_data, block, min_val, max_val);
        }
    }
    
    void quantize_symmetric_block(const std::vector<float>& block_data, QuantizedBlock& block,
                                 float min_val, float max_val) {
        float abs_max = std::max(std::abs(min_val), std::abs(max_val));
        if (abs_max == 0.0f) {
            block.scale = 1.0f;
            block.zero_point = 0.0f;
            return;
        }
        
        // For 4-bit symmetric: range is [-7, 7]
        block.scale = abs_max / 7.0f;
        block.zero_point = 0.0f;
        
        // Quantize and pack values
        for (size_t i = 0; i < block_data.size(); i += 2) {
            int8_t q1 = static_cast<int8_t>(std::round(block_data[i] / block.scale));
            q1 = std::clamp(q1, static_cast<int8_t>(-7), static_cast<int8_t>(7));
            
            int8_t q2 = 0;
            if (i + 1 < block_data.size()) {
                q2 = static_cast<int8_t>(std::round(block_data[i + 1] / block.scale));
                q2 = std::clamp(q2, static_cast<int8_t>(-7), static_cast<int8_t>(7));
            }
            
            // Pack two 4-bit values into one byte
            // Convert signed to unsigned for storage: add 8
            uint8_t packed = ((q1 + 8) & 0x0F) | (((q2 + 8) & 0x0F) << 4);
            block.data[i / 2] = packed;
        }
    }
    
    void quantize_asymmetric_block(const std::vector<float>& block_data, QuantizedBlock& block,
                                  float min_val, float max_val) {
        if (max_val == min_val) {
            block.scale = 1.0f;
            block.zero_point = min_val;
            return;
        }
        
        // For 4-bit asymmetric: range is [0, 15]
        block.scale = (max_val - min_val) / 15.0f;
        block.zero_point = min_val;
        
        // Quantize and pack values
        for (size_t i = 0; i < block_data.size(); i += 2) {
            uint8_t q1 = static_cast<uint8_t>(std::round((block_data[i] - min_val) / block.scale));
            q1 = std::clamp(q1, static_cast<uint8_t>(0), static_cast<uint8_t>(15));
            
            uint8_t q2 = 0;
            if (i + 1 < block_data.size()) {
                q2 = static_cast<uint8_t>(std::round((block_data[i + 1] - min_val) / block.scale));
                q2 = std::clamp(q2, static_cast<uint8_t>(0), static_cast<uint8_t>(15));
            }
            
            // Pack two 4-bit values into one byte
            uint8_t packed = (q1 & 0x0F) | ((q2 & 0x0F) << 4);
            block.data[i / 2] = packed;
        }
    }
    
    void dequantize_block(const QuantizedBlock& block, size_t block_idx, std::vector<float>& output) {
        size_t start_idx = block_idx * BLOCK_SIZE;
        size_t end_idx = std::min(start_idx + BLOCK_SIZE, output.size());
        
        for (size_t i = start_idx; i < end_idx; i += 2) {
            size_t packed_idx = (i - start_idx) / 2;
            uint8_t packed = block.data[packed_idx];
            
            if (use_symmetric_quantization_) {
                // Unpack and convert back to signed
                int8_t q1 = static_cast<int8_t>((packed & 0x0F) - 8);
                output[i] = q1 * block.scale;
                
                if (i + 1 < end_idx) {
                    int8_t q2 = static_cast<int8_t>(((packed >> 4) & 0x0F) - 8);
                    output[i + 1] = q2 * block.scale;
                }
            } else {
                // Asymmetric dequantization
                uint8_t q1 = packed & 0x0F;
                output[i] = q1 * block.scale + block.zero_point;
                
                if (i + 1 < end_idx) {
                    uint8_t q2 = (packed >> 4) & 0x0F;
                    output[i + 1] = q2 * block.scale + block.zero_point;
                }
            }
        }
    }
    
    float dequantize_single_value(const QuantizedTensor& tensor, size_t global_idx) {
        size_t block_idx = global_idx / BLOCK_SIZE;
        size_t local_idx = global_idx % BLOCK_SIZE;
        
        if (block_idx >= tensor.blocks.size()) return 0.0f;
        
        const QuantizedBlock& block = tensor.blocks[block_idx];
        size_t packed_idx = local_idx / 2;
        bool is_second = (local_idx % 2) == 1;
        
        uint8_t packed = block.data[packed_idx];
        
        if (tensor.is_symmetric) {
            int8_t q = is_second ? static_cast<int8_t>(((packed >> 4) & 0x0F) - 8)
                                 : static_cast<int8_t>((packed & 0x0F) - 8);
            return q * block.scale;
        } else {
            uint8_t q = is_second ? ((packed >> 4) & 0x0F) : (packed & 0x0F);
            return q * block.scale + block.zero_point;
        }
    }
    
    float compute_block_dot_product(const float* input, const QuantizedBlock& weight_block, size_t size) {
        float sum = 0.0f;
        
        #ifdef __AVX2__
        if (size >= 8) {
            __m256 acc = _mm256_setzero_ps();
            __m256 scale_vec = _mm256_set1_ps(weight_block.scale);
            __m256 zero_vec = _mm256_set1_ps(weight_block.zero_point);
            
            for (size_t i = 0; i + 8 <= size; i += 8) {
                // Load input values
                __m256 input_vec = _mm256_loadu_ps(&input[i]);
                
                // Dequantize weights (simplified for demonstration)
                __m256 weight_vec = dequantize_8_weights_avx2(weight_block, i, scale_vec, zero_vec);
                
                // Multiply and accumulate
                __m256 prod = _mm256_mul_ps(input_vec, weight_vec);
                acc = _mm256_add_ps(acc, prod);
            }
            
            // Horizontal sum
            __m128 sum_quad = _mm_add_ps(_mm256_extractf128_ps(acc, 1), _mm256_extractf128_ps(acc, 0));
            __m128 sum_dual = _mm_add_ps(sum_quad, _mm_shuffle_ps(sum_quad, sum_quad, _MM_SHUFFLE(2, 3, 0, 1)));
            __m128 sum_single = _mm_add_ss(sum_dual, _mm_shuffle_ps(sum_dual, sum_dual, _MM_SHUFFLE(1, 0, 3, 2)));
            sum = _mm_cvtss_f32(sum_single);
            
            // Handle remaining elements
            for (size_t i = (size / 8) * 8; i < size; ++i) {
                float weight = dequantize_single_value_from_block(weight_block, i);
                sum += input[i] * weight;
            }
        } else
        #endif
        {
            for (size_t i = 0; i < size; ++i) {
                float weight = dequantize_single_value_from_block(weight_block, i);
                sum += input[i] * weight;
            }
        }
        
        return sum;
    }
    
    float dequantize_single_value_from_block(const QuantizedBlock& block, size_t local_idx) {
        size_t packed_idx = local_idx / 2;
        bool is_second = (local_idx % 2) == 1;
        
        uint8_t packed = block.data[packed_idx];
        
        if (use_symmetric_quantization_) {
            int8_t q = is_second ? static_cast<int8_t>(((packed >> 4) & 0x0F) - 8)
                                 : static_cast<int8_t>((packed & 0x0F) - 8);
            return q * block.scale;
        } else {
            uint8_t q = is_second ? ((packed >> 4) & 0x0F) : (packed & 0x0F);
            return q * block.scale + block.zero_point;
        }
    }
    
    #ifdef __AVX2__
    __m256 dequantize_8_weights_avx2(const QuantizedBlock& block, size_t start_idx,
                                    __m256 scale_vec, __m256 zero_vec) {
        // Load 4 packed bytes (8 quantized values)
        uint32_t packed_data;
        std::memcpy(&packed_data, &block.data[start_idx / 2], sizeof(uint32_t));
        
        // Extract 8 4-bit values
        alignas(32) float weights[8];
        for (int i = 0; i < 8; ++i) {
            uint8_t q = (packed_data >> (i * 4)) & 0x0F;
            if (use_symmetric_quantization_) {
                weights[i] = static_cast<float>(static_cast<int8_t>(q - 8));
            } else {
                weights[i] = static_cast<float>(q);
            }
        }
        
        __m256 weight_vec = _mm256_load_ps(weights);
        weight_vec = _mm256_mul_ps(weight_vec, scale_vec);
        
        if (!use_symmetric_quantization_) {
            weight_vec = _mm256_add_ps(weight_vec, zero_vec);
        }
        
        return weight_vec;
    }
    #endif
    
    void handle_outliers(std::vector<float>& data) {
        if (data.empty()) return;
        
        // Calculate mean and standard deviation
        float mean = 0.0f;
        for (float val : data) mean += val;
        mean /= data.size();
        
        float variance = 0.0f;
        for (float val : data) {
            float diff = val - mean;
            variance += diff * diff;
        }
        variance /= data.size();
        float std_dev = std::sqrt(variance);
        
        // Clip outliers
        float lower_bound = mean - outlier_threshold_ * std_dev;
        float upper_bound = mean + outlier_threshold_ * std_dev;
        
        for (float& val : data) {
            val = std::clamp(val, lower_bound, upper_bound);
        }
    }
};

// Utility functions for model conversion
class ModelQuantizer {
public:
    struct QuantizationConfig {
        bool quantize_embeddings = false;    // Usually keep full precision
        bool quantize_attention = true;      // Safe to quantize
        bool quantize_feedforward = true;    // Safe to quantize
        bool quantize_output = false;        // Usually keep full precision
        bool use_symmetric = false;          // Asymmetric usually better
        float outlier_threshold = 3.0f;
        
        QuantizationConfig() = default;
    };
    
private:
    INT4BlockQuantizer quantizer_;
    QuantizationConfig config_;
    
public:
    ModelQuantizer(const QuantizationConfig& config = QuantizationConfig())
        : quantizer_(config.use_symmetric, config.outlier_threshold), config_(config) {}
    
    // Quantize specific layer types
    INT4BlockQuantizer::QuantizedTensor quantize_linear_layer(
        const std::vector<float>& weights, const std::vector<size_t>& shape) {
        return quantizer_.quantize(weights, shape);
    }
    
    INT4BlockQuantizer::QuantizedTensor quantize_attention_weights(
        const std::vector<float>& weights, const std::vector<size_t>& shape) {
        if (!config_.quantize_attention) {
            // Return dummy quantized tensor - in practice, return original weights
            return INT4BlockQuantizer::QuantizedTensor();
        }
        return quantizer_.quantize(weights, shape);
    }
    
    INT4BlockQuantizer::QuantizedTensor quantize_feedforward_weights(
        const std::vector<float>& weights, const std::vector<size_t>& shape) {
        if (!config_.quantize_feedforward) {
            return INT4BlockQuantizer::QuantizedTensor();
        }
        return quantizer_.quantize(weights, shape);
    }
    
    // Calculate quantization metrics
    struct QuantizationMetrics {
        float compression_ratio;
        float memory_saved_mb;
        size_t original_size;
        size_t quantized_size;
        
        QuantizationMetrics() : compression_ratio(1.0f), memory_saved_mb(0.0f),
                               original_size(0), quantized_size(0) {}
    };
    
    QuantizationMetrics calculate_metrics(const std::vector<INT4BlockQuantizer::QuantizedTensor>& quantized_layers,
                                        const std::vector<size_t>& original_sizes) {
        QuantizationMetrics metrics;
        
        for (size_t i = 0; i < quantized_layers.size() && i < original_sizes.size(); ++i) {
            metrics.original_size += original_sizes[i] * sizeof(float);
            metrics.quantized_size += quantized_layers[i].memory_usage();
        }
        
        if (metrics.original_size > 0) {
            metrics.compression_ratio = static_cast<float>(metrics.original_size) / metrics.quantized_size;
            metrics.memory_saved_mb = static_cast<float>(metrics.original_size - metrics.quantized_size) / (1024 * 1024);
        }
        
        return metrics;
    }
};

} // namespace lightgpt 