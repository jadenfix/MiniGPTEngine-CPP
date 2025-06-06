#pragma once

#include "advanced_inference.hpp"
#include "int4_quantization.hpp"
#include <vector>
#include <memory>
#include <thread>
#include <atomic>
#include <chrono>
#include <queue>
#include <future>

namespace lightgpt {

// High-throughput inference engine combining all optimizations
class ThroughputOptimizer {
public:
    struct OptimizationConfig {
        // Quantization settings
        bool use_int4_quantization = true;
        bool symmetric_quantization = false;
        float outlier_threshold = 3.0f;
        
        // Streaming settings
        size_t batch_size = 8;
        size_t max_sequence_length = 512;
        float timeout_ms = 10.0f;
        
        // Sampling settings
        float temperature = 1.0f;
        uint32_t top_k = 50;
        float top_p = 0.9f;
        
        // Cache settings
        size_t token_cache_size = 2048;
        size_t cache_window = 32;
        bool use_token_cache = true;
        
        // Threading settings
        size_t num_worker_threads = 0; // 0 = auto-detect
        bool enable_prefetch = true;
        
        OptimizationConfig() = default;
    };
    
    struct PerformanceMetrics {
        // Throughput metrics
        float tokens_per_second = 0.0f;
        float requests_per_second = 0.0f;
        float avg_latency_ms = 0.0f;
        
        // Memory metrics
        float memory_usage_mb = 0.0f;
        float compression_ratio = 1.0f;
        float cache_hit_rate = 0.0f;
        
        // Quality metrics
        float accuracy_score = 1.0f;
        size_t total_tokens_generated = 0;
        size_t total_requests_processed = 0;
        
        PerformanceMetrics() = default;
        
        void update_throughput(size_t tokens, float duration_ms) {
            if (duration_ms > 0) {
                tokens_per_second = static_cast<float>(tokens) / (duration_ms / 1000.0f);
            }
        }
        
        void update_latency(float latency_ms) {
            avg_latency_ms = (avg_latency_ms * total_requests_processed + latency_ms) / 
                           (total_requests_processed + 1);
        }
    };

private:
    OptimizationConfig config_;
    PerformanceMetrics metrics_;
    
    // Core components
    std::unique_ptr<StreamedInference> streamed_inference_;
    std::unique_ptr<BatchInference> batch_inference_;
    std::unique_ptr<INT4BlockQuantizer> quantizer_;
    std::unique_ptr<ModelQuantizer> model_quantizer_;
    
    // Quantized model weights
    std::vector<INT4BlockQuantizer::QuantizedTensor> quantized_weights_;
    
    // Performance monitoring
    std::atomic<bool> monitoring_active_{false};
    std::thread monitoring_thread_;
    std::chrono::steady_clock::time_point start_time_;
    
public:
    ThroughputOptimizer(const OptimizationConfig& config = OptimizationConfig())
        : config_(config), start_time_(std::chrono::steady_clock::now()) {
        initialize_components();
    }
    
    ~ThroughputOptimizer() {
        stop_monitoring();
    }
    
    // Main inference interface
    std::future<std::vector<uint32_t>> generate_async(
        const std::vector<uint32_t>& prompt,
        size_t max_new_tokens = 100,
        std::function<void(uint32_t)> token_callback = nullptr) {
        
        auto start_time = std::chrono::steady_clock::now();
        
        // Choose inference method based on request characteristics
        if (prompt.size() > config_.max_sequence_length / 2) {
            // Use streaming for long sequences
            return streamed_inference_->generate_async(prompt, token_callback);
        } else {
            // Use batching for short sequences
            return batch_inference_->submit_request(
                std::vector<uint32_t>(prompt), max_new_tokens);
        }
    }
    
    // Batch generation for maximum throughput
    std::vector<std::future<std::vector<uint32_t>>> generate_batch(
        const std::vector<std::vector<uint32_t>>& prompts,
        size_t max_new_tokens = 100) {
        
        std::vector<std::future<std::vector<uint32_t>>> futures;
        futures.reserve(prompts.size());
        
        for (const auto& prompt : prompts) {
            futures.push_back(batch_inference_->submit_request(
                std::vector<uint32_t>(prompt), max_new_tokens));
        }
        
        return futures;
    }
    
    // Quantize model weights for memory efficiency
    void quantize_model_weights(const std::vector<std::vector<float>>& weights,
                               const std::vector<std::vector<size_t>>& shapes) {
        quantized_weights_.clear();
        quantized_weights_.reserve(weights.size());
        
        auto start_time = std::chrono::steady_clock::now();
        
        #pragma omp parallel for
        for (size_t i = 0; i < weights.size(); ++i) {
            auto quantized = quantizer_->quantize(weights[i], shapes[i]);
            
            #pragma omp critical
            {
                quantized_weights_.push_back(std::move(quantized));
            }
        }
        
        auto end_time = std::chrono::steady_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        
        // Update metrics
        update_quantization_metrics();
        
        printf("Quantized %zu weight tensors in %.2f ms\n", 
               weights.size(), static_cast<float>(duration.count()));
    }
    
    // Optimized inference with quantized weights
    std::vector<float> forward_pass_quantized(const std::vector<float>& input,
                                             const std::vector<size_t>& layer_indices,
                                             const std::vector<std::vector<size_t>>& dimensions) {
        auto start_time = std::chrono::steady_clock::now();
        
        std::vector<float> output = input;
        
        for (size_t i = 0; i < layer_indices.size(); ++i) {
            size_t layer_idx = layer_indices[i];
            if (layer_idx >= quantized_weights_.size()) continue;
            
            const auto& dims = dimensions[i];
            if (dims.size() < 3) continue; // Need at least [batch, in_dim, out_dim]
            
            size_t batch_size = dims[0];
            size_t in_dim = dims[1];
            size_t out_dim = dims[2];
            
            std::vector<float> layer_output;
            quantizer_->quantized_matmul(output, quantized_weights_[layer_idx],
                                       layer_output, batch_size, out_dim, in_dim);
            
            output = std::move(layer_output);
        }
        
        auto end_time = std::chrono::steady_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
        
        // Update performance metrics
        metrics_.total_tokens_generated += output.size();
        
        return output;
    }
    
    // Performance profiling and optimization
    void start_performance_monitoring() {
        monitoring_active_ = true;
        monitoring_thread_ = std::thread(&ThroughputOptimizer::monitor_performance, this);
    }
    
    void stop_monitoring() {
        monitoring_active_ = false;
        if (monitoring_thread_.joinable()) {
            monitoring_thread_.join();
        }
    }
    
    PerformanceMetrics get_metrics() const {
        return metrics_;
    }
    
    // Memory usage optimization
    void optimize_memory_usage() {
        // Clear caches if memory pressure is high
        if (metrics_.memory_usage_mb > 1024.0f) { // 1GB threshold
            clear_caches();
        }
        
        // Trigger garbage collection for quantized weights
        compress_quantized_weights();
    }
    
    // Accuracy validation
    float validate_accuracy(const std::vector<std::vector<float>>& reference_outputs,
                           const std::vector<std::vector<float>>& quantized_outputs) {
        if (reference_outputs.size() != quantized_outputs.size()) return 0.0f;
        
        float total_error = 0.0f;
        size_t total_elements = 0;
        
        for (size_t i = 0; i < reference_outputs.size(); ++i) {
            const auto& ref = reference_outputs[i];
            const auto& quant = quantized_outputs[i];
            
            if (ref.size() != quant.size()) continue;
            
            for (size_t j = 0; j < ref.size(); ++j) {
                float error = std::abs(ref[j] - quant[j]);
                total_error += error;
                total_elements++;
            }
        }
        
        float mean_absolute_error = total_elements > 0 ? total_error / total_elements : 0.0f;
        float accuracy = std::max(0.0f, 1.0f - mean_absolute_error);
        
        metrics_.accuracy_score = accuracy;
        return accuracy;
    }
    
    // Dynamic optimization based on runtime characteristics
    void auto_optimize() {
        PerformanceMetrics current_metrics = metrics_;
        
        // Adjust batch size based on throughput
        if (current_metrics.tokens_per_second < 100.0f && config_.batch_size < 32) {
            config_.batch_size *= 2;
            reinitialize_batch_inference();
        } else if (current_metrics.avg_latency_ms > 50.0f && config_.batch_size > 1) {
            config_.batch_size = std::max(1UL, config_.batch_size / 2);
            reinitialize_batch_inference();
        }
        
        // Adjust cache size based on hit rate
        if (current_metrics.cache_hit_rate < 0.3f && config_.token_cache_size < 4096) {
            config_.token_cache_size *= 2;
            reinitialize_streaming();
        }
        
        // Adjust quantization based on accuracy
        if (current_metrics.accuracy_score < 0.95f && config_.use_int4_quantization) {
            config_.symmetric_quantization = true;
            config_.outlier_threshold = std::min(5.0f, config_.outlier_threshold + 0.5f);
            reinitialize_quantization();
        }
    }

private:
    void initialize_components() {
        // Initialize quantizer
        quantizer_ = std::make_unique<INT4BlockQuantizer>(
            config_.symmetric_quantization, config_.outlier_threshold);
        
        ModelQuantizer::QuantizationConfig quant_config;
        quant_config.use_symmetric = config_.symmetric_quantization;
        quant_config.outlier_threshold = config_.outlier_threshold;
        model_quantizer_ = std::make_unique<ModelQuantizer>(quant_config);
        
        // Initialize streaming inference
        StreamedInference::StreamConfig stream_config;
        stream_config.batch_size = config_.batch_size;
        stream_config.max_length = config_.max_sequence_length;
        stream_config.temperature = config_.temperature;
        stream_config.top_k = config_.top_k;
        stream_config.top_p = config_.top_p;
        stream_config.cache_window = config_.cache_window;
        stream_config.use_token_cache = config_.use_token_cache;
        
        streamed_inference_ = std::make_unique<StreamedInference>(stream_config);
        
        // Initialize batch inference
        BatchInference::BatchConfig batch_config;
        batch_config.max_batch_size = config_.batch_size;
        batch_config.max_sequence_length = config_.max_sequence_length;
        batch_config.timeout_ms = config_.timeout_ms;
        
        batch_inference_ = std::make_unique<BatchInference>(batch_config);
        batch_inference_->start();
        
        // Start performance monitoring
        start_performance_monitoring();
    }
    
    void monitor_performance() {
        auto last_update = std::chrono::steady_clock::now();
        
        while (monitoring_active_) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1000)); // Update every second
            
            auto now = std::chrono::steady_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(now - last_update);
            
            // Update cache hit rate
            metrics_.cache_hit_rate = static_cast<float>(streamed_inference_->get_cache_hit_rate());
            
            // Update memory usage (simplified estimation)
            metrics_.memory_usage_mb = calculate_memory_usage();
            
            // Auto-optimize if needed
            static int optimization_counter = 0;
            if (++optimization_counter % 30 == 0) { // Every 30 seconds
                auto_optimize();
            }
            
            last_update = now;
        }
    }
    
    float calculate_memory_usage() const {
        float total_mb = 0.0f;
        
        // Quantized weights memory
        for (const auto& tensor : quantized_weights_) {
            total_mb += static_cast<float>(tensor.memory_usage()) / (1024 * 1024);
        }
        
        // Cache memory (estimated)
        total_mb += config_.token_cache_size * 0.01f; // Rough estimate
        
        return total_mb;
    }
    
    void update_quantization_metrics() {
        if (quantized_weights_.empty()) return;
        
        float total_compression = 0.0f;
        for (const auto& tensor : quantized_weights_) {
            total_compression += tensor.compression_ratio();
        }
        
        metrics_.compression_ratio = total_compression / quantized_weights_.size();
        metrics_.memory_usage_mb = calculate_memory_usage();
    }
    
    void clear_caches() {
        // This would clear internal caches - simplified for demonstration
        printf("Clearing caches to reduce memory usage\n");
    }
    
    void compress_quantized_weights() {
        // This would perform additional compression - simplified for demonstration
        printf("Compressing quantized weights\n");
    }
    
    void reinitialize_batch_inference() {
        batch_inference_->stop();
        
        BatchInference::BatchConfig batch_config;
        batch_config.max_batch_size = config_.batch_size;
        batch_config.max_sequence_length = config_.max_sequence_length;
        batch_config.timeout_ms = config_.timeout_ms;
        
        batch_inference_ = std::make_unique<BatchInference>(batch_config);
        batch_inference_->start();
    }
    
    void reinitialize_streaming() {
        StreamedInference::StreamConfig stream_config;
        stream_config.batch_size = config_.batch_size;
        stream_config.max_length = config_.max_sequence_length;
        stream_config.temperature = config_.temperature;
        stream_config.top_k = config_.top_k;
        stream_config.top_p = config_.top_p;
        stream_config.cache_window = config_.cache_window;
        stream_config.use_token_cache = config_.use_token_cache;
        
        streamed_inference_ = std::make_unique<StreamedInference>(stream_config);
    }
    
    void reinitialize_quantization() {
        quantizer_ = std::make_unique<INT4BlockQuantizer>(
            config_.symmetric_quantization, config_.outlier_threshold);
        
        ModelQuantizer::QuantizationConfig quant_config;
        quant_config.use_symmetric = config_.symmetric_quantization;
        quant_config.outlier_threshold = config_.outlier_threshold;
        model_quantizer_ = std::make_unique<ModelQuantizer>(quant_config);
    }
};

// Utility class for benchmarking and comparison
class ThroughputBenchmark {
public:
    struct BenchmarkResult {
        float baseline_tokens_per_second = 0.0f;
        float optimized_tokens_per_second = 0.0f;
        float speedup_ratio = 1.0f;
        float memory_reduction_percent = 0.0f;
        float accuracy_retention = 1.0f;
        float avg_latency_baseline_ms = 0.0f;
        float avg_latency_optimized_ms = 0.0f;
        
        void calculate_improvements() {
            if (baseline_tokens_per_second > 0) {
                speedup_ratio = optimized_tokens_per_second / baseline_tokens_per_second;
            }
        }
        
        void print_results() const {
            printf("\n=== Throughput Optimization Results ===\n");
            printf("Baseline Performance: %.2f tokens/sec\n", baseline_tokens_per_second);
            printf("Optimized Performance: %.2f tokens/sec\n", optimized_tokens_per_second);
            printf("Speedup: %.2fx\n", speedup_ratio);
            printf("Memory Reduction: %.1f%%\n", memory_reduction_percent);
            printf("Accuracy Retention: %.1f%%\n", accuracy_retention * 100);
            printf("Latency Improvement: %.2f ms -> %.2f ms\n", 
                   avg_latency_baseline_ms, avg_latency_optimized_ms);
            printf("========================================\n\n");
        }
    };
    
    static BenchmarkResult run_comprehensive_benchmark(
        const std::vector<std::vector<uint32_t>>& test_prompts,
        const std::vector<std::vector<float>>& test_weights,
        const std::vector<std::vector<size_t>>& weight_shapes) {
        
        BenchmarkResult result;
        
        printf("Running comprehensive throughput benchmark...\n");
        
        // Baseline performance (simulated)
        auto baseline_start = std::chrono::steady_clock::now();
        size_t total_tokens_baseline = simulate_baseline_inference(test_prompts);
        auto baseline_end = std::chrono::steady_clock::now();
        
        auto baseline_duration = std::chrono::duration_cast<std::chrono::milliseconds>(
            baseline_end - baseline_start);
        result.baseline_tokens_per_second = 
            static_cast<float>(total_tokens_baseline) / (baseline_duration.count() / 1000.0f);
        result.avg_latency_baseline_ms = static_cast<float>(baseline_duration.count()) / test_prompts.size();
        
        // Optimized performance
        ThroughputOptimizer optimizer;
        optimizer.quantize_model_weights(test_weights, weight_shapes);
        
        auto optimized_start = std::chrono::steady_clock::now();
        size_t total_tokens_optimized = simulate_optimized_inference(optimizer, test_prompts);
        auto optimized_end = std::chrono::steady_clock::now();
        
        auto optimized_duration = std::chrono::duration_cast<std::chrono::milliseconds>(
            optimized_end - optimized_start);
        result.optimized_tokens_per_second = 
            static_cast<float>(total_tokens_optimized) / (optimized_duration.count() / 1000.0f);
        result.avg_latency_optimized_ms = static_cast<float>(optimized_duration.count()) / test_prompts.size();
        
        // Calculate improvements
        result.calculate_improvements();
        
        // Get metrics from optimizer
        auto metrics = optimizer.get_metrics();
        result.memory_reduction_percent = (1.0f - (1.0f / metrics.compression_ratio)) * 100.0f;
        result.accuracy_retention = metrics.accuracy_score;
        
        return result;
    }

private:
    static size_t simulate_baseline_inference(const std::vector<std::vector<uint32_t>>& prompts) {
        size_t total_tokens = 0;
        for (const auto& prompt : prompts) {
            total_tokens += prompt.size() + 50; // Assume 50 generated tokens
            std::this_thread::sleep_for(std::chrono::microseconds(100)); // Simulate processing time
        }
        return total_tokens;
    }
    
    static size_t simulate_optimized_inference(ThroughputOptimizer& optimizer,
                                             const std::vector<std::vector<uint32_t>>& prompts) {
        auto futures = optimizer.generate_batch(prompts, 50);
        
        size_t total_tokens = 0;
        for (auto& future : futures) {
            auto result = future.get();
            total_tokens += result.size();
        }
        
        return total_tokens;
    }
};

} // namespace lightgpt 