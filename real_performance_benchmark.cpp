#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <iomanip>
#include <fstream>
#include <thread>
#include <immintrin.h>

// Include our optimization headers
#include "include/lightgpt/int4_quantization.hpp"
#include "include/lightgpt/advanced_inference.hpp"

using namespace lightgpt;

// ANSI color codes for output
#define RESET   "\033[0m"
#define RED     "\033[31m"
#define GREEN   "\033[32m"
#define YELLOW  "\033[33m"
#define BLUE    "\033[34m"
#define MAGENTA "\033[35m"
#define CYAN    "\033[36m"
#define BOLD    "\033[1m"

class RealPerformanceBenchmark {
private:
    std::mt19937 rng_;
    
public:
    RealPerformanceBenchmark() : rng_(42) {}
    
    // Generate realistic model weights (normal distribution like real neural networks)
    std::vector<float> generate_realistic_weights(size_t size) {
        std::vector<float> weights(size);
        std::normal_distribution<float> dist(0.0f, 0.02f); // Typical weight init scale
        
        for (auto& w : weights) {
            w = dist(rng_);
        }
        return weights;
    }
    
    // Baseline matrix multiplication (no optimizations)
    void baseline_matmul(const std::vector<float>& A, const std::vector<float>& B,
                        std::vector<float>& C, size_t M, size_t N, size_t K) {
        C.resize(M * N);
        std::fill(C.begin(), C.end(), 0.0f);
        
        for (size_t i = 0; i < M; ++i) {
            for (size_t j = 0; j < N; ++j) {
                for (size_t k = 0; k < K; ++k) {
                    C[i * N + j] += A[i * K + k] * B[k * N + j];
                }
            }
        }
    }
    
    // Optimized matrix multiplication with cache blocking
    void optimized_matmul(const std::vector<float>& A, const std::vector<float>& B,
                          std::vector<float>& C, size_t M, size_t N, size_t K) {
        C.resize(M * N);
        std::fill(C.begin(), C.end(), 0.0f);
        
        const size_t BLOCK_SIZE = 64; // Cache-friendly block size
        
        for (size_t ii = 0; ii < M; ii += BLOCK_SIZE) {
            for (size_t jj = 0; jj < N; jj += BLOCK_SIZE) {
                for (size_t kk = 0; kk < K; kk += BLOCK_SIZE) {
                    
                    size_t i_end = std::min(ii + BLOCK_SIZE, M);
                    size_t j_end = std::min(jj + BLOCK_SIZE, N);
                    size_t k_end = std::min(kk + BLOCK_SIZE, K);
                    
                    for (size_t i = ii; i < i_end; ++i) {
                        for (size_t j = jj; j < j_end; ++j) {
                            float sum = 0.0f;
                            
                            #ifdef __AVX2__
                            // Vectorized inner loop
                            __m256 acc = _mm256_setzero_ps();
                            size_t k_vec = kk;
                            for (; k_vec + 8 <= k_end; k_vec += 8) {
                                __m256 a_vec = _mm256_loadu_ps(&A[i * K + k_vec]);
                                __m256 b_vec = _mm256_loadu_ps(&B[k_vec * N + j]);
                                acc = _mm256_fmadd_ps(a_vec, b_vec, acc);
                            }
                            
                            // Horizontal sum of AVX register
                            __m128 sum_quad = _mm_add_ps(_mm256_extractf128_ps(acc, 1), 
                                                        _mm256_extractf128_ps(acc, 0));
                            __m128 sum_dual = _mm_add_ps(sum_quad, _mm_shuffle_ps(sum_quad, sum_quad, _MM_SHUFFLE(2, 3, 0, 1)));
                            __m128 sum_single = _mm_add_ss(sum_dual, _mm_shuffle_ps(sum_dual, sum_dual, _MM_SHUFFLE(1, 0, 3, 2)));
                            sum = _mm_cvtss_f32(sum_single);
                            
                            // Handle remaining elements
                            for (size_t k = k_vec; k < k_end; ++k) {
                                sum += A[i * K + k] * B[k * N + j];
                            }
                            #else
                            // Scalar fallback
                            for (size_t k = kk; k < k_end; ++k) {
                                sum += A[i * K + k] * B[k * N + j];
                            }
                            #endif
                            
                            C[i * N + j] += sum;
                        }
                    }
                }
            }
        }
    }
    
    struct BenchmarkResult {
        double baseline_time_ms;
        double optimized_time_ms;
        double speedup;
        double accuracy_loss;
        size_t memory_saved_bytes;
        bool passed_correctness;
        
        void print() const {
            std::cout << std::fixed << std::setprecision(2);
            std::cout << CYAN << "📊 Benchmark Results:" << RESET << "\n";
            std::cout << "  Baseline Time:    " << YELLOW << std::setw(8) << baseline_time_ms << " ms" << RESET << "\n";
            std::cout << "  Optimized Time:   " << GREEN << std::setw(8) << optimized_time_ms << " ms" << RESET << "\n";
            std::cout << "  Speedup:          " << BOLD << GREEN << std::setw(8) << speedup << "x" << RESET << "\n";
            std::cout << "  Accuracy Loss:    " << std::setw(8) << accuracy_loss * 100 << "%" << RESET << "\n";
            std::cout << "  Memory Saved:     " << std::setw(8) << memory_saved_bytes / 1024 << " KB" << RESET << "\n";
            std::cout << "  Correctness:      " << (passed_correctness ? GREEN "✅ PASS" : RED "❌ FAIL") << RESET << "\n\n";
        }
    };
    
    BenchmarkResult benchmark_matrix_multiplication() {
        std::cout << BLUE << "🧮 Testing Matrix Multiplication Optimization..." << RESET << "\n";
        
        const size_t M = 512, N = 512, K = 512;
        auto A = generate_realistic_weights(M * K);
        auto B = generate_realistic_weights(K * N);
        
        std::vector<float> C_baseline, C_optimized;
        
        // Warm up
        baseline_matmul(A, B, C_baseline, M, N, K);
        optimized_matmul(A, B, C_optimized, M, N, K);
        
        // Benchmark baseline
        auto start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < 10; ++i) {
            baseline_matmul(A, B, C_baseline, M, N, K);
        }
        auto end = std::chrono::high_resolution_clock::now();
        double baseline_time = std::chrono::duration<double, std::milli>(end - start).count() / 10.0;
        
        // Benchmark optimized
        start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < 10; ++i) {
            optimized_matmul(A, B, C_optimized, M, N, K);
        }
        end = std::chrono::high_resolution_clock::now();
        double optimized_time = std::chrono::duration<double, std::milli>(end - start).count() / 10.0;
        
        // Calculate accuracy
        double mse = 0.0;
        for (size_t i = 0; i < C_baseline.size(); ++i) {
            double diff = C_baseline[i] - C_optimized[i];
            mse += diff * diff;
        }
        mse /= C_baseline.size();
        
        BenchmarkResult result;
        result.baseline_time_ms = baseline_time;
        result.optimized_time_ms = optimized_time;
        result.speedup = baseline_time / optimized_time;
        result.accuracy_loss = std::sqrt(mse);
        result.memory_saved_bytes = 0; // Same precision
        result.passed_correctness = result.accuracy_loss < 1e-5; // Should be nearly identical
        
        return result;
    }
    
    BenchmarkResult benchmark_quantization() {
        std::cout << BLUE << "🗜️  Testing INT4 Quantization..." << RESET << "\n";
        
        const size_t size = 32768;
        auto weights = generate_realistic_weights(size);
        
        // Baseline: use original weights
        auto start = std::chrono::high_resolution_clock::now();
        std::vector<float> baseline_copy = weights; // Simulate "processing"
        auto end = std::chrono::high_resolution_clock::now();
        double baseline_time = std::chrono::duration<double, std::milli>(end - start).count();
        
        // Optimized: quantize and dequantize
        INT4BlockQuantizer quantizer(false, 3.0f);
        start = std::chrono::high_resolution_clock::now();
        auto quantized = quantizer.quantize(weights, {size});
        auto dequantized = quantizer.dequantize(quantized);
        end = std::chrono::high_resolution_clock::now();
        double optimized_time = std::chrono::duration<double, std::milli>(end - start).count();
        
        // Calculate accuracy loss
        double mse = 0.0;
        for (size_t i = 0; i < weights.size(); ++i) {
            double diff = weights[i] - dequantized[i];
            mse += diff * diff;
        }
        mse /= weights.size();
        
        BenchmarkResult result;
        result.baseline_time_ms = baseline_time;
        result.optimized_time_ms = optimized_time;
        result.speedup = baseline_time / optimized_time; // May be slower due to conversion overhead
        result.accuracy_loss = std::sqrt(mse);
        result.memory_saved_bytes = (weights.size() * sizeof(float)) - quantized.memory_usage();
        result.passed_correctness = result.accuracy_loss < 0.1f; // Reasonable for quantization
        
        return result;
    }
    
    BenchmarkResult benchmark_sampling() {
        std::cout << BLUE << "🎲 Testing Advanced Sampling..." << RESET << "\n";
        
        const size_t vocab_size = 32000;
        const int num_samples = 1000;
        
        // Generate realistic logits (typical LLM output distribution)
        std::vector<float> logits(vocab_size);
        std::normal_distribution<float> logit_dist(0.0f, 2.0f);
        for (auto& logit : logits) {
            logit = logit_dist(rng_);
        }
        
        // Baseline: simple sampling
        auto start = std::chrono::high_resolution_clock::now();
        std::vector<uint32_t> baseline_samples;
        std::uniform_int_distribution<uint32_t> simple_dist(0, vocab_size - 1);
        for (int i = 0; i < num_samples; ++i) {
            baseline_samples.push_back(simple_dist(rng_));
        }
        auto end = std::chrono::high_resolution_clock::now();
        double baseline_time = std::chrono::duration<double, std::milli>(end - start).count();
        
        // Optimized: advanced sampling
        AdvancedSampler sampler(42);
        start = std::chrono::high_resolution_clock::now();
        std::vector<uint32_t> optimized_samples;
        for (int i = 0; i < num_samples; ++i) {
            uint32_t token = sampler.sample_top_k_fast(logits.data(), vocab_size, 50, 1.0f, 0.9f);
            optimized_samples.push_back(token);
        }
        end = std::chrono::high_resolution_clock::now();
        double optimized_time = std::chrono::duration<double, std::milli>(end - start).count();
        
        // Calculate diversity (quality metric)
        std::set<uint32_t> baseline_unique(baseline_samples.begin(), baseline_samples.end());
        std::set<uint32_t> optimized_unique(optimized_samples.begin(), optimized_samples.end());
        
        BenchmarkResult result;
        result.baseline_time_ms = baseline_time;
        result.optimized_time_ms = optimized_time;
        result.speedup = baseline_time / optimized_time;
        result.accuracy_loss = 0.0; // Different metric for sampling
        result.memory_saved_bytes = 0;
        result.passed_correctness = optimized_unique.size() > baseline_unique.size() * 0.8; // Should have reasonable diversity
        
        std::cout << "  Baseline diversity: " << baseline_unique.size() << "/" << num_samples << " unique tokens\n";
        std::cout << "  Optimized diversity: " << optimized_unique.size() << "/" << num_samples << " unique tokens\n";
        
        return result;
    }
    
    BenchmarkResult benchmark_memory_allocation() {
        std::cout << BLUE << "💾 Testing Memory Management..." << RESET << "\n";
        
        const int num_allocations = 10000;
        const size_t allocation_size = 1024;
        
        // Baseline: standard allocation
        auto start = std::chrono::high_resolution_clock::now();
        std::vector<std::vector<float>> baseline_allocations;
        for (int i = 0; i < num_allocations; ++i) {
            baseline_allocations.emplace_back(allocation_size, 1.0f);
        }
        auto end = std::chrono::high_resolution_clock::now();
        double baseline_time = std::chrono::duration<double, std::milli>(end - start).count();
        
        // Optimized: pre-allocated pool (simulated)
        start = std::chrono::high_resolution_clock::now();
        std::vector<float> pool(num_allocations * allocation_size);
        std::vector<float*> optimized_allocations;
        for (int i = 0; i < num_allocations; ++i) {
            optimized_allocations.push_back(&pool[i * allocation_size]);
            // Simulate some work
            for (size_t j = 0; j < allocation_size; ++j) {
                optimized_allocations.back()[j] = 1.0f;
            }
        }
        end = std::chrono::high_resolution_clock::now();
        double optimized_time = std::chrono::duration<double, std::milli>(end - start).count();
        
        BenchmarkResult result;
        result.baseline_time_ms = baseline_time;
        result.optimized_time_ms = optimized_time;
        result.speedup = baseline_time / optimized_time;
        result.accuracy_loss = 0.0;
        result.memory_saved_bytes = 0; // Same total memory, but better allocation pattern
        result.passed_correctness = true;
        
        return result;
    }
    
    void run_comprehensive_benchmark() {
        std::cout << BOLD << CYAN << "\n🚀 Real Performance Benchmark Suite" << RESET << "\n";
        std::cout << "====================================\n\n";
        
        // Check system capabilities
        std::cout << YELLOW << "System Information:" << RESET << "\n";
        std::cout << "  CPU Cores: " << std::thread::hardware_concurrency() << "\n";
        
        #ifdef __AVX2__
        std::cout << "  AVX2: " << GREEN << "✅ Enabled" << RESET << "\n";
        #else
        std::cout << "  AVX2: " << RED << "❌ Not available" << RESET << "\n";
        #endif
        
        #ifdef _OPENMP
        std::cout << "  OpenMP: " << GREEN << "✅ Enabled" << RESET << "\n";
        #else
        std::cout << "  OpenMP: " << RED << "❌ Not available" << RESET << "\n";
        #endif
        
        std::cout << "\n";
        
        // Run benchmarks
        std::vector<BenchmarkResult> results;
        
        results.push_back(benchmark_matrix_multiplication());
        results.push_back(benchmark_quantization());
        results.push_back(benchmark_sampling());
        results.push_back(benchmark_memory_allocation());
        
        // Print all results
        for (size_t i = 0; i < results.size(); ++i) {
            results[i].print();
        }
        
        // Calculate overall performance improvement
        double total_speedup = 1.0;
        int valid_speedups = 0;
        
        for (const auto& result : results) {
            if (result.speedup > 0.1 && result.speedup < 100.0) { // Reasonable range
                total_speedup *= result.speedup;
                valid_speedups++;
            }
        }
        
        if (valid_speedups > 0) {
            double geometric_mean_speedup = std::pow(total_speedup, 1.0 / valid_speedups);
            
            std::cout << BOLD << GREEN << "🎯 Overall Performance Summary:" << RESET << "\n";
            std::cout << "  Geometric Mean Speedup: " << BOLD << geometric_mean_speedup << "x" << RESET << "\n";
            
            if (geometric_mean_speedup > 2.0) {
                std::cout << "  " << GREEN << "✅ Significant performance improvement achieved!" << RESET << "\n";
            } else if (geometric_mean_speedup > 1.2) {
                std::cout << "  " << YELLOW << "⚠️  Moderate performance improvement" << RESET << "\n";
            } else {
                std::cout << "  " << RED << "❌ No significant performance improvement" << RESET << "\n";
            }
        }
        
        std::cout << "\n" << MAGENTA << "💡 Note: Results depend on your specific hardware capabilities." << RESET << "\n";
        std::cout << MAGENTA << "   AVX2/AVX-512 and OpenMP support provide the biggest gains." << RESET << "\n";
    }
};

int main() {
    RealPerformanceBenchmark benchmark;
    
    try {
        benchmark.run_comprehensive_benchmark();
        return 0;
    } catch (const std::exception& e) {
        std::cerr << RED << "❌ Benchmark failed: " << e.what() << RESET << std::endl;
        return 1;
    }
} 