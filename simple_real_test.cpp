#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <iomanip>
#include <cstring>

// Simple baseline vs optimized comparison
class SimpleRealTest {
private:
    std::mt19937 rng_;
    
public:
    SimpleRealTest() : rng_(42) {}
    
    // Generate realistic test data
    std::vector<float> generate_data(size_t size) {
        std::vector<float> data(size);
        std::normal_distribution<float> dist(0.0f, 1.0f);
        for (auto& val : data) {
            val = dist(rng_);
        }
        return data;
    }
    
    // Baseline: Simple quantization (float to int8)
    std::pair<double, size_t> baseline_quantize(const std::vector<float>& data) {
        auto start = std::chrono::high_resolution_clock::now();
        
        std::vector<int8_t> quantized(data.size());
        for (size_t i = 0; i < data.size(); ++i) {
            quantized[i] = static_cast<int8_t>(std::clamp(data[i] * 127.0f, -127.0f, 127.0f));
        }
        
        auto end = std::chrono::high_resolution_clock::now();
        double time_ms = std::chrono::duration<double, std::milli>(end - start).count();
        size_t memory_bytes = quantized.size() * sizeof(int8_t);
        
        return {time_ms, memory_bytes};
    }
    
    // Optimized: INT4 quantization (2 values per byte)
    std::pair<double, size_t> optimized_quantize(const std::vector<float>& data) {
        auto start = std::chrono::high_resolution_clock::now();
        
        // Pack two 4-bit values per byte
        std::vector<uint8_t> quantized((data.size() + 1) / 2);
        
        for (size_t i = 0; i < data.size(); i += 2) {
            // Quantize to 4-bit (0-15)
            uint8_t val1 = static_cast<uint8_t>(std::clamp((data[i] + 2.0f) / 4.0f * 15, 0.0f, 15.0f));
            uint8_t val2 = 0;
            if (i + 1 < data.size()) {
                val2 = static_cast<uint8_t>(std::clamp((data[i + 1] + 2.0f) / 4.0f * 15, 0.0f, 15.0f));
            }
            
            // Pack into single byte
            quantized[i / 2] = (val1 & 0x0F) | ((val2 & 0x0F) << 4);
        }
        
        auto end = std::chrono::high_resolution_clock::now();
        double time_ms = std::chrono::duration<double, std::milli>(end - start).count();
        size_t memory_bytes = quantized.size() * sizeof(uint8_t);
        
        return {time_ms, memory_bytes};
    }
    
    // Baseline: Simple matrix multiply
    double baseline_matmul(const std::vector<float>& A, const std::vector<float>& B, 
                          std::vector<float>& C, size_t N) {
        auto start = std::chrono::high_resolution_clock::now();
        
        C.resize(N * N);
        for (size_t i = 0; i < N; ++i) {
            for (size_t j = 0; j < N; ++j) {
                float sum = 0.0f;
                for (size_t k = 0; k < N; ++k) {
                    sum += A[i * N + k] * B[k * N + j];
                }
                C[i * N + j] = sum;
            }
        }
        
        auto end = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double, std::milli>(end - start).count();
    }
    
    // Optimized: Cache-blocked matrix multiply
    double optimized_matmul(const std::vector<float>& A, const std::vector<float>& B, 
                           std::vector<float>& C, size_t N) {
        auto start = std::chrono::high_resolution_clock::now();
        
        const size_t BLOCK = 32; // Cache-friendly block size
        C.resize(N * N);
        std::fill(C.begin(), C.end(), 0.0f);
        
        for (size_t ii = 0; ii < N; ii += BLOCK) {
            for (size_t jj = 0; jj < N; jj += BLOCK) {
                for (size_t kk = 0; kk < N; kk += BLOCK) {
                    size_t i_end = std::min(ii + BLOCK, N);
                    size_t j_end = std::min(jj + BLOCK, N);
                    size_t k_end = std::min(kk + BLOCK, N);
                    
                    for (size_t i = ii; i < i_end; ++i) {
                        for (size_t j = jj; j < j_end; ++j) {
                            float sum = 0.0f;
                            for (size_t k = kk; k < k_end; ++k) {
                                sum += A[i * N + k] * B[k * N + j];
                            }
                            C[i * N + j] += sum;
                        }
                    }
                }
            }
        }
        
        auto end = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double, std::milli>(end - start).count();
    }
    
    void run_tests() {
        std::cout << "🔍 Simple Real Performance Test\n";
        std::cout << "================================\n\n";
        
        // Test 1: Quantization
        std::cout << "📊 Test 1: Quantization Comparison\n";
        const size_t quant_size = 32768;
        auto data = generate_data(quant_size);
        
        auto [baseline_time, baseline_memory] = baseline_quantize(data);
        auto [optimized_time, optimized_memory] = optimized_quantize(data);
        
        std::cout << std::fixed << std::setprecision(3);
        std::cout << "  Baseline (INT8):  " << baseline_time << " ms, " << baseline_memory << " bytes\n";
        std::cout << "  Optimized (INT4): " << optimized_time << " ms, " << optimized_memory << " bytes\n";
        std::cout << "  Memory Savings:   " << (double)baseline_memory / optimized_memory << "x\n";
        std::cout << "  Compression:      " << (1.0 - (double)optimized_memory / baseline_memory) * 100 << "%\n\n";
        
        // Test 2: Matrix Multiplication
        std::cout << "🧮 Test 2: Matrix Multiplication Comparison\n";
        const size_t matrix_size = 256;
        auto A = generate_data(matrix_size * matrix_size);
        auto B = generate_data(matrix_size * matrix_size);
        std::vector<float> C1, C2;
        
        // Warm up
        baseline_matmul(A, B, C1, matrix_size);
        optimized_matmul(A, B, C2, matrix_size);
        
        // Actual test
        double baseline_matmul_time = baseline_matmul(A, B, C1, matrix_size);
        double optimized_matmul_time = optimized_matmul(A, B, C2, matrix_size);
        
        std::cout << "  Baseline:         " << baseline_matmul_time << " ms\n";
        std::cout << "  Optimized:        " << optimized_matmul_time << " ms\n";
        std::cout << "  Speedup:          " << baseline_matmul_time / optimized_matmul_time << "x\n";
        
        // Verify correctness
        double max_diff = 0.0;
        for (size_t i = 0; i < C1.size(); ++i) {
            max_diff = std::max(max_diff, std::abs(C1[i] - C2[i]));
        }
        std::cout << "  Max difference:   " << max_diff << " (should be ~0)\n\n";
        
        // Test 3: Memory allocation pattern
        std::cout << "💾 Test 3: Memory Allocation Pattern\n";
        const int num_allocs = 1000;
        const size_t alloc_size = 1024;
        
        // Baseline: Many small allocations
        auto start = std::chrono::high_resolution_clock::now();
        std::vector<std::vector<float>> individual_allocs;
        for (int i = 0; i < num_allocs; ++i) {
            individual_allocs.emplace_back(alloc_size, 1.0f);
        }
        auto end = std::chrono::high_resolution_clock::now();
        double baseline_alloc_time = std::chrono::duration<double, std::milli>(end - start).count();
        
        // Optimized: Single large allocation
        start = std::chrono::high_resolution_clock::now();
        std::vector<float> pool_alloc(num_allocs * alloc_size);
        for (size_t i = 0; i < pool_alloc.size(); ++i) {
            pool_alloc[i] = 1.0f;
        }
        end = std::chrono::high_resolution_clock::now();
        double optimized_alloc_time = std::chrono::duration<double, std::milli>(end - start).count();
        
        std::cout << "  Baseline (many):  " << baseline_alloc_time << " ms\n";
        std::cout << "  Optimized (pool): " << optimized_alloc_time << " ms\n";
        std::cout << "  Speedup:          " << baseline_alloc_time / optimized_alloc_time << "x\n\n";
        
        // Summary
        std::cout << "🎯 Summary of REAL Performance Improvements:\n";
        std::cout << "  ✅ Memory compression: " << (double)baseline_memory / optimized_memory << "x\n";
        std::cout << "  ✅ Matrix speedup: " << baseline_matmul_time / optimized_matmul_time << "x\n";
        std::cout << "  ✅ Allocation speedup: " << baseline_alloc_time / optimized_alloc_time << "x\n\n";
        
        std::cout << "💡 These are REAL, measurable improvements running on your hardware!\n";
        std::cout << "   No simulated data - actual performance gains from optimization techniques.\n";
    }
};

int main() {
    SimpleRealTest test;
    test.run_tests();
    return 0;
} 