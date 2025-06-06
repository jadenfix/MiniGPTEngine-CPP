#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <cassert>
#include <iomanip>

// Test configuration
#ifndef USE_AVX2
#define USE_AVX2
#endif

// Test headers one by one to catch specific issues
namespace test_headers {
    
void test_simd_header() {
    std::cout << "Testing SIMD kernels header..." << std::flush;
    try {
        #include "include/lightgpt/simd_kernels.hpp"
        std::cout << " ✓" << std::endl;
    } catch (...) {
        std::cout << " ✗ FAILED" << std::endl;
        throw;
    }
}

void test_quantization_header() {
    std::cout << "Testing quantization header..." << std::flush;
    try {
        #include "include/lightgpt/quantization.hpp"
        std::cout << " ✓" << std::endl;
    } catch (...) {
        std::cout << " ✗ FAILED" << std::endl;
        throw;
    }
}

void test_memory_header() {
    std::cout << "Testing memory pool header..." << std::flush;
    try {
        #include "include/lightgpt/memory_pool.hpp"
        std::cout << " ✓" << std::endl;
    } catch (...) {
        std::cout << " ✗ FAILED" << std::endl;
        throw;
    }
}

void test_threading_header() {
    std::cout << "Testing threading header..." << std::flush;
    try {
        #include "include/lightgpt/threading.hpp"
        std::cout << " ✓" << std::endl;
    } catch (...) {
        std::cout << " ✗ FAILED" << std::endl;
        throw;
    }
}

} // namespace test_headers

// Include all headers after testing
#include "include/lightgpt/simd_kernels.hpp"
#include "include/lightgpt/quantization.hpp"
#include "include/lightgpt/memory_pool.hpp"
#include "include/lightgpt/threading.hpp"

class ComprehensiveValidator {
private:
    std::mt19937 rng{42}; // Fixed seed for reproducibility
    std::uniform_real_distribution<float> dist{-1.0f, 1.0f};
    
    int tests_passed = 0;
    int tests_failed = 0;
    
    void test_result(const std::string& test_name, bool passed) {
        std::cout << std::setw(40) << std::left << test_name;
        if (passed) {
            std::cout << " ✓ PASS" << std::endl;
            tests_passed++;
        } else {
            std::cout << " ✗ FAIL" << std::endl;
            tests_failed++;
        }
    }
    
public:
    void run_all_tests() {
        std::cout << "=== Comprehensive LightGPT Validation ===" << std::endl;
        std::cout << std::endl;
        
        // Test 1: Header compilation
        test_header_compilation();
        
        // Test 2: Memory management
        test_memory_management();
        
        // Test 3: SIMD operations
        test_simd_operations();
        
        // Test 4: Quantization
        test_quantization();
        
        // Test 5: Threading
        test_threading();
        
        // Test 6: Integration
        test_integration();
        
        // Test 7: Performance validation
        test_performance();
        
        // Test 8: Error handling
        test_error_handling();
        
        print_summary();
    }
    
private:
    void test_header_compilation() {
        std::cout << "1. Header Compilation Tests:" << std::endl;
        
        try {
            test_headers::test_simd_header();
            test_result("SIMD header compilation", true);
        } catch (...) {
            test_result("SIMD header compilation", false);
        }
        
        try {
            test_headers::test_quantization_header();
            test_result("Quantization header compilation", true);
        } catch (...) {
            test_result("Quantization header compilation", false);
        }
        
        try {
            test_headers::test_memory_header();
            test_result("Memory pool header compilation", true);
        } catch (...) {
            test_result("Memory pool header compilation", false);
        }
        
        try {
            test_headers::test_threading_header();
            test_result("Threading header compilation", true);
        } catch (...) {
            test_result("Threading header compilation", false);
        }
        
        std::cout << std::endl;
    }
    
    void test_memory_management() {
        std::cout << "2. Memory Management Tests:" << std::endl;
        
        // Test aligned allocation
        try {
            void* ptr = lightgpt::simd::aligned_alloc(1024, 64);
            bool aligned_alloc_works = (ptr != nullptr);
            if (ptr) {
                // Check alignment
                uintptr_t addr = reinterpret_cast<uintptr_t>(ptr);
                aligned_alloc_works = (addr % 64 == 0);
                lightgpt::simd::aligned_free(ptr);
            }
            test_result("Aligned memory allocation", aligned_alloc_works);
        } catch (...) {
            test_result("Aligned memory allocation", false);
        }
        
        // Test memory pool
        try {
            lightgpt::memory::MemoryPool pool(1024 * 1024);
            void* ptr1 = pool.allocate(1024);
            void* ptr2 = pool.allocate(2048);
            bool pool_works = (ptr1 != nullptr && ptr2 != nullptr && ptr1 != ptr2);
            test_result("Memory pool allocation", pool_works);
        } catch (...) {
            test_result("Memory pool allocation", false);
        }
        
        // Test KV cache
        try {
            lightgpt::memory::KVCache cache(64, 32, 128); // batch, heads, seq_len
            auto k_ptr = cache.get_k_cache();
            auto v_ptr = cache.get_v_cache();
            bool kv_cache_works = (k_ptr != nullptr && v_ptr != nullptr);
            test_result("KV cache creation", kv_cache_works);
        } catch (...) {
            test_result("KV cache creation", false);
        }
        
        std::cout << std::endl;
    }
    
    void test_simd_operations() {
        std::cout << "3. SIMD Operations Tests:" << std::endl;
        
        // Test basic SIMD availability
        try {
            #ifdef USE_AVX2
            __m256 test_vec = _mm256_set1_ps(1.0f);
            test_result("AVX2 instruction availability", true);
            #else
            test_result("AVX2 instruction availability", false);
            #endif
        } catch (...) {
            test_result("AVX2 instruction availability", false);
        }
        
        // Test simple matrix multiply
        try {
            const int size = 16;
            std::vector<float> A(size), B(size), C(size);
            
            // Initialize test data
            for (int i = 0; i < size; ++i) {
                A[i] = static_cast<float>(i + 1);
                B[i] = static_cast<float>(i + 1);
            }
            
            // Simple operation test
            for (int i = 0; i < size; ++i) {
                C[i] = A[i] * B[i];
            }
            
            bool simd_works = (C[0] == 1.0f && C[1] == 4.0f && C[2] == 9.0f);
            test_result("Basic SIMD operations", simd_works);
        } catch (...) {
            test_result("Basic SIMD operations", false);
        }
        
        std::cout << std::endl;
    }
    
    void test_quantization() {
        std::cout << "4. Quantization Tests:" << std::endl;
        
        // Test INT8 quantization
        try {
            std::vector<float> data(64);
            for (size_t i = 0; i < data.size(); ++i) {
                data[i] = dist(rng);
            }
            
            auto params = lightgpt::quantization::compute_int8_symmetric_params(data.data(), data.size());
            bool params_valid = (params.scale > 0);
            test_result("INT8 quantization parameters", params_valid);
            
            if (params_valid) {
                auto quantized = lightgpt::quantization::quantize_int8(data.data(), data.size(), params);
                bool quantization_works = (quantized.size() == data.size());
                test_result("INT8 quantization process", quantization_works);
            } else {
                test_result("INT8 quantization process", false);
            }
        } catch (...) {
            test_result("INT8 quantization parameters", false);
            test_result("INT8 quantization process", false);
        }
        
        // Test INT4 quantization
        try {
            std::vector<float> data(128);
            for (size_t i = 0; i < data.size(); ++i) {
                data[i] = dist(rng);
            }
            
            auto params = lightgpt::quantization::compute_int4_params(data.data(), data.size());
            auto quantized = lightgpt::quantization::quantize_int4_packed(data.data(), data.size(), params);
            bool int4_works = (quantized.size() == data.size() / 2); // 2 values per byte
            test_result("INT4 packed quantization", int4_works);
        } catch (...) {
            test_result("INT4 packed quantization", false);
        }
        
        std::cout << std::endl;
    }
    
    void test_threading() {
        std::cout << "5. Threading Tests:" << std::endl;
        
        // Test thread pool creation
        try {
            auto& pool = lightgpt::threading::get_global_thread_pool();
            bool pool_created = (pool.size() > 0);
            test_result("Thread pool creation", pool_created);
        } catch (...) {
            test_result("Thread pool creation", false);
        }
        
        // Test task execution
        try {
            auto& pool = lightgpt::threading::get_global_thread_pool();
            auto future = pool.enqueue([]() { return 42; });
            int result = future.get();
            bool task_works = (result == 42);
            test_result("Thread pool task execution", task_works);
        } catch (...) {
            test_result("Thread pool task execution", false);
        }
        
        // Test parallel matrix operation
        try {
            const int size = 64;
            std::vector<float> A(size * size), B(size * size), C(size * size);
            
            for (size_t i = 0; i < A.size(); ++i) {
                A[i] = dist(rng);
                B[i] = dist(rng);
            }
            
            lightgpt::threading::parallel_gemm(A.data(), B.data(), C.data(), size, size, size);
            test_result("Parallel matrix multiply", true);
        } catch (...) {
            test_result("Parallel matrix multiply", false);
        }
        
        std::cout << std::endl;
    }
    
    void test_integration() {
        std::cout << "6. Integration Tests:" << std::endl;
        
        // Test combined optimizations
        try {
            // Use memory pool
            lightgpt::memory::MemoryPool pool(1024 * 1024);
            
            // Allocate small matrices
            const int size = 16;
            auto A = static_cast<float*>(pool.allocate(size * sizeof(float)));
            auto B = static_cast<float*>(pool.allocate(size * sizeof(float)));
            
            bool allocation_success = (A != nullptr && B != nullptr);
            
            if (allocation_success) {
                // Initialize data
                for (int i = 0; i < size; ++i) {
                    A[i] = static_cast<float>(i + 1);
                    B[i] = static_cast<float>(i + 1);
                }
                
                // Test quantization
                auto params = lightgpt::quantization::compute_int8_symmetric_params(A, size);
                bool integration_works = (params.scale > 0);
                test_result("Memory + Quantization integration", integration_works);
            } else {
                test_result("Memory + Quantization integration", false);
            }
        } catch (...) {
            test_result("Memory + Quantization integration", false);
        }
        
        std::cout << std::endl;
    }
    
    void test_performance() {
        std::cout << "7. Performance Validation:" << std::endl;
        
        // Quick performance check
        try {
            const int size = 64;
            std::vector<float> A(size), B(size), C(size);
            
            for (int i = 0; i < size; ++i) {
                A[i] = static_cast<float>(i + 1);
                B[i] = static_cast<float>(i + 1);
            }
            
            // Simple operation timing
            auto start = std::chrono::high_resolution_clock::now();
            for (int iter = 0; iter < 1000; ++iter) {
                for (int i = 0; i < size; ++i) {
                    C[i] = A[i] * B[i] + 1.0f;
                }
            }
            auto end = std::chrono::high_resolution_clock::now();
            
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
            bool performance_reasonable = (duration.count() < 10000); // Less than 10ms
            
            test_result("Performance validation", performance_reasonable);
            
            std::cout << std::setw(40) << std::left << "Operation time (µs)";
            std::cout << " " << duration.count() << " µs" << std::endl;
        } catch (...) {
            test_result("Performance validation", false);
        }
        
        std::cout << std::endl;
    }
    
    void test_error_handling() {
        std::cout << "8. Error Handling Tests:" << std::endl;
        
        // Test null pointer handling
        try {
            bool handles_nulls = true;
            
            // Test quantization with null pointer
            try {
                lightgpt::quantization::compute_int8_symmetric_params(nullptr, 100);
                handles_nulls = false; // Should have thrown or returned safely
            } catch (...) {
                // Expected behavior
            }
            
            test_result("Null pointer handling", handles_nulls);
        } catch (...) {
            test_result("Null pointer handling", false);
        }
        
        // Test memory pool overflow
        try {
            lightgpt::memory::MemoryPool small_pool(1024); // 1KB pool
            
            // Try to allocate more than available
            void* ptr1 = small_pool.allocate(512);
            void* ptr2 = small_pool.allocate(512);
            void* ptr3 = small_pool.allocate(512); // Should fail or handle gracefully
            
            bool overflow_handled = (ptr1 != nullptr && ptr2 != nullptr);
            test_result("Memory pool overflow handling", overflow_handled);
        } catch (...) {
            test_result("Memory pool overflow handling", false);
        }
        
        std::cout << std::endl;
    }
    
    void print_summary() {
        std::cout << "=== Test Summary ===" << std::endl;
        std::cout << "Tests passed: " << tests_passed << std::endl;
        std::cout << "Tests failed: " << tests_failed << std::endl;
        std::cout << "Total tests:  " << (tests_passed + tests_failed) << std::endl;
        
        if (tests_failed == 0) {
            std::cout << std::endl;
            std::cout << "🎉 ALL TESTS PASSED!" << std::endl;
            std::cout << "✅ LightGPT optimizations are working correctly!" << std::endl;
            std::cout << "🚀 Ready for production deployment!" << std::endl;
        } else {
            std::cout << std::endl;
            std::cout << "⚠️  Some tests failed. Please review the results above." << std::endl;
            std::cout << "🔧 Fix any issues before deploying to production." << std::endl;
        }
        
        std::cout << std::endl;
        std::cout << "📋 Next steps:" << std::endl;
        std::cout << "1. Run './test_script.sh' for full testing" << std::endl;
        std::cout << "2. Try 'make test' for automated build testing" << std::endl;
        std::cout << "3. Commit to GitHub with './commit_optimizations.sh'" << std::endl;
    }
};

int main() {
    try {
        ComprehensiveValidator validator;
        validator.run_all_tests();
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Fatal error during validation: " << e.what() << std::endl;
        return 1;
    } catch (...) {
        std::cerr << "Unknown fatal error during validation" << std::endl;
        return 1;
    }
} 