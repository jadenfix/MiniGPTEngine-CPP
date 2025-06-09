#include <iostream>
#include <chrono>
#include <vector>
#include <arm_neon.h>
#include <cstdlib>
#include <cstring>
#include <cassert>

// Optional: Include Accelerate framework if available
#ifdef __APPLE__
#include <Accelerate/Accelerate.h>
#endif

class AppleSiliconVectorOptimizer {
private:
    static const size_t ALIGNMENT = 64; // 64-byte alignment for optimal cache performance
    
    // Aligned memory allocation
    float* allocate_aligned(size_t N) {
        void* ptr = nullptr;
        if (posix_memalign(&ptr, ALIGNMENT, N * sizeof(float)) != 0) {
            return nullptr;
        }
        return static_cast<float*>(ptr);
    }

public:
    // Test 1: Simple vector addition with auto-vectorization hints
    void test_auto_vectorized_addition(size_t N = 8*1024*1024) {
        float* a = allocate_aligned(N);
        float* b = allocate_aligned(N);
        float* out_scalar = allocate_aligned(N);
        float* out_vector = allocate_aligned(N);
        
        // Initialize data
        for (size_t i = 0; i < N; ++i) {
            a[i] = static_cast<float>(i) * 0.5f;
            b[i] = static_cast<float>(i) * 0.3f + 1.0f;
        }
        
        std::cout << "🧪 Test 1: Auto-vectorized Addition (N=" << N << ")\n";
        
        // Scalar version
        auto start = std::chrono::high_resolution_clock::now();
        for (size_t i = 0; i < N; ++i) {
            out_scalar[i] = a[i] + b[i];
        }
        auto end = std::chrono::high_resolution_clock::now();
        auto scalar_time = std::chrono::duration<double, std::micro>(end - start).count();
        
        // Vector version with pragma hints
        start = std::chrono::high_resolution_clock::now();
        #pragma clang loop vectorize(enable)
        for (size_t i = 0; i < N; ++i) {
            out_vector[i] = a[i] + b[i];
        }
        end = std::chrono::high_resolution_clock::now();
        auto vector_time = std::chrono::duration<double, std::micro>(end - start).count();
        
        // Verify correctness
        for (size_t i = 0; i < std::min(N, size_t(100)); ++i) {
            assert(std::abs(out_scalar[i] - out_vector[i]) < 1e-6f);
        }
        
        double speedup = scalar_time / vector_time;
        std::cout << "   Scalar time: " << scalar_time << " μs\n";
        std::cout << "   Vector time: " << vector_time << " μs\n";
        std::cout << "   Speedup: " << speedup << "×\n\n";
        
        free(a); free(b); free(out_scalar); free(out_vector);
    }
    
    // Test 2: Explicit NEON intrinsics - vector addition
    void test_neon_intrinsics_addition(size_t N = 8*1024*1024) {
        float* a = allocate_aligned(N);
        float* b = allocate_aligned(N);
        float* out_scalar = allocate_aligned(N);
        float* out_neon = allocate_aligned(N);
        
        // Initialize data
        for (size_t i = 0; i < N; ++i) {
            a[i] = static_cast<float>(i) * 0.5f;
            b[i] = static_cast<float>(i) * 0.3f + 1.0f;
        }
        
        std::cout << "🧪 Test 2: NEON Intrinsics Addition (N=" << N << ")\n";
        
        // Scalar version
        auto start = std::chrono::high_resolution_clock::now();
        for (size_t i = 0; i < N; ++i) {
            out_scalar[i] = a[i] + b[i];
        }
        auto end = std::chrono::high_resolution_clock::now();
        auto scalar_time = std::chrono::duration<double, std::micro>(end - start).count();
        
        // NEON intrinsics version
        start = std::chrono::high_resolution_clock::now();
        size_t i = 0;
        for (; i + 4 <= N; i += 4) {
            float32x4_t va = vld1q_f32(a + i);
            float32x4_t vb = vld1q_f32(b + i);
            float32x4_t vc = vaddq_f32(va, vb);
            vst1q_f32(out_neon + i, vc);
        }
        // Handle remaining elements
        for (; i < N; ++i) {
            out_neon[i] = a[i] + b[i];
        }
        end = std::chrono::high_resolution_clock::now();
        auto neon_time = std::chrono::duration<double, std::micro>(end - start).count();
        
        // Verify correctness
        for (size_t i = 0; i < std::min(N, size_t(100)); ++i) {
            assert(std::abs(out_scalar[i] - out_neon[i]) < 1e-6f);
        }
        
        double speedup = scalar_time / neon_time;
        std::cout << "   Scalar time: " << scalar_time << " μs\n";
        std::cout << "   NEON time: " << neon_time << " μs\n";
        std::cout << "   Speedup: " << speedup << "×\n\n";
        
        free(a); free(b); free(out_scalar); free(out_neon);
    }
    
    // Test 3: Complex computation - FMA (Fused Multiply-Add)
    void test_neon_fma(size_t N = 8*1024*1024) {
        float* a = allocate_aligned(N);
        float* b = allocate_aligned(N);
        float* c = allocate_aligned(N);
        float* out_scalar = allocate_aligned(N);
        float* out_neon = allocate_aligned(N);
        
        // Initialize data
        for (size_t i = 0; i < N; ++i) {
            a[i] = static_cast<float>(i) * 0.5f;
            b[i] = static_cast<float>(i) * 0.3f + 1.0f;
            c[i] = static_cast<float>(i) * 0.7f - 2.0f;
        }
        
        std::cout << "🧪 Test 3: NEON FMA (a*b + c) (N=" << N << ")\n";
        
        // Scalar version
        auto start = std::chrono::high_resolution_clock::now();
        for (size_t i = 0; i < N; ++i) {
            out_scalar[i] = a[i] * b[i] + c[i];
        }
        auto end = std::chrono::high_resolution_clock::now();
        auto scalar_time = std::chrono::duration<double, std::micro>(end - start).count();
        
        // NEON FMA version
        start = std::chrono::high_resolution_clock::now();
        size_t i = 0;
        for (; i + 4 <= N; i += 4) {
            float32x4_t va = vld1q_f32(a + i);
            float32x4_t vb = vld1q_f32(b + i);
            float32x4_t vc = vld1q_f32(c + i);
            float32x4_t vresult = vfmaq_f32(vc, va, vb); // c + a*b
            vst1q_f32(out_neon + i, vresult);
        }
        // Handle remaining elements
        for (; i < N; ++i) {
            out_neon[i] = a[i] * b[i] + c[i];
        }
        end = std::chrono::high_resolution_clock::now();
        auto neon_time = std::chrono::duration<double, std::micro>(end - start).count();
        
        // Verify correctness
        for (size_t i = 0; i < std::min(N, size_t(100)); ++i) {
            assert(std::abs(out_scalar[i] - out_neon[i]) < 1e-6f);
        }
        
        double speedup = scalar_time / neon_time;
        std::cout << "   Scalar time: " << scalar_time << " μs\n";
        std::cout << "   NEON FMA time: " << neon_time << " μs\n";
        std::cout << "   Speedup: " << speedup << "×\n\n";
        
        free(a); free(b); free(c); free(out_scalar); free(out_neon);
    }
    
    // Test 4: Matrix multiplication with cache blocking
    void test_matrix_multiply(size_t M = 512, size_t N = 512, size_t K = 512) {
        float* A = allocate_aligned(M * K);
        float* B = allocate_aligned(K * N);
        float* C_scalar = allocate_aligned(M * N);
        float* C_neon = allocate_aligned(M * N);
        
        // Initialize matrices
        for (size_t i = 0; i < M * K; ++i) A[i] = (float)(rand() % 100) / 100.0f;
        for (size_t i = 0; i < K * N; ++i) B[i] = (float)(rand() % 100) / 100.0f;
        
        std::cout << "🧪 Test 4: Matrix Multiplication (" << M << "×" << K << " × " << K << "×" << N << ")\n";
        
        // Scalar version
        auto start = std::chrono::high_resolution_clock::now();
        for (size_t i = 0; i < M; ++i) {
            for (size_t j = 0; j < N; ++j) {
                float sum = 0.0f;
                for (size_t k = 0; k < K; ++k) {
                    sum += A[i * K + k] * B[k * N + j];
                }
                C_scalar[i * N + j] = sum;
            }
        }
        auto end = std::chrono::high_resolution_clock::now();
        auto scalar_time = std::chrono::duration<double, std::milli>(end - start).count();
        
        // NEON optimized version with cache blocking
        start = std::chrono::high_resolution_clock::now();
        const size_t BLOCK_SIZE = 32;
        
        // Initialize C_neon to zero
        memset(C_neon, 0, M * N * sizeof(float));
        
        for (size_t ii = 0; ii < M; ii += BLOCK_SIZE) {
            for (size_t jj = 0; jj < N; jj += BLOCK_SIZE) {
                for (size_t kk = 0; kk < K; kk += BLOCK_SIZE) {
                    // Process block
                    size_t i_end = std::min(ii + BLOCK_SIZE, M);
                    size_t j_end = std::min(jj + BLOCK_SIZE, N);
                    size_t k_end = std::min(kk + BLOCK_SIZE, K);
                    
                    for (size_t i = ii; i < i_end; ++i) {
                        for (size_t j = jj; j < j_end; j += 4) {
                            float32x4_t c_vec = vld1q_f32(&C_neon[i * N + j]);
                            
                            for (size_t k = kk; k < k_end; ++k) {
                                float32x4_t a_vec = vdupq_n_f32(A[i * K + k]);
                                float32x4_t b_vec = vld1q_f32(&B[k * N + j]);
                                c_vec = vfmaq_f32(c_vec, a_vec, b_vec);
                            }
                            
                            vst1q_f32(&C_neon[i * N + j], c_vec);
                        }
                    }
                }
            }
        }
        end = std::chrono::high_resolution_clock::now();
        auto neon_time = std::chrono::duration<double, std::milli>(end - start).count();
        
        // Verify correctness (sample check)
        float max_diff = 0.0f;
        for (size_t i = 0; i < std::min(M, size_t(10)); ++i) {
            for (size_t j = 0; j < std::min(N, size_t(10)); ++j) {
                float diff = std::abs(C_scalar[i * N + j] - C_neon[i * N + j]);
                max_diff = std::max(max_diff, diff);
            }
        }
        
        double speedup = scalar_time / neon_time;
        std::cout << "   Scalar time: " << scalar_time << " ms\n";
        std::cout << "   NEON time: " << neon_time << " ms\n";
        std::cout << "   Speedup: " << speedup << "×\n";
        std::cout << "   Max difference: " << max_diff << "\n\n";
        
        free(A); free(B); free(C_scalar); free(C_neon);
    }
    
#ifdef __APPLE__
    // Test 5: Apple Accelerate framework comparison
    void test_accelerate_framework(size_t N = 8*1024*1024) {
        float* a = allocate_aligned(N);
        float* b = allocate_aligned(N);
        float* out_manual = allocate_aligned(N);
        float* out_accelerate = allocate_aligned(N);
        
        // Initialize data
        for (size_t i = 0; i < N; ++i) {
            a[i] = static_cast<float>(i) * 0.5f;
            b[i] = static_cast<float>(i) * 0.3f + 1.0f;
        }
        
        std::cout << "🧪 Test 5: Apple Accelerate Framework (N=" << N << ")\n";
        
        // Manual NEON version
        auto start = std::chrono::high_resolution_clock::now();
        size_t i = 0;
        for (; i + 4 <= N; i += 4) {
            float32x4_t va = vld1q_f32(a + i);
            float32x4_t vb = vld1q_f32(b + i);
            float32x4_t vc = vaddq_f32(va, vb);
            vst1q_f32(out_manual + i, vc);
        }
        for (; i < N; ++i) {
            out_manual[i] = a[i] + b[i];
        }
        auto end = std::chrono::high_resolution_clock::now();
        auto manual_time = std::chrono::duration<double, std::micro>(end - start).count();
        
        // Accelerate framework version
        start = std::chrono::high_resolution_clock::now();
        vDSP_vadd(a, 1, b, 1, out_accelerate, 1, N);
        end = std::chrono::high_resolution_clock::now();
        auto accelerate_time = std::chrono::duration<double, std::micro>(end - start).count();
        
        // Verify correctness
        for (size_t i = 0; i < std::min(N, size_t(100)); ++i) {
            assert(std::abs(out_manual[i] - out_accelerate[i]) < 1e-6f);
        }
        
        double speedup = manual_time / accelerate_time;
        std::cout << "   Manual NEON time: " << manual_time << " μs\n";
        std::cout << "   Accelerate time: " << accelerate_time << " μs\n";
        std::cout << "   Accelerate speedup: " << speedup << "×\n\n";
        
        free(a); free(b); free(out_manual); free(out_accelerate);
    }
#endif
    
    // Test 6: Vector reduction operations
    void test_vector_reduction(size_t N = 8*1024*1024) {
        float* data = allocate_aligned(N);
        
        // Initialize data
        for (size_t i = 0; i < N; ++i) {
            data[i] = static_cast<float>(i % 1000) * 0.001f;
        }
        
        std::cout << "🧪 Test 6: Vector Sum Reduction (N=" << N << ")\n";
        
        // Scalar version
        auto start = std::chrono::high_resolution_clock::now();
        float scalar_sum = 0.0f;
        for (size_t i = 0; i < N; ++i) {
            scalar_sum += data[i];
        }
        auto end = std::chrono::high_resolution_clock::now();
        auto scalar_time = std::chrono::duration<double, std::micro>(end - start).count();
        
        // NEON reduction version
        start = std::chrono::high_resolution_clock::now();
        float32x4_t vec_sum = vdupq_n_f32(0.0f);
        size_t i = 0;
        for (; i + 4 <= N; i += 4) {
            float32x4_t vec_data = vld1q_f32(data + i);
            vec_sum = vaddq_f32(vec_sum, vec_data);
        }
        
        // Horizontal sum of the vector
        float neon_sum = vaddvq_f32(vec_sum);
        
        // Handle remaining elements
        for (; i < N; ++i) {
            neon_sum += data[i];
        }
        end = std::chrono::high_resolution_clock::now();
        auto neon_time = std::chrono::duration<double, std::micro>(end - start).count();
        
        // Verify correctness
        assert(std::abs(scalar_sum - neon_sum) < 1e-3f);
        
        double speedup = scalar_time / neon_time;
        std::cout << "   Scalar time: " << scalar_time << " μs (sum=" << scalar_sum << ")\n";
        std::cout << "   NEON time: " << neon_time << " μs (sum=" << neon_sum << ")\n";
        std::cout << "   Speedup: " << speedup << "×\n\n";
        
        free(data);
    }
    
    void run_all_tests() {
        std::cout << "🚀 Apple Silicon M2 Vector Unit Performance Tests\n";
        std::cout << "================================================\n\n";
        
        test_auto_vectorized_addition();
        test_neon_intrinsics_addition();
        test_neon_fma();
        test_matrix_multiply();
        
#ifdef __APPLE__
        test_accelerate_framework();
#endif
        
        test_vector_reduction();
        
        std::cout << "✅ All tests completed!\n";
        std::cout << "🎯 Target: 2.0× speedup or better\n";
    }
};

int main() {
    AppleSiliconVectorOptimizer optimizer;
    optimizer.run_all_tests();
    return 0;
} 