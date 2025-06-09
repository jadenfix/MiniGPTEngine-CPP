#include <iostream>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <cassert>
#include <arm_neon.h>

// Apple Accelerate framework
#ifdef __APPLE__
#include <Accelerate/Accelerate.h>
#endif

class AppleSiliconM2Optimizer {
private:
    static const size_t ALIGNMENT = 64;
    
    float* allocate_aligned(size_t N) {
        void* ptr = nullptr;
        if (posix_memalign(&ptr, ALIGNMENT, N * sizeof(float)) != 0) {
            return nullptr;
        }
        return static_cast<float*>(ptr);
    }

public:
    // Test 1: Simple regular loops with auto-vectorization
    void test_auto_vectorization(size_t N = 8*1024*1024) {
        float* a = allocate_aligned(N);
        float* b = allocate_aligned(N);
        float* out_scalar = allocate_aligned(N);
        float* out_vector = allocate_aligned(N);
        
        for (size_t i = 0; i < N; ++i) {
            a[i] = static_cast<float>(i) * 0.5f;
            b[i] = static_cast<float>(i) * 0.3f + 1.0f;
        }
        
        std::cout << "🧪 Test 1: Auto-vectorization with pragmas (N=" << N << ")\n";
        
        // Scalar baseline
        auto start = std::chrono::high_resolution_clock::now();
        for (size_t i = 0; i < N; ++i) {
            out_scalar[i] = a[i] + b[i];
        }
        auto end = std::chrono::high_resolution_clock::now();
        auto scalar_time = std::chrono::duration<double, std::micro>(end - start).count();
        
        // Auto-vectorized with pragma
        start = std::chrono::high_resolution_clock::now();
        #pragma clang loop vectorize(enable)
        for (size_t i = 0; i < N; ++i) {
            out_vector[i] = a[i] + b[i];
        }
        end = std::chrono::high_resolution_clock::now();
        auto vector_time = std::chrono::duration<double, std::micro>(end - start).count();
        
        double speedup = scalar_time / vector_time;
        std::cout << "   Scalar: " << scalar_time << " μs\n";
        std::cout << "   Vector: " << vector_time << " μs\n";
        std::cout << "   Speedup: " << speedup << "×\n\n";
        
        free(a); free(b); free(out_scalar); free(out_vector);
    }
    
    // Test 2: Manual NEON intrinsics
    void test_neon_intrinsics(size_t N = 8*1024*1024) {
        float* a = allocate_aligned(N);
        float* b = allocate_aligned(N);
        float* out_scalar = allocate_aligned(N);
        float* out_neon = allocate_aligned(N);
        
        for (size_t i = 0; i < N; ++i) {
            a[i] = static_cast<float>(i) * 0.5f;
            b[i] = static_cast<float>(i) * 0.3f + 1.0f;
        }
        
        std::cout << "🧪 Test 2: Manual NEON intrinsics (N=" << N << ")\n";
        
        // Scalar baseline
        auto start = std::chrono::high_resolution_clock::now();
        for (size_t i = 0; i < N; ++i) {
            out_scalar[i] = a[i] + b[i];
        }
        auto end = std::chrono::high_resolution_clock::now();
        auto scalar_time = std::chrono::duration<double, std::micro>(end - start).count();
        
        // NEON intrinsics
        start = std::chrono::high_resolution_clock::now();
        size_t i = 0;
        for (; i + 4 <= N; i += 4) {
            float32x4_t va = vld1q_f32(a + i);
            float32x4_t vb = vld1q_f32(b + i);
            float32x4_t vc = vaddq_f32(va, vb);
            vst1q_f32(out_neon + i, vc);
        }
        for (; i < N; ++i) {
            out_neon[i] = a[i] + b[i];
        }
        end = std::chrono::high_resolution_clock::now();
        auto neon_time = std::chrono::duration<double, std::micro>(end - start).count();
        
        double speedup = scalar_time / neon_time;
        std::cout << "   Scalar: " << scalar_time << " μs\n";
        std::cout << "   NEON: " << neon_time << " μs\n";
        std::cout << "   Speedup: " << speedup << "×\n\n";
        
        free(a); free(b); free(out_scalar); free(out_neon);
    }
    
    // Test 3: FMA operations
    void test_fma_operations(size_t N = 8*1024*1024) {
        float* a = allocate_aligned(N);
        float* b = allocate_aligned(N);
        float* c = allocate_aligned(N);
        float* out_scalar = allocate_aligned(N);
        float* out_neon = allocate_aligned(N);
        
        for (size_t i = 0; i < N; ++i) {
            a[i] = static_cast<float>(i) * 0.5f;
            b[i] = static_cast<float>(i) * 0.3f + 1.0f;
            c[i] = static_cast<float>(i) * 0.7f - 2.0f;
        }
        
        std::cout << "🧪 Test 3: FMA operations (a*b + c) (N=" << N << ")\n";
        
        // Scalar
        auto start = std::chrono::high_resolution_clock::now();
        for (size_t i = 0; i < N; ++i) {
            out_scalar[i] = a[i] * b[i] + c[i];
        }
        auto end = std::chrono::high_resolution_clock::now();
        auto scalar_time = std::chrono::duration<double, std::micro>(end - start).count();
        
        // NEON FMA
        start = std::chrono::high_resolution_clock::now();
        size_t i = 0;
        for (; i + 4 <= N; i += 4) {
            float32x4_t va = vld1q_f32(a + i);
            float32x4_t vb = vld1q_f32(b + i);
            float32x4_t vc = vld1q_f32(c + i);
            float32x4_t vresult = vfmaq_f32(vc, va, vb);
            vst1q_f32(out_neon + i, vresult);
        }
        for (; i < N; ++i) {
            out_neon[i] = a[i] * b[i] + c[i];
        }
        end = std::chrono::high_resolution_clock::now();
        auto neon_time = std::chrono::duration<double, std::micro>(end - start).count();
        
        double speedup = scalar_time / neon_time;
        std::cout << "   Scalar: " << scalar_time << " μs\n";
        std::cout << "   NEON FMA: " << neon_time << " μs\n";
        std::cout << "   Speedup: " << speedup << "×\n\n";
        
        free(a); free(b); free(c); free(out_scalar); free(out_neon);
    }

#ifdef __APPLE__
    // Test 4: Apple Accelerate framework
    void test_accelerate_framework(size_t N = 8*1024*1024) {
        float* a = allocate_aligned(N);
        float* b = allocate_aligned(N);
        float* out_manual = allocate_aligned(N);
        float* out_accelerate = allocate_aligned(N);
        
        for (size_t i = 0; i < N; ++i) {
            a[i] = static_cast<float>(i) * 0.5f;
            b[i] = static_cast<float>(i) * 0.3f + 1.0f;
        }
        
        std::cout << "🧪 Test 4: Apple Accelerate framework (N=" << N << ")\n";
        
        // Manual NEON
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
        
        // Accelerate framework
        start = std::chrono::high_resolution_clock::now();
        vDSP_vadd(a, 1, b, 1, out_accelerate, 1, N);
        end = std::chrono::high_resolution_clock::now();
        auto accelerate_time = std::chrono::duration<double, std::micro>(end - start).count();
        
        double speedup = manual_time / accelerate_time;
        std::cout << "   Manual NEON: " << manual_time << " μs\n";
        std::cout << "   Accelerate: " << accelerate_time << " μs\n";
        std::cout << "   Accelerate speedup: " << speedup << "×\n\n";
        
        free(a); free(b); free(out_manual); free(out_accelerate);
    }
#endif
    
    void run_comprehensive_tests() {
        std::cout << "🚀 Apple Silicon M2 Comprehensive Vector Tests\n";
        std::cout << "============================================\n\n";
        
        test_auto_vectorization();
        test_neon_intrinsics();
        test_fma_operations();
        
#ifdef __APPLE__
        test_accelerate_framework();
#endif
        
        std::cout << "✅ Target: 2.0× speedup or better\n";
    }
};

int main() {
    AppleSiliconM2Optimizer optimizer;
    optimizer.run_comprehensive_tests();
    return 0;
} 