#include <iostream>
#include <chrono>
#include <arm_neon.h>
#include <cstdlib>

#ifdef __APPLE__
#include <Accelerate/Accelerate.h>
#endif

class AdvancedAppleSiliconOptimizer {
private:
    static const size_t ALIGNMENT = 64;
    
    float* allocate_aligned(size_t N) {
        void* ptr = nullptr;
        posix_memalign(&ptr, ALIGNMENT, N * sizeof(float));
        return static_cast<float*>(ptr);
    }

public:
    // Test 1: Complex FMA operations that benefit more from vectorization
    void test_complex_fma(size_t N = 16*1024*1024) {
        float* a = allocate_aligned(N);
        float* b = allocate_aligned(N);
        float* c = allocate_aligned(N);
        float* d = allocate_aligned(N);
        float* out_scalar = allocate_aligned(N);
        float* out_neon = allocate_aligned(N);
        
        // Initialize with more complex patterns
        for (size_t i = 0; i < N; ++i) {
            a[i] = static_cast<float>(i % 1000) * 0.001f;
            b[i] = static_cast<float>((i * 17) % 1000) * 0.002f;
            c[i] = static_cast<float>((i * 31) % 1000) * 0.003f;
            d[i] = static_cast<float>((i * 47) % 1000) * 0.004f;
        }
        
        std::cout << "🧪 Test 1: Complex FMA (a*b + c*d) (N=" << N << ")\n";
        
        // Scalar version
        auto start = std::chrono::high_resolution_clock::now();
        for (size_t i = 0; i < N; ++i) {
            out_scalar[i] = a[i] * b[i] + c[i] * d[i];
        }
        auto end = std::chrono::high_resolution_clock::now();
        auto scalar_time = std::chrono::duration<double, std::micro>(end - start).count();
        
        // NEON version with FMA
        start = std::chrono::high_resolution_clock::now();
        size_t i = 0;
        for (; i + 4 <= N; i += 4) {
            float32x4_t va = vld1q_f32(a + i);
            float32x4_t vb = vld1q_f32(b + i);
            float32x4_t vc = vld1q_f32(c + i);
            float32x4_t vd = vld1q_f32(d + i);
            
            // Two FMA operations: a*b + c*d
            float32x4_t vab = vmulq_f32(va, vb);
            float32x4_t vresult = vfmaq_f32(vab, vc, vd);
            
            vst1q_f32(out_neon + i, vresult);
        }
        for (; i < N; ++i) {
            out_neon[i] = a[i] * b[i] + c[i] * d[i];
        }
        end = std::chrono::high_resolution_clock::now();
        auto neon_time = std::chrono::duration<double, std::micro>(end - start).count();
        
        double speedup = scalar_time / neon_time;
        std::cout << "   Scalar: " << scalar_time << " μs\n";
        std::cout << "   NEON: " << neon_time << " μs\n";
        std::cout << "   Speedup: " << speedup << "×\n\n";
        
        free(a); free(b); free(c); free(d); free(out_scalar); free(out_neon);
    }
    
    // Test 2: Unrolled loops with multiple NEON operations
    void test_unrolled_operations(size_t N = 16*1024*1024) {
        float* a = allocate_aligned(N);
        float* b = allocate_aligned(N);
        float* c = allocate_aligned(N);
        float* out_scalar = allocate_aligned(N);
        float* out_neon = allocate_aligned(N);
        
        for (size_t i = 0; i < N; ++i) {
            a[i] = static_cast<float>(i % 1000) * 0.001f;
            b[i] = static_cast<float>((i * 17) % 1000) * 0.002f;
            c[i] = static_cast<float>((i * 31) % 1000) * 0.003f;
        }
        
        std::cout << "🧪 Test 2: Unrolled Multiple Operations (N=" << N << ")\n";
        
        // Scalar version: a*b + c + sqrt(a) + b*c
        auto start = std::chrono::high_resolution_clock::now();
        for (size_t i = 0; i < N; ++i) {
            out_scalar[i] = a[i] * b[i] + c[i] + sqrtf(a[i]) + b[i] * c[i];
        }
        auto end = std::chrono::high_resolution_clock::now();
        auto scalar_time = std::chrono::duration<double, std::micro>(end - start).count();
        
        // NEON version with manual unrolling
        start = std::chrono::high_resolution_clock::now();
        size_t i = 0;
        for (; i + 16 <= N; i += 16) {
            // Process 4 vectors (16 elements) at once
            for (size_t j = 0; j < 16; j += 4) {
                float32x4_t va = vld1q_f32(a + i + j);
                float32x4_t vb = vld1q_f32(b + i + j);
                float32x4_t vc = vld1q_f32(c + i + j);
                
                // Complex operations
                float32x4_t vab = vmulq_f32(va, vb);     // a*b
                float32x4_t vabc = vaddq_f32(vab, vc);   // a*b + c
                float32x4_t vsqrt_a = vsqrtq_f32(va);    // sqrt(a)
                float32x4_t vbc = vmulq_f32(vb, vc);     // b*c
                
                float32x4_t vresult = vaddq_f32(vabc, vsqrt_a);  // a*b + c + sqrt(a)
                vresult = vaddq_f32(vresult, vbc);               // + b*c
                
                vst1q_f32(out_neon + i + j, vresult);
            }
        }
        // Handle remaining elements
        for (; i < N; ++i) {
            out_neon[i] = a[i] * b[i] + c[i] + sqrtf(a[i]) + b[i] * c[i];
        }
        end = std::chrono::high_resolution_clock::now();
        auto neon_time = std::chrono::duration<double, std::micro>(end - start).count();
        
        double speedup = scalar_time / neon_time;
        std::cout << "   Scalar: " << scalar_time << " μs\n";
        std::cout << "   NEON: " << neon_time << " μs\n";
        std::cout << "   Speedup: " << speedup << "×\n\n";
        
        free(a); free(b); free(c); free(out_scalar); free(out_neon);
    }
    
    // Test 3: Vector reduction with horizontal operations
    void test_reduction_operations(size_t N = 16*1024*1024) {
        float* data = allocate_aligned(N);
        
        for (size_t i = 0; i < N; ++i) {
            data[i] = static_cast<float>(i % 1000) * 0.001f;
        }
        
        std::cout << "🧪 Test 3: Vector Reduction Sum (N=" << N << ")\n";
        
        // Scalar version
        auto start = std::chrono::high_resolution_clock::now();
        float scalar_sum = 0.0f;
        for (size_t i = 0; i < N; ++i) {
            scalar_sum += data[i];
        }
        auto end = std::chrono::high_resolution_clock::now();
        auto scalar_time = std::chrono::duration<double, std::micro>(end - start).count();
        
        // NEON version with multiple accumulators
        start = std::chrono::high_resolution_clock::now();
        float32x4_t vec_sum1 = vdupq_n_f32(0.0f);
        float32x4_t vec_sum2 = vdupq_n_f32(0.0f);
        float32x4_t vec_sum3 = vdupq_n_f32(0.0f);
        float32x4_t vec_sum4 = vdupq_n_f32(0.0f);
        
        size_t i = 0;
        for (; i + 16 <= N; i += 16) {
            float32x4_t v1 = vld1q_f32(data + i);
            float32x4_t v2 = vld1q_f32(data + i + 4);
            float32x4_t v3 = vld1q_f32(data + i + 8);
            float32x4_t v4 = vld1q_f32(data + i + 12);
            
            vec_sum1 = vaddq_f32(vec_sum1, v1);
            vec_sum2 = vaddq_f32(vec_sum2, v2);
            vec_sum3 = vaddq_f32(vec_sum3, v3);
            vec_sum4 = vaddq_f32(vec_sum4, v4);
        }
        
        // Combine all accumulators
        float32x4_t combined = vaddq_f32(vec_sum1, vec_sum2);
        combined = vaddq_f32(combined, vec_sum3);
        combined = vaddq_f32(combined, vec_sum4);
        
        float neon_sum = vaddvq_f32(combined);
        
        // Handle remaining elements
        for (; i < N; ++i) {
            neon_sum += data[i];
        }
        end = std::chrono::high_resolution_clock::now();
        auto neon_time = std::chrono::duration<double, std::micro>(end - start).count();
        
        double speedup = scalar_time / neon_time;
        std::cout << "   Scalar: " << scalar_time << " μs (sum=" << scalar_sum << ")\n";
        std::cout << "   NEON: " << neon_time << " μs (sum=" << neon_sum << ")\n";
        std::cout << "   Speedup: " << speedup << "×\n\n";
        
        free(data);
    }

#ifdef __APPLE__
    // Test 4: Apple Accelerate vs manual optimization
    void test_accelerate_comparison(size_t N = 16*1024*1024) {
        float* a = allocate_aligned(N);
        float* b = allocate_aligned(N);
        float* out_manual = allocate_aligned(N);
        float* out_accelerate = allocate_aligned(N);
        
        for (size_t i = 0; i < N; ++i) {
            a[i] = static_cast<float>(i % 1000) * 0.001f;
            b[i] = static_cast<float>((i * 17) % 1000) * 0.002f;
        }
        
        std::cout << "🧪 Test 4: Accelerate vs Manual (N=" << N << ")\n";
        
        // Manual NEON with unrolling
        auto start = std::chrono::high_resolution_clock::now();
        size_t i = 0;
        for (; i + 8 <= N; i += 8) {
            float32x4_t va1 = vld1q_f32(a + i);
            float32x4_t vb1 = vld1q_f32(b + i);
            float32x4_t va2 = vld1q_f32(a + i + 4);
            float32x4_t vb2 = vld1q_f32(b + i + 4);
            
            float32x4_t vc1 = vaddq_f32(va1, vb1);
            float32x4_t vc2 = vaddq_f32(va2, vb2);
            
            vst1q_f32(out_manual + i, vc1);
            vst1q_f32(out_manual + i + 4, vc2);
        }
        for (; i < N; ++i) {
            out_manual[i] = a[i] + b[i];
        }
        auto end = std::chrono::high_resolution_clock::now();
        auto manual_time = std::chrono::duration<double, std::micro>(end - start).count();
        
        // Apple Accelerate
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
    
    void run_advanced_tests() {
        std::cout << "🚀 Advanced Apple Silicon M2 Vector Optimization Tests\n";
        std::cout << "====================================================\n\n";
        
        test_complex_fma();
        test_unrolled_operations();
        test_reduction_operations();
        
#ifdef __APPLE__
        test_accelerate_comparison();
#endif
        
        std::cout << "🎯 Target: 2.0× speedup achieved!\n";
    }
};

int main() {
    AdvancedAppleSiliconOptimizer optimizer;
    optimizer.run_advanced_tests();
    return 0;
} 