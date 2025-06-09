#include <iostream>
#include <chrono>
#include <arm_neon.h>
#include <cstdlib>
#include <cmath>

int main() {
    const size_t N = 16*1024*1024;  // Larger dataset
    
    // Aligned memory allocation
    void* ptr_a = nullptr;
    void* ptr_b = nullptr;
    void* ptr_c = nullptr;
    void* ptr_out_scalar = nullptr;
    void* ptr_out_neon = nullptr;
    
    posix_memalign(&ptr_a, 64, N * sizeof(float));
    posix_memalign(&ptr_b, 64, N * sizeof(float));
    posix_memalign(&ptr_c, 64, N * sizeof(float));
    posix_memalign(&ptr_out_scalar, 64, N * sizeof(float));
    posix_memalign(&ptr_out_neon, 64, N * sizeof(float));
    
    float* a = static_cast<float*>(ptr_a);
    float* b = static_cast<float*>(ptr_b);
    float* c = static_cast<float*>(ptr_c);
    float* out_scalar = static_cast<float*>(ptr_out_scalar);
    float* out_neon = static_cast<float*>(ptr_out_neon);
    
    // Initialize with varied data patterns
    for (size_t i = 0; i < N; ++i) {
        a[i] = static_cast<float>(i % 1000) * 0.001f;
        b[i] = static_cast<float>((i * 17) % 1000) * 0.002f;
        c[i] = static_cast<float>((i * 31) % 1000) * 0.003f;
    }
    
    std::cout << "🚀 Advanced Apple Silicon M2 Test: Complex FMA (N=" << N << ")\n";
    std::cout << "Formula: a*b + c*sqrt(a) + b*c\n\n";
    
    // Scalar version - more complex computation
    auto start = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < N; ++i) {
        out_scalar[i] = a[i] * b[i] + c[i] * sqrtf(a[i]) + b[i] * c[i];
    }
    auto end = std::chrono::high_resolution_clock::now();
    auto scalar_time = std::chrono::duration<double, std::micro>(end - start).count();
    
    // NEON version with complex operations
    start = std::chrono::high_resolution_clock::now();
    size_t i = 0;
    for (; i + 8 <= N; i += 8) {
        // Process 8 elements at once (2 NEON vectors)
        float32x4_t va1 = vld1q_f32(a + i);
        float32x4_t vb1 = vld1q_f32(b + i);
        float32x4_t vc1 = vld1q_f32(c + i);
        
        float32x4_t va2 = vld1q_f32(a + i + 4);
        float32x4_t vb2 = vld1q_f32(b + i + 4);
        float32x4_t vc2 = vld1q_f32(c + i + 4);
        
        // Complex computation: a*b + c*sqrt(a) + b*c
        float32x4_t vab1 = vmulq_f32(va1, vb1);         // a*b
        float32x4_t vsqrt_a1 = vsqrtq_f32(va1);         // sqrt(a)
        float32x4_t vc_sqrt1 = vmulq_f32(vc1, vsqrt_a1); // c*sqrt(a)
        float32x4_t vbc1 = vmulq_f32(vb1, vc1);         // b*c
        
        float32x4_t vab2 = vmulq_f32(va2, vb2);
        float32x4_t vsqrt_a2 = vsqrtq_f32(va2);
        float32x4_t vc_sqrt2 = vmulq_f32(vc2, vsqrt_a2);
        float32x4_t vbc2 = vmulq_f32(vb2, vc2);
        
        // Combine using FMA
        float32x4_t vresult1 = vfmaq_f32(vab1, vc_sqrt1, vld1q_dup_f32(&(float){1.0f}));
        vresult1 = vaddq_f32(vresult1, vbc1);
        
        float32x4_t vresult2 = vfmaq_f32(vab2, vc_sqrt2, vld1q_dup_f32(&(float){1.0f}));
        vresult2 = vaddq_f32(vresult2, vbc2);
        
        vst1q_f32(out_neon + i, vresult1);
        vst1q_f32(out_neon + i + 4, vresult2);
    }
    
    // Handle remaining elements
    for (; i < N; ++i) {
        out_neon[i] = a[i] * b[i] + c[i] * sqrtf(a[i]) + b[i] * c[i];
    }
    end = std::chrono::high_resolution_clock::now();
    auto neon_time = std::chrono::duration<double, std::micro>(end - start).count();
    
    double speedup = scalar_time / neon_time;
    std::cout << "Scalar time: " << scalar_time << " μs\n";
    std::cout << "NEON time: " << neon_time << " μs\n";
    std::cout << "Speedup: " << speedup << "×\n";
    std::cout << "🎯 Target: 2.0× speedup\n";
    
    if (speedup >= 2.0) {
        std::cout << "🎉 TARGET ACHIEVED! " << speedup << "× >= 2.0×\n";
    } else {
        std::cout << "📈 Progress: " << (speedup/2.0)*100 << "% of target\n";
    }
    
    free(a); free(b); free(c); free(out_scalar); free(out_neon);
    return 0;
} 