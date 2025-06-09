#include <iostream>
#include <chrono>
#include <arm_neon.h>
#include <cstdlib>
#include <cmath>

int main() {
    const size_t N = 16*1024*1024;
    
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
    
    // Initialize data
    for (size_t i = 0; i < N; ++i) {
        a[i] = static_cast<float>(i % 1000) * 0.001f + 1.0f;
        b[i] = static_cast<float>((i * 17) % 1000) * 0.002f + 0.5f;
        c[i] = static_cast<float>((i * 31) % 1000) * 0.003f + 0.1f;
    }
    
    std::cout << "🚀 Complex Apple Silicon M2 Test (N=" << N << ")\n";
    std::cout << "Formula: a*b + c + sqrt(a) + b*c + a/c\n\n";
    
    // Scalar version
    auto start = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < N; ++i) {
        out_scalar[i] = a[i] * b[i] + c[i] + sqrtf(a[i]) + b[i] * c[i] + a[i] / c[i];
    }
    auto end = std::chrono::high_resolution_clock::now();
    auto scalar_time = std::chrono::duration<double, std::micro>(end - start).count();
    
    // NEON version
    start = std::chrono::high_resolution_clock::now();
    size_t i = 0;
    for (; i + 4 <= N; i += 4) {
        float32x4_t va = vld1q_f32(a + i);
        float32x4_t vb = vld1q_f32(b + i);
        float32x4_t vc = vld1q_f32(c + i);
        
        float32x4_t vab = vmulq_f32(va, vb);      // a*b
        float32x4_t result = vaddq_f32(vab, vc);  // a*b + c
        
        float32x4_t vsqrt_a = vsqrtq_f32(va);     // sqrt(a)
        result = vaddq_f32(result, vsqrt_a);      // + sqrt(a)
        
        float32x4_t vbc = vmulq_f32(vb, vc);      // b*c
        result = vaddq_f32(result, vbc);          // + b*c
        
        float32x4_t va_div_c = vdivq_f32(va, vc); // a/c
        result = vaddq_f32(result, va_div_c);     // + a/c
        
        vst1q_f32(out_neon + i, result);
    }
    
    for (; i < N; ++i) {
        out_neon[i] = a[i] * b[i] + c[i] + sqrtf(a[i]) + b[i] * c[i] + a[i] / c[i];
    }
    end = std::chrono::high_resolution_clock::now();
    auto neon_time = std::chrono::duration<double, std::micro>(end - start).count();
    
    double speedup = scalar_time / neon_time;
    std::cout << "Scalar time: " << scalar_time << " μs\n";
    std::cout << "NEON time: " << neon_time << " μs\n";
    std::cout << "Speedup: " << speedup << "×\n";
    
    if (speedup >= 2.0) {
        std::cout << "🎉 TARGET ACHIEVED! " << speedup << "× >= 2.0×\n";
    } else {
        std::cout << "📈 Progress: " << (speedup/2.0)*100 << "% of target\n";
    }
    
    free(a); free(b); free(c); free(out_scalar); free(out_neon);
    return 0;
} 