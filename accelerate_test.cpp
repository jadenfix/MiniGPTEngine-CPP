#include <iostream>
#include <chrono>
#include <arm_neon.h>
#include <cstdlib>
#include <cmath>

#ifdef __APPLE__
#include <Accelerate/Accelerate.h>
#endif

int main() {
    const size_t N = 16*1024*1024;
    
    // Aligned memory allocation
    void* ptr_a = nullptr;
    void* ptr_b = nullptr;
    void* ptr_c = nullptr;
    void* ptr_out_manual = nullptr;
    void* ptr_out_accelerate = nullptr;
    
    posix_memalign(&ptr_a, 64, N * sizeof(float));
    posix_memalign(&ptr_b, 64, N * sizeof(float));
    posix_memalign(&ptr_c, 64, N * sizeof(float));
    posix_memalign(&ptr_out_manual, 64, N * sizeof(float));
    posix_memalign(&ptr_out_accelerate, 64, N * sizeof(float));
    
    float* a = static_cast<float*>(ptr_a);
    float* b = static_cast<float*>(ptr_b);
    float* c = static_cast<float*>(ptr_c);
    float* out_manual = static_cast<float*>(ptr_out_manual);
    float* out_accelerate = static_cast<float*>(ptr_out_accelerate);
    
    // Initialize data
    for (size_t i = 0; i < N; ++i) {
        a[i] = static_cast<float>(i % 1000) * 0.001f + 1.0f;
        b[i] = static_cast<float>((i * 17) % 1000) * 0.002f + 0.5f;
        c[i] = static_cast<float>((i * 31) % 1000) * 0.003f + 0.1f;
    }
    
    std::cout << "🚀 Apple Accelerate Framework vs Manual NEON (N=" << N << ")\n\n";
    
    // Manual NEON version with unrolling
    auto start = std::chrono::high_resolution_clock::now();
    size_t i = 0;
    for (; i + 8 <= N; i += 8) {
        // Process 8 elements at once (2 NEON vectors)
        float32x4_t va1 = vld1q_f32(a + i);
        float32x4_t vb1 = vld1q_f32(b + i);
        float32x4_t vc1 = vld1q_f32(c + i);
        
        float32x4_t va2 = vld1q_f32(a + i + 4);
        float32x4_t vb2 = vld1q_f32(b + i + 4);
        float32x4_t vc2 = vld1q_f32(c + i + 4);
        
        // Complex computation: a*b + c*sqrt(a)
        float32x4_t vab1 = vmulq_f32(va1, vb1);         // a*b
        float32x4_t vsqrt1 = vsqrtq_f32(va1);           // sqrt(a)
        float32x4_t vc_sqrt1 = vmulq_f32(vc1, vsqrt1);  // c*sqrt(a)
        float32x4_t result1 = vaddq_f32(vab1, vc_sqrt1); // a*b + c*sqrt(a)
        
        float32x4_t vab2 = vmulq_f32(va2, vb2);
        float32x4_t vsqrt2 = vsqrtq_f32(va2);
        float32x4_t vc_sqrt2 = vmulq_f32(vc2, vsqrt2);
        float32x4_t result2 = vaddq_f32(vab2, vc_sqrt2);
        
        vst1q_f32(out_manual + i, result1);
        vst1q_f32(out_manual + i + 4, result2);
    }
    
    // Handle remaining elements
    for (; i < N; ++i) {
        out_manual[i] = a[i] * b[i] + c[i] * sqrtf(a[i]);
    }
    auto end = std::chrono::high_resolution_clock::now();
    auto manual_time = std::chrono::duration<double, std::micro>(end - start).count();
    
#ifdef __APPLE__
    // Apple Accelerate version (hand-tuned for Apple Silicon)
    start = std::chrono::high_resolution_clock::now();
    
    // Temporary arrays for intermediate results
    float* temp_ab = static_cast<float*>(malloc(N * sizeof(float)));
    float* temp_sqrt = static_cast<float*>(malloc(N * sizeof(float)));
    float* temp_c_sqrt = static_cast<float*>(malloc(N * sizeof(float)));
    
    // Use vDSP for optimized operations
    vDSP_vmul(a, 1, b, 1, temp_ab, 1, N);              // a*b
    int n_int = static_cast<int>(N);
    vvsqrtf(temp_sqrt, a, &n_int);                      // sqrt(a) - vectorized sqrt
    vDSP_vmul(c, 1, temp_sqrt, 1, temp_c_sqrt, 1, N);  // c*sqrt(a)
    vDSP_vadd(temp_ab, 1, temp_c_sqrt, 1, out_accelerate, 1, N); // a*b + c*sqrt(a)
    
    free(temp_ab);
    free(temp_sqrt);
    free(temp_c_sqrt);
    
    end = std::chrono::high_resolution_clock::now();
    auto accelerate_time = std::chrono::duration<double, std::micro>(end - start).count();
    
    double speedup = manual_time / accelerate_time;
    
    std::cout << "Manual NEON time: " << manual_time << " μs\n";
    std::cout << "Accelerate time: " << accelerate_time << " μs\n";
    std::cout << "Accelerate speedup: " << speedup << "×\n";
    
    if (speedup >= 2.0) {
        std::cout << "🎉 TARGET ACHIEVED! " << speedup << "× >= 2.0×\n";
    } else {
        std::cout << "📈 Accelerate progress: " << (speedup/2.0)*100 << "% of target\n";
    }
#else
    std::cout << "❌ Accelerate framework not available (not on Apple platform)\n";
#endif
    
    // Test 2: Simple operations comparison
    std::cout << "\n🔬 Test 2: Simple vector addition comparison\n";
    
    // Manual NEON addition
    start = std::chrono::high_resolution_clock::now();
    i = 0;
    for (; i + 4 <= N; i += 4) {
        float32x4_t va = vld1q_f32(a + i);
        float32x4_t vb = vld1q_f32(b + i);
        float32x4_t result = vaddq_f32(va, vb);
        vst1q_f32(out_manual + i, result);
    }
    for (; i < N; ++i) {
        out_manual[i] = a[i] + b[i];
    }
    end = std::chrono::high_resolution_clock::now();
    auto neon_add_time = std::chrono::duration<double, std::micro>(end - start).count();
    
#ifdef __APPLE__
    // Accelerate addition
    start = std::chrono::high_resolution_clock::now();
    vDSP_vadd(a, 1, b, 1, out_accelerate, 1, N);
    end = std::chrono::high_resolution_clock::now();
    auto accelerate_add_time = std::chrono::duration<double, std::micro>(end - start).count();
    
    double add_speedup = neon_add_time / accelerate_add_time;
    
    std::cout << "Manual NEON addition: " << neon_add_time << " μs\n";
    std::cout << "Accelerate addition: " << accelerate_add_time << " μs\n";
    std::cout << "Addition speedup: " << add_speedup << "×\n";
    
    if (add_speedup >= 2.0) {
        std::cout << "🎉 ADDITION TARGET ACHIEVED! " << add_speedup << "× >= 2.0×\n";
    }
#endif
    
    free(a); free(b); free(c); free(out_manual); free(out_accelerate);
    return 0;
} 