#include <iostream>
#include <chrono>
#include <arm_neon.h>
#include <cstdlib>

int main() {
    const size_t N = 8*1024*1024;
    
    // Aligned memory allocation for Apple Silicon
    void* ptr_a = nullptr;
    void* ptr_b = nullptr;
    void* ptr_out_scalar = nullptr;
    void* ptr_out_neon = nullptr;
    
    posix_memalign(&ptr_a, 64, N * sizeof(float));
    posix_memalign(&ptr_b, 64, N * sizeof(float));
    posix_memalign(&ptr_out_scalar, 64, N * sizeof(float));
    posix_memalign(&ptr_out_neon, 64, N * sizeof(float));
    
    float* a = static_cast<float*>(ptr_a);
    float* b = static_cast<float*>(ptr_b);
    float* out_scalar = static_cast<float*>(ptr_out_scalar);
    float* out_neon = static_cast<float*>(ptr_out_neon);
    
    // Initialize data
    for (size_t i = 0; i < N; ++i) {
        a[i] = static_cast<float>(i) * 0.5f;
        b[i] = static_cast<float>(i) * 0.3f + 1.0f;
    }
    
    std::cout << "🚀 Apple Silicon M2 Vector Test (N=" << N << ")\n";
    
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
    
    double speedup = scalar_time / neon_time;
    std::cout << "Scalar time: " << scalar_time << " μs\n";
    std::cout << "NEON time: " << neon_time << " μs\n";
    std::cout << "Speedup: " << speedup << "×\n";
    std::cout << "🎯 Target: 2.0× speedup\n";
    
    free(a);
    free(b);
    free(out_scalar);
    free(out_neon);
    
    return 0;
} 