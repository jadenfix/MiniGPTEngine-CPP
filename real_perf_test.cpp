#include <iostream>
#include <chrono>
#include <vector>
#include <random>
#include <arm_neon.h>

int main() {
    std::cout << "🚀 REAL-WORLD PERFORMANCE TEST\n";
    std::cout << "===============================\n\n";
    
    // MUCH larger test - realistic for LLM operations
    const size_t size = 1024 * 1024;  // 1M elements = 4MB
    std::vector<float> a(size), b(size), c_scalar(size), c_neon(size);
    
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    
    for (size_t i = 0; i < size; i++) {
        a[i] = dist(rng);
        b[i] = dist(rng);
    }
    
    // Scalar version
    auto start = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < size; i++) {
        c_scalar[i] = a[i] * b[i] + 0.5f;
    }
    auto scalar_time = std::chrono::duration<double, std::milli>(
        std::chrono::high_resolution_clock::now() - start).count();
    
    // NEON version  
    start = std::chrono::high_resolution_clock::now();
    const float32x4_t half = vdupq_n_f32(0.5f);
    for (size_t i = 0; i < size; i += 4) {
        float32x4_t va = vld1q_f32(&a[i]);
        float32x4_t vb = vld1q_f32(&b[i]);
        float32x4_t vc = vfmaq_f32(half, va, vb);
        vst1q_f32(&c_neon[i], vc);
    }
    auto neon_time = std::chrono::duration<double, std::milli>(
        std::chrono::high_resolution_clock::now() - start).count();
    
    float speedup = scalar_time / neon_time;
    float throughput = (size * 4 * 3) / (neon_time / 1000.0); // 3 arrays, bytes/sec
    
    std::cout << "📊 REAL-WORLD RESULTS (1M elements = 4MB):\n";
    std::cout << "   Scalar time:   " << scalar_time << " ms\n";
    std::cout << "   NEON time:     " << neon_time << " ms\n";
    std::cout << "   Speedup:       " << speedup << "x\n";
    std::cout << "   Throughput:    " << throughput / 1e9 << " GB/s\n\n";
    
    if (speedup > 2.0) {
        std::cout << "🏆 EXCELLENT! NEON delivering serious performance gains!\n";
    } else if (speedup > 1.5) {
        std::cout << "✅ GOOD! NEON providing solid performance improvement!\n";
    } else {
        std::cout << "⚠️  MARGINAL: Need larger operations for SIMD benefits\n";
    }
    
    return 0;
} 