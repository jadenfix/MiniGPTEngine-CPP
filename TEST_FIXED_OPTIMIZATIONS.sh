#!/bin/bash

echo "🔧 Testing FIXED OPTIMIZATIONS based on analysis"
echo "================================================="
echo "Targeting specific issues:"
echo "1. SIMD blocking too small -> 128x128 tiles"
echo "2. Memory pool contention -> Thread-local pools"
echo "3. Quantization overhead -> Hierarchical approach"
echo "4. Measurement errors -> Fixed timing"
echo ""

# Build with proper Apple Silicon flags
echo "Building with Apple Silicon optimizations..."
clang++ -std=c++20 -O3 -march=native -mcpu=apple-m2 \
        -ffast-math -funroll-loops -ftree-vectorize \
        -DNDEBUG -flto \
        FIXED_OPTIMIZATIONS.cpp \
        -o fixed_optimizations \
        -framework Accelerate

if [ $? -ne 0 ]; then
    echo "❌ Build failed!"
    exit 1
fi

echo "✅ Build successful!"
echo ""

# Run the fixed optimizations
echo "Running fixed optimizations test..."
echo "==================================="
./fixed_optimizations

exit_code=$?

echo ""
if [ $exit_code -eq 0 ]; then
    echo "🏆 FIXED OPTIMIZATIONS TEST COMPLETED"
    echo "Check results above to see which fixes worked"
else
    echo "❌ Some fixes still need work"
    echo "Review output above for specific issues"
fi

exit $exit_code 