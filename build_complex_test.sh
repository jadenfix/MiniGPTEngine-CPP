#!/bin/bash

echo "🚀 Building Complex Mathematical Apple Silicon M2 Test"
echo "====================================================="

# Apple Silicon M2 specific flags with aggressive optimizations
APPLE_M2_FLAGS="-O3 -mcpu=apple-m2 -ffast-math"
VECTORIZATION_FLAGS="-Rpass=loop-vectorize -Rpass-missed=loop-vectorize"
AGGRESSIVE_FLAGS="-march=native -flto -fno-strict-aliasing -funroll-loops"
MATH_FLAGS="-fno-math-errno -funsafe-math-optimizations"

echo "🔧 Optimizations:"
echo "   M2 Specific: $APPLE_M2_FLAGS"
echo "   Vectorization: $VECTORIZATION_FLAGS"
echo "   Aggressive: $AGGRESSIVE_FLAGS"
echo "   Math: $MATH_FLAGS"
echo ""

# Compile complex mathematical test
clang++ $APPLE_M2_FLAGS $VECTORIZATION_FLAGS $AGGRESSIVE_FLAGS $MATH_FLAGS \
    -o complex_test complex_test.cpp

if [ $? -eq 0 ]; then
    echo "✅ Build successful!"
    echo "🏃 Running complex mathematical test..."
    echo ""
    ./complex_test
else
    echo "❌ Build failed"
    exit 1
fi 