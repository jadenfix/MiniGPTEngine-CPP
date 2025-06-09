#!/bin/bash

echo "🚀 Building Apple Accelerate Framework Test"
echo "==========================================="

# Apple Silicon M2 flags with Accelerate framework
APPLE_M2_FLAGS="-O3 -mcpu=apple-m2 -ffast-math"
VECTORIZATION_FLAGS="-Rpass=loop-vectorize -Rpass-missed=loop-vectorize"
AGGRESSIVE_FLAGS="-march=native -flto -fno-strict-aliasing -funroll-loops"
ACCELERATE_FLAGS="-framework Accelerate"

echo "🔧 Optimizations + Accelerate Framework:"
echo "   M2 Specific: $APPLE_M2_FLAGS"
echo "   Vectorization: $VECTORIZATION_FLAGS"
echo "   Aggressive: $AGGRESSIVE_FLAGS"
echo "   Accelerate: $ACCELERATE_FLAGS"
echo ""

# Compile with Accelerate framework
clang++ $APPLE_M2_FLAGS $VECTORIZATION_FLAGS $AGGRESSIVE_FLAGS $ACCELERATE_FLAGS \
    -o accelerate_test accelerate_test.cpp

if [ $? -eq 0 ]; then
    echo "✅ Build successful!"
    echo "🏃 Running Accelerate framework test..."
    echo ""
    ./accelerate_test
else
    echo "❌ Build failed"
    exit 1
fi 