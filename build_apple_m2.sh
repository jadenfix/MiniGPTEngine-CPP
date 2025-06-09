#!/bin/bash

echo "🚀 Building Apple Silicon M2 Optimized Vector Test"
echo "================================================="

# Apple Silicon M2 specific compiler flags
APPLE_M2_FLAGS="-O3 -mcpu=apple-m2 -ffast-math"

# Vectorization diagnostics
VECTORIZATION_FLAGS="-Rpass=loop-vectorize -Rpass-missed=loop-vectorize"

# Additional optimization flags
OPTIMIZATION_FLAGS="-march=native -flto -fno-strict-aliasing -funroll-loops"

echo "🔧 Compiler flags:"
echo "   Core: $APPLE_M2_FLAGS"
echo "   Vectorization: $VECTORIZATION_FLAGS"
echo "   Additional: $OPTIMIZATION_FLAGS"
echo ""

# Compile with all Apple Silicon optimizations
clang++ $APPLE_M2_FLAGS $VECTORIZATION_FLAGS $OPTIMIZATION_FLAGS \
    -o apple_m2_test apple_m2_test.cpp

if [ $? -eq 0 ]; then
    echo "✅ Build successful!"
    echo "🏃 Running test..."
    echo ""
    ./apple_m2_test
else
    echo "❌ Build failed"
    exit 1
fi 