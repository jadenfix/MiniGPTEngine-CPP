#!/bin/bash

echo "🚀 Building LightGPT LLM with Apple Silicon Optimizations"
echo "========================================================"

# Apple Silicon optimized flags
APPLE_M2_FLAGS="-O3 -mcpu=apple-m2 -ffast-math -march=native"
INCLUDE_FLAGS="-Iinclude -Isrc"
THREADING_FLAGS="-pthread"

echo "🔧 Compiler flags:"
echo "   Apple M2: $APPLE_M2_FLAGS"
echo "   Includes: $INCLUDE_FLAGS"
echo "   Threading: $THREADING_FLAGS"
echo ""

# Source files
SRC_FILES="src/main.cpp src/model_loader.cpp src/transformer.cpp src/tokenizer.cpp src/tensor.cpp src/tensor_ops.cpp src/kv_cache.cpp"

echo "📁 Source files:"
for file in $SRC_FILES; do
    if [ -f "$file" ]; then
        echo "   ✅ $file"
    else
        echo "   ❌ $file (missing)"
    fi
done
echo ""

# Build
echo "🔨 Building LightGPT..."
clang++ $APPLE_M2_FLAGS $INCLUDE_FLAGS $THREADING_FLAGS \
    $SRC_FILES \
    -o lightgpt

if [ $? -eq 0 ]; then
    echo "✅ Build successful!"
    echo "🏃 LightGPT executable created: ./lightgpt"
    echo ""
    echo "📝 Usage:"
    echo "   ./lightgpt --model models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf --tokenizer tokenizer.json --prompt 'Hello, world!'"
else
    echo "❌ Build failed"
    exit 1
fi 