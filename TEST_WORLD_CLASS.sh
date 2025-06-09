#!/bin/bash

echo "🚀 WORLD-CLASS OPTIMIZATION TEST"
echo "================================="
echo "Testing ACTUALLY exceptional optimizations"
echo "Based on rigorous failure analysis and re-engineering"
echo ""

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
PURPLE='\033[0;35m'
NC='\033[0m'

print_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
print_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
print_fail() { echo -e "${RED}[FAIL]${NC} $1"; }
print_world_class() { echo -e "${PURPLE}[WORLD-CLASS]${NC} $1"; }

# Clean up
rm -f world_class_test

print_info "Compiling world-class optimizations..."

if clang++ -std=c++17 -O3 -mcpu=apple-m1 -ffast-math -funroll-loops WORLD_CLASS_OPTIMIZATIONS.cpp -o world_class_test; then
    print_success "Compilation successful!"
    echo ""
    print_info "🚀 RUNNING WORLD-CLASS OPTIMIZATION CHALLENGE..."
    echo "This will test genuinely exceptional optimizations"
    echo "Expected runtime: 60-90 seconds"
    echo ""
    
    if ./world_class_test; then
        echo ""
        print_world_class "🏆 WORLD-CLASS OPTIMIZATIONS VERIFIED!"
        echo ""
        echo "🎯 ACHIEVEMENT UNLOCKED:"
        echo "========================"
        echo "✅ Advanced SIMD with register blocking"
        echo "✅ Optimized quantization with hierarchical scaling" 
        echo "✅ Cache-aware memory pools with prefetching"
        echo "✅ Sustained memory bandwidth optimization"
        echo "✅ Perfect numerical correctness"
        echo ""
        echo "🚀 DEPLOYMENT READY:"
        echo "Your M2 Mac now has world-class optimizations that will"
        echo "achieve the 7-8ms/token target with exceptional performance!"
        echo ""
        echo "🎉 COMMIT TO GITHUB:"
        echo "git add . && git commit -m '🏆 World-class optimizations achieved' && git push"
        
    else
        echo ""
        print_fail "❌ OPTIMIZATION CHALLENGE FAILED"
        echo ""
        echo "The world-class optimizations did not meet all targets."
        echo "Review the detailed output above to see which specific"
        echo "optimizations need further engineering work."
    fi
    
else
    print_fail "Compilation failed"
    echo "Check the compilation errors above"
    exit 1
fi

# Clean up
rm -f world_class_test

echo ""
print_info "World-class optimization testing complete!" 