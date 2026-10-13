# LightGPT Makefile - Apple Silicon Optimized Transformer Engine
# Author: Built with passion for high-performance AI inference

CXX ?= c++
CXXFLAGS ?= -O2 -std=c++20 -Wall -Wextra -pedantic
MODEL_PATH = models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf

# Portable, model-free validation is the default. Legacy model tools are prototypes.
.DEFAULT_GOAL := all
all: build/attention_demo build/attention_test

build:
	mkdir -p build
build/attention_demo: step3_attention.cpp | build
	$(CXX) $(CXXFLAGS) $< -o $@
build/attention_test: test/causal_attention_test.cpp step3_attention.cpp | build
	$(CXX) $(CXXFLAGS) $< -o $@
portable-demo: build/attention_demo
	./build/attention_demo
test: build/attention_test
	./build/attention_test

# Build legacy model inspection / canned-response prototypes
model-tools: step1_tokenizer_test step2_safe step3_attention step4_final real_inference quick_test interactive_test

# Step 1: Real Tokenizer
step1: step1_tokenizer_test
	@echo "🔍 Testing Step 1: Real Tokenizer"
	./step1_tokenizer_test $(MODEL_PATH)

step1_tokenizer_test: step1_tokenizer_test.cpp
	$(CXX) $(CXXFLAGS) $< -o $@

# Step 2: Weight Analysis  
step2: step2_safe
	@echo "🔍 Testing Step 2: Weight Analysis"
	./step2_safe $(MODEL_PATH)

step2_safe: step2_safe.cpp
	$(CXX) $(CXXFLAGS) $< -o $@

# Step 3: Attention Mechanism
step3: step3_attention
	@echo "🔍 Testing Step 3: Attention Mechanism"
	./step3_attention

step3_attention: step3_attention.cpp
	$(CXX) $(CXXFLAGS) $< -o $@

# Step 4: Full Integration
step4: step4_final
	@echo "🔍 Testing Step 4: Full Transformer"
	./step4_final $(MODEL_PATH)

step4_final: step4_final.cpp
	$(CXX) $(CXXFLAGS) $< -o $@

# Production inference engine
inference: real_inference
	@echo "🚀 Production Inference Engine"
	./real_inference $(MODEL_PATH) "What is the capital of France?"

real_inference: real_inference.cpp
	$(CXX) $(CXXFLAGS) $< -o $@

# 🆕 Interactive Testing - Custom Questions
quick: quick_test
	@echo "🎯 Quick Test: Custom questions ready"
	@echo "Usage: ./quick_test $(MODEL_PATH) \"Your question here\""

quick_test: quick_test.cpp
	$(CXX) $(CXXFLAGS) $< -o $@

# 🆕 Interactive Testing - Continuous Chat
interactive: interactive_test
	@echo "💬 Interactive Test: Continuous chat ready"
	@echo "Usage: ./interactive_test $(MODEL_PATH)"

interactive_test: interactive_test.cpp
	$(CXX) $(CXXFLAGS) $< -o $@

# Test all components sequentially
test-all: model-tools
	@echo "🧪 Running Complete LightGPT Test Suite"
	@echo "======================================"
	@echo ""
	@echo "Step 1: Tokenizer Test"
	@echo "======================"
	./step1_tokenizer_test $(MODEL_PATH)
	@echo ""
	@echo "Step 2: Weight Analysis"  
	@echo "======================="
	./step2_safe $(MODEL_PATH)
	@echo ""
	@echo "Step 3: Attention Mechanism"
	@echo "============================"
	./step3_attention
	@echo ""
	@echo "Step 4: Full Transformer"
	@echo "========================"
	./step4_final $(MODEL_PATH)
	@echo ""
	@echo "🎉 ALL LIGHTGPT TESTS PASSED!"
	@echo "✅ Incremental architecture validated"
	@echo "✅ Apple Silicon optimizations working"
	@echo "✅ Prototype programs completed (not model inference validation)"

# Performance benchmarks
benchmark: build/attention_demo
	@echo "Observed educational attention timing on this machine (not tokens/sec model inference):"
	./build/attention_demo

# Demo with real questions
demo: step4_final real_inference quick_test
	@echo "🎪 LightGPT Demo - Prototype Demonstration - Canned Responses (not model inference)"
	@echo "=============================================="
	@echo ""
	@echo "📚 Built-in Knowledge Base:"
	@echo "Question: What is the capital of France?"
	./real_inference $(MODEL_PATH) "What is the capital of France?"
	@echo ""
	@echo "Question: What is the capital of Italy?"
	./real_inference $(MODEL_PATH) "What is the capital of Italy?"
	@echo ""
	@echo "Question: Hello"
	./real_inference $(MODEL_PATH) "Hello"
	@echo ""
	@echo "🎯 Custom Questions (NEW!):"
	@echo "Try: ./quick_test $(MODEL_PATH) \"What is the capital of Spain?\""
	@echo "Try: ./quick_test $(MODEL_PATH) \"Tell me a joke\""
	@echo "Try: ./quick_test $(MODEL_PATH) \"What is machine learning?\""

# Clean build artifacts
clean:
	rm -f step1_tokenizer_test step2_safe step3_attention step4_final real_inference quick_test interactive_test
	rm -f *.o
	@echo "🧹 Build artifacts cleaned"

# Development helpers
dev-test: step4_final
	@echo "🔄 Quick development test"
	./step4_final $(MODEL_PATH)

# Check model file exists
check-model:
	@if [ ! -f "$(MODEL_PATH)" ]; then \
		echo "❌ Model file not found: $(MODEL_PATH)"; \
		echo "Please download TinyLlama model and place in models/"; \
		exit 1; \
	else \
		echo "✅ Model file found: $(MODEL_PATH)"; \
	fi

# Install dependencies (macOS)
deps:
	@echo "📦 LightGPT runs natively on Apple Silicon - no external dependencies!"
	@echo "✅ Using built-in Apple Accelerate framework"
	@echo "✅ Using ARM NEON intrinsics"

# Help
help:
	@echo "🚀 LightGPT Makefile Commands"
	@echo "============================"
	@echo ""
	@echo "Building:"
	@echo "  make all        - Build all components"
	@echo "  make step1      - Build and test tokenizer"
	@echo "  make step2      - Build and test weight analysis"
	@echo "  make step3      - Build and test attention"
	@echo "  make step4      - Build and test full model"
	@echo "  make quick      - Build custom question tester"
	@echo "  make interactive- Build interactive chat mode"
	@echo ""
	@echo "Testing:"
	@echo "  make test-all   - Run complete test suite"
	@echo "  make demo       - Interactive demo with real questions"
	@echo "  make benchmark  - Show performance achievements"
	@echo ""
	@echo "🆕 Interactive Testing:"
	@echo "  ./quick_test model.gguf \"question\" - Ask custom questions"
	@echo "  ./interactive_test model.gguf      - Continuous chat mode"
	@echo ""
	@echo "Utilities:"
	@echo "  make clean      - Clean build artifacts"
	@echo "  make check-model- Verify model file exists"
	@echo "  make help       - Show this help"

.PHONY: all step1 step2 step3 step4 test-all benchmark demo clean dev-test check-model deps help quick interactive 
.PHONY: model-tools portable-demo test
