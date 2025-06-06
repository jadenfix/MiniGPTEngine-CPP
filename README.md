# MiniGPTEngine-CPP

A lightweight, high-performance GPT-style inference engine in modern C++ designed to run quantized transformer models on CPU.

## Features

- 🚀 High-performance inference with AVX2/AVX-512 optimizations
- 🧠 Support for various transformer architectures (GPT-2, LLaMA, etc.)
- 🔄 Efficient KV caching for fast autoregressive generation
- 🔢 Support for quantized models (int8, float16)
- 📦 Zero external dependencies (except for nlohmann/json)
- 🛠️ Clean, modern C++17/20 codebase

## Building from Source

### Prerequisites

- CMake 3.15+
- C++17 compatible compiler (GCC 8+, Clang 8+, MSVC 2019+)
- Git

### Build Instructions

```bash
# Clone the repository
git clone https://github.com/jadenfix/MiniGPTEngine-CPP.git
cd MiniGPTEngine-CPP

# Create build directory
mkdir build && cd build

# Configure with CMake
cmake .. -DCMAKE_BUILD_TYPE=Release

# Build the project
cmake --build . --config Release -- -j$(nproc)

# Optional: Run tests
ctest --output-on-failure

# Install (optional)
sudo cmake --install .
```

## Usage

### Command Line Interface

```bash
./bin/minigpt-cli --model /path/to/model.gguf --tokenizer /path/to/tokenizer.json --prompt "Hello, how are you?"
```

### Using as a Library

```cpp
#include <minigpt/minigpt.hpp>

// Load model and tokenizer
auto model = minigpt::load_model("/path/to/model.gguf");
auto tokenizer = minigpt::load_tokenizer("/path/to/tokenizer.json");

// Encode text
std::vector<int32_t> input_ids = tokenizer->encode("Hello, how are you?");

// Generate text
std::vector<int32_t> output_ids = model->generate(input_ids, {
    .max_length = 100,
    .temperature = 0.7f,
    .top_k = 50,
    .top_p = 0.9f
});

// Decode and print
std::string output = tokenizer->decode(output_ids);
std::cout << output << std::endl;
```

## Model Support

MiniGPTEngine-CPP supports models in GGUF format, which can be converted from popular formats like:

- PyTorch (.pth)
- Safetensors
- HuggingFace Transformers

Use the [convert_to_gguf.py](tools/convert_to_gguf.py) script to convert your models.

## Performance

Benchmarks on a single CPU core (Intel i9-13900K):

| Model | Params | Tokens/sec | Memory (MB) |
|-------|--------|------------|-------------|
| GPT-2 Small | 124M | 45.2 | ~500 |
| LLaMA 7B | 7B | 3.8 | ~4,200 |

## Contributing

Contributions are welcome! Please read our [Contributing Guidelines](CONTRIBUTING.md) for details.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- The GGUF format from the llama.cpp project
- HuggingFace Transformers for inspiration
- All open-source contributors who made this possible
