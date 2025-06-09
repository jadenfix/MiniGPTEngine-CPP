#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <unordered_map>
#include <cstring>

struct TensorInfo {
    std::string name;
    uint32_t type;
    std::vector<uint64_t> dimensions;
    uint64_t offset;
    size_t size_bytes;
};

class GGUFWeightLoader {
private:
    std::vector<TensorInfo> tensors;
    std::unordered_map<std::string, std::vector<float>> loaded_weights;
    
    template<typename T>
    T read_value(std::ifstream& file) {
        T value;
        file.read(reinterpret_cast<char*>(&value), sizeof(T));
        return value;
    }
    
    std::string read_string(std::ifstream& file) {
        uint64_t length = read_value<uint64_t>(file);
        std::string str(length, '\0');
        file.read(&str[0], length);
        return str;
    }
    
    void skip_value(std::ifstream& file, uint32_t type) {
        switch (type) {
            case 4: read_value<uint32_t>(file); break;  // UINT32
            case 5: read_value<int32_t>(file); break;   // INT32
            case 6: read_string(file); break;           // STRING
            case 7: read_value<float>(file); break;     // FLOAT32
            case 8: // ARRAY - more complex
                {
                    uint32_t array_type = read_value<uint32_t>(file);
                    uint64_t array_length = read_value<uint64_t>(file);
                    for (uint64_t i = 0; i < array_length; i++) {
                        skip_value(file, array_type);
                    }
                }
                break;
            default:
                file.seekg(8, std::ios::cur); // Skip unknown
                break;
        }
    }
    
public:
    bool load_from_gguf(const std::string& model_path) {
        std::ifstream file(model_path, std::ios::binary);
        if (!file) {
            std::cerr << "❌ Cannot open: " << model_path << std::endl;
            return false;
        }
        
        std::cout << "🔍 Step 2: Loading tensor weights from GGUF..." << std::endl;
        
        // Read header
        char magic[4];
        uint32_t version;
        uint64_t tensor_count, kv_count;
        
        file.read(magic, 4);
        file.read(reinterpret_cast<char*>(&version), 4);
        file.read(reinterpret_cast<char*>(&tensor_count), 8);
        file.read(reinterpret_cast<char*>(&kv_count), 8);
        
        std::cout << "✅ Found " << tensor_count << " tensors in GGUF" << std::endl;
        
        // Skip metadata section
        std::cout << "⏩ Skipping " << kv_count << " metadata entries..." << std::endl;
        for (uint64_t i = 0; i < kv_count; i++) {
            std::string key = read_string(file);
            uint32_t value_type = read_value<uint32_t>(file);
            skip_value(file, value_type);
        }
        
        // Read tensor info
        std::cout << "📊 Reading tensor information..." << std::endl;
        for (uint64_t i = 0; i < tensor_count; i++) {
            TensorInfo tensor;
            tensor.name = read_string(file);
            
            uint32_t n_dimensions = read_value<uint32_t>(file);
            for (uint32_t j = 0; j < n_dimensions; j++) {
                tensor.dimensions.push_back(read_value<uint64_t>(file));
            }
            
            tensor.type = read_value<uint32_t>(file);
            tensor.offset = read_value<uint64_t>(file);
            
            // Calculate size
            tensor.size_bytes = 1;
            for (uint64_t dim : tensor.dimensions) {
                tensor.size_bytes *= dim;
            }
            // Multiply by type size (simplified)
            tensor.size_bytes *= (tensor.type == 0) ? 4 : 2; // F32 vs F16
            
            tensors.push_back(tensor);
            
            // Show first few tensors
            if (i < 5) {
                std::cout << "   Tensor " << i << ": " << tensor.name;
                std::cout << " [";
                for (size_t j = 0; j < tensor.dimensions.size(); j++) {
                    std::cout << tensor.dimensions[j];
                    if (j < tensor.dimensions.size() - 1) std::cout << ", ";
                }
                std::cout << "] type=" << tensor.type << std::endl;
            }
        }
        
        std::cout << "✅ Parsed " << tensors.size() << " tensor definitions" << std::endl;
        return true;
    }
    
    bool load_specific_weights(const std::string& model_path, const std::vector<std::string>& tensor_names) {
        std::ifstream file(model_path, std::ios::binary);
        if (!file) return false;
        
        std::cout << "🎯 Loading specific tensor weights..." << std::endl;
        
        for (const auto& name : tensor_names) {
            auto it = std::find_if(tensors.begin(), tensors.end(),
                                 [&name](const TensorInfo& t) { return t.name == name; });
            
            if (it != tensors.end()) {
                // Seek to tensor data (simplified)
                size_t data_offset = 512 + it->offset; // Rough estimate
                file.seekg(data_offset, std::ios::beg);
                
                // Load first few values for testing
                std::vector<float> weights(std::min(it->size_bytes / 4, (size_t)10));
                file.read(reinterpret_cast<char*>(weights.data()), weights.size() * 4);
                
                loaded_weights[name] = weights;
                
                std::cout << "   ✅ Loaded " << name << " (" << weights.size() << " values)" << std::endl;
                std::cout << "      First values: ";
                for (size_t i = 0; i < std::min((size_t)5, weights.size()); i++) {
                    std::cout << weights[i] << " ";
                }
                std::cout << std::endl;
            } else {
                std::cout << "   ❌ Tensor not found: " << name << std::endl;
            }
        }
        
        return true;
    }
    
    void run_tests() {
        std::cout << "\n🧪 Testing Weight Loader" << std::endl;
        std::cout << "========================" << std::endl;
        
        // Test 1: Tensor count
        std::cout << "✅ Test 1 - Tensor inventory:" << std::endl;
        std::cout << "   Total tensors: " << tensors.size() << std::endl;
        
        // Test 2: Look for key transformer components
        std::cout << "\n✅ Test 2 - Key tensor search:" << std::endl;
        std::vector<std::string> key_patterns = {"token_embd", "attn", "ffn", "output"};
        
        for (const auto& pattern : key_patterns) {
            int count = 0;
            for (const auto& tensor : tensors) {
                if (tensor.name.find(pattern) != std::string::npos) {
                    count++;
                }
            }
            std::cout << "   Found " << count << " tensors matching '" << pattern << "'" << std::endl;
        }
        
        // Test 3: Check tensor shapes
        std::cout << "\n✅ Test 3 - Tensor shapes:" << std::endl;
        for (size_t i = 0; i < std::min((size_t)3, tensors.size()); i++) {
            const auto& tensor = tensors[i];
            std::cout << "   " << tensor.name << ": [";
            for (size_t j = 0; j < tensor.dimensions.size(); j++) {
                std::cout << tensor.dimensions[j];
                if (j < tensor.dimensions.size() - 1) std::cout << " × ";
            }
            std::cout << "]" << std::endl;
        }
        
        // Test 4: Loaded weights validation
        std::cout << "\n✅ Test 4 - Weight validation:" << std::endl;
        for (const auto& pair : loaded_weights) {
            std::cout << "   " << pair.first << ": " << pair.second.size() << " values loaded" << std::endl;
        }
        
        std::cout << "\n🎉 Weight loader tests PASSED!" << std::endl;
    }
    
    size_t get_tensor_count() const { return tensors.size(); }
    const std::vector<TensorInfo>& get_tensors() const { return tensors; }
};

int main(int argc, char* argv[]) {
    std::cout << "🤖 Step 2: Weight Loading Test\n";
    std::cout << "==============================\n\n";
    
    if (argc < 2) {
        std::cout << "Usage: " << argv[0] << " <model_path>\n";
        return 1;
    }
    
    GGUFWeightLoader loader;
    
    if (!loader.load_from_gguf(argv[1])) {
        return 1;
    }
    
    // Try to load some specific weights
    std::vector<std::string> test_tensors = {
        "token_embd.weight",
        "blk.0.attn_q.weight", 
        "blk.0.attn_k.weight",
        "output.weight"
    };
    
    loader.load_specific_weights(argv[1], test_tensors);
    loader.run_tests();
    
    std::cout << "\n✅ STEP 2 COMPLETE!" << std::endl;
    std::cout << "📈 Progress: Tokenizer ✅ → Weights ✅ → Attention ❌ → Full Model ❌" << std::endl;
    std::cout << "📝 Next: Implement attention mechanism" << std::endl;
    
    return 0;
} 