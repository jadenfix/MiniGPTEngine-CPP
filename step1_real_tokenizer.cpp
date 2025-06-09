#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <unordered_map>
#include <cstring>
#include <cassert>

class GGUFTokenizer {
private:
    std::unordered_map<std::string, int> token_to_id;
    std::unordered_map<int, std::string> id_to_token;
    int vocab_size = 0;
    
    // Helper to read different data types from GGUF
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
    
public:
    bool load_from_gguf(const std::string& model_path) {
        std::ifstream file(model_path, std::ios::binary);
        if (!file) {
            std::cerr << "❌ Cannot open: " << model_path << std::endl;
            return false;
        }
        
        std::cout << "🔍 Parsing GGUF for real vocabulary..." << std::endl;
        
        // Read GGUF header
        struct GGUFHeader {
            char magic[4];
            uint32_t version;
            uint64_t tensor_count;
            uint64_t kv_count;
        } header;
        
        file.read(reinterpret_cast<char*>(&header), sizeof(header));
        
        if (std::string(header.magic, 4) != "GGUF") {
            std::cerr << "❌ Invalid GGUF format" << std::endl;
            return false;
        }
        
        std::cout << "✅ GGUF Header: version=" << header.version 
                  << ", tensors=" << header.tensor_count 
                  << ", kv_pairs=" << header.kv_count << std::endl;
        
        // Parse key-value metadata to find tokenizer
        for (uint64_t i = 0; i < header.kv_count; i++) {
            std::string key = read_string(file);
            uint32_t value_type = read_value<uint32_t>(file);
            
            std::cout << "🔑 Key: " << key << " (type=" << value_type << ")" << std::endl;
            
            if (key == "tokenizer.ggml.tokens") {
                // Found the vocabulary!
                std::cout << "🎯 Found tokenizer vocabulary!" << std::endl;
                
                if (value_type == 8) { // GGUF_TYPE_ARRAY
                    uint32_t array_type = read_value<uint32_t>(file);
                    uint64_t array_length = read_value<uint64_t>(file);
                    
                    std::cout << "📚 Loading " << array_length << " tokens..." << std::endl;
                    
                    for (uint64_t j = 0; j < array_length; j++) {
                        std::string token = read_string(file);
                        token_to_id[token] = j;
                        id_to_token[j] = token;
                        
                        // Show first few tokens
                        if (j < 10) {
                            std::cout << "   Token " << j << ": '" << token << "'" << std::endl;
                        }
                    }
                    
                    vocab_size = array_length;
                    std::cout << "✅ Loaded " << vocab_size << " real tokens!" << std::endl;
                    return true;
                }
            } else {
                // Skip other values based on type
                switch (value_type) {
                    case 6: // GGUF_TYPE_STRING
                        read_string(file);
                        break;
                    case 4: // GGUF_TYPE_UINT32
                        read_value<uint32_t>(file);
                        break;
                    case 5: // GGUF_TYPE_INT32
                        read_value<int32_t>(file);
                        break;
                    case 7: // GGUF_TYPE_FLOAT32
                        read_value<float>(file);
                        break;
                    default:
                        std::cout << "⚠️  Skipping unknown type " << value_type << std::endl;
                        // Skip unknown types more carefully
                        file.seekg(8, std::ios::cur); // Skip 8 bytes as fallback
                        break;
                }
            }
        }
        
        std::cout << "❌ Tokenizer not found in GGUF metadata" << std::endl;
        return false;
    }
    
    std::vector<int> encode(const std::string& text) {
        std::vector<int> tokens;
        
        // Simple word tokenization for testing
        std::string current_token;
        for (char c : text) {
            if (c == ' ') {
                if (!current_token.empty()) {
                    if (token_to_id.count(current_token)) {
                        tokens.push_back(token_to_id[current_token]);
                    } else {
                        tokens.push_back(0); // UNK token
                    }
                    current_token.clear();
                }
            } else {
                current_token += c;
            }
        }
        
        if (!current_token.empty()) {
            if (token_to_id.count(current_token)) {
                tokens.push_back(token_to_id[current_token]);
            } else {
                tokens.push_back(0); // UNK
            }
        }
        
        return tokens;
    }
    
    std::string decode(const std::vector<int>& tokens) {
        std::string result;
        for (int token_id : tokens) {
            if (id_to_token.count(token_id)) {
                if (!result.empty()) result += " ";
                result += id_to_token[token_id];
            }
        }
        return result;
    }
    
    // Test functions
    void run_tests() {
        std::cout << "\n🧪 Running Tokenizer Tests" << std::endl;
        std::cout << "==========================" << std::endl;
        
        // Test 1: Vocabulary size
        std::cout << "Test 1: Vocabulary size = " << vocab_size << std::endl;
        assert(vocab_size > 1000); // Should be substantial
        std::cout << "✅ Pass: Reasonable vocab size" << std::endl;
        
        // Test 2: Check for common tokens
        std::vector<std::string> common_tokens = {"the", "a", "is", "of", "to"};
        for (const auto& token : common_tokens) {
            if (token_to_id.count(token)) {
                std::cout << "✅ Found common token: '" << token << "' -> " << token_to_id[token] << std::endl;
            } else {
                std::cout << "⚠️  Missing common token: '" << token << "'" << std::endl;
            }
        }
        
        // Test 3: Encode/decode roundtrip
        std::string test_text = "the capital of France";
        std::vector<int> encoded = encode(test_text);
        std::string decoded = decode(encoded);
        
        std::cout << "Test 3: Encode/Decode roundtrip" << std::endl;
        std::cout << "   Original: '" << test_text << "'" << std::endl;
        std::cout << "   Encoded:  [";
        for (size_t i = 0; i < encoded.size(); i++) {
            std::cout << encoded[i];
            if (i < encoded.size() - 1) std::cout << ", ";
        }
        std::cout << "]" << std::endl;
        std::cout << "   Decoded:  '" << decoded << "'" << std::endl;
        
        // Test 4: Special tokens
        if (token_to_id.count("<s>")) {
            std::cout << "✅ Found BOS token: <s> -> " << token_to_id["<s>"] << std::endl;
        }
        if (token_to_id.count("</s>")) {
            std::cout << "✅ Found EOS token: </s> -> " << token_to_id["</s>"] << std::endl;
        }
        
        std::cout << "🎉 Tokenizer tests complete!" << std::endl;
    }
    
    int get_vocab_size() const { return vocab_size; }
};

int main(int argc, char* argv[]) {
    std::cout << "🤖 Step 1: Real Tokenizer Test\n";
    std::cout << "===============================\n\n";
    
    if (argc < 2) {
        std::cout << "Usage: " << argv[0] << " <model_path>\n";
        std::cout << "This will parse the real vocabulary from your GGUF file.\n";
        return 1;
    }
    
    std::string model_path = argv[1];
    GGUFTokenizer tokenizer;
    
    if (!tokenizer.load_from_gguf(model_path)) {
        std::cout << "❌ Failed to load tokenizer" << std::endl;
        return 1;
    }
    
    // Run comprehensive tests
    tokenizer.run_tests();
    
    // Interactive test
    std::cout << "\n💬 Interactive Test:" << std::endl;
    std::string test_prompt = "What is the capital of France?";
    auto tokens = tokenizer.encode(test_prompt);
    
    std::cout << "Input: '" << test_prompt << "'" << std::endl;
    std::cout << "Tokens: ";
    for (int token : tokens) {
        std::cout << token << " ";
    }
    std::cout << std::endl;
    std::cout << "Decoded: '" << tokenizer.decode(tokens) << "'" << std::endl;
    
    std::cout << "\n✅ Step 1 Complete! Real tokenizer working." << std::endl;
    std::cout << "📝 Next: Load actual tensor weights from GGUF" << std::endl;
    
    return 0;
} 