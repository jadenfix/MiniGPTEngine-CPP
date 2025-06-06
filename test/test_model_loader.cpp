#include <gtest/gtest.h>
#include <lightgpt/model_loader.hpp>
#include <filesystem>
#include <fstream>
#include <cstring>

using namespace lightgpt;

// Simple binary writer helper class
class BinaryWriter {
    std::ofstream ofs_;
    
public:
    BinaryWriter(const std::string& filename) : ofs_(filename, std::ios::binary) {}
    
    // Write a single value of type T
    template<typename T>
    void write(const T& value) {
        ofs_.write(reinterpret_cast<const char*>(&value), sizeof(T));
    }
    
    // Write raw data with size
    void write(const void* data, size_t size) {
        ofs_.write(static_cast<const char*>(data), size);
    }
    
    void write_string(const std::string& str) {
        uint64_t len = str.size();
        write(len);
        ofs_.write(str.data(), len);
        // Align to 8 bytes
        size_t padding = (8 - (len % 8)) % 8;
        for (size_t i = 0; i < padding; ++i) {
            ofs_.put(0);
        }
    }
    
    void write_zeros(size_t count) {
        std::vector<char> zeros(count, 0);
        write(zeros.data(), count);
    }
    
    size_t tellp() { return ofs_.tellp(); }
    void seekp(size_t pos) { ofs_.seekp(pos); }
};

class ModelLoaderTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Create a temporary test model file
        test_model_path_ = "test_model.gguf";
        
        // Create a simple GGUF file for testing
        BinaryWriter writer(test_model_path_);
        
        // Write GGUF header
        writer.write("GGUF", 4);  // Magic
        writer.write<uint32_t>(1);  // Version
        writer.write<uint64_t>(0);  // n_tensors (simplified - no tensors for now)
        writer.write<uint64_t>(2);  // n_kv (now 2 KV pairs)
        
        // Write key-value pairs
        // Each KV pair is: key_len (uint64_t), key (char[key_len]), value_type (uint32_t), value
        
        // general.architecture (string)
        std::string arch_key = "general.architecture";
        std::string arch_value = "llama";
        writer.write<uint64_t>(arch_key.size());
        writer.write(arch_key.c_str(), arch_key.size());
        // Pad to 8-byte alignment after key
        size_t key_padding = (8 - (arch_key.size() % 8)) % 8;
        writer.write_zeros(key_padding);
        writer.write<uint32_t>(8);  // GGUF_TYPE_STRING
        writer.write<uint64_t>(arch_value.size());
        writer.write(arch_value.c_str(), arch_value.size());
        // Pad to 8-byte alignment after value
        size_t value_padding = (8 - (arch_value.size() % 8)) % 8;
        writer.write_zeros(value_padding);
        
        // general.file_type (uint32)
        std::string file_type_key = "general.file_type";
        writer.write<uint64_t>(file_type_key.size());
        writer.write(file_type_key.c_str(), file_type_key.size());
        // Pad to 8-byte alignment after key
        size_t key_padding2 = (8 - (file_type_key.size() % 8)) % 8;
        writer.write_zeros(key_padding2);
        writer.write<uint32_t>(4);  // GGUF_TYPE_UINT32
        writer.write<uint32_t>(1);   // F32
        
        // No tensors for this simplified test
    }
    
    void write_kv_string(BinaryWriter& writer, const std::string& key, const std::string& value) {
        writer.write<uint64_t>(key.size());
        writer.write_string(key);
        writer.write<uint32_t>(0);  // String type
        writer.write<uint64_t>(value.size());
        writer.write_string(value);
    }
    
    void write_kv_uint64(BinaryWriter& writer, const std::string& key, uint64_t value) {
        writer.write<uint64_t>(key.size());
        writer.write_string(key);
        writer.write<uint32_t>(4);  // UINT64 type
        writer.write<uint64_t>(value);
    }
    
    void write_tensor_metadata(BinaryWriter& writer, const std::string& name, 
                              const std::vector<int64_t>& shape, uint32_t type, size_t offset) {
        writer.write<uint64_t>(name.size());
        writer.write_string(name);
        writer.write<uint32_t>(shape.size());  // n_dims
        for (size_t i = 0; i < 4; ++i) {  // Always write 4 dims, pad with 1 if needed
            writer.write<uint64_t>(i < shape.size() ? shape[i] : 1);
        }
        writer.write<uint32_t>(type);  // F32
        writer.write<uint64_t>(offset);
    }

    void TearDown() override {
        // Clean up the test model file
        if (std::filesystem::exists(test_model_path_)) {
            std::filesystem::remove(test_model_path_);
        }
    }

    std::string test_model_path_;
    
    // Helper functions for writing GGUF format are implemented in the class body above
};

TEST_F(ModelLoaderTest, LoadModel) {
    std::cout << "Testing model loading from: " << test_model_path_ << std::endl;
    
    // Verify the test file exists and has content
    std::ifstream file(test_model_path_, std::ios::binary | std::ios::ate);
    ASSERT_TRUE(file.is_open()) << "Failed to open test model file";
    std::streamsize size = file.tellg();
    file.close();
    ASSERT_GT(size, 0) << "Test model file is empty";
    std::cout << "Test model file size: " << size << " bytes" << std::endl;
    
    // Test loading the model
    EXPECT_NO_THROW({
        std::cout << "Creating ModelLoader instance..." << std::endl;
        ModelLoader loader(test_model_path_);
        
        // Verify the model is marked as loaded
        EXPECT_TRUE(loader.is_loaded()) << "Model should be marked as loaded";
        
        // Print some debug info
        std::cout << "Model architecture: " << loader.get_architecture() << std::endl;
        std::cout << "Number of tensors: " << loader.get_tensor_count() << std::endl;
        
        // Verify we can read the architecture
        EXPECT_EQ(loader.get_architecture(), "llama") << "Architecture should be 'llama'";
        
    }) << "Model loading threw an exception";
}

TEST_F(ModelLoaderTest, LoadNonexistentFile) {
    // Test loading a non-existent file
    EXPECT_THROW({
        ModelLoader loader("nonexistent_file.gguf");
    }, std::system_error);
}

TEST_F(ModelLoaderTest, LoadInvalidFile) {
    // Create an invalid model file
    std::string invalid_model_path = "invalid_model.gguf";
    std::ofstream ofs(invalid_model_path);
    ofs << "INVALID FORMAT\n";
    ofs.close();
    
    EXPECT_THROW({
        ModelLoader loader(invalid_model_path);
    }, std::runtime_error);
    
    // Clean up
    std::filesystem::remove(invalid_model_path);
}

// Add more tests for specific tensor loading, error cases, etc.

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
