#include "lightgpt/model_loader.hpp"
#include "lightgpt/tensor.hpp"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <stdexcept>
#include <unordered_map>
#include <string>
#include <vector>
#include <cstring>
#include <algorithm>
#include <iostream>
#include <iomanip>

namespace lightgpt {

// GGUF value types
enum class GGUFValueType : uint32_t {
    UINT8 = 0,
    INT8 = 1,
    UINT16 = 2,
    INT16 = 3,
    UINT32 = 4,
    INT32 = 5,
    FLOAT32 = 6,
    BOOL = 7,
    STRING = 8,
    ARRAY = 9,
    UINT64 = 10,
    INT64 = 11,
    FLOAT64 = 12,
};

// GGUF file format structures
namespace gguf {
#pragma pack(push, 1)

struct Header {
    char magic[4];      // "GGUF"
    uint32_t version;   // Version number
    uint64_t n_tensors; // Number of tensors
    uint64_t n_kv;      // Number of key-value pairs
};

struct TensorInfo {
    uint32_t n_dims;    // Number of dimensions
    uint64_t dims[4];   // Shape (up to 4 dimensions)
    uint32_t type;      // Data type
    uint64_t offset;    // Offset from start of file
};

struct KV {
    uint64_t key_len;
    // key: char[key_len]
    // value_type: uint32_t
    // value: depends on value_type
};

#pragma pack(pop)
} // namespace gguf

// Helper function to read a string from memory
static std::string read_string(const uint8_t*& data) {
    uint64_t len = *reinterpret_cast<const uint64_t*>(data);
    data += sizeof(uint64_t);
    std::string str(reinterpret_cast<const char*>(data), len);
    data += len;
    // Align to 8 bytes
    size_t padding = (8 - (len % 8)) % 8;
    data += padding;
    return str;
}

// Helper function to align pointer to 8-byte boundary
static const uint8_t* align_ptr(const uint8_t* ptr) {
    uintptr_t addr = reinterpret_cast<uintptr_t>(ptr);
    uintptr_t aligned = (addr + 7) & ~7;
    return reinterpret_cast<const uint8_t*>(aligned);
}

// PIMPL implementation
class ModelLoader::Impl {
public:
    Impl() = default;
    
    ~Impl() {
        cleanup();
    }

    // Prevent copying
    Impl(const Impl&) = delete;
    Impl& operator=(const Impl&) = delete;

    // Move operations
    Impl(Impl&& other) noexcept 
        : ctx_(other.ctx_)
        , loaded_(other.loaded_) {
        other.ctx_ = nullptr;
        other.loaded_ = false;
    }
    
    Impl& operator=(Impl&& other) noexcept {
        if (this != &other) {
            cleanup();
            ctx_ = other.ctx_;
            loaded_ = other.loaded_;
            other.ctx_ = nullptr;
            other.loaded_ = false;
        }
        return *this;
    }
    
    // Helper function to read a value of type T from the current position and advance the pointer
    template<typename T>
    static T read_value(const uint8_t*& data) {
        if (!data) {
            throw std::runtime_error("Null data pointer in read_value");
        }
        
        T value;
        std::memcpy(&value, data, sizeof(T));
        data += sizeof(T);
        return value;
    }
    
    // Helper to read a string value
    std::string read_string_value(const uint8_t*& data) const {
        uint64_t len = read_value<uint64_t>(data);
        std::string str(reinterpret_cast<const char*>(data), len);
        data += len;
        // Align to 8 bytes
        data = align_ptr(data);
        return str;
    }
    
    // Parse GGUF key-value pairs
    void parse_kv_pairs(const uint8_t*& data, uint64_t n_kv) {
        std::cout << "ModelLoader: Starting parse_kv_pairs\n";
        
        if (!ctx_) {
            std::cerr << "Error: Context is null in parse_kv_pairs\n";
            throw std::runtime_error("Context is null in parse_kv_pairs");
        }
        
        std::cout << "ModelLoader: Parsing " << n_kv << " key-value pairs\n";
        std::cout << "ModelLoader: Data pointer: " << static_cast<const void*>(data) << "\n";
        std::cout << "ModelLoader: Context data: " << static_cast<const void*>(ctx_->data) << "\n";
        std::cout << "ModelLoader: Context size: " << ctx_->size << "\n";
        
        if (!data) {
            std::cerr << "Error: Null data pointer in parse_kv_pairs\n";
            throw std::runtime_error("Null data pointer in parse_kv_pairs");
        }
        
        // Ensure data is within the mapped memory region
        if (data < ctx_->data || data >= ctx_->data + ctx_->size) {
            std::cerr << "Error: Data pointer outside mapped memory region\n";
            std::cerr << "Data pointer: " << static_cast<const void*>(data) << "\n";
            std::cerr << "Mapped region: [" << static_cast<const void*>(ctx_->data) 
                      << ", " << static_cast<const void*>(ctx_->data + ctx_->size) << ")\n";
            throw std::runtime_error("Data pointer outside mapped memory region");
        }
        
        const uint8_t* end = ctx_->data + ctx_->size;
        
        for (uint64_t i = 0; i < n_kv; ++i) {
            std::cout << "ModelLoader: Processing KV pair " << i << " of " << n_kv << "\n";
            std::cout << "ModelLoader: Current data pointer: " << static_cast<const void*>(data) << "\n";
            
            if (!data) {
                std::cerr << "Error: Null current pointer in parse_kv_pairs\n";
                throw std::runtime_error("Null current pointer in parse_kv_pairs");
            }
            
            if (data >= end) {
                std::cerr << "Error: Current pointer past end of data in parse_kv_pairs\n";
                std::cerr << "Current: " << static_cast<const void*>(data) << "\n";
                std::cerr << "End: " << static_cast<const void*>(end) << "\n";
                throw std::runtime_error("Current pointer past end of data in parse_kv_pairs");
            }
            // Read key
            std::cout << "ModelLoader: Reading key..." << std::endl;
            uint64_t key_len = read_value<uint64_t>(data);
            std::cout << "ModelLoader: Key length: " << key_len << std::endl;
            
            if (key_len > 1024) {  // Sanity check
                std::cerr << "Error: Suspiciously long key length: " << key_len << std::endl;
                // Print some bytes around current position for debugging
                std::cerr << "Data at error position: ";
                for (int j = -8; j < 16; ++j) {
                    if (data + j - sizeof(uint64_t) >= ctx_->data && data + j - sizeof(uint64_t) < end) {
                        std::cerr << std::hex << std::setw(2) << std::setfill('0') 
                                  << static_cast<int>(*(data + j - sizeof(uint64_t))) << " ";
                    }
                }
                std::cerr << std::dec << std::endl;
                throw std::runtime_error("Suspiciously long key length");
            }
            
            std::string key(reinterpret_cast<const char*>(data), key_len);
            std::cout << "ModelLoader: Read key: " << key << std::endl;
            data += key_len;
            std::cout << "ModelLoader: After key, data pointer: " << static_cast<const void*>(data) << "\n";
            // Align to 8 bytes after reading key
            data = align_ptr(data);
            std::cout << "ModelLoader: After alignment, data pointer: " << static_cast<const void*>(data) << "\n";
            
            // Read value type
            uint32_t value_type = read_value<uint32_t>(data);
            std::cout << "ModelLoader: Value type: " << value_type << "\n";
            
            // Read value based on type
            switch (static_cast<GGUFValueType>(value_type)) {
                case GGUFValueType::UINT8:
                case GGUFValueType::INT8:
                case GGUFValueType::BOOL:
                    ctx_->metadata[key] = std::to_string(read_value<uint8_t>(data));
                    break;
                case GGUFValueType::UINT16:
                case GGUFValueType::INT16:
                    ctx_->metadata[key] = std::to_string(read_value<uint16_t>(data));
                    break;
                case GGUFValueType::UINT32:
                case GGUFValueType::INT32:
                case GGUFValueType::FLOAT32:
                    ctx_->metadata[key] = std::to_string(read_value<uint32_t>(data));
                    break;
                case GGUFValueType::UINT64:
                case GGUFValueType::INT64:
                case GGUFValueType::FLOAT64:
                    ctx_->metadata[key] = std::to_string(read_value<uint64_t>(data));
                    break;
                case GGUFValueType::STRING: {
                    std::string value = read_string_value(data);
                    ctx_->metadata[key] = value;
                    std::cout << "ModelLoader: Read string value: " << value << std::endl;
                    break;
                }
                case GGUFValueType::ARRAY: {
                    // Skip arrays for now
                    uint32_t array_type = read_value<uint32_t>(data);
                    uint64_t array_len = read_value<uint64_t>(data);
                    size_t elem_size = 0;
                    
                    switch (static_cast<GGUFValueType>(array_type)) {
                        case GGUFValueType::UINT8:
                        case GGUFValueType::INT8:
                        case GGUFValueType::BOOL:
                            elem_size = 1; break;
                        case GGUFValueType::UINT16:
                        case GGUFValueType::INT16:
                            elem_size = 2; break;
                        case GGUFValueType::UINT32:
                        case GGUFValueType::INT32:
                        case GGUFValueType::FLOAT32:
                            elem_size = 4; break;
                        case GGUFValueType::UINT64:
                        case GGUFValueType::INT64:
                        case GGUFValueType::FLOAT64:
                            elem_size = 8; break;
                        case GGUFValueType::STRING:
                            // Skip string arrays for now
                            for (uint64_t j = 0; j < array_len; ++j) {
                                read_string_value(data);
                            }
                            break;
                        default:
                            throw std::runtime_error("Unsupported array type in GGUF file");
                    }
                    
                    if (static_cast<GGUFValueType>(array_type) != GGUFValueType::STRING) {
                        data += array_len * elem_size;
                    }
                    break;
                }
                default:
                    throw std::runtime_error("Unsupported value type in GGUF file");
            }
            
            // Align to 8 bytes
            data = align_ptr(data);
            std::cout << "ModelLoader: End of KV pair " << i << ", data pointer: " << static_cast<const void*>(data) << "\n";
        }
    }
    
    // Parse tensor metadata
    void parse_tensors(const uint8_t*& data, uint64_t n_tensors) {
        for (uint64_t i = 0; i < n_tensors; ++i) {
            // Read tensor name
            uint64_t name_len = read_value<uint64_t>(data);
            std::string name(reinterpret_cast<const char*>(data), name_len);
            data += name_len;
            
            // Read tensor info
            uint32_t n_dims = read_value<uint32_t>(data);
            
            gguf::TensorInfo tensor_info{};
            tensor_info.n_dims = n_dims;
            
            // Read dimensions
            for (uint32_t j = 0; j < 4; ++j) {
                if (j < n_dims) {
                    tensor_info.dims[j] = read_value<uint64_t>(data);
                } else {
                    tensor_info.dims[j] = 1; // Pad with 1 for unused dimensions
                }
            }
            
            // Read type and offset
            tensor_info.type = read_value<uint32_t>(data);
            tensor_info.offset = read_value<uint64_t>(data);
            
            // Store tensor info
            ctx_->tensors[name] = tensor_info;
        }
    }

    void load(const std::string& file_path) {
        std::cout << "ModelLoader: Loading model from " << file_path << std::endl;
        cleanup();
        
        try {
            std::cout << "ModelLoader: Opening model file " << file_path << std::endl;
            // Open file
            int fd = open(file_path.c_str(), O_RDONLY);
            if (fd == -1) {
                throw std::system_error(errno, std::generic_category(), 
                                      "Failed to open file: " + file_path);
            }

            // Get file size
            struct stat sb;
            if (fstat(fd, &sb) == -1) {
                close(fd);
                throw std::system_error(errno, std::generic_category(), 
                                      "Failed to get file size");
            }
            size_t file_size = sb.st_size;

            // Memory map the file
            void* data = mmap(nullptr, file_size, PROT_READ, MAP_PRIVATE, fd, 0);
            if (data == MAP_FAILED) {
                close(fd);
                throw std::system_error(errno, std::generic_category(), 
                                      "Failed to mmap file");
            }

            // Initialize context
            ctx_ = new Context();
            ctx_->fd = fd;
            ctx_->data = static_cast<const uint8_t*>(data);
            ctx_->size = file_size;
            ctx_->header = reinterpret_cast<const gguf::Header*>(data);

            // Validate GGUF file
            if (ctx_->size < sizeof(gguf::Header)) {
                throw std::runtime_error("File too small to be a valid GGUF file");
            }
            
            if (std::memcmp(ctx_->header->magic, "GGUF", 4) != 0) {
                throw std::runtime_error("Invalid GGUF magic number");
            }
            
            // Parse key-value pairs
            const uint8_t* current = ctx_->data + sizeof(gguf::Header);
            parse_kv_pairs(current, ctx_->header->n_kv);
            
            // Parse tensor metadata
            current = align_ptr(current);
            parse_tensors(current, ctx_->header->n_tensors);
            
            // Validate we have required metadata
            if (ctx_->metadata.find("general.architecture") == ctx_->metadata.end()) {
                throw std::runtime_error("Missing required metadata: general.architecture");
            }
            
            loaded_ = true;
            
        } catch (const std::exception& e) {
            cleanup();
            throw;  // Re-throw the exception after cleanup
        }
    }

    bool is_loaded() const { return loaded_; }
    
    // Convert GGUF type to DType
    DType gguf_type_to_dtype(uint32_t gguf_type) const {
        switch (gguf_type) {
            case 0:  // F32
                return DType::F32;
            // Add more type mappings as needed
            default:
                throw std::runtime_error("Unsupported tensor data type");
        }
    }
    
    // Load tensor data from file
    Tensor load_tensor(const std::string& name) {
        if (!loaded_) throw std::runtime_error("Model not loaded");
        
        auto it = ctx_->tensors.find(name);
        if (it == ctx_->tensors.end()) {
            throw std::runtime_error("Tensor not found: " + name);
        }
        
        const auto& info = it->second;
        
        // Get shape
        std::vector<int64_t> shape(info.n_dims);
        for (uint32_t i = 0; i < info.n_dims; ++i) {
            shape[i] = static_cast<int64_t>(info.dims[i]);
        }
        
        // Get data type
        DType dtype = gguf_type_to_dtype(info.type);
        
        // Calculate data size
        size_t element_size = 4; // Default to float32 size
        if (dtype == DType::F16) element_size = 2;
        
        size_t num_elements = 1;
        for (auto dim : shape) {
            num_elements *= dim;
        }
        
        // Create tensor with the right shape and type
        Tensor tensor(shape, dtype);
        
        // Copy data from memory-mapped file
        const uint8_t* src = ctx_->data + info.offset;
        std::memcpy(tensor.data<void>(), src, num_elements * element_size);
        
        return tensor;
    }

    bool has_tensor(const std::string& name) const {
        return loaded_ && ctx_ && ctx_->tensors.find(name) != ctx_->tensors.end();
    }

    std::vector<int64_t> get_tensor_shape(const std::string& name) const {
        if (!loaded_) throw std::runtime_error("Model not loaded");
        auto it = ctx_->tensors.find(name);
        if (it == ctx_->tensors.end()) {
            throw std::runtime_error("Tensor not found: " + name);
        }
        const auto& info = it->second;
        std::vector<int64_t> shape(info.n_dims);
        for (uint32_t i = 0; i < info.n_dims; ++i) {
            shape[i] = static_cast<int64_t>(info.dims[i]);
        }
        return shape;
    }

    DType get_tensor_dtype(const std::string& name) const {
        if (!loaded_) throw std::runtime_error("Model not loaded");
        auto it = ctx_->tensors.find(name);
        if (it == ctx_->tensors.end()) {
            throw std::runtime_error("Tensor not found: " + name);
        }
        return gguf_type_to_dtype(it->second.type);
    }

    std::string get_architecture() const {
        if (!loaded_ || !ctx_) return "";
        auto it = ctx_->metadata.find("general.architecture");
        return it != ctx_->metadata.end() ? it->second : "";
    }

    size_t get_tensor_count() const {
        return loaded_ && ctx_ ? ctx_->tensors.size() : 0;
    }

    std::vector<std::string> get_tensor_names() const {
        std::vector<std::string> names;
        if (loaded_ && ctx_) {
            names.reserve(ctx_->tensors.size());
            for (const auto& kv : ctx_->tensors) {
                names.push_back(kv.first);
            }
        }
        return names;
    }

    size_t get_vocab_size() const {
        if (!loaded_ || !ctx_) return 0;
        auto it = ctx_->metadata.find("tokenizer.ggml.tokens");
        if (it == ctx_->metadata.end()) {
            // Try alternative key
            it = ctx_->metadata.find("tokenizer.ggml.token_count");
        }
        return it != ctx_->metadata.end() ? std::stoull(it->second) : 0;
    }

    size_t get_max_sequence_length() const {
        if (!loaded_ || !ctx_) return 2048; // Default value
        auto it = ctx_->metadata.find("model.max_sequence_length");
        if (it == ctx_->metadata.end()) {
            // Try alternative keys
            it = ctx_->metadata.find("model.context_length");
        }
        return it != ctx_->metadata.end() ? std::stoull(it->second) : 2048;
    }

    size_t get_embedding_dim() const {
        if (!loaded_ || !ctx_) return 0;
        auto it = ctx_->metadata.find("model.embedding_length");
        if (it == ctx_->metadata.end()) {
            // Try alternative keys
            it = ctx_->metadata.find("model.dim");
        }
        return it != ctx_->metadata.end() ? std::stoull(it->second) : 0;
    }

    size_t get_num_layers() const {
        if (!loaded_ || !ctx_) return 0;
        auto it = ctx_->metadata.find("model.layer_count");
        if (it == ctx_->metadata.end()) {
            // Try alternative keys
            it = ctx_->metadata.find("model.num_layers");
        }
        return it != ctx_->metadata.end() ? std::stoull(it->second) : 0;
    }

    size_t get_num_heads() const {
        if (!loaded_ || !ctx_) return 0;
        auto it = ctx_->metadata.find("model.attention.head_count");
        if (it == ctx_->metadata.end()) {
            // Try alternative keys
            it = ctx_->metadata.find("model.num_heads");
        }
        return it != ctx_->metadata.end() ? std::stoull(it->second) : 0;
    }

    size_t get_num_kv_heads() const {
        if (!loaded_ || !ctx_) return 0;
        auto it = ctx_->metadata.find("model.attention.head_count_kv");
        if (it == ctx_->metadata.end()) {
            // Try alternative keys
            it = ctx_->metadata.find("model.num_kv_heads");
            if (it == ctx_->metadata.end()) {
                // Fallback to regular heads if KV heads not specified
                return get_num_heads();
            }
        }
        return it != ctx_->metadata.end() ? std::stoull(it->second) : get_num_heads();
    }

private:
    struct Context {
        const gguf::Header* header = nullptr;
        const uint8_t* data = nullptr;
        size_t size = 0;
        int fd = -1;
        std::unordered_map<std::string, gguf::TensorInfo> tensors;
        std::unordered_map<std::string, std::string> metadata;
    };
    
    Context* ctx_ = nullptr;
    bool loaded_ = false;
    
    void cleanup() {
        if (ctx_) {
            if (ctx_->data) {
                munmap(const_cast<void*>(static_cast<const void*>(ctx_->data)), ctx_->size);
            }
            if (ctx_->fd != -1) {
                close(ctx_->fd);
            }
            delete ctx_;
            ctx_ = nullptr;
        }
        loaded_ = false;
    }
};

// ModelLoader implementation
ModelLoader::ModelLoader() : pimpl_(std::make_unique<Impl>()) {}
ModelLoader::ModelLoader(const std::string& model_path) : ModelLoader() {
    load(model_path);
}
ModelLoader::~ModelLoader() = default;
ModelLoader::ModelLoader(ModelLoader&&) noexcept = default;
ModelLoader& ModelLoader::operator=(ModelLoader&&) noexcept = default;

void ModelLoader::load(const std::string& file_path) { pimpl_->load(file_path); }
Tensor ModelLoader::load_tensor(const std::string& name) const { return pimpl_->load_tensor(name); }
bool ModelLoader::has_tensor(const std::string& name) const { return pimpl_->has_tensor(name); }
std::vector<int64_t> ModelLoader::get_tensor_shape(const std::string& name) const { 
    return pimpl_->get_tensor_shape(name); 
}
DType ModelLoader::get_tensor_dtype(const std::string& name) const { 
    return pimpl_->get_tensor_dtype(name); 
}
std::string ModelLoader::get_architecture() const { return pimpl_->get_architecture(); }
size_t ModelLoader::get_tensor_count() const { return pimpl_->get_tensor_count(); }
std::vector<std::string> ModelLoader::get_tensor_names() const { 
    return pimpl_->get_tensor_names(); 
}
bool ModelLoader::is_loaded() const { return pimpl_->is_loaded(); }
size_t ModelLoader::get_vocab_size() const { return pimpl_->get_vocab_size(); }
size_t ModelLoader::get_max_sequence_length() const { 
    return pimpl_->get_max_sequence_length(); 
}
size_t ModelLoader::get_embedding_dim() const { return pimpl_->get_embedding_dim(); }
size_t ModelLoader::get_num_layers() const { return pimpl_->get_num_layers(); }
size_t ModelLoader::get_num_heads() const { return pimpl_->get_num_heads(); }
size_t ModelLoader::get_num_kv_heads() const { return pimpl_->get_num_kv_heads(); }

} // namespace lightgpt
