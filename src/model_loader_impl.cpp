#include "lightgpt/model_loader.hpp"
#include "lightgpt/tensor.hpp"
#include <fstream>
#include <stdexcept>
#include <system_error>

// GGUF file format structures
struct GGUFHeader {
    char magic[4];      // "GGUF"
    uint32_t version;   // Version number
    uint64_t n_tensors; // Number of tensors
    uint64_t n_kv;      // Number of key-value pairs
};

struct GGUFMetadataTensorInfo {
    uint32_t n_dims;        // Number of dimensions
    uint64_t shape[4];      // Shape (up to 4 dimensions)
    uint32_t type;          // Data type
    uint64_t offset;        // Offset from start of file
};

namespace lightgpt {

class ModelLoaderImpl {
public:
    ModelLoaderImpl() = default;
    ~ModelLoaderImpl();

    void load(const std::string& file_path);
    Tensor load_tensor(const std::string& name) const;
    bool has_tensor(const std::string& name) const;
    std::vector<int64_t> get_tensor_shape(const std::string& name) const;
    DType get_tensor_dtype(const std::string& name) const;
    std::string get_architecture() const;
    size_t get_tensor_count() const;
    std::vector<std::string> get_tensor_names() const;
    bool is_loaded() const { return loaded_; }

private:
    struct GGUFContext {
        int fd = -1;
        void* data = nullptr;
        size_t size = 0;
        const GGUFHeader* header = nullptr;
        const void* tensor_data = nullptr;
        std::unordered_map<std::string, const GGUFMetadataTensorInfo*> tensor_index;
        std::unordered_map<std::string, std::string> metadata;
    };

    std::unique_ptr<GGUFContext> ctx_;
    bool loaded_ = false;

    void validate_gguf_file() const;
    void load_metadata();
    static size_t gguf_type_size(uint32_t type);
    static DType gguf_type_to_dtype(uint32_t type);
};

// Implementation of ModelLoader methods
ModelLoader::ModelLoader() : pimpl_(std::make_unique<ModelLoaderImpl>()) {}
ModelLoader::ModelLoader(const std::string& file_path) : ModelLoader() { load(file_path); }
ModelLoader::~ModelLoader() = default;
ModelLoader::ModelLoader(ModelLoader&&) noexcept = default;
ModelLoader& ModelLoader::operator=(ModelLoader&&) noexcept = default;

void ModelLoader::load(const std::string& file_path) { pimpl_->load(file_path); }
Tensor ModelLoader::load_tensor(const std::string& name) const { return pimpl_->load_tensor(name); }
bool ModelLoader::has_tensor(const std::string& name) const { return pimpl_->has_tensor(name); }
std::vector<int64_t> ModelLoader::get_tensor_shape(const std::string& name) const { return pimpl_->get_tensor_shape(name); }
DType ModelLoader::get_tensor_dtype(const std::string& name) const { return pimpl_->get_tensor_dtype(name); }
std::string ModelLoader::get_architecture() const { return pimpl_->get_architecture(); }
size_t ModelLoader::get_tensor_count() const { return pimpl_->get_tensor_count(); }
std::vector<std::string> ModelLoader::get_tensor_names() const { return pimpl_->get_tensor_names(); }
bool ModelLoader::is_loaded() const { return pimpl_->is_loaded(); }

// Implementation of ModelLoaderImpl methods
ModelLoaderImpl::~ModelLoaderImpl() {
    if (ctx_) {
        if (ctx_->data) {
            munmap(const_cast<void*>(ctx_->data), ctx_->size);
        }
        if (ctx_->fd != -1) {
            close(ctx_->fd);
        }
    }
}

void ModelLoaderImpl::load(const std::string& file_path) {
    // Open file
    int fd = open(file_path.c_str(), O_RDONLY);
    if (fd == -1) {
        throw std::system_error(errno, std::generic_category(), "Failed to open file");
    }

    // Get file size
    struct stat sb;
    if (fstat(fd, &sb) == -1) {
        close(fd);
        throw std::system_error(errno, std::generic_category(), "Failed to get file size");
    }
    size_t file_size = sb.st_size;

    // Memory map the file
    void* data = mmap(nullptr, file_size, PROT_READ, MAP_PRIVATE, fd, 0);
    if (data == MAP_FAILED) {
        close(fd);
        throw std::system_error(errno, std::generic_category(), "Failed to memory map file");
    }

    // Create context
    ctx_ = std::make_unique<GGUFContext>();
    ctx_->fd = fd;
    ctx_->data = data;
    ctx_->size = file_size;
    ctx_->header = static_cast<const GGUFHeader*>(data);

    // Validate GGUF file
    validate_gguf_file();
    
    // Load metadata
    load_metadata();
    loaded_ = true;
}

void ModelLoaderImpl::validate_gguf_file() const {
    if (!ctx_ || !ctx_->header) {
        throw std::runtime_error("Invalid GGUF context");
    }
    
    if (ctx_->size < sizeof(GGUFHeader)) {
        throw std::runtime_error("File too small to be a valid GGUF file");
    }
    
    if (memcmp(ctx_->header->magic, "GGUF", 4) != 0) {
        throw std::runtime_error("Invalid GGUF file format");
    }
    
    if (ctx_->header->version != 1) {
        throw std::runtime_error("Unsupported GGUF version");
    }
}

void ModelLoaderImpl::load_metadata() {
    if (!ctx_ || !ctx_->header) return;
    
    // Implementation depends on GGUF format details
    // This is a placeholder for actual metadata parsing
}

Tensor ModelLoaderImpl::load_tensor(const std::string& name) const {
    if (!loaded_) {
        throw std::runtime_error("No model loaded");
    }
    
    auto it = ctx_->tensor_index.find(name);
    if (it == ctx_->tensor_index.end()) {
        throw std::runtime_error("Tensor not found: " + name);
    }
    
    const GGUFMetadataTensorInfo* info = it->second;
    std::vector<int64_t> shape(info->n_dims);
    for (uint32_t i = 0; i < info->n_dims; ++i) {
        shape[i] = static_cast<int64_t>(info->shape[i]);
    }
    
    const void* data = static_cast<const uint8_t*>(ctx_->data) + info->offset;
    return Tensor(shape, gguf_type_to_dtype(info->type), const_cast<void*>(data));
}

bool ModelLoaderImpl::has_tensor(const std::string& name) const {
    return loaded_ && ctx_ && ctx_->tensor_index.find(name) != ctx_->tensor_index.end();
}

std::vector<int64_t> ModelLoaderImpl::get_tensor_shape(const std::string& name) const {
    if (!loaded_) {
        throw std::runtime_error("No model loaded");
    }
    
    auto it = ctx_->tensor_index.find(name);
    if (it == ctx_->tensor_index.end()) {
        throw std::runtime_error("Tensor not found: " + name);
    }
    
    const GGUFMetadataTensorInfo* info = it->second;
    std::vector<int64_t> shape(info->n_dims);
    for (uint32_t i = 0; i < info->n_dims; ++i) {
        shape[i] = static_cast<int64_t>(info->shape[i]);
    }
    return shape;
}

DType ModelLoaderImpl::get_tensor_dtype(const std::string& name) const {
    if (!loaded_) {
        throw std::runtime_error("No model loaded");
    }
    
    auto it = ctx_->tensor_index.find(name);
    if (it == ctx_->tensor_index.end()) {
        throw std::runtime_error("Tensor not found: " + name);
    }
    
    return gguf_type_to_dtype(it->second->type);
}

std::string ModelLoaderImpl::get_architecture() const {
    if (!loaded_ || !ctx_) return "unknown";
    auto it = ctx_->metadata.find("general.architecture");
    return it != ctx_->metadata.end() ? it->second : "unknown";
}

size_t ModelLoaderImpl::get_tensor_count() const {
    return loaded_ && ctx_ ? ctx_->tensor_index.size() : 0;
}

std::vector<std::string> ModelLoaderImpl::get_tensor_names() const {
    std::vector<std::string> names;
    if (loaded_ && ctx_) {
        names.reserve(ctx_->tensor_index.size());
        for (const auto& pair : ctx_->tensor_index) {
            names.push_back(pair.first);
        }
    }
    return names;
}

size_t ModelLoaderImpl::gguf_type_size(uint32_t type) {
    // Map GGUF types to their sizes
    switch (type) {
        case 0: return 1;  // UINT8
        case 1: return 1;  // INT8
        case 2: return 2;  // INT16
        case 3: return 4;  // INT32
        case 4: return 4;  // FLOAT32
        case 5: return 1;  // BOOL
        case 6: return 2;  // FLOAT16
        case 7: return 8;  // FLOAT64
        case 8: return 8;  // UINT16
        case 9: return 8;  // UINT32
        default: return 0; // Unknown type
    }
}

DType ModelLoaderImpl::gguf_type_to_dtype(uint32_t type) {
    // Map GGUF types to DType
    switch (type) {
        case 0: return DType::UINT8;
        case 1: return DType::INT8;
        case 2: return DType::INT16;
        case 3: return DType::INT32;
        case 4: return DType::FLOAT32;
        case 5: return DType::BOOL;
        case 6: return DType::FLOAT16;
        case 7: return DType::FLOAT64;
        case 8: return DType::UINT16;
        case 9: return DType::UINT32;
        default: return DType::UNKNOWN;
    }
}

} // namespace lightgpt
