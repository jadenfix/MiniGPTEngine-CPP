#pragma once

#include "lightgpt/tensor.hpp"
#include <string>
#include <vector>
#include <memory>
#include <cstdint>
#include <stdexcept>
#include <unordered_map>
#include <system_error>

namespace lightgpt {

/**
 * @brief Loads and manages LightGPT model files in GGUF format
 * 
 * This class provides an interface to load and access model parameters
 * from GGUF format files. It uses the PIMPL idiom to hide implementation
 * details and maintain ABI stability.
 */
class ModelLoader {
    class Impl; // Forward declaration of implementation class
    std::unique_ptr<Impl> pimpl_;

public:
    /**
     * @brief Default constructor
     */
    ModelLoader();
    
    /**
     * @brief Construct and load a model from file
     * @param file_path Path to the GGUF model file
     * @throws std::system_error if file operations fail
     * @throws std::runtime_error if the file is not a valid GGUF file
     */
    explicit ModelLoader(const std::string& file_path);
    
    /**
     * @brief Destructor
     */
    ~ModelLoader();
    
    // Prevent copying
    ModelLoader(const ModelLoader&) = delete;
    ModelLoader& operator=(const ModelLoader&) = delete;
    
    // Move operations
    ModelLoader(ModelLoader&&) noexcept;
    ModelLoader& operator=(ModelLoader&&) noexcept;
    
    /**
     * @brief Factory method to create a ModelLoader from a file
     * @param file_path Path to the GGUF model file
     * @return ModelLoader instance
     * @throws std::system_error if file operations fail
     * @throws std::runtime_error if the file is not a valid GGUF file
     */
    static ModelLoader from_file(const std::string& file_path) {
        return ModelLoader(file_path);
    }
    
    /**
     * @brief Load a model from file
     * @param file_path Path to the GGUF model file
     * @throws std::system_error if file operations fail
     * @throws std::runtime_error if the file is not a valid GGUF file
     */
    void load(const std::string& file_path);
    
    /**
     * @brief Load a tensor by name
     * @param name Name of the tensor to load
     * @return Tensor containing the loaded data
     * @throws std::runtime_error if tensor is not found or model is not loaded
     */
    Tensor load_tensor(const std::string& name) const;
    
    /**
     * @brief Check if a tensor exists in the model
     * @param name Name of the tensor to check
     * @return true if tensor exists, false otherwise
     */
    bool has_tensor(const std::string& name) const;
    
    /**
     * @brief Get the shape of a tensor
     * @param name Name of the tensor
     * @return Vector of int64_t representing the tensor shape
     * @throws std::runtime_error if tensor is not found or model is not loaded
     */
    std::vector<int64_t> get_tensor_shape(const std::string& name) const;
    
    /**
     * @brief Get the data type of a tensor
     * @param name Name of the tensor
     * @return DType enum representing the tensor's data type
     * @throws std::runtime_error if tensor is not found or model is not loaded
     */
    DType get_tensor_dtype(const std::string& name) const;
    
    /**
     * @brief Get the model architecture name
     * @return String representing the model architecture (e.g., "gpt2")
     */
    std::string get_architecture() const;
    
    /**
     * @brief Get the number of tensors in the model
     * @return Number of tensors
     */
    size_t get_tensor_count() const;
    
    /**
     * @brief Get the names of all tensors in the model
     * @return Vector of tensor names
     */
    std::vector<std::string> get_tensor_names() const;
    
    /**
     * @brief Check if a model is loaded
     * @return true if a model is loaded, false otherwise
     */
    bool is_loaded() const;
    
    /**
     * @brief Get the vocabulary size of the model
     * @return Vocabulary size
     * @throws std::runtime_error if model is not loaded
     */
    size_t get_vocab_size() const;
    
    /**
     * @brief Get the maximum sequence length the model can handle
     * @return Maximum sequence length
     * @throws std::runtime_error if model is not loaded
     */
    size_t get_max_sequence_length() const;
    
    /**
     * @brief Get the embedding dimension of the model
     * @return Embedding dimension
     * @throws std::runtime_error if model is not loaded
     */
    size_t get_embedding_dim() const;
    
    /**
     * @brief Get the number of transformer layers in the model
     * @return Number of transformer layers
     * @throws std::runtime_error if model is not loaded
     */
    size_t get_num_layers() const;
    
    /**
     * @brief Get the number of attention heads in the model
     * @return Number of attention heads
     * @throws std::runtime_error if model is not loaded
     */
    size_t get_num_heads() const;
    
    /**
     * @brief Get the number of key/value heads in the model (for multi-query attention)
     * @return Number of key/value heads
     * @throws std::runtime_error if model is not loaded or not applicable
     */
    size_t get_num_kv_heads() const;
};

} // namespace lightgpt
