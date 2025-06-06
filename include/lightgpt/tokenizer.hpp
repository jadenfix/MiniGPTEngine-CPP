#pragma once

#include <string>
#include <vector>
#include <unordered_map>
#include <memory>

namespace lightgpt {

class Tokenizer {
public:
    Tokenizer() = default;
    ~Tokenizer() = default;

    // Load tokenizer from tokenizer.json
    static std::unique_ptr<Tokenizer> from_file(const std::string& path);

    // Encode text to token IDs
    std::vector<int32_t> encode(const std::string& text) const;

    // Decode token IDs to text
    std::string decode(const std::vector<int32_t>& ids) const;

    // Get vocabulary size
    size_t vocab_size() const { return vocab_.size(); }

    // Get special tokens
    int32_t bos_token_id() const { return bos_token_id_; }
    int32_t eos_token_id() const { return eos_token_id_; }
    int32_t pad_token_id() const { return pad_token_id_; }

private:
    struct BPE_Rule {
        int32_t x;
        int32_t y;
        float score;
    };

    std::unordered_map<std::string, int32_t> vocab_;
    std::vector<std::string> id_to_token_;
    std::vector<BPE_Rule> bpe_rules_;
    
    // Special tokens
    int32_t bos_token_id_ = -1;
    int32_t eos_token_id_ = -1;
    int32_t pad_token_id_ = -1;

    // Helper functions
    std::vector<std::string> bpe_tokenize(const std::string& text) const;
};

} // namespace lightgpt
