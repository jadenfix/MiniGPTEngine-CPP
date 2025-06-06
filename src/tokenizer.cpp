#include "lightgpt/tokenizer.hpp"
#include <fstream>
#include <sstream>
#include <algorithm>
#include <cassert>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

namespace lightgpt {

namespace {

// Helper function to split a string on whitespace
std::vector<std::string> split_into_words(const std::string& text) {
    std::vector<std::string> words;
    std::istringstream iss(text);
    std::string word;
    while (iss >> word) {
        words.push_back(word);
    }
    return words;
}

// Helper function to load JSON from file
json load_json(const std::string& path) {
    std::ifstream f(path);
    if (!f.is_open()) {
        throw std::runtime_error("Failed to open tokenizer file: " + path);
    }
    return json::parse(f);
}

} // anonymous namespace

/* static */
std::unique_ptr<Tokenizer> Tokenizer::from_file(const std::string& path) {
    auto tokenizer = std::make_unique<Tokenizer>();
    json j = load_json(path);
    
    // Load vocabulary
    const auto& vocab = j["model"]["vocab"];
    for (auto it = vocab.begin(); it != vocab.end(); ++it) {
        int32_t id = it.value().get<int32_t>();
        tokenizer->vocab_[it.key()] = id;
        if (id >= static_cast<int32_t>(tokenizer->id_to_token_.size())) {
            tokenizer->id_to_token_.resize(id + 1);
        }
        tokenizer->id_to_token_[id] = it.key();
    }
    
    // Load BPE merges
    if (j["model"].contains("merges")) {
        const auto& merges = j["model"]["merges"];
        for (const auto& merge : merges.items()) {
            std::istringstream iss(merge.key());
            std::string x_str, y_str;
            float score = merge.value().get<float>();
            
            iss >> x_str >> y_str;
            tokenizer->bpe_rules_.push_back({
                tokenizer->vocab_.at(x_str),
                tokenizer->vocab_.at(y_str),
                score
            });
        }
    }
    
    // Load special tokens
    auto get_special_token = [&](const std::string& key) -> int32_t {
        if (j["model"]["special_tokens"].contains(key)) {
            return j["model"]["special_tokens"][key];
        }
        return -1;
    };
    
    tokenizer->bos_token_id_ = get_special_token("<s>");
    tokenizer->eos_token_id_ = get_special_token("</s>");
    tokenizer->pad_token_id_ = get_special_token("<pad>");
    
    return tokenizer;
}

std::vector<std::string> Tokenizer::bpe_tokenize(const std::string& text) const {
    // Simple BPE tokenization - in practice you'd want to use the actual BPE algorithm
    // This is a simplified version that just splits on whitespace and punctuation
    std::vector<std::string> tokens;
    std::string current;
    
    for (char c : text) {
        if (std::isspace(c)) {
            if (!current.empty()) {
                tokens.push_back(current);
                current.clear();
            }
        } else if (std::ispunct(c)) {
            if (!current.empty()) {
                tokens.push_back(current);
                current.clear();
            }
            tokens.push_back(std::string(1, c));
        } else {
            current += c;
        }
    }
    
    if (!current.empty()) {
        tokens.push_back(current);
    }
    
    return tokens;
}

std::vector<int32_t> Tokenizer::encode(const std::string& text) const {
    std::vector<int32_t> ids;
    auto tokens = bpe_tokenize(text);
    
    for (const auto& token : tokens) {
        auto it = vocab_.find(token);
        if (it != vocab_.end()) {
            ids.push_back(it->second);
        } else {
            // Handle out-of-vocabulary tokens (use UNK or split into subwords)
            // For now, we'll just skip them
            continue;
        }
    }
    
    return ids;
}

std::string Tokenizer::decode(const std::vector<int32_t>& ids) const {
    std::string result;
    for (size_t i = 0; i < ids.size(); ++i) {
        int32_t id = ids[i];
        if (id < 0 || id >= static_cast<int32_t>(id_to_token_.size())) {
            continue;  // Skip invalid token IDs
        }
        
        const std::string& token = id_to_token_[id];
        
        // Add space before token if it's not punctuation
        if (!result.empty() && !std::ispunct(token[0])) {
            result += " ";
        }
        result += token;
    }
    
    return result;
}

} // namespace lightgpt
