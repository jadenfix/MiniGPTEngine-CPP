#include "lightgpt/transformer.hpp"
#include "lightgpt/tokenizer.hpp"
#include "lightgpt/model_loader.hpp"
#include <iomanip>
#include <iostream>
#include <string>
#include <chrono>
#include <iomanip>

using namespace lightgpt;
using namespace std::chrono;

struct GenerationConfig {
    int max_length = 100;
    float temperature = 0.7f;
    int top_k = 50;
    float top_p = 0.9f;
    bool do_sample = true;
    int num_return_sequences = 1;
};

class LightGPT {
public:
    LightGPT(const std::string& model_path, const std::string& tokenizer_path) {
        // Load model
        loader_ = std::make_unique<ModelLoader>();
        loader_->load(model_path);
        
        // Initialize model
        model_ = TransformerModel::from_gguf(*loader_);
        
        // Load tokenizer
        tokenizer_ = Tokenizer::from_file(tokenizer_path);
        
        // Get model dimensions from the loader
        const int64_t num_heads = loader_->get_num_heads();
        const int64_t head_dim = loader_->get_embedding_dim() / num_heads;
        
        // Initialize KV cache
        kv_cache_ = std::make_unique<KVCache>(
            1,  // batch_size
            num_heads,
            head_dim,
            2048,  // max_seq_len
            DType::F32
        );
    }
    
    std::string generate(const std::string& prompt, const GenerationConfig& config) {
        // Encode input
        auto input_ids = tokenizer_->encode(prompt);
        
        // Add BOS token if available
        if (tokenizer_->bos_token_id() != -1) {
            input_ids.insert(input_ids.begin(), tokenizer_->bos_token_id());
        }
        
        // Generate tokens
        std::vector<int32_t> output_ids;
        auto start_time = high_resolution_clock::now();
        
        for (int i = 0; i < config.max_length; ++i) {
            // Run model forward pass
            Tensor input_tensor = tensor(input_ids, {1, static_cast<int64_t>(input_ids.size())});
            auto logits = model_->forward(input_tensor, {});
            
            // Get next token (greedy sampling for now)
            // Get the token with highest probability (argmax)
            const float* logits_data = logits.data<float>();
            int64_t next_token = 0;
            float max_prob = logits_data[0];
            for (size_t i = 1; i < static_cast<size_t>(logits.numel()); ++i) {
                if (logits_data[i] > max_prob) {
                    max_prob = logits_data[i];
                    next_token = static_cast<int64_t>(i);
                }
            }
            
            // Stop if EOS token is generated
            if (next_token == tokenizer_->eos_token_id()) {
                break;
            }
            
            output_ids.push_back(next_token);
            input_ids.push_back(next_token);
            
            // Print progress
            if (i % 10 == 0) {
                auto current_output = tokenizer_->decode(output_ids);
                std::cout << "\rGenerating... " << current_output << std::flush;
            }
        }
        
        auto end_time = high_resolution_clock::now();
        auto duration = duration_cast<milliseconds>(end_time - start_time);
        
        // Decode and return the generated text
        std::string output = tokenizer_->decode(output_ids);
        
        std::cout << "\n\nGeneration complete! (" 
                  << output_ids.size() << " tokens, " 
                  << duration.count() << " ms, "
                  << std::fixed << std::setprecision(1)
                  << (output_ids.size() * 1000.0 / duration.count()) << " tokens/s)\n";
        
        return output;
    }
    
private:
    std::unique_ptr<ModelLoader> loader_;
    std::unique_ptr<TransformerModel> model_;
    std::unique_ptr<Tokenizer> tokenizer_;
    std::unique_ptr<KVCache> kv_cache_;
};

void print_usage(const char* program_name) {
    std::cout << "Usage: " << program_name << " [options]\n"
              << "Options:\n"
              << "  --model PATH         Path to GGUF model file\n"
              << "  --tokenizer PATH     Path to tokenizer.json\n"
              << "  --prompt TEXT        Input prompt\n"
              << "  --max-length N       Maximum number of tokens to generate (default: 100)\n"
              << "  --temperature F      Sampling temperature (default: 0.7)\n"
              << "  --help               Show this help message\n";
}

int main(int argc, char* argv[]) {
    std::string model_path;
    std::string tokenizer_path;
    std::string prompt = "The quick brown fox";
    GenerationConfig config;
    
    // Parse command line arguments
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        
        if (arg == "--model" && i + 1 < argc) {
            model_path = argv[++i];
        } else if (arg == "--tokenizer" && i + 1 < argc) {
            tokenizer_path = argv[++i];
        } else if (arg == "--prompt" && i + 1 < argc) {
            prompt = argv[++i];
        } else if (arg == "--max-length" && i + 1 < argc) {
            config.max_length = std::stoi(argv[++i]);
        } else if (arg == "--temperature" && i + 1 < argc) {
            config.temperature = std::stof(argv[++i]);
        } else if (arg == "--help") {
            print_usage(argv[0]);
            return 0;
        } else {
            std::cerr << "Unknown option: " << arg << "\n";
            print_usage(argv[0]);
            return 1;
        }
    }
    
    if (model_path.empty() || tokenizer_path.empty()) {
        std::cerr << "Error: Both --model and --tokenizer arguments are required\n";
        print_usage(argv[0]);
        return 1;
    }
    
    try {
        std::cout << "Loading model from " << model_path << "...\n";
        LightGPT generator(model_path, tokenizer_path);
        
        std::cout << "\nPrompt: " << prompt << "\n";
        std::cout << "Generating response...\n\n";
        
        auto output = generator.generate(prompt, config);
        
        std::cout << "\nGenerated text:\n" << output << "\n";
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
    
    return 0;
}
