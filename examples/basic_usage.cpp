#include "../include/lightgpt/transformer.hpp"
#include "../include/lightgpt/tokenizer.hpp"
#include "../include/lightgpt/model_loader.hpp"
#include <iostream>
#include <chrono>
#include <iomanip>

int main(int argc, char** argv) {
    if (argc < 4) {
        std::cerr << "Usage: " << argv[0] << " <model_path> <tokenizer_path> <prompt>\n";
        return 1;
    }

    const std::string model_path = argv[1];
    const std::string tokenizer_path = argv[2];
    const std::string prompt = argv[3];

    try {
        std::cout << "Loading model from " << model_path << "...\n";
        auto model = lightgpt::TransformerModel::from_gguf(lightgpt::ModelLoader(model_path));
        
        std::cout << "Loading tokenizer from " << tokenizer_path << "...\n";
        auto tokenizer = lightgpt::Tokenizer::from_file(tokenizer_path);
        
        std::cout << "\nPrompt: " << prompt << "\n";
        
        // Encode the prompt
        auto input_ids = tokenizer->encode(prompt);
        
        // Add BOS token if available
        if (tokenizer->bos_token_id() != -1) {
            input_ids.insert(input_ids.begin(), tokenizer->bos_token_id());
        }
        
        std::cout << "Generating response...\n\n";
        
        // Generate response
        auto start_time = std::chrono::high_resolution_clock::now();
        std::vector<int32_t> output_ids;
        
        for (int i = 0; i < 50; ++i) {  // Generate up to 50 tokens
            // Create input tensor
            lightgpt::Tensor input_tensor = lightgpt::tensor(
                input_ids, 
                {1, static_cast<int64_t>(input_ids.size())}
            );
            
            // Forward pass
            auto logits = model->forward(input_tensor, {});
            
            // Greedy decoding (select most likely next token)
            int32_t next_token = static_cast<int32_t>(logits.argmax(-1));
            
            // Stop if EOS token is generated
            if (next_token == tokenizer->eos_token_id()) {
                break;
            }
            
            output_ids.push_back(next_token);
            input_ids.push_back(next_token);
            
            // Print progress
            std::cout << tokenizer->decode(std::vector<int32_t>{next_token}) << std::flush;
        }
        
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
            end_time - start_time
        );
        
        std::cout << "\n\nGeneration complete! (" 
                  << output_ids.size() << " tokens, " 
                  << duration.count() << " ms, "
                  << std::fixed << std::setprecision(1)
                  << (output_ids.size() * 1000.0 / duration.count()) << " tokens/s)\n";
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
    
    return 0;
}
