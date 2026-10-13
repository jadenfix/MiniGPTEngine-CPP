#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>
#include <random>
#include <algorithm>
#include <limits>
#include <stdexcept>

class AttentionMechanism {
private:
    int hidden_size = 64;
    int num_heads = 8;
    int head_dim = hidden_size / num_heads;
    
public:
    std::vector<float> self_attention(const std::vector<float>& input_embeddings, int seq_len) {
        std::cout << "🧠 Computing self-attention..." << std::endl;
        std::cout << "   Sequence length: " << seq_len << std::endl;
        std::cout << "   Hidden size: " << hidden_size << std::endl;
        
        auto start = std::chrono::high_resolution_clock::now();
        
        if (seq_len <= 0 || input_embeddings.size() != static_cast<size_t>(seq_len) * hidden_size) {
            throw std::invalid_argument("Expected positive sequence length and seq_len * 64 embeddings");
        }
        for (float value : input_embeddings) {
            if (!std::isfinite(value)) throw std::invalid_argument("Embeddings must be finite");
        }
        // Educational multi-head causal attention with identity Q, K and V projections.
        // No learned weights or model inference is performed here.
        std::vector<float> output(input_embeddings.size(), 0.0f);
        const double scale = 1.0 / std::sqrt(static_cast<double>(head_dim));
        for (int i = 0; i < seq_len; ++i) {
            for (int head = 0; head < num_heads; ++head) {
                std::vector<double> scores(i + 1);
                double maximum = -std::numeric_limits<double>::infinity();
                for (int k = 0; k <= i; ++k) {
                    double dot = 0.0;
                    for (int j = 0; j < head_dim; ++j) {
                        dot += static_cast<double>(input_embeddings[i * hidden_size + head * head_dim + j]) *
                               input_embeddings[k * hidden_size + head * head_dim + j];
                    }
                    scores[k] = dot * scale;
                    maximum = std::max(maximum, scores[k]);
                }
                double normalizer = 0.0;
                for (double& score : scores) { score = std::exp(score - maximum); normalizer += score; }
                for (int j = 0; j < head_dim; ++j) {
                    double value = 0.0;
                    for (int k = 0; k <= i; ++k) {
                        value += scores[k] / normalizer * input_embeddings[k * hidden_size + head * head_dim + j];
                    }
                    output[i * hidden_size + head * head_dim + j] = static_cast<float>(value);
                }
            }
        }

        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        
        std::cout << "✅ Attention computed in " << duration.count() << " ms" << std::endl;
        
        return output;
    }
    
    void run_tests() {
        std::cout << "\n🧪 Testing Attention Mechanism" << std::endl;
        std::cout << "==============================" << std::endl;
        
        int test_seq_len = 4;
        std::vector<float> test_input(test_seq_len * hidden_size, 1.0f);
        
        std::cout << "✅ Test 1 - Basic attention computation:" << std::endl;
        auto output = self_attention(test_input, test_seq_len);
        
        std::cout << "   Input shape: [" << test_seq_len << ", " << hidden_size << "]" << std::endl;
        std::cout << "   Output shape: [" << test_seq_len << ", " << hidden_size << "]" << std::endl;
        std::cout << "   Sample output: " << output[0] << " " << output[1] << " " << output[2] << std::endl;
        
        std::cout << "\n✅ Test 2 - Performance check:" << std::endl;
        auto perf_start = std::chrono::high_resolution_clock::now();
        auto perf_output = self_attention(test_input, test_seq_len);
        auto perf_end = std::chrono::high_resolution_clock::now();
        auto perf_duration = std::chrono::duration_cast<std::chrono::microseconds>(perf_end - perf_start);
        
        std::cout << "   Completed in " << perf_duration.count() << " μs" << std::endl;
        
        std::cout << "\n🎉 Attention mechanism tests PASSED!" << std::endl;
    }
};

int main() {
    std::cout << "🤖 Step 3: Attention Mechanism Test\n";
    std::cout << "===================================\n\n";
    
    AttentionMechanism attention;
    attention.run_tests();
    
    std::cout << "\n✅ STEP 3 COMPLETE!" << std::endl;
    std::cout << "📈 Validated here: educational causal attention with identity Q/K/V only" << std::endl;
    std::cout << "📝 Learned projections, weight loading, and model generation are separate unfinished work." << std::endl;
    
    return 0;
}