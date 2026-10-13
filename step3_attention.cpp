#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>

// A dependency-free reference kernel. Q, K and V are identity projections;
// trained projection weights and model decoding are deliberately not implied.
class AttentionMechanism {
public:
    static constexpr int hidden_size = 64;
    static constexpr int num_heads = 8;
    static constexpr int head_dim = hidden_size / num_heads;

    std::vector<float> self_attention(const std::vector<float>& input, int seq_len) const {
        if (seq_len <= 0 || input.size() != static_cast<size_t>(seq_len) * hidden_size)
            throw std::invalid_argument("input must have shape [positive sequence length, 64]");
        for (float value : input)
            if (!std::isfinite(value)) throw std::invalid_argument("input must be finite");
        std::vector<float> output(input.size(), 0.0f);
        std::vector<double> scores(seq_len);
        for (int head = 0; head < num_heads; ++head) {
            for (int query = 0; query < seq_len; ++query) {
                double maximum = -INFINITY;
                for (int key = 0; key <= query; ++key) {
                    double dot = 0;
                    for (int dim = 0; dim < head_dim; ++dim) {
                        const int channel = head * head_dim + dim;
                        dot += static_cast<double>(input[query * hidden_size + channel]) * input[key * hidden_size + channel];
                    }
                    scores[key] = dot / std::sqrt(static_cast<double>(head_dim));
                    maximum = std::max(maximum, scores[key]);
                }
                double denominator = 0;
                for (int key = 0; key <= query; ++key) {
                    scores[key] = std::exp(scores[key] - maximum);
                    denominator += scores[key];
                }
                for (int dim = 0; dim < head_dim; ++dim) {
                    const int channel = head * head_dim + dim;
                    double value = 0;
                    for (int key = 0; key <= query; ++key)
                        value += scores[key] / denominator * input[key * hidden_size + channel];
                    output[query * hidden_size + channel] = static_cast<float>(value);
                }
            }
        }
        return output;
    }
};

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

int main() {
    try {
        AttentionMechanism attention;
        std::vector<float> input(3 * 64, 1.0f);
        auto constant = attention.self_attention(input, 3);
        for (float value : constant) require(std::abs(value - 1.0f) < 1e-6f, "constant values must be preserved");
        std::fill(input.begin() + 64, input.begin() + 128, 2.0f);
        std::fill(input.begin() + 128, input.end(), 3.0f);
        auto output = attention.self_attention(input, 3);
        require(output[0] == 1.0f, "first token must not see future tokens");
        double weight = std::exp(-2.0 * std::sqrt(8.0));
        double reference = (weight + 2.0) / (weight + 1.0);
        require(std::abs(output[64] - reference) < 1e-6, "scaled softmax must match independent reference");
        std::fill(input.begin() + 128, input.end(), 1000.0f);
        auto changed = attention.self_attention(input, 3);
        for (int i = 0; i < 128; ++i) require(changed[i] == output[i], "causal mask must isolate future values");
        require(std::isfinite(changed[128]), "stable softmax must stay finite");
        bool rejected = false;
        try { attention.self_attention({}, 0); } catch (const std::invalid_argument&) { rejected = true; }
        require(rejected, "invalid shapes must be rejected");
        std::cout << "PASS: 8-head causal scaled dot-product attention, stable softmax, reference agreement and shape validation\n";
        std::cout << "Educational kernel: identity Q/K/V projections, no trained-model generation\n";
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
