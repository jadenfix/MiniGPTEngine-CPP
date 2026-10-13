#define main attention_demo_main
#include "../step3_attention.cpp"
#undef main
#include <cassert>

int main() {
    AttentionMechanism attention;
    std::vector<float> input(3 * 64, 1.0f);
    auto output = attention.self_attention(input, 3);
    for (float value : output) assert(std::abs(value - 1.0f) < 1e-6f);
    // A future token must never affect an earlier output.
    std::fill(input.begin() + 128, input.end(), 1000.0f);
    output = attention.self_attention(input, 3);
    for (int i = 0; i < 128; ++i) assert(std::abs(output[i] - 1.0f) < 1e-6f);
    // Two token reference: first vector is zero, second is one.
    input.assign(128, 0.0f);
    std::fill(input.begin() + 64, input.end(), 1.0f);
    output = attention.self_attention(input, 2);
    const double expected = 1.0 / (1.0 + std::exp(-std::sqrt(8.0)));
    for (int i = 0; i < 64; ++i) {
        assert(output[i] == 0.0f);
        assert(std::abs(output[64 + i] - expected) < 1e-6);
    }
    // Heads operate independently: modifying head 0 leaves head 1 unchanged.
    auto baseline = output;
    for (int i = 0; i < 8; ++i) input[64 + i] = 2.0f;
    output = attention.self_attention(input, 2);
    for (int i = 8; i < 64; ++i) assert(output[64 + i] == baseline[64 + i]);
    auto rejected = [&](std::vector<float> values, int length) {
        bool threw = false;
        try { attention.self_attention(values, length); }
        catch (const std::invalid_argument&) { threw = true; }
        assert(threw);
    };
    rejected({}, 0);
    rejected({}, -1);
    rejected({1.0f}, 1);
    rejected(std::vector<float>(64, std::numeric_limits<float>::quiet_NaN()), 1);
    rejected(std::vector<float>(64, std::numeric_limits<float>::infinity()), 1);
}
