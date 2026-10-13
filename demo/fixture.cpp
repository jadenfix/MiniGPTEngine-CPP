#define main attention_demo_main
#include "../step3_attention.cpp"
#undef main
#include <iomanip>
int main() {
    AttentionMechanism kernel;
    std::vector<float> x(192);
    for(int token=0;token<3;++token) std::fill(x.begin()+token*64,x.begin()+(token+1)*64,float(token+1));
    auto before=kernel.self_attention(x,3);
    std::fill(x.begin()+128,x.end(),1000.0f);
    auto after=kernel.self_attention(x,3);
    for(int i=0;i<128;++i) require(before[i]==after[i], "future leakage");
    for(float v:after) require(std::isfinite(v), "nonfinite output");
    std::cout << std::setprecision(9) << "{\"before\":[" << before[0] << "," << before[64] << "," << before[128] << "],\"after\":[" << after[0] << "," << after[64] << "," << after[128] << "],\"earlier_channels_unchanged\":128,\"finite\":true}\n";
}
