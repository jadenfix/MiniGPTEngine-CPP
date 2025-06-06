#pragma once

#include <vector>
#include <deque>
#include <unordered_map>
#include <memory>
#include <future>
#include <atomic>
#include <random>
#include <algorithm>
#include <string>
#include <functional>
#include <chrono>
#include <cmath>
#include <immintrin.h>

namespace lightgpt {

// Token cache for repeated sequence optimization
class TokenCache {
private:
    struct CacheEntry {
        std::vector<float> hidden_states;
        std::vector<float> attention_weights;
        uint64_t timestamp;
        uint32_t access_count;
        
        CacheEntry() : timestamp(0), access_count(0) {}
    };
    
    std::unordered_map<uint64_t, CacheEntry> cache_;
    std::deque<uint64_t> lru_order_;
    size_t max_cache_size_;
    size_t hidden_dim_;
    std::atomic<uint64_t> hit_count_{0};
    std::atomic<uint64_t> miss_count_{0};
    
    uint64_t hash_sequence(const std::vector<uint32_t>& tokens, size_t start, size_t len) const {
        uint64_t hash = 0xcbf29ce484222325ULL; // FNV-1a base
        for (size_t i = start; i < start + len && i < tokens.size(); ++i) {
            hash ^= tokens[i];
            hash *= 0x100000001b3ULL;
        }
        return hash;
    }
    
    void evict_lru() {
        if (lru_order_.empty()) return;
        
        uint64_t oldest_key = lru_order_.front();
        lru_order_.pop_front();
        cache_.erase(oldest_key);
    }
    
public:
    TokenCache(size_t max_size = 1024, size_t hidden_dim = 2048) 
        : max_cache_size_(max_size), hidden_dim_(hidden_dim) {}
    
    bool lookup(const std::vector<uint32_t>& tokens, size_t pos, size_t window,
                std::vector<float>& hidden_states, std::vector<float>& attention_weights) {
        if (window == 0 || pos + window > tokens.size()) return false;
        
        uint64_t key = hash_sequence(tokens, pos, window);
        auto it = cache_.find(key);
        
        if (it != cache_.end()) {
            hidden_states = it->second.hidden_states;
            attention_weights = it->second.attention_weights;
            it->second.access_count++;
            it->second.timestamp = std::chrono::steady_clock::now().time_since_epoch().count();
            
            // Move to end of LRU
            lru_order_.erase(std::find(lru_order_.begin(), lru_order_.end(), key));
            lru_order_.push_back(key);
            
            hit_count_++;
            return true;
        }
        
        miss_count_++;
        return false;
    }
    
    void store(const std::vector<uint32_t>& tokens, size_t pos, size_t window,
               const std::vector<float>& hidden_states, const std::vector<float>& attention_weights) {
        if (window == 0 || pos + window > tokens.size()) return;
        
        uint64_t key = hash_sequence(tokens, pos, window);
        
        if (cache_.size() >= max_cache_size_) {
            evict_lru();
        }
        
        CacheEntry& entry = cache_[key];
        entry.hidden_states = hidden_states;
        entry.attention_weights = attention_weights;
        entry.timestamp = std::chrono::steady_clock::now().time_since_epoch().count();
        entry.access_count = 1;
        
        lru_order_.push_back(key);
    }
    
    void clear() {
        cache_.clear();
        lru_order_.clear();
        hit_count_ = 0;
        miss_count_ = 0;
    }
    
    double hit_rate() const {
        uint64_t total = hit_count_ + miss_count_;
        return total > 0 ? static_cast<double>(hit_count_) / total : 0.0;
    }
    
    size_t size() const { return cache_.size(); }
};

// Top-K sampling with temperature and nucleus (top-p) sampling
class AdvancedSampler {
private:
    std::mt19937 rng_;
    
public:
    AdvancedSampler(uint32_t seed = std::random_device{}()) : rng_(seed) {}
    
    uint32_t sample_top_k(const std::vector<float>& logits, uint32_t k = 50, 
                          float temperature = 1.0f, float top_p = 0.9f) {
        if (logits.empty()) return 0;
        
        std::vector<std::pair<float, uint32_t>> scored_tokens;
        scored_tokens.reserve(logits.size());
        
        // Apply temperature scaling and collect scores
        for (size_t i = 0; i < logits.size(); ++i) {
            float score = logits[i] / temperature;
            scored_tokens.emplace_back(score, static_cast<uint32_t>(i));
        }
        
        // Sort by score (descending)
        std::partial_sort(scored_tokens.begin(), 
                         scored_tokens.begin() + std::min(k, static_cast<uint32_t>(scored_tokens.size())),
                         scored_tokens.end(),
                         [](const auto& a, const auto& b) { return a.first > b.first; });
        
        // Limit to top-k
        size_t effective_k = std::min(static_cast<size_t>(k), scored_tokens.size());
        scored_tokens.resize(effective_k);
        
        // Apply softmax and compute cumulative probabilities for nucleus sampling
        float max_score = scored_tokens[0].first;
        float sum_exp = 0.0f;
        
        for (auto& pair : scored_tokens) {
            pair.first = std::exp(pair.first - max_score);
            sum_exp += pair.first;
        }
        
        // Normalize probabilities
        for (auto& pair : scored_tokens) {
            pair.first /= sum_exp;
        }
        
        // Apply nucleus (top-p) sampling
        if (top_p < 1.0f) {
            float cumulative_prob = 0.0f;
            size_t nucleus_size = 0;
            
            for (size_t i = 0; i < scored_tokens.size(); ++i) {
                cumulative_prob += scored_tokens[i].first;
                nucleus_size = i + 1;
                if (cumulative_prob >= top_p) break;
            }
            
            scored_tokens.resize(nucleus_size);
            
            // Renormalize after nucleus filtering
            float nucleus_sum = 0.0f;
            for (const auto& pair : scored_tokens) {
                nucleus_sum += pair.first;
            }
            for (auto& pair : scored_tokens) {
                pair.first /= nucleus_sum;
            }
        }
        
        // Sample from the filtered distribution
        std::uniform_real_distribution<float> dist(0.0f, 1.0f);
        float random_val = dist(rng_);
        
        float cumulative_prob = 0.0f;
        for (const auto& pair : scored_tokens) {
            cumulative_prob += pair.first;
            if (random_val <= cumulative_prob) {
                return pair.second;
            }
        }
        
        return scored_tokens.back().second;
    }
    
    // Specialized sampling for very large vocabularies
    uint32_t sample_top_k_fast(const float* logits, size_t vocab_size, uint32_t k = 50,
                               float temperature = 1.0f, float top_p = 0.9f) {
        thread_local std::vector<std::pair<float, uint32_t>> scored_tokens;
        scored_tokens.clear();
        scored_tokens.reserve(std::min(static_cast<size_t>(k * 2), vocab_size));
        
        // Find approximate top-k using partial selection
        const float inv_temp = 1.0f / temperature;
        
        // Use AVX2 for faster processing if available
        #ifdef __AVX2__
        if (vocab_size >= 8) {
            const __m256 temp_vec = _mm256_set1_ps(inv_temp);
            
            for (size_t i = 0; i + 8 <= vocab_size; i += 8) {
                __m256 logits_vec = _mm256_loadu_ps(&logits[i]);
                __m256 scores = _mm256_mul_ps(logits_vec, temp_vec);
                
                float scores_array[8];
                _mm256_storeu_ps(scores_array, scores);
                
                for (int j = 0; j < 8; ++j) {
                    scored_tokens.emplace_back(scores_array[j], static_cast<uint32_t>(i + j));
                }
            }
            
            // Handle remaining elements
            for (size_t i = (vocab_size / 8) * 8; i < vocab_size; ++i) {
                scored_tokens.emplace_back(logits[i] * inv_temp, static_cast<uint32_t>(i));
            }
        } else
        #endif
        {
            for (size_t i = 0; i < vocab_size; ++i) {
                scored_tokens.emplace_back(logits[i] * inv_temp, static_cast<uint32_t>(i));
            }
        }
        
        // Use nth_element for O(n) top-k selection
        size_t effective_k = std::min(static_cast<size_t>(k), scored_tokens.size());
        std::nth_element(scored_tokens.begin(), 
                        scored_tokens.begin() + effective_k,
                        scored_tokens.end(),
                        [](const auto& a, const auto& b) { return a.first > b.first; });
        
        scored_tokens.resize(effective_k);
        
        // Continue with softmax and sampling as before
        return sample_from_distribution(scored_tokens, top_p);
    }
    
private:
    uint32_t sample_from_distribution(std::vector<std::pair<float, uint32_t>>& scored_tokens, float top_p) {
        if (scored_tokens.empty()) return 0;
        
        // Apply softmax
        float max_score = scored_tokens[0].first;
        float sum_exp = 0.0f;
        
        for (auto& pair : scored_tokens) {
            pair.first = std::exp(pair.first - max_score);
            sum_exp += pair.first;
        }
        
        for (auto& pair : scored_tokens) {
            pair.first /= sum_exp;
        }
        
        // Apply nucleus sampling if needed
        if (top_p < 1.0f) {
            float cumulative_prob = 0.0f;
            size_t nucleus_size = 0;
            
            for (size_t i = 0; i < scored_tokens.size(); ++i) {
                cumulative_prob += scored_tokens[i].first;
                nucleus_size = i + 1;
                if (cumulative_prob >= top_p) break;
            }
            
            scored_tokens.resize(nucleus_size);
        }
        
        // Sample
        std::uniform_real_distribution<float> dist(0.0f, 1.0f);
        float random_val = dist(rng_);
        
        float cumulative_prob = 0.0f;
        for (const auto& pair : scored_tokens) {
            cumulative_prob += pair.first;
            if (random_val <= cumulative_prob) {
                return pair.second;
            }
        }
        
        return scored_tokens.back().second;
    }
};

// Streamed inference for real-time generation
class StreamedInference {
public:
    struct StreamConfig {
        size_t batch_size = 1;
        size_t max_length = 512;
        float temperature = 1.0f;
        uint32_t top_k = 50;
        float top_p = 0.9f;
        size_t cache_window = 32;
        bool use_token_cache = true;
        size_t prefill_chunk_size = 64;
        
        StreamConfig() = default;
    };
    
    struct StreamState {
        std::vector<uint32_t> tokens;
        std::vector<float> hidden_states;
        std::vector<float> attention_cache;
        size_t position = 0;
        bool is_prefilling = true;
        std::chrono::steady_clock::time_point start_time;
        
        StreamState() : start_time(std::chrono::steady_clock::now()) {}
    };
    
private:
    StreamConfig config_;
    TokenCache token_cache_;
    AdvancedSampler sampler_;
    std::atomic<bool> should_stop_{false};
    
public:
    StreamedInference(const StreamConfig& config = StreamConfig()) 
        : config_(config), token_cache_(1024, 2048) {}
    
    // Async streaming generation
    std::future<std::vector<uint32_t>> generate_async(
        const std::vector<uint32_t>& prompt,
        std::function<void(uint32_t)> token_callback = nullptr) {
        
        return std::async(std::launch::async, [this, prompt, token_callback]() {
            return this->generate_stream(prompt, token_callback);
        });
    }
    
    std::vector<uint32_t> generate_stream(
        const std::vector<uint32_t>& prompt,
        std::function<void(uint32_t)> token_callback = nullptr) {
        
        StreamState state;
        state.tokens = prompt;
        should_stop_ = false;
        
        // Prefill phase - process prompt in chunks for better cache utilization
        while (state.is_prefilling && state.position < prompt.size()) {
            size_t chunk_end = std::min(state.position + config_.prefill_chunk_size, prompt.size());
            
            // Try to use token cache for this chunk
            std::vector<float> cached_hidden, cached_attention;
            bool cache_hit = false;
            
            if (config_.use_token_cache && chunk_end - state.position >= config_.cache_window) {
                cache_hit = token_cache_.lookup(state.tokens, state.position, 
                                              config_.cache_window, cached_hidden, cached_attention);
            }
            
            if (cache_hit) {
                state.hidden_states = std::move(cached_hidden);
                state.attention_cache = std::move(cached_attention);
                state.position += config_.cache_window;
            } else {
                // Process chunk and cache result
                process_chunk(state, state.position, chunk_end);
                
                if (config_.use_token_cache && chunk_end - state.position >= config_.cache_window) {
                    token_cache_.store(state.tokens, state.position, config_.cache_window,
                                     state.hidden_states, state.attention_cache);
                }
                
                state.position = chunk_end;
            }
            
            if (chunk_end >= prompt.size()) {
                state.is_prefilling = false;
            }
        }
        
        // Generation phase
        while (state.tokens.size() < config_.max_length && !should_stop_) {
            // Forward pass for current position
            std::vector<float> logits = forward_pass(state);
            
            // Sample next token
            uint32_t next_token = sampler_.sample_top_k_fast(
                logits.data(), logits.size(), config_.top_k, config_.temperature, config_.top_p);
            
            state.tokens.push_back(next_token);
            state.position++;
            
            // Call user callback if provided
            if (token_callback) {
                token_callback(next_token);
            }
            
            // Check for EOS token (assuming 0 is EOS)
            if (next_token == 0) break;
        }
        
        return state.tokens;
    }
    
    void stop() {
        should_stop_ = true;
    }
    
    double get_cache_hit_rate() const {
        return token_cache_.hit_rate();
    }
    
private:
    void process_chunk(StreamState& state, size_t start, size_t end) {
        // Simulate processing chunk - this would call your actual transformer
        // For now, just initialize with dummy data
        size_t chunk_size = end - start;
        state.hidden_states.resize(chunk_size * 2048);
        state.attention_cache.resize(chunk_size * 1024);
        
        // Fill with small random values to simulate processing
        std::uniform_real_distribution<float> dist(-0.1f, 0.1f);
        for (auto& val : state.hidden_states) {
            val = dist(sampler_.rng_);
        }
        for (auto& val : state.attention_cache) {
            val = dist(sampler_.rng_);
        }
    }
    
    std::vector<float> forward_pass(const StreamState& state) {
        // Simulate forward pass - return logits for vocabulary
        const size_t vocab_size = 32000; // Typical for LLaMA
        std::vector<float> logits(vocab_size);
        
        std::uniform_real_distribution<float> dist(-5.0f, 5.0f);
        for (auto& logit : logits) {
            logit = dist(sampler_.rng_);
        }
        
        return logits;
    }
};

// Batch processing for maximum throughput
class BatchInference {
public:
    struct BatchConfig {
        size_t max_batch_size = 8;
        size_t max_sequence_length = 512;
        bool dynamic_batching = true;
        float timeout_ms = 10.0f;
        bool use_padding = true;
        
        BatchConfig() = default;
    };
    
    struct BatchRequest {
        std::vector<uint32_t> tokens;
        std::promise<std::vector<uint32_t>> promise;
        std::chrono::steady_clock::time_point submit_time;
        size_t max_new_tokens = 100;
        
        BatchRequest(std::vector<uint32_t> input_tokens) 
            : tokens(std::move(input_tokens)), submit_time(std::chrono::steady_clock::now()) {}
    };
    
private:
    BatchConfig config_;
    std::deque<std::unique_ptr<BatchRequest>> pending_requests_;
    std::atomic<bool> running_{false};
    std::thread worker_thread_;
    std::mutex queue_mutex_;
    std::condition_variable queue_cv_;
    AdvancedSampler sampler_;
    
public:
    BatchInference(const BatchConfig& config = BatchConfig()) : config_(config) {}
    
    ~BatchInference() {
        stop();
    }
    
    void start() {
        running_ = true;
        worker_thread_ = std::thread(&BatchInference::process_batches, this);
    }
    
    void stop() {
        running_ = false;
        queue_cv_.notify_all();
        if (worker_thread_.joinable()) {
            worker_thread_.join();
        }
    }
    
    std::future<std::vector<uint32_t>> submit_request(std::vector<uint32_t> tokens, size_t max_new_tokens = 100) {
        auto request = std::make_unique<BatchRequest>(std::move(tokens));
        request->max_new_tokens = max_new_tokens;
        auto future = request->promise.get_future();
        
        {
            std::lock_guard<std::mutex> lock(queue_mutex_);
            pending_requests_.push_back(std::move(request));
        }
        queue_cv_.notify_one();
        
        return future;
    }
    
private:
    void process_batches() {
        while (running_) {
            std::vector<std::unique_ptr<BatchRequest>> current_batch;
            
            // Collect batch
            {
                std::unique_lock<std::mutex> lock(queue_mutex_);
                
                auto timeout_point = std::chrono::steady_clock::now() + 
                                   std::chrono::duration<float, std::milli>(config_.timeout_ms);
                
                queue_cv_.wait_until(lock, timeout_point, [this] {
                    return !pending_requests_.empty() || !running_;
                });
                
                if (!running_) break;
                
                // Collect up to max_batch_size requests
                while (!pending_requests_.empty() && current_batch.size() < config_.max_batch_size) {
                    current_batch.push_back(std::move(pending_requests_.front()));
                    pending_requests_.pop_front();
                }
            }
            
            if (!current_batch.empty()) {
                process_batch(current_batch);
            }
        }
    }
    
    void process_batch(std::vector<std::unique_ptr<BatchRequest>>& batch) {
        if (batch.empty()) return;
        
        // Prepare padded batch
        size_t max_input_len = 0;
        for (const auto& req : batch) {
            max_input_len = std::max(max_input_len, req->tokens.size());
        }
        
        size_t padded_len = ((max_input_len + 31) / 32) * 32; // Round up to 32 for SIMD
        
        // Create batched input
        std::vector<std::vector<uint32_t>> batched_tokens(batch.size());
        for (size_t i = 0; i < batch.size(); ++i) {
            batched_tokens[i] = batch[i]->tokens;
            if (config_.use_padding) {
                batched_tokens[i].resize(padded_len, 0); // Pad with 0
            }
        }
        
        // Process batch (simplified simulation)
        for (size_t i = 0; i < batch.size(); ++i) {
            auto& request = batch[i];
            
            // Simulate generation
            std::vector<uint32_t> generated = request->tokens;
            for (size_t j = 0; j < request->max_new_tokens; ++j) {
                uint32_t next_token = sampler_.sample_top_k({1.0f, 2.0f, 3.0f}, 3);
                generated.push_back(next_token);
                if (next_token == 0) break; // EOS
            }
            
            request->promise.set_value(std::move(generated));
        }
    }
};

} // namespace lightgpt 