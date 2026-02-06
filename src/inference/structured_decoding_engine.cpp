#include "structured_decoding_engine.h"
#include <spdlog/spdlog.h>

namespace liquid {

StructuredDecodingEngine::StructuredDecodingEngine(const lfm_vocab* vocab, const std::string& grammar_str, const std::string& root_rule)
    : vocab_(vocab), grammar_str_(grammar_str), root_rule_(root_rule) {
    
    // Initialize sampler
    if (!grammar_str_.empty()) {
        sampler_ = lfm_sampler_init_grammar(vocab_, grammar_str_.c_str(), root_rule_.c_str());
        if (!sampler_) {
            spdlog::error("Failed to initialize grammar sampler.");
        }
    }
}

StructuredDecodingEngine::~StructuredDecodingEngine() {
    if (sampler_) {
        lfm_sampler_free(sampler_);
    }
}

void StructuredDecodingEngine::Apply(lfm_token_data_array& candidates) {
    if (sampler_) {
        lfm_sampler_apply(sampler_, &candidates);
    }
}

void StructuredDecodingEngine::Accept(lfm_token token) {
    if (sampler_) {
        lfm_sampler_accept(sampler_, token);
    }
}

void StructuredDecodingEngine::Reset() {
    if (sampler_) {
        lfm_sampler_free(sampler_);
        sampler_ = lfm_sampler_init_grammar(vocab_, grammar_str_.c_str(), root_rule_.c_str());
    }
}

} // namespace liquid
