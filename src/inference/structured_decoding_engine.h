#pragma once

#include "lfm_inference.h"
#include <string>
#include <memory>

namespace liquid {

class StructuredDecodingEngine {
public:
    StructuredDecodingEngine(const lfm_vocab* vocab, const std::string& grammar_str, const std::string& root_rule = "root");
    ~StructuredDecodingEngine();

    // Apply grammar constraints to the logits/candidates
    // This modifies the candidates array by setting probability of invalid tokens to -infinity (or filtering them out)
    void Apply(lfm_token_data_array& candidates);

    // Update the grammar state with the selected token
    void Accept(lfm_token token);

    // Reset the grammar to initial state
    void Reset();

private:
    const lfm_vocab* vocab_;
    lfm_sampler* sampler_ = nullptr;
    std::string grammar_str_;
    std::string root_rule_;
};

} // namespace liquid
