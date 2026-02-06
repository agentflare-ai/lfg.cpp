#pragma once

#include "lfm_inference.h"

#include <string>
#include <vector>
#include <memory>

// pre-tokenization types
enum lfm_vocab_pre_type {
    LFM_VOCAB_PRE_TYPE_DEFAULT         = 0,
    LFM_VOCAB_PRE_TYPE_LIQUID3          = 1,
    LFM_VOCAB_PRE_TYPE_DEEPSEEK_LLM    = 2,
    LFM_VOCAB_PRE_TYPE_DEEPSEEK_CODER  = 3,
    LFM_VOCAB_PRE_TYPE_FALCON          = 4,
    LFM_VOCAB_PRE_TYPE_MPT             = 5,
    LFM_VOCAB_PRE_TYPE_STARCODER       = 6,
    LFM_VOCAB_PRE_TYPE_GPT2            = 7,
    LFM_VOCAB_PRE_TYPE_REFACT          = 8,
    LFM_VOCAB_PRE_TYPE_COMMAND_R       = 9,
    LFM_VOCAB_PRE_TYPE_STABLELM2       = 10,
    LFM_VOCAB_PRE_TYPE_QWEN2           = 11,
    LFM_VOCAB_PRE_TYPE_OLMO            = 12,
    LFM_VOCAB_PRE_TYPE_DBRX            = 13,
    LFM_VOCAB_PRE_TYPE_SMAUG           = 14,
    LFM_VOCAB_PRE_TYPE_PORO            = 15,
    LFM_VOCAB_PRE_TYPE_CHATGLM3        = 16,
    LFM_VOCAB_PRE_TYPE_CHATGLM4        = 17,
    LFM_VOCAB_PRE_TYPE_VIKING          = 18,
    LFM_VOCAB_PRE_TYPE_JAIS            = 19,
    LFM_VOCAB_PRE_TYPE_TEKKEN          = 20,
    LFM_VOCAB_PRE_TYPE_SMOLLM          = 21,
    LFM_VOCAB_PRE_TYPE_CODESHELL       = 22,
    LFM_VOCAB_PRE_TYPE_BLOOM           = 23,
    LFM_VOCAB_PRE_TYPE_GPT3_FINNISH    = 24,
    LFM_VOCAB_PRE_TYPE_EXAONE          = 25,
    LFM_VOCAB_PRE_TYPE_CHAMELEON       = 26,
    LFM_VOCAB_PRE_TYPE_MINERVA         = 27,
    LFM_VOCAB_PRE_TYPE_DEEPSEEK3_LLM   = 28,
    LFM_VOCAB_PRE_TYPE_GPT4O           = 29,
    LFM_VOCAB_PRE_TYPE_SUPERBPE        = 30,
    LFM_VOCAB_PRE_TYPE_TRILLION        = 31,
    LFM_VOCAB_PRE_TYPE_BAILINGMOE      = 32,
    LFM_VOCAB_PRE_TYPE_LIQUID4          = 33,
    LFM_VOCAB_PRE_TYPE_PIXTRAL         = 34,
    LFM_VOCAB_PRE_TYPE_SEED_CODER      = 35,
    LFM_VOCAB_PRE_TYPE_HUNYUAN         = 36,
    LFM_VOCAB_PRE_TYPE_KIMI_K2         = 37,
    LFM_VOCAB_PRE_TYPE_HUNYUAN_DENSE   = 38,
    LFM_VOCAB_PRE_TYPE_GROK_2          = 39,
    LFM_VOCAB_PRE_TYPE_GRANITE_DOCLING = 40,
    LFM_VOCAB_PRE_TYPE_MINIMAX_M2      = 41,
    LFM_VOCAB_PRE_TYPE_AFMOE           = 42,
    LFM_VOCAB_PRE_TYPE_SOLAR_OPEN      = 43,
    LFM_VOCAB_PRE_TYPE_YOUTU           = 44,
    LFM_VOCAB_PRE_TYPE_EXAONE_MOE      = 45,
};

struct LLM_KV;
struct lfm_model_loader;

struct lfm_vocab {
    struct token_data {
        std::string      text;
        float            score;
        lfm_token_attr attr;
    };

    lfm_vocab();
    ~lfm_vocab();

    void load(lfm_model_loader & ml, const LLM_KV & kv);

    std::string get_tokenizer_model() const;
    std::string get_tokenizer_pre() const;

    enum lfm_vocab_type     get_type()     const;
    enum lfm_vocab_pre_type get_pre_type() const;

    uint32_t n_tokens() const;
    uint32_t n_token_types() const;

    std::string type_name() const;

    bool is_normal      (lfm_token id) const;
    bool is_unknown     (lfm_token id) const;
    bool is_control     (lfm_token id) const;
    bool is_byte        (lfm_token id) const;
    bool is_user_defined(lfm_token id) const;
    bool is_unused      (lfm_token id) const;
    bool is_eog         (lfm_token id) const;

    uint8_t     token_to_byte(lfm_token id) const;
    lfm_token byte_to_token(uint8_t ch)     const;

    lfm_token text_to_token(const std::string & text) const;

    const token_data & get_token_data(lfm_token id) const;

    const char *     token_get_text (lfm_token id) const;
    float            token_get_score(lfm_token id) const;
    lfm_token_attr token_get_attr (lfm_token id) const;

    lfm_token token_bos() const;
    lfm_token token_eos() const;
    lfm_token token_eot() const;
    lfm_token token_eom() const;
    lfm_token token_unk() const;
    lfm_token token_sep() const;
    lfm_token token_nl () const;
    lfm_token token_pad() const;
    lfm_token token_mask() const;

    lfm_token token_prefix() const;
    lfm_token token_middle() const;
    lfm_token token_suffix() const;

    lfm_token token_fim_pre() const;
    lfm_token token_fim_suf() const;
    lfm_token token_fim_mid() const;
    lfm_token token_fim_pad() const;
    lfm_token token_fim_rep() const;
    lfm_token token_fim_sep() const;

    bool get_add_space_prefix          () const;
    bool get_add_bos                   () const;
    bool get_add_eos                   () const;
    bool get_add_sep                   () const;
    bool get_ignore_merges             () const;
    bool get_clean_spaces              () const;
    bool get_remove_extra_whitespaces  () const;
    bool get_escape_whitespaces        () const;
    bool get_treat_whitespace_as_suffix() const;

    int max_token_len() const;

    int find_bpe_rank(const std::string & token_left, const std::string & token_right) const;
    std::vector<std::string> get_bpe_merges() const;

    std::vector<char> get_precompiled_charsmap() const;

    int32_t tokenize(
                   const char * text,
                      int32_t   text_len,
                  lfm_token * tokens,
                      int32_t   n_tokens_max,
                         bool   add_special,
                         bool   parse_special) const;

    std::vector<lfm_token> tokenize(
            const std::string & raw_text,
                         bool   add_special,
                         bool   parse_special = false) const;

    // does not write null-terminator to buf
    int32_t token_to_piece(
                  lfm_token   token,
                         char * buf,
                      int32_t   length,
                      int32_t   lstrip,
                         bool   special) const;

    // use cached data
    const std::string & token_to_piece(lfm_token token) const;

    int32_t detokenize(
            const lfm_token * tokens,
                      int32_t   n_tokens,
                         char * text,
                      int32_t   text_len_max,
                         bool   remove_special,
                         bool   unparse_special) const;

    std::string detokenize(
            const std::vector<lfm_token> & tokens,
                                      bool   special) const;

    void print_info() const;

private:
    struct impl;
    std::unique_ptr<impl> pimpl;
};
