#pragma once

#include "lfg_inference.h"
#include "lfg_arch.h"

#include <vector>

struct lfg_model_saver {
    struct gguf_context * gguf_ctx = nullptr;
    const struct lfg_model & model;
    const struct LFG_KV lfg_kv_enum;

    lfg_model_saver(const struct lfg_model & model);
    ~lfg_model_saver();

    void add_kv(enum lfg_kv_enum key, uint32_t     value);
    void add_kv(enum lfg_kv_enum key, int32_t      value);
    void add_kv(enum lfg_kv_enum key, float        value);
    void add_kv(enum lfg_kv_enum key, bool         value);
    void add_kv(enum lfg_kv_enum key, const char * value);

    [[noreturn]]
    void add_kv(enum lfg_kv_enum key, char value); // needed to make the template below compile

    template <typename Container>
    void add_kv(enum lfg_kv_enum key, const Container & value, bool per_layer = false);

    void add_kv(enum lfg_kv_enum key, const std::vector<std::string> & value);

    void add_tensor(const struct ggml_tensor * tensor);

    void add_kv_from_model();

    void add_tensors_from_model();

    void save(const std::string & path_model);
};
