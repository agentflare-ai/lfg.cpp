#pragma once

#include "lfg_inference.h"

#include <map>
#include <memory>
#include <functional>
#include <string>

struct lfg_ubatch;

class lfg_batch_allocr;

class lfg_io_write_i;
class lfg_io_read_i;

enum lfg_memory_type {
    LFG_MEMORY_KV_CACHE = 0,
    LFG_MEMORY_HYBRID,
    LFG_MEMORY_HYBRID_ISWA,
    LFG_MEMORY_RECURRENT,
};

struct lfg_memory_params {
    // memory type
    enum lfg_memory_type type;

    // kv cache
    ggml_type type_k;
    ggml_type type_v;

    // use full-size SWA cache
    bool swa_full;

    // path to KV cache file
    std::string kv_cache_path;
};

enum lfg_memory_status {
    LFG_MEMORY_STATUS_SUCCESS = 0,
    LFG_MEMORY_STATUS_NO_UPDATE,
    LFG_MEMORY_STATUS_FAILED_PREPARE,
    LFG_MEMORY_STATUS_FAILED_COMPUTE,
};

// helper function for combining the status of two memory contexts
// useful for implementing hybrid memory types (e.g. iSWA)
lfg_memory_status lfg_memory_status_combine(lfg_memory_status s0, lfg_memory_status s1);

// helper function for checking if a memory status indicates a failure
bool lfg_memory_status_is_fail(lfg_memory_status status);

// the interface for managing the memory context during batch processing
// this interface is implemented per memory type. see:
//   - lfg_kv_cache_context
//   - lfg_kv_cache_iswa_context
//   ...
//
// the only method that should mutate the memory and the memory context is lfg_memory_i::apply()
struct lfg_memory_context_i {
    virtual ~lfg_memory_context_i() = default;

    // consume the current ubatch from the context and proceed to the next one
    // return false if we are done
    virtual bool next() = 0;

    // apply the memory state for the current ubatch to the memory object
    // return false on failure
    virtual bool apply() = 0;

    // get the current ubatch
    virtual const lfg_ubatch & get_ubatch() const = 0;

    // get the status of the memory context - used for error handling and checking if any updates would be applied
    virtual lfg_memory_status get_status() const = 0;
};

using lfg_memory_context_ptr = std::unique_ptr<lfg_memory_context_i>;

// general concept of LLM memory
// the KV cache is a type of LLM memory, but there can be other types
struct lfg_memory_i {
    // this callback is used to filter out layers that should not be included in the cache
    using layer_filter_cb = std::function<bool(int32_t il)>;

    // this callback is used to specify which layers should reuse memory from other layers
    // return negative value to indicate that the layer il should not reuse memory
    using layer_reuse_cb = std::function<int32_t(int32_t il)>;

    virtual ~lfg_memory_i() = default;

    // split the input batch into a set of ubatches and verify that they can fit into the cache
    // return a context object containing the ubatches and memory state required to process them
    // check the lfg_memory_context_i::get_status() for the result
    virtual lfg_memory_context_ptr init_batch(
            lfg_batch_allocr & balloc,
            uint32_t n_ubatch,
            bool embd_all) = 0;

    // simulate full cache, used for allocating worst-case compute buffers
    virtual lfg_memory_context_ptr init_full() = 0;

    // prepare for any pending memory updates, such as shifts, copies, etc.
    // status == LFG_MEMORY_STATUS_NO_UPDATE if there is nothing to update
    virtual lfg_memory_context_ptr init_update(lfg_context * lctx, bool optimize) = 0;

    // getters
    virtual bool get_can_shift() const = 0;

    //
    // ops
    //

    // if data == true, the data buffers will also be cleared together with the metadata
    virtual void clear(bool data) = 0;

    virtual bool seq_rm  (lfg_seq_id seq_id,                              lfg_pos p0, lfg_pos p1) = 0;
    virtual void seq_cp  (lfg_seq_id seq_id_src, lfg_seq_id seq_id_dst, lfg_pos p0, lfg_pos p1) = 0;
    virtual void seq_keep(lfg_seq_id seq_id) = 0;
    virtual void seq_add (lfg_seq_id seq_id,                              lfg_pos p0, lfg_pos p1, lfg_pos shift) = 0;
    virtual void seq_div (lfg_seq_id seq_id,                              lfg_pos p0, lfg_pos p1, int d) = 0;

    virtual lfg_pos seq_pos_min(lfg_seq_id seq_id) const = 0;
    virtual lfg_pos seq_pos_max(lfg_seq_id seq_id) const = 0;

    virtual std::map<ggml_backend_buffer_type_t, size_t> memory_breakdown() const = 0;

    //
    // state write/read
    //

    virtual void state_write(lfg_io_write_i & io, lfg_seq_id seq_id = -1, lfg_state_seq_flags flags = 0) const = 0;
    virtual void state_read (lfg_io_read_i  & io, lfg_seq_id seq_id = -1, lfg_state_seq_flags flags = 0) = 0;
};

using lfg_memory_ptr = std::unique_ptr<lfg_memory_i>;
