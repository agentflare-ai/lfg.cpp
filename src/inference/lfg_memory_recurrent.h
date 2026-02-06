#pragma once

#include "lfg_batch.h"
#include "lfg_graph.h"
#include "lfg_memory.h"

#include <map>
#include <set>
#include <vector>

//
// lfg_memory_recurrent
//

// TODO: extract the cache state used for graph computation into lfg_memory_recurrent_context_i
//       see the implementation of lfg_kv_cache_context_i for an example how to do it
class lfg_memory_recurrent : public lfg_memory_i {
public:
    lfg_memory_recurrent(
            const lfg_model & model,
                    ggml_type   type_r,
                    ggml_type   type_s,
                         bool   offload,
                     uint32_t   mem_size,
                     uint32_t   n_seq_max,
        const layer_filter_cb & filter);

    ~lfg_memory_recurrent() = default;

    //
    // lfg_memory_i
    //

    lfg_memory_context_ptr init_batch(
            lfg_batch_allocr & balloc,
            uint32_t n_ubatch,
            bool embd_all) override;

    lfg_memory_context_ptr init_full() override;

    lfg_memory_context_ptr init_update(lfg_context * lctx, bool optimize) override;

    void clear(bool data) override;

    bool seq_rm  (lfg_seq_id seq_id,                              lfg_pos p0, lfg_pos p1) override;
    void seq_cp  (lfg_seq_id seq_id_src, lfg_seq_id seq_id_dst, lfg_pos p0, lfg_pos p1) override;
    void seq_keep(lfg_seq_id seq_id)                                                          override;
    void seq_add (lfg_seq_id seq_id,                              lfg_pos p0, lfg_pos p1, lfg_pos shift) override;
    void seq_div (lfg_seq_id seq_id,                              lfg_pos p0, lfg_pos p1, int d) override;

    lfg_pos seq_pos_min(lfg_seq_id seq_id) const override;
    lfg_pos seq_pos_max(lfg_seq_id seq_id) const override;

    std::map<ggml_backend_buffer_type_t, size_t> memory_breakdown() const override;

    bool prepare(const std::vector<lfg_ubatch> & ubatches);

    // find a contiguous slot of memory cells and emplace the ubatch there
    bool find_slot(const lfg_ubatch & ubatch);

    bool get_can_shift() const override;

    // state write/load

    void state_write(lfg_io_write_i & io, lfg_seq_id seq_id = -1, lfg_state_seq_flags flags = 0) const override;
    void state_read (lfg_io_read_i  & io, lfg_seq_id seq_id = -1, lfg_state_seq_flags flags = 0) override;

    uint32_t head = 0; // the location where the batch will be placed in the cache (see find_slot())
    uint32_t size = 0; // total number of cells, shared across all sequences
    uint32_t used = 0; // used cells (i.e. at least one seq_id)

    // computed before each graph build
    uint32_t n = 0;

    // first zero-ed state
    int32_t rs_z = -1;

    // TODO: optimize for recurrent state needs
    struct mem_cell {
        lfg_pos pos  = -1;
        int32_t   src  = -1; // used to know where states should be copied from
        int32_t   src0 = -1; // like src, but only used when setting the inputs (allowing to copy once)
        int32_t   tail = -1;

        std::set<lfg_seq_id> seq_id;

        bool has_seq_id(const lfg_seq_id & id) const {
            return seq_id.find(id) != seq_id.end();
        }

        bool is_empty() const {
            return seq_id.empty();
        }

        bool is_same_seq(const mem_cell & other) const {
            return seq_id == other.seq_id;
        }
    };

    std::vector<mem_cell> cells;

    // per layer
    std::vector<ggml_tensor *> r_l;
    std::vector<ggml_tensor *> s_l;

private:
    //const lfg_model & model;
    const lfg_hparams & hparams;

    const uint32_t n_seq_max = 1;

    // ggml contexts for the KV cache along with the allocated backend buffers:
    std::vector<std::pair<ggml_context_ptr, ggml_backend_buffer_ptr>> ctxs_bufs;

    size_t total_size() const;

    size_t size_r_bytes() const;
    size_t size_s_bytes() const;

    void state_write_meta(lfg_io_write_i & io, const std::vector<std::pair<uint32_t, uint32_t>> & cell_ranges, lfg_seq_id seq_id = -1) const;
    void state_write_data(lfg_io_write_i & io, const std::vector<std::pair<uint32_t, uint32_t>> & cell_ranges) const;

    bool state_read_meta(lfg_io_read_i & io, uint32_t cell_count, lfg_seq_id dest_seq_id = -1);
    bool state_read_data(lfg_io_read_i & io, uint32_t cell_count);
};

class lfg_memory_recurrent_context : public lfg_memory_context_i {
public:
    // used for errors
    lfg_memory_recurrent_context(lfg_memory_status status);

    // used to create a full-cache or update context
    lfg_memory_recurrent_context(
            lfg_memory_recurrent * mem);

    // used to create a batch processing context from a batch
    lfg_memory_recurrent_context(
            lfg_memory_recurrent * mem,
            std::vector<lfg_ubatch> ubatches);

    virtual ~lfg_memory_recurrent_context();

    //
    // lfg_memory_context_i
    //

    bool next()  override;
    bool apply() override;

    lfg_memory_status  get_status() const override;
    const lfg_ubatch & get_ubatch() const override;

    //
    // lfg_memory_recurrent_context specific API
    //

    uint32_t get_n_rs() const;
    uint32_t get_head() const;
    int32_t  get_rs_z() const;
    uint32_t get_size() const;

    ggml_tensor * get_r_l(int32_t il) const;
    ggml_tensor * get_s_l(int32_t il) const;

    int32_t s_copy(int i) const;

private:
    const lfg_memory_status status;

    lfg_memory_recurrent * mem;

    size_t i_next = 0;

    std::vector<lfg_ubatch> ubatches;

    //
    // data needed for building the compute graph for the current ubatch:
    // TODO: extract all the state like `head` and `n` here
    //

    const bool is_full = false;
};
