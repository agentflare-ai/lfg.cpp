#pragma once

#include "lfm_batch.h"
#include "lfm_graph.h"
#include "lfm_kv_cache_iswa.h"
#include "lfm_memory.h"
#include "lfm_memory_recurrent.h"

#include <memory>
#include <vector>

//
// lfm_memory_hybrid_iswa
//

// utilizes instances of lfm_memory_recurrent and lfm_kv_cache_iswa to
//   support models where each layer may be either attention-based (with SWA support) or recurrent

class lfm_memory_hybrid_iswa : public lfm_memory_i {
public:
    lfm_memory_hybrid_iswa(
        const lfm_model & model,
                            /* attn */
                ggml_type   type_k,
                ggml_type   type_v,
                     bool   v_trans,
                     bool   swa_full,
                 uint32_t   kv_size,
                 uint32_t   n_ubatch,
                 uint32_t   n_pad,
                            /* recurrent */
                ggml_type   type_r,
                ggml_type   type_s,
                 uint32_t   rs_size,
                            /* common */
                 uint32_t   n_seq_max,
                     bool   offload,
                     bool   unified,
                            /* layer filters */
    const layer_filter_cb & filter_attn = nullptr,
    const layer_filter_cb & filter_recr = nullptr);

    ~lfm_memory_hybrid_iswa() = default;

    //
    // lfm_memory_i
    //

    lfm_memory_context_ptr init_batch(
            lfm_batch_allocr & balloc,
            uint32_t n_ubatch,
            bool embd_all) override;

    lfm_memory_context_ptr init_full() override;

    lfm_memory_context_ptr init_update(lfm_context * lctx, bool optimize) override;

    bool get_can_shift() const override;

    void clear(bool data) override;

    bool seq_rm  (lfm_seq_id seq_id,                              lfm_pos p0, lfm_pos p1) override;
    void seq_cp  (lfm_seq_id seq_id_src, lfm_seq_id seq_id_dst, lfm_pos p0, lfm_pos p1) override;
    void seq_keep(lfm_seq_id seq_id)                                                          override;
    void seq_add (lfm_seq_id seq_id,                              lfm_pos p0, lfm_pos p1, lfm_pos shift) override;
    void seq_div (lfm_seq_id seq_id,                              lfm_pos p0, lfm_pos p1, int d) override;

    lfm_pos seq_pos_min(lfm_seq_id seq_id) const override;
    lfm_pos seq_pos_max(lfm_seq_id seq_id) const override;

    std::map<ggml_backend_buffer_type_t, size_t> memory_breakdown() const override;

    // state write/load

    void state_write(lfm_io_write_i & io, lfm_seq_id seq_id = -1, lfm_state_seq_flags flags = 0) const override;
    void state_read (lfm_io_read_i  & io, lfm_seq_id seq_id = -1, lfm_state_seq_flags flags = 0)       override;

    //
    // lfm_memory_hybrid_iswa specific API
    //

    lfm_kv_cache_iswa * get_mem_attn() const;
    lfm_memory_recurrent * get_mem_recr() const;

private:
    const lfm_hparams & hparams;

    const std::unique_ptr<lfm_kv_cache_iswa> mem_attn;
    const std::unique_ptr<lfm_memory_recurrent> mem_recr;
};

class lfm_memory_hybrid_iswa_context : public lfm_memory_context_i {
public:
    using slot_info_vec_t = lfm_kv_cache::slot_info_vec_t;

    // init failure
    explicit lfm_memory_hybrid_iswa_context(lfm_memory_status status);

    // init full
    explicit lfm_memory_hybrid_iswa_context(lfm_memory_hybrid_iswa * mem);

    // init update
    explicit lfm_memory_hybrid_iswa_context(
        lfm_memory_hybrid_iswa * mem,
                   lfm_context * lctx,
                            bool   optimize);

    // init success
    lfm_memory_hybrid_iswa_context(
           lfm_memory_hybrid_iswa * mem,
                    slot_info_vec_t   sinfos_base,
                    slot_info_vec_t   sinfos_swa,
          std::vector<lfm_ubatch>   ubatches);

    ~lfm_memory_hybrid_iswa_context() = default;

    bool next()  override;
    bool apply() override;

    lfm_memory_status  get_status() const override;
    const lfm_ubatch & get_ubatch() const override;

    //
    // lfm_memory_hybrid_iswa_context
    //

    const lfm_kv_cache_iswa_context * get_attn() const;
    const lfm_memory_recurrent_context * get_recr() const;

private:
    // the index of the next ubatch to process
    size_t i_next = 0;

    std::vector<lfm_ubatch> ubatches;

    const lfm_memory_context_ptr ctx_attn;
    const lfm_memory_context_ptr ctx_recr;

    const lfm_memory_status status;
};
