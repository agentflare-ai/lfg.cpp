#pragma once

#include "lfg_batch.h"
#include "lfg_graph.h"
#include "lfg_kv_cache_iswa.h"
#include "lfg_memory.h"
#include "lfg_memory_recurrent.h"

#include <memory>
#include <vector>

//
// lfg_memory_hybrid_iswa
//

// utilizes instances of lfg_memory_recurrent and lfg_kv_cache_iswa to
//   support models where each layer may be either attention-based (with SWA support) or recurrent

class lfg_memory_hybrid_iswa : public lfg_memory_i {
public:
    lfg_memory_hybrid_iswa(
        const lfg_model & model,
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

    ~lfg_memory_hybrid_iswa() = default;

    //
    // lfg_memory_i
    //

    lfg_memory_context_ptr init_batch(
            lfg_batch_allocr & balloc,
            uint32_t n_ubatch,
            bool embd_all) override;

    lfg_memory_context_ptr init_full() override;

    lfg_memory_context_ptr init_update(lfg_context * lctx, bool optimize) override;

    bool get_can_shift() const override;

    void clear(bool data) override;

    bool seq_rm  (lfg_seq_id seq_id,                              lfg_pos p0, lfg_pos p1) override;
    void seq_cp  (lfg_seq_id seq_id_src, lfg_seq_id seq_id_dst, lfg_pos p0, lfg_pos p1) override;
    void seq_keep(lfg_seq_id seq_id)                                                          override;
    void seq_add (lfg_seq_id seq_id,                              lfg_pos p0, lfg_pos p1, lfg_pos shift) override;
    void seq_div (lfg_seq_id seq_id,                              lfg_pos p0, lfg_pos p1, int d) override;

    lfg_pos seq_pos_min(lfg_seq_id seq_id) const override;
    lfg_pos seq_pos_max(lfg_seq_id seq_id) const override;

    std::map<ggml_backend_buffer_type_t, size_t> memory_breakdown() const override;

    // state write/load

    void state_write(lfg_io_write_i & io, lfg_seq_id seq_id = -1, lfg_state_seq_flags flags = 0) const override;
    void state_read (lfg_io_read_i  & io, lfg_seq_id seq_id = -1, lfg_state_seq_flags flags = 0)       override;

    //
    // lfg_memory_hybrid_iswa specific API
    //

    lfg_kv_cache_iswa * get_mem_attn() const;
    lfg_memory_recurrent * get_mem_recr() const;

private:
    const lfg_hparams & hparams;

    const std::unique_ptr<lfg_kv_cache_iswa> mem_attn;
    const std::unique_ptr<lfg_memory_recurrent> mem_recr;
};

class lfg_memory_hybrid_iswa_context : public lfg_memory_context_i {
public:
    using slot_info_vec_t = lfg_kv_cache::slot_info_vec_t;

    // init failure
    explicit lfg_memory_hybrid_iswa_context(lfg_memory_status status);

    // init full
    explicit lfg_memory_hybrid_iswa_context(lfg_memory_hybrid_iswa * mem);

    // init update
    explicit lfg_memory_hybrid_iswa_context(
        lfg_memory_hybrid_iswa * mem,
                   lfg_context * lctx,
                            bool   optimize);

    // init success
    lfg_memory_hybrid_iswa_context(
           lfg_memory_hybrid_iswa * mem,
                    slot_info_vec_t   sinfos_base,
                    slot_info_vec_t   sinfos_swa,
          std::vector<lfg_ubatch>   ubatches);

    ~lfg_memory_hybrid_iswa_context() = default;

    bool next()  override;
    bool apply() override;

    lfg_memory_status  get_status() const override;
    const lfg_ubatch & get_ubatch() const override;

    //
    // lfg_memory_hybrid_iswa_context
    //

    const lfg_kv_cache_iswa_context * get_attn() const;
    const lfg_memory_recurrent_context * get_recr() const;

private:
    // the index of the next ubatch to process
    size_t i_next = 0;

    std::vector<lfg_ubatch> ubatches;

    const lfg_memory_context_ptr ctx_attn;
    const lfg_memory_context_ptr ctx_recr;

    const lfg_memory_status status;
};
