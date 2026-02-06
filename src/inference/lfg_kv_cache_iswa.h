#pragma once

#include "lfg_kv_cache.h"

#include <vector>

//
// lfg_kv_cache_iswa
//

// utilizes two instances of lfg_kv_cache
//   the first instance is for the non-SWA layers of the model and the second instance is for the SWA layers

class lfg_kv_cache_iswa : public lfg_memory_i {
public:
    lfg_kv_cache_iswa(
            const lfg_model & model,
                    ggml_type   type_k,
                    ggml_type   type_v,
                         bool   v_trans,
                         bool   offload,
                         bool   swa_full,
                         bool   unified,
                     uint32_t   kv_size,
                     uint32_t   n_seq_max,
                     uint32_t   n_ubatch,
                     uint32_t   n_pad,
        const layer_filter_cb & filter,
        const  layer_reuse_cb & reuse);

    ~lfg_kv_cache_iswa() = default;

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
    void state_read (lfg_io_read_i  & io, lfg_seq_id seq_id = -1, lfg_state_seq_flags flags = 0) override;

    //
    // lfg_kv_cache_iswa specific API
    //

    lfg_kv_cache * get_base() const;
    lfg_kv_cache * get_swa () const;

private:
    const lfg_hparams & hparams;

    const bool unified;

    std::unique_ptr<lfg_kv_cache> kv_base;
    std::unique_ptr<lfg_kv_cache> kv_swa;
};

class lfg_kv_cache_iswa_context : public lfg_memory_context_i {
public:
    using slot_info_vec_t = lfg_kv_cache::slot_info_vec_t;

    // used for errors
    lfg_kv_cache_iswa_context(lfg_memory_status status);

    // used to create a full-cache context
    lfg_kv_cache_iswa_context(
            lfg_kv_cache_iswa * kv);

    // used to create an update context
    lfg_kv_cache_iswa_context(
            lfg_kv_cache_iswa * kv,
            lfg_context * lctx,
            bool optimize);

    // used to create a batch processing context from a batch
    lfg_kv_cache_iswa_context(
            lfg_kv_cache_iswa * kv,
            slot_info_vec_t sinfos_base,
            slot_info_vec_t sinfos_swa,
            std::vector<lfg_ubatch> ubatches);

    virtual ~lfg_kv_cache_iswa_context();

    //
    // lfg_memory_context_i
    //

    bool next()  override;
    bool apply() override;

    lfg_memory_status  get_status() const override;
    const lfg_ubatch & get_ubatch() const override;

    //
    // lfg_kv_cache_iswa_context specific API
    //

    const lfg_kv_cache_context * get_base() const;
    const lfg_kv_cache_context * get_swa()  const;

private:
    //lfg_kv_cache_iswa * kv;

    // the index of the next ubatch to process
    size_t i_next = 0;

    std::vector<lfg_ubatch> ubatches;

    const lfg_memory_context_ptr ctx_base;
    const lfg_memory_context_ptr ctx_swa;

    const lfg_memory_status status;
};
