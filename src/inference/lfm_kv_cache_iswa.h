#pragma once

#include "lfm_kv_cache.h"

#include <vector>

//
// lfm_kv_cache_iswa
//

// utilizes two instances of lfm_kv_cache
//   the first instance is for the non-SWA layers of the model and the second instance is for the SWA layers

class lfm_kv_cache_iswa : public lfm_memory_i {
public:
    lfm_kv_cache_iswa(
            const lfm_model & model,
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

    ~lfm_kv_cache_iswa() = default;

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
    void state_read (lfm_io_read_i  & io, lfm_seq_id seq_id = -1, lfm_state_seq_flags flags = 0) override;

    //
    // lfm_kv_cache_iswa specific API
    //

    lfm_kv_cache * get_base() const;
    lfm_kv_cache * get_swa () const;

private:
    const lfm_hparams & hparams;

    const bool unified;

    std::unique_ptr<lfm_kv_cache> kv_base;
    std::unique_ptr<lfm_kv_cache> kv_swa;
};

class lfm_kv_cache_iswa_context : public lfm_memory_context_i {
public:
    using slot_info_vec_t = lfm_kv_cache::slot_info_vec_t;

    // used for errors
    lfm_kv_cache_iswa_context(lfm_memory_status status);

    // used to create a full-cache context
    lfm_kv_cache_iswa_context(
            lfm_kv_cache_iswa * kv);

    // used to create an update context
    lfm_kv_cache_iswa_context(
            lfm_kv_cache_iswa * kv,
            lfm_context * lctx,
            bool optimize);

    // used to create a batch processing context from a batch
    lfm_kv_cache_iswa_context(
            lfm_kv_cache_iswa * kv,
            slot_info_vec_t sinfos_base,
            slot_info_vec_t sinfos_swa,
            std::vector<lfm_ubatch> ubatches);

    virtual ~lfm_kv_cache_iswa_context();

    //
    // lfm_memory_context_i
    //

    bool next()  override;
    bool apply() override;

    lfm_memory_status  get_status() const override;
    const lfm_ubatch & get_ubatch() const override;

    //
    // lfm_kv_cache_iswa_context specific API
    //

    const lfm_kv_cache_context * get_base() const;
    const lfm_kv_cache_context * get_swa()  const;

private:
    //lfm_kv_cache_iswa * kv;

    // the index of the next ubatch to process
    size_t i_next = 0;

    std::vector<lfm_ubatch> ubatches;

    const lfm_memory_context_ptr ctx_base;
    const lfm_memory_context_ptr ctx_swa;

    const lfm_memory_status status;
};
