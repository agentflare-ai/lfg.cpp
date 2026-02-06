#include "lfm_memory_hybrid_iswa.h"

#include "lfm_impl.h"
#include "lfm_model.h"
#include "lfm_context.h"

//
// lfm_memory_hybrid_iswa
//

lfm_memory_hybrid_iswa::lfm_memory_hybrid_iswa(
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
    const layer_filter_cb & filter_attn,
    const layer_filter_cb & filter_recr) :
    hparams(model.hparams),
    mem_attn(new lfm_kv_cache_iswa(
        model,
        type_k,
        type_v,
        v_trans,
        offload,
        swa_full,
        unified,
        kv_size,
        n_seq_max,
        n_ubatch,
        n_pad,
        filter_attn == nullptr ?
            [&](int32_t il) { return !hparams.is_recurrent(il); }
            : filter_attn,
        nullptr
    )),
    mem_recr(new lfm_memory_recurrent(
        model,
        type_r,
        type_s,
        offload,
        rs_size,
        n_seq_max,
        filter_recr == nullptr ?
            [&](int32_t il) { return hparams.is_recurrent(il); }
            : filter_recr
    )) {}

lfm_memory_context_ptr lfm_memory_hybrid_iswa::init_batch(lfm_batch_allocr & balloc, uint32_t n_ubatch, bool embd_all) {
    do {
        balloc.split_reset();

        // follow the recurrent pattern for creating the ubatch splits
        std::vector<lfm_ubatch> ubatches;

        while (true) {
            lfm_ubatch ubatch;

            if (embd_all) {
                // if all tokens are output, split by sequence
                ubatch = balloc.split_seq(n_ubatch);
            } else {
                // TODO: non-sequential equal split can be done if using unified KV cache
                //       for simplicity, we always use sequential equal split for now
                ubatch = balloc.split_equal(n_ubatch, true);
            }

            if (ubatch.n_tokens == 0) {
                break;
            }

            ubatches.push_back(std::move(ubatch)); // NOLINT
        }

        if (balloc.get_n_used() < balloc.get_n_tokens()) {
            // failed to find a suitable split
            break;
        }

        // prepare the recurrent batches first
        if (!mem_recr->prepare(ubatches)) {
            // TODO: will the recurrent cache be in an undefined context at this point?
            LFM_LOG_ERROR("%s: failed to prepare recurrent ubatches\n", __func__);
            return std::make_unique<lfm_memory_hybrid_iswa_context>(LFM_MEMORY_STATUS_FAILED_PREPARE);
        }

        // prepare the attention cache (iswa version returns both base and swa slot infos)
        auto sinfos_base = mem_attn->get_base()->prepare(ubatches);
        if (sinfos_base.empty()) {
            LFM_LOG_ERROR("%s: failed to prepare attention base ubatches\n", __func__);
            return std::make_unique<lfm_memory_hybrid_iswa_context>(LFM_MEMORY_STATUS_FAILED_PREPARE);
        }

        auto sinfos_swa = mem_attn->get_swa()->prepare(ubatches);
        if (sinfos_swa.empty()) {
            LFM_LOG_ERROR("%s: failed to prepare attention swa ubatches\n", __func__);
            return std::make_unique<lfm_memory_hybrid_iswa_context>(LFM_MEMORY_STATUS_FAILED_PREPARE);
        }

        return std::make_unique<lfm_memory_hybrid_iswa_context>(
                this, std::move(sinfos_base), std::move(sinfos_swa), std::move(ubatches));
    } while(false);

    return std::make_unique<lfm_memory_hybrid_iswa_context>(LFM_MEMORY_STATUS_FAILED_PREPARE);
}

lfm_memory_context_ptr lfm_memory_hybrid_iswa::init_full() {
    return std::make_unique<lfm_memory_hybrid_iswa_context>(this);
}

lfm_memory_context_ptr lfm_memory_hybrid_iswa::init_update(lfm_context * lctx, bool optimize) {
    return std::make_unique<lfm_memory_hybrid_iswa_context>(this, lctx, optimize);
}

bool lfm_memory_hybrid_iswa::get_can_shift() const {
    // Shifting is trivially supported for recurrent
    return mem_attn->get_can_shift();
}

void lfm_memory_hybrid_iswa::clear(bool data) {
    mem_attn->clear(data);
    mem_recr->clear(data);
}

bool lfm_memory_hybrid_iswa::seq_rm(lfm_seq_id seq_id, lfm_pos p0, lfm_pos p1) {
    // Try removing from the recurrent cache first since it may fail. If it does
    // fail, the cache will not have been mutated.
    if (!mem_recr->seq_rm(seq_id, p0, p1)) {
        return false;
    }
    return mem_attn->seq_rm(seq_id, p0, p1);
}

void lfm_memory_hybrid_iswa::seq_cp(lfm_seq_id seq_id_src, lfm_seq_id seq_id_dst, lfm_pos p0, lfm_pos p1) {
    mem_attn->seq_cp(seq_id_src, seq_id_dst, p0, p1);
    mem_recr->seq_cp(seq_id_src, seq_id_dst, p0, p1);
}

void lfm_memory_hybrid_iswa::seq_keep(lfm_seq_id seq_id) {
    mem_attn->seq_keep(seq_id);
    mem_recr->seq_keep(seq_id);
}

void lfm_memory_hybrid_iswa::seq_add(lfm_seq_id seq_id, lfm_pos p0, lfm_pos p1, lfm_pos shift) {
    mem_attn->seq_add(seq_id, p0, p1, shift);
    mem_recr->seq_add(seq_id, p0, p1, shift);
}

void lfm_memory_hybrid_iswa::seq_div(lfm_seq_id seq_id, lfm_pos p0, lfm_pos p1, int d) {
    mem_attn->seq_div(seq_id, p0, p1, d);
    mem_recr->seq_div(seq_id, p0, p1, d);
}

lfm_pos lfm_memory_hybrid_iswa::seq_pos_min(lfm_seq_id seq_id) const {
    // the min of the total cache is the max of the two caches' min values
    return std::max(mem_attn->seq_pos_min(seq_id), mem_recr->seq_pos_min(seq_id));
}

lfm_pos lfm_memory_hybrid_iswa::seq_pos_max(lfm_seq_id seq_id) const {
    // the max of the total cache is the min of the two caches' max values
    return std::min(mem_attn->seq_pos_max(seq_id), mem_recr->seq_pos_max(seq_id));
}

std::map<ggml_backend_buffer_type_t, size_t> lfm_memory_hybrid_iswa::memory_breakdown() const {
    std::map<ggml_backend_buffer_type_t, size_t> mb = mem_attn->memory_breakdown();
    for (const auto & buft_size : mem_recr->memory_breakdown()) {
        mb[buft_size.first] += buft_size.second;
    }
    return mb;
}

void lfm_memory_hybrid_iswa::state_write(lfm_io_write_i & io, lfm_seq_id seq_id, lfm_state_seq_flags flags) const {
    mem_attn->state_write(io, seq_id, flags);
    mem_recr->state_write(io, seq_id, flags);
}

void lfm_memory_hybrid_iswa::state_read(lfm_io_read_i & io, lfm_seq_id seq_id, lfm_state_seq_flags flags) {
    mem_attn->state_read(io, seq_id, flags);
    mem_recr->state_read(io, seq_id, flags);
}

lfm_kv_cache_iswa * lfm_memory_hybrid_iswa::get_mem_attn() const {
    return mem_attn.get();
}

lfm_memory_recurrent * lfm_memory_hybrid_iswa::get_mem_recr() const {
    return mem_recr.get();
}

//
// lfm_memory_hybrid_iswa_context
//

lfm_memory_hybrid_iswa_context::lfm_memory_hybrid_iswa_context(lfm_memory_status status) : status(status) {}

lfm_memory_hybrid_iswa_context::lfm_memory_hybrid_iswa_context(lfm_memory_hybrid_iswa * mem) :
    ctx_attn(mem->get_mem_attn()->init_full()),
    ctx_recr(mem->get_mem_recr()->init_full()),
    status(lfm_memory_status_combine(ctx_attn->get_status(), ctx_recr->get_status())) {
}

lfm_memory_hybrid_iswa_context::lfm_memory_hybrid_iswa_context(
        lfm_memory_hybrid_iswa * mem,
                   lfm_context * lctx,
                            bool   optimize) :
    ctx_attn(mem->get_mem_attn()->init_update(lctx, optimize)),
    ctx_recr(mem->get_mem_recr()->init_update(lctx, optimize)),
    status(lfm_memory_status_combine(ctx_attn->get_status(), ctx_recr->get_status())) {
}

lfm_memory_hybrid_iswa_context::lfm_memory_hybrid_iswa_context(
           lfm_memory_hybrid_iswa * mem,
                    slot_info_vec_t   sinfos_base,
                    slot_info_vec_t   sinfos_swa,
          std::vector<lfm_ubatch>   ubatches) :
    ubatches(std::move(ubatches)),
    // note: here we copy the ubatches. not sure if this is ideal
    ctx_attn(new lfm_kv_cache_iswa_context(mem->get_mem_attn(), std::move(sinfos_base), std::move(sinfos_swa), this->ubatches)),
    ctx_recr(new lfm_memory_recurrent_context(mem->get_mem_recr(), this->ubatches)),
    status(lfm_memory_status_combine(ctx_attn->get_status(), ctx_recr->get_status())) {
}

bool lfm_memory_hybrid_iswa_context::next() {
    assert(status == LFM_MEMORY_STATUS_SUCCESS);

    ctx_attn->next();
    ctx_recr->next();

    if (++i_next >= ubatches.size()) {
        return false;
    }

    return true;
}

bool lfm_memory_hybrid_iswa_context::apply() {
    assert(!lfm_memory_status_is_fail(status));

    bool res = true;

    res = res & ctx_attn->apply();
    res = res & ctx_recr->apply();

    return res;
}

lfm_memory_status lfm_memory_hybrid_iswa_context::get_status() const {
    return status;
}

const lfm_ubatch & lfm_memory_hybrid_iswa_context::get_ubatch() const {
    assert(status == LFM_MEMORY_STATUS_SUCCESS);
    return ubatches[i_next];
}

const lfm_kv_cache_iswa_context * lfm_memory_hybrid_iswa_context::get_attn() const {
    return static_cast<const lfm_kv_cache_iswa_context *>(ctx_attn.get());
}

const lfm_memory_recurrent_context * lfm_memory_hybrid_iswa_context::get_recr() const {
    return static_cast<const lfm_memory_recurrent_context *>(ctx_recr.get());
}
