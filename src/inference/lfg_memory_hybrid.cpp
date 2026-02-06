#include "lfg_memory_hybrid.h"

#include "lfg_impl.h"
#include "lfg_model.h"
#include "lfg_context.h"

//
// lfg_memory_hybrid
//

lfg_memory_hybrid::lfg_memory_hybrid(
        const lfg_model & model,
                            /* attn */
                ggml_type   type_k,
                ggml_type   type_v,
                     bool   v_trans,
                 uint32_t   kv_size,
                 uint32_t   n_pad,
                 uint32_t   n_swa,
           lfg_swa_type   swa_type,
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
    mem_attn(new lfg_kv_cache(
        model,
        type_k,
        type_v,
        v_trans,
        offload,
        unified,
        kv_size,
        n_seq_max,
        n_pad,
        n_swa,
        swa_type,
        filter_attn == nullptr ?
            [&](int32_t il) { return !hparams.is_recurrent(il); }
            : filter_attn,
        nullptr
    )),
    mem_recr(new lfg_memory_recurrent(
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

lfg_memory_context_ptr lfg_memory_hybrid::init_batch(lfg_batch_allocr & balloc, uint32_t n_ubatch, bool embd_all) {
    do {
        balloc.split_reset();

        // follow the recurrent pattern for creating the ubatch splits
        std::vector<lfg_ubatch> ubatches;

        while (true) {
            lfg_ubatch ubatch;

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
            LFG_LOG_ERROR("%s: failed to prepare recurrent ubatches\n", __func__);
            return std::make_unique<lfg_memory_hybrid_context>(LFG_MEMORY_STATUS_FAILED_PREPARE);
        }

        // prepare the attention cache
        auto heads_attn = mem_attn->prepare(ubatches);
        if (heads_attn.empty()) {
            LFG_LOG_ERROR("%s: failed to prepare attention ubatches\n", __func__);
            return std::make_unique<lfg_memory_hybrid_context>(LFG_MEMORY_STATUS_FAILED_PREPARE);
        }

        return std::make_unique<lfg_memory_hybrid_context>(
                this, std::move(heads_attn), std::move(ubatches));
    } while(false);

    return std::make_unique<lfg_memory_hybrid_context>(LFG_MEMORY_STATUS_FAILED_PREPARE);
}

lfg_memory_context_ptr lfg_memory_hybrid::init_full() {
    return std::make_unique<lfg_memory_hybrid_context>(this);
}

lfg_memory_context_ptr lfg_memory_hybrid::init_update(lfg_context * lctx, bool optimize) {
    return std::make_unique<lfg_memory_hybrid_context>(this, lctx, optimize);
}

bool lfg_memory_hybrid::get_can_shift() const {
    // Shifting is trivially supported for recurrent
    return mem_attn->get_can_shift();
}

void lfg_memory_hybrid::clear(bool data) {
    mem_attn->clear(data);
    mem_recr->clear(data);
}

bool lfg_memory_hybrid::seq_rm(lfg_seq_id seq_id, lfg_pos p0, lfg_pos p1) {
    // Try removing from the recurrent cache first since it may fail. If it does
    // fail, the cache will not have been mutated.
    if (!mem_recr->seq_rm(seq_id, p0, p1)) {
        return false;
    }
    return mem_attn->seq_rm(seq_id, p0, p1);
}

void lfg_memory_hybrid::seq_cp(lfg_seq_id seq_id_src, lfg_seq_id seq_id_dst, lfg_pos p0, lfg_pos p1) {
    mem_attn->seq_cp(seq_id_src, seq_id_dst, p0, p1);
    mem_recr->seq_cp(seq_id_src, seq_id_dst, p0, p1);
}

void lfg_memory_hybrid::seq_keep(lfg_seq_id seq_id) {
    mem_attn->seq_keep(seq_id);
    mem_recr->seq_keep(seq_id);
}

void lfg_memory_hybrid::seq_add(lfg_seq_id seq_id, lfg_pos p0, lfg_pos p1, lfg_pos shift) {
    mem_attn->seq_add(seq_id, p0, p1, shift);
    mem_recr->seq_add(seq_id, p0, p1, shift);
}

void lfg_memory_hybrid::seq_div(lfg_seq_id seq_id, lfg_pos p0, lfg_pos p1, int d) {
    mem_attn->seq_div(seq_id, p0, p1, d);
    mem_recr->seq_div(seq_id, p0, p1, d);
}

lfg_pos lfg_memory_hybrid::seq_pos_min(lfg_seq_id seq_id) const {
    // the min of the total cache is the max of the two caches' min values
    return std::max(mem_attn->seq_pos_min(seq_id), mem_recr->seq_pos_min(seq_id));
}

lfg_pos lfg_memory_hybrid::seq_pos_max(lfg_seq_id seq_id) const {
    // the max of the total cache is the min of the two caches' max values
    return std::min(mem_attn->seq_pos_max(seq_id), mem_recr->seq_pos_max(seq_id));
}

std::map<ggml_backend_buffer_type_t, size_t> lfg_memory_hybrid::memory_breakdown() const {
    std::map<ggml_backend_buffer_type_t, size_t> mb = mem_attn->memory_breakdown();
    for (const auto & buft_size : mem_recr->memory_breakdown()) {
        mb[buft_size.first] += buft_size.second;
    }
    return mb;
}

void lfg_memory_hybrid::state_write(lfg_io_write_i & io, lfg_seq_id seq_id, lfg_state_seq_flags flags) const {
    if ((flags & LFG_STATE_SEQ_FLAGS_PARTIAL_ONLY) == 0) {
        mem_attn->state_write(io, seq_id, flags);
    }
    mem_recr->state_write(io, seq_id, flags);
}

void lfg_memory_hybrid::state_read(lfg_io_read_i & io, lfg_seq_id seq_id, lfg_state_seq_flags flags) {
    if ((flags & LFG_STATE_SEQ_FLAGS_PARTIAL_ONLY) == 0) {
        mem_attn->state_read(io, seq_id, flags);
    }
    mem_recr->state_read(io, seq_id, flags);
}

lfg_kv_cache * lfg_memory_hybrid::get_mem_attn() const {
    return mem_attn.get();
}

lfg_memory_recurrent * lfg_memory_hybrid::get_mem_recr() const {
    return mem_recr.get();
}

lfg_memory_hybrid_context::lfg_memory_hybrid_context(lfg_memory_status status) : status(status) {}

lfg_memory_hybrid_context::lfg_memory_hybrid_context(lfg_memory_hybrid * mem) :
    ctx_attn(mem->get_mem_attn()->init_full()),
    ctx_recr(mem->get_mem_recr()->init_full()),
    status(lfg_memory_status_combine(ctx_attn->get_status(), ctx_recr->get_status())) {
}

lfg_memory_hybrid_context::lfg_memory_hybrid_context(
        lfg_memory_hybrid * mem,
              lfg_context * lctx,
                       bool   optimize) :
    ctx_attn(mem->get_mem_attn()->init_update(lctx, optimize)),
    ctx_recr(mem->get_mem_recr()->init_update(lctx, optimize)),
    status(lfg_memory_status_combine(ctx_attn->get_status(), ctx_recr->get_status())) {
}

lfg_memory_hybrid_context::lfg_memory_hybrid_context(
              lfg_memory_hybrid * mem,
                  slot_info_vec_t   sinfos_attn,
        std::vector<lfg_ubatch>   ubatches) :
    ubatches(std::move(ubatches)),
    // note: here we copy the ubatches. not sure if this is ideal
    ctx_attn(new lfg_kv_cache_context(mem->get_mem_attn(), std::move(sinfos_attn), this->ubatches)),
    ctx_recr(new lfg_memory_recurrent_context(mem->get_mem_recr(), this->ubatches)),
    status(lfg_memory_status_combine(ctx_attn->get_status(), ctx_recr->get_status())) {
}

bool lfg_memory_hybrid_context::next() {
    assert(status == LFG_MEMORY_STATUS_SUCCESS);

    ctx_attn->next();
    ctx_recr->next();

    if (++i_next >= ubatches.size()) {
        return false;
    }

    return true;
}

bool lfg_memory_hybrid_context::apply() {
    assert(!lfg_memory_status_is_fail(status));

    bool res = true;

    res = res & ctx_attn->apply();
    res = res & ctx_recr->apply();

    return res;
}

lfg_memory_status lfg_memory_hybrid_context::get_status() const {
    return status;
}

const lfg_ubatch & lfg_memory_hybrid_context::get_ubatch() const {
    assert(status == LFG_MEMORY_STATUS_SUCCESS);
    return ubatches[i_next];
}

const lfg_kv_cache_context * lfg_memory_hybrid_context::get_attn() const {
    return static_cast<const lfg_kv_cache_context *>(ctx_attn.get());
}

const lfg_memory_recurrent_context * lfg_memory_hybrid_context::get_recr() const {
    return static_cast<const lfg_memory_recurrent_context *>(ctx_recr.get());
}
