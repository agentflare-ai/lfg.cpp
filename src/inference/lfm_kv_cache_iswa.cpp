#include "lfm_kv_cache_iswa.h"

#include "lfm_impl.h"
#include "lfm_batch.h"
#include "lfm_model.h"

#include <algorithm>
#include <cassert>

//
// lfm_kv_cache_iswa
//

lfm_kv_cache_iswa::lfm_kv_cache_iswa(
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
    const  layer_reuse_cb & reuse) : hparams(model.hparams), unified(unified) {

    // chain filters
    const layer_filter_cb filter_base = [&](int32_t il) {
        if (filter && !filter(il)) {
            return false;
        }

        return !model.hparams.is_swa(il);
    };

    const layer_filter_cb filter_swa  = [&](int32_t il) {
        if (filter && !filter(il)) {
            return false;
        }

        return  model.hparams.is_swa(il);
    };

    const uint32_t size_base = kv_size;

    // note: the SWA cache is always padded to 256 for performance
    //       https://github.com/ggml-org/liquid.cpp/issues/17037
    uint32_t size_swa = GGML_PAD(std::min(size_base, hparams.n_swa*(unified ? n_seq_max : 1) + n_ubatch), 256);

    // when using full-size SWA cache, we set the SWA cache size to be equal to the base cache size
    if (swa_full) {
        LFM_LOG_WARN("%s: using full-size SWA cache (ref: %s)\n",
                __func__, "https://github.com/ggml-org/liquid.cpp/pull/13194#issuecomment-2868343055");

        size_swa = size_base;
    }

    LFM_LOG_INFO("%s: creating non-SWA KV cache, size = %u cells\n", __func__, size_base);

    kv_base = std::make_unique<lfm_kv_cache>(
            model, type_k, type_v,
            v_trans, offload, unified, size_base, n_seq_max, n_pad,
            0, LFM_SWA_TYPE_NONE, filter_base, reuse);

    LFM_LOG_INFO("%s: creating     SWA KV cache, size = %u cells\n", __func__, size_swa);

    kv_swa = std::make_unique<lfm_kv_cache>(
            model, type_k, type_v,
            v_trans, offload, unified, size_swa, n_seq_max, n_pad,
            hparams.n_swa, hparams.swa_type, filter_swa, reuse);
}

void lfm_kv_cache_iswa::clear(bool data) {
    kv_base->clear(data);
    kv_swa ->clear(data);
}

bool lfm_kv_cache_iswa::seq_rm(lfm_seq_id seq_id, lfm_pos p0, lfm_pos p1) {
    bool res = true;

    res = res & kv_base->seq_rm(seq_id, p0, p1);
    res = res & kv_swa ->seq_rm(seq_id, p0, p1);

    return res;
}

void lfm_kv_cache_iswa::seq_cp(lfm_seq_id seq_id_src, lfm_seq_id seq_id_dst, lfm_pos p0, lfm_pos p1) {
    kv_base->seq_cp(seq_id_src, seq_id_dst, p0, p1);
    kv_swa ->seq_cp(seq_id_src, seq_id_dst, p0, p1);
}

void lfm_kv_cache_iswa::seq_keep(lfm_seq_id seq_id) {
    kv_base->seq_keep(seq_id);
    kv_swa ->seq_keep(seq_id);
}

void lfm_kv_cache_iswa::seq_add(lfm_seq_id seq_id, lfm_pos p0, lfm_pos p1, lfm_pos shift) {
    kv_base->seq_add(seq_id, p0, p1, shift);
    kv_swa ->seq_add(seq_id, p0, p1, shift);
}

void lfm_kv_cache_iswa::seq_div(lfm_seq_id seq_id, lfm_pos p0, lfm_pos p1, int d) {
    kv_base->seq_div(seq_id, p0, p1, d);
    kv_swa ->seq_div(seq_id, p0, p1, d);
}

lfm_pos lfm_kv_cache_iswa::seq_pos_min(lfm_seq_id seq_id) const {
    // the base cache is a superset of the SWA cache, so we can just check the SWA cache
    return kv_swa->seq_pos_min(seq_id);
}

lfm_pos lfm_kv_cache_iswa::seq_pos_max(lfm_seq_id seq_id) const {
    return kv_swa->seq_pos_max(seq_id);
}

std::map<ggml_backend_buffer_type_t, size_t> lfm_kv_cache_iswa::memory_breakdown() const {
    std::map<ggml_backend_buffer_type_t, size_t> mb = kv_base->memory_breakdown();
    for (const auto & buft_size : kv_swa->memory_breakdown()) {
        mb[buft_size.first] += buft_size.second;
    }
    return mb;
}

lfm_memory_context_ptr lfm_kv_cache_iswa::init_batch(lfm_batch_allocr & balloc, uint32_t n_ubatch, bool embd_all) {
    GGML_UNUSED(embd_all);

    // first try simple split
    do {
        if (!unified) {
            // requires equal splits, so we skip the simple split
            break;
        }

        balloc.split_reset();

        std::vector<lfm_ubatch> ubatches;
        while (true) {
            auto ubatch = balloc.split_simple(n_ubatch);

            if (ubatch.n_tokens == 0) {
                break;
            }

            ubatches.push_back(std::move(ubatch)); // NOLINT
        }

        if (balloc.get_n_used() < balloc.get_n_tokens()) {
            // failed to find a suitable split
            break;
        }

        auto sinfos_base = kv_base->prepare(ubatches);
        if (sinfos_base.empty()) {
            break;
        }

        auto sinfos_swa = kv_swa->prepare(ubatches);
        if (sinfos_swa.empty()) {
            break;
        }

        assert(sinfos_base.size() == sinfos_swa.size());

        return std::make_unique<lfm_kv_cache_iswa_context>(
                this, std::move(sinfos_base), std::move(sinfos_swa), std::move(ubatches));
    } while (false);

    // if it fails, try equal split
    do {
        balloc.split_reset();

        std::vector<lfm_ubatch> ubatches;
        while (true) {
            auto ubatch = balloc.split_equal(n_ubatch, !unified);

            if (ubatch.n_tokens == 0) {
                break;
            }

            ubatches.push_back(std::move(ubatch)); // NOLINT
        }

        if (balloc.get_n_used() < balloc.get_n_tokens()) {
            // failed to find a suitable split
            break;
        }

        auto sinfos_base = kv_base->prepare(ubatches);
        if (sinfos_base.empty()) {
            break;
        }

        auto sinfos_swa = kv_swa->prepare(ubatches);
        if (sinfos_swa.empty()) {
            break;
        }

        assert(sinfos_base.size() == sinfos_swa.size());

        return std::make_unique<lfm_kv_cache_iswa_context>(
                this, std::move(sinfos_base), std::move(sinfos_swa), std::move(ubatches));
    } while (false);

    // TODO: if we fail again, we should attempt different splitting strategies
    //       but to do that properly, we first have to refactor the batches to be more flexible

    return std::make_unique<lfm_kv_cache_iswa_context>(LFM_MEMORY_STATUS_FAILED_PREPARE);
}

lfm_memory_context_ptr lfm_kv_cache_iswa::init_full() {
    return std::make_unique<lfm_kv_cache_iswa_context>(this);
}

lfm_memory_context_ptr lfm_kv_cache_iswa::init_update(lfm_context * lctx, bool optimize) {
    return std::make_unique<lfm_kv_cache_iswa_context>(this, lctx, optimize);
}

bool lfm_kv_cache_iswa::get_can_shift() const {
    return kv_base->get_size() == kv_swa->get_size();
}

void lfm_kv_cache_iswa::state_write(lfm_io_write_i & io, lfm_seq_id seq_id, lfm_state_seq_flags flags) const {
    if ((flags & LFM_STATE_SEQ_FLAGS_PARTIAL_ONLY) == 0) {
        kv_base->state_write(io, seq_id, flags);
    }

    kv_swa->state_write(io, seq_id, flags);
}

void lfm_kv_cache_iswa::state_read(lfm_io_read_i & io, lfm_seq_id seq_id, lfm_state_seq_flags flags) {
    if ((flags & LFM_STATE_SEQ_FLAGS_PARTIAL_ONLY) == 0) {
        kv_base->state_read(io, seq_id, flags);
    }

    kv_swa->state_read(io, seq_id, flags);
}

lfm_kv_cache * lfm_kv_cache_iswa::get_base() const {
    return kv_base.get();
}

lfm_kv_cache * lfm_kv_cache_iswa::get_swa() const {
    return kv_swa.get();
}

//
// lfm_kv_cache_iswa_context
//

lfm_kv_cache_iswa_context::lfm_kv_cache_iswa_context(lfm_memory_status status) : status(status) {}

lfm_kv_cache_iswa_context::lfm_kv_cache_iswa_context(
        lfm_kv_cache_iswa * kv) :
    ctx_base(kv->get_base()->init_full()),
    ctx_swa (kv->get_swa ()->init_full()),
    status(lfm_memory_status_combine(ctx_base->get_status(), ctx_swa->get_status())) {
}

lfm_kv_cache_iswa_context::lfm_kv_cache_iswa_context(
        lfm_kv_cache_iswa * kv,
        lfm_context * lctx,
        bool optimize) :
    ctx_base(kv->get_base()->init_update(lctx, optimize)),
    ctx_swa (kv->get_swa ()->init_update(lctx, optimize)),
    status(lfm_memory_status_combine(ctx_base->get_status(), ctx_swa->get_status())) {
}

lfm_kv_cache_iswa_context::lfm_kv_cache_iswa_context(
        lfm_kv_cache_iswa * kv,
        slot_info_vec_t sinfos_base,
        slot_info_vec_t sinfos_swa,
        std::vector<lfm_ubatch> ubatches) :
    ubatches(std::move(ubatches)),
    // note: here we copy the ubatches. not sure if this is ideal
    ctx_base(new lfm_kv_cache_context(kv->get_base(), std::move(sinfos_base), this->ubatches)),
    ctx_swa (new lfm_kv_cache_context(kv->get_swa (), std::move(sinfos_swa),  this->ubatches)),
    status(lfm_memory_status_combine(ctx_base->get_status(), ctx_swa->get_status())) {
}

lfm_kv_cache_iswa_context:: ~lfm_kv_cache_iswa_context() = default;

bool lfm_kv_cache_iswa_context::next() {
    assert(status == LFM_MEMORY_STATUS_SUCCESS);

    ctx_base->next();
    ctx_swa ->next();

    if (++i_next >= ubatches.size()) {
        return false;
    }

    return true;
}

bool lfm_kv_cache_iswa_context::apply() {
    assert(!lfm_memory_status_is_fail(status));

    bool res = true;

    res = res & ctx_base->apply();
    res = res & ctx_swa ->apply();

    return res;
}

lfm_memory_status lfm_kv_cache_iswa_context::get_status() const {
    return status;
}

const lfm_ubatch & lfm_kv_cache_iswa_context::get_ubatch() const {
    assert(status == LFM_MEMORY_STATUS_SUCCESS);

    return ubatches[i_next];
}

const lfm_kv_cache_context * lfm_kv_cache_iswa_context::get_base() const {
    assert(status == LFM_MEMORY_STATUS_SUCCESS);

    return static_cast<const lfm_kv_cache_context *>(ctx_base.get());
}

const lfm_kv_cache_context * lfm_kv_cache_iswa_context::get_swa()  const {
    assert(status == LFM_MEMORY_STATUS_SUCCESS);

    return static_cast<const lfm_kv_cache_context *>(ctx_swa.get());
}
