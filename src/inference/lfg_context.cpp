#include "lfg_context.h"

#include "lfg_arch.h"
#include "lfg_impl.h"
#include "lfg_batch.h"
#include "lfg_io.h"
#include "lfg_memory.h"
#include "lfg_mmap.h"
#include "lfg_model.h"

#include <cinttypes>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>

namespace {
inline bool lfg_require_ctx(const lfg_context * ctx, const char * fn) {
    return lfg_check_ptr(ctx, LFG_ERROR_INVALID_ARGUMENT, fn, "ctx");
}
}

//
// lfg_context
//

lfg_context::lfg_context(
        const lfg_model & model,
              lfg_context_params params) :
    model(model),
    balloc(std::make_unique<lfg_batch_allocr>(model.hparams.n_pos_per_embd())) {
    //     may need to be backend-dependent
    LFG_LOG_INFO("%s: constructing lfg_context\n", __func__);

    t_start_us = model.t_start_us;
    t_load_us  = model.t_load_us;

    const auto & hparams = model.hparams;

    cparams.n_seq_max = std::max(1u, params.n_seq_max);
    if (cparams.n_seq_max > LFG_MAX_SEQ) {
        throw std::runtime_error("n_seq_max must be <= " + std::to_string(LFG_MAX_SEQ));
    }

    cparams.n_threads        = params.n_threads;
    cparams.n_threads_batch  = params.n_threads_batch;
    cparams.yarn_ext_factor  = params.yarn_ext_factor  >= 0.0f ? params.yarn_ext_factor  : hparams.yarn_ext_factor;
    cparams.yarn_attn_factor = params.yarn_attn_factor >= 0.0f ? params.yarn_attn_factor : hparams.yarn_attn_factor;
    cparams.yarn_beta_fast   = params.yarn_beta_fast   >= 0.0f ? params.yarn_beta_fast   : hparams.yarn_beta_fast;
    cparams.yarn_beta_slow   = params.yarn_beta_slow   >= 0.0f ? params.yarn_beta_slow   : hparams.yarn_beta_slow;
    cparams.embeddings       = params.embeddings;
    cparams.offload_kqv      = params.offload_kqv;
    cparams.no_perf          = params.no_perf;
    cparams.pooling_type     = params.pooling_type;
    cparams.warmup           = false;

    cparams.n_ctx            = params.n_ctx           == 0    ? hparams.n_ctx_train           : params.n_ctx;
    cparams.rope_freq_base   = params.rope_freq_base  == 0.0f ? hparams.rope_freq_base_train  : params.rope_freq_base;
    cparams.rope_freq_scale  = params.rope_freq_scale == 0.0f ? hparams.rope_freq_scale_train : params.rope_freq_scale;

    cparams.n_ctx_orig_yarn  = params.yarn_orig_ctx    != 0 ? params.yarn_orig_ctx    :
                               hparams.n_ctx_orig_yarn != 0 ? hparams.n_ctx_orig_yarn :
                                                              hparams.n_ctx_train;

    cparams.cb_eval           = params.cb_eval;
    cparams.cb_eval_user_data = params.cb_eval_user_data;

    // Initialize backend samplers here so they are part of the sampling graph
    // before the reserve passes run later in this function. This avoids a later
    // re-reserve when graph nodes change.
    if (params.samplers != nullptr && params.n_samplers > 0) {
        for (size_t i = 0; i < params.n_samplers; ++i) {
            const auto & config = params.samplers[i];

            if (lfg_sampler_chain_get(config.sampler, -1) == nullptr) {
                throw std::runtime_error("the backend samplers must be of type lfg_sampler_chain");
            }

            if (set_sampler(config.seq_id, config.sampler)) {
                const int n_samplers = lfg_sampler_chain_n(config.sampler);

                LFG_LOG_INFO("%s: setting backend sampler for seq_id %d (n = %d)\n", __func__, config.seq_id, n_samplers);
            }
        }
    }

    auto rope_scaling_type = params.rope_scaling_type;
    if (rope_scaling_type == LFG_ROPE_SCALING_TYPE_UNSPECIFIED) {
        rope_scaling_type = hparams.rope_scaling_type_train;
    }

    if (rope_scaling_type == LFG_ROPE_SCALING_TYPE_NONE) {
        cparams.rope_freq_scale = 1.0f; // never scale if scaling type is none
    }

    if (cparams.yarn_ext_factor < 0.0f) { // negative indicates 'not set'
        cparams.yarn_ext_factor = rope_scaling_type == LFG_ROPE_SCALING_TYPE_YARN ? 1.0f : 0.0f;
    }

    if (cparams.yarn_ext_factor != 0) {
        static auto get_mscale = [](float scale, float mscale) {
            return scale <= 1.0f ? 1.0f : (0.1f * mscale * logf(scale) + 1.0f);
        };

        const float factor = 1.0f / cparams.rope_freq_scale;

        // ref: https://github.com/huggingface/transformers/blob/6d00f6b0a5679c36510f203e4226e36f517c3032/src/transformers/modeling_rope_utils.py#L336-L348
        if (hparams.rope_yarn_log_mul != 0.0f) {
            // note: here we assume `mscale == 1.0f`
            // TODO: start reading the actual value of mscale and handle the case where it is not 1.0f
                  float mscale          = 1.0f;
            const float mscale_all_dims = hparams.rope_yarn_log_mul;

            // [TAG_DEEPSEEK2_YARN_LOG_MUL_FIX]
            // special-case DEEPSEEK v2:
            // https://huggingface.co/deepseek-ai/DeepSeek-V2-Lite-Chat/blob/main/config.json#L42-L43
            if (false && mscale_all_dims != 1.0f) {
                mscale = mscale_all_dims;
            }

            cparams.yarn_attn_factor = get_mscale(factor, mscale) / get_mscale(factor, mscale_all_dims);

            LFG_LOG_WARN("%s: setting new yarn_attn_factor = %.4f (mscale == %.1f, mscale_all_dim = %.1f)\n",
                    __func__, cparams.yarn_attn_factor, mscale, mscale_all_dims);
        } else {
            cparams.yarn_attn_factor = get_mscale(factor, 1.0f);
        }

        // when YARN is applied with yarn_ext_factor != 0.0f, we need to cancel this factor:
        // https://github.com/ggml-org/liquid.cpp/blob/a81a569577cc38b32558958b048228150be63eae/ggml/src/ggml-cpu/ops.cpp#L5541-L5544
        //
        // ref: https://github.com/ggml-org/liquid.cpp/discussions/7416
        //      https://github.com/ggml-org/liquid.cpp/pull/17945
        cparams.yarn_attn_factor *= 1.0f / (1.0f + 0.1f * logf(factor));
    }

    cparams.yarn_attn_factor *= hparams.rope_attn_factor;

    if (cparams.pooling_type == LFG_POOLING_TYPE_UNSPECIFIED) {
        if (hparams.pooling_type == LFG_POOLING_TYPE_UNSPECIFIED) {
            cparams.pooling_type = LFG_POOLING_TYPE_NONE;
        } else {
            cparams.pooling_type = hparams.pooling_type;
        }
    }

    if (params.attention_type == LFG_ATTENTION_TYPE_UNSPECIFIED) {
        cparams.causal_attn = hparams.causal_attn;
    } else {
        cparams.causal_attn = params.attention_type == LFG_ATTENTION_TYPE_CAUSAL;
    }

    cparams.flash_attn = params.flash_attn_type != LFG_FLASH_ATTN_TYPE_DISABLED;
    cparams.auto_fa    = params.flash_attn_type == LFG_FLASH_ATTN_TYPE_AUTO;

    // with causal attention, the batch size is limited by the context size
    cparams.n_batch = cparams.causal_attn ? std::min(cparams.n_ctx, params.n_batch) : params.n_batch;

    cparams.n_ubatch = std::min(cparams.n_batch, params.n_ubatch == 0 ? params.n_batch : params.n_ubatch);

    cparams.op_offload = params.op_offload;
    cparams.kv_unified = params.kv_unified;

    // intialized later
    cparams.pipeline_parallel = false;

    {
        const char * LFG_GRAPH_REUSE_DISABLE = getenv("LFG_GRAPH_REUSE_DISABLE");
        graph_reuse_disable = LFG_GRAPH_REUSE_DISABLE ? (atoi(LFG_GRAPH_REUSE_DISABLE) != 0) : graph_reuse_disable;

        if (graph_reuse_disable) {
            LFG_LOG_WARN("%s: graph reuse disabled\n", __func__);
        }
    }

    // ref: https://github.com/ggml-org/liquid.cpp/pull/17046#discussion_r2503085732
    cparams.n_ctx = static_cast<uint32_t>(GGML_PAD(cparams.n_ctx, 256));

    if (cparams.kv_unified) {
        cparams.n_ctx_seq = cparams.n_ctx;
    } else {
        cparams.n_ctx_seq = cparams.n_ctx / cparams.n_seq_max;
        cparams.n_ctx_seq = static_cast<uint32_t>(GGML_PAD(cparams.n_ctx_seq, 256));

        if (cparams.n_ctx_seq == 0) {
            throw std::runtime_error("n_ctx_seq == 0");
        }

        if (cparams.n_ctx != cparams.n_ctx_seq * cparams.n_seq_max) {
            cparams.n_ctx =  cparams.n_ctx_seq * cparams.n_seq_max;
            LFG_LOG_WARN("%s: n_ctx is not divisible by n_seq_max - rounding down to %u\n", __func__, cparams.n_ctx);
        }
    }

    LFG_LOG_INFO("%s: n_seq_max     = %u\n",   __func__, cparams.n_seq_max);
    LFG_LOG_INFO("%s: n_ctx         = %u\n",   __func__, cparams.n_ctx);
    LFG_LOG_INFO("%s: n_ctx_seq     = %u\n",   __func__, cparams.n_ctx_seq);
    LFG_LOG_INFO("%s: n_batch       = %u\n",   __func__, cparams.n_batch);
    LFG_LOG_INFO("%s: n_ubatch      = %u\n",   __func__, cparams.n_ubatch);
    LFG_LOG_INFO("%s: causal_attn   = %d\n",   __func__, cparams.causal_attn);
    LFG_LOG_INFO("%s: flash_attn    = %s\n",   __func__, lfg_flash_attn_type_name(params.flash_attn_type));
    LFG_LOG_INFO("%s: kv_unified    = %s\n",   __func__, cparams.kv_unified ? "true" : "false");
    LFG_LOG_INFO("%s: freq_base     = %.1f\n", __func__, cparams.rope_freq_base);
    LFG_LOG_INFO("%s: freq_scale    = %g\n",   __func__, cparams.rope_freq_scale);

    if (cparams.n_ctx_seq < hparams.n_ctx_train) {
        LFG_LOG_WARN("%s: n_ctx_seq (%u) < n_ctx_train (%u) -- the full capacity of the model will not be utilized\n",
                __func__, cparams.n_ctx_seq, hparams.n_ctx_train);
    }

    if (cparams.n_ctx_seq > hparams.n_ctx_train) {
        LFG_LOG_WARN("%s: n_ctx_seq (%u) > n_ctx_train (%u) -- possible training context overflow\n",
                __func__, cparams.n_ctx_seq, hparams.n_ctx_train);
    }

    if (!hparams.vocab_only) {
        // GPU backends
        for (auto * dev : model.devices) {
            ggml_backend_t backend = ggml_backend_dev_init(dev, nullptr);
            if (backend == nullptr) {
                throw std::runtime_error(lfg_format("failed to initialize %s backend", ggml_backend_dev_name(dev)));
            }
            backends.emplace_back(backend);
        }

        // add ACCEL backends (such as BLAS)
        for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
            ggml_backend_dev_t dev = ggml_backend_dev_get(i);
            if (ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_ACCEL) {
                LFG_LOG_WARN("%s: initializing ACCEL backend: %s\n", __func__, ggml_backend_dev_name(dev));
                ggml_backend_t backend = ggml_backend_dev_init(dev, nullptr);
                if (backend == nullptr) {
                    throw std::runtime_error(lfg_format("failed to initialize %s backend", ggml_backend_dev_name(dev)));
                }
                backends.emplace_back(backend);
            }
        }

        // add CPU backend
        backend_cpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
        if (backend_cpu == nullptr) {
            throw std::runtime_error("failed to initialize CPU backend");
        }
        backends.emplace_back(backend_cpu);

        // create a list of the set_n_threads functions in the backends
        for (auto & backend : backends) {
            ggml_backend_dev_t dev = ggml_backend_get_device(backend.get());
            ggml_backend_reg_t reg = dev ? ggml_backend_dev_backend_reg(dev) : nullptr;
            if (reg) {
                auto ggml_backend_set_n_threads_fn = (ggml_backend_set_n_threads_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_set_n_threads");
                if (ggml_backend_set_n_threads_fn) {
                    set_n_threads_fns.emplace_back(backend.get(), ggml_backend_set_n_threads_fn);
                }
            }
        }

        lfg_set_abort_callback(this, params.abort_callback, params.abort_callback_data);

        // graph outputs buffer
        {
            // resized during inference when a batch uses more outputs
            // Create a dummy batch for initialization.
            lfg_batch dummy_batch = {};
            dummy_batch.n_tokens = 0;
            if (output_reserve(params.n_seq_max, dummy_batch) < params.n_seq_max) {
                throw std::runtime_error("failed to reserve initial output buffer");
            }

            LFG_LOG_INFO("%s: %10s  output buffer size = %8.2f MiB\n", __func__,
                    ggml_backend_buffer_name    (buf_output.get()),
                    ggml_backend_buffer_get_size(buf_output.get()) / 1024.0 / 1024.0);
        }
    }

    // init the memory module
    if (!hparams.vocab_only) {
        lfg_memory_params params_mem = {
            /*.type     =*/ LFG_MEMORY_KV_CACHE,
            /*.type_k   =*/ params.type_k,
            /*.type_v   =*/ params.type_v,
            /*.swa_full =*/ params.swa_full,
            /*.kv_cache_path =*/ "",
        };

        memory.reset(model.create_memory(params_mem, cparams));
    }

    // init backends
    if (!hparams.vocab_only) {
        LFG_LOG_DEBUG("%s: enumerating backends\n", __func__);

        backend_buft.clear();
        backend_ptrs.clear();
        backend_buf_exp_size.clear();

        for (auto & backend : backends) {
            auto * buft = ggml_backend_get_default_buffer_type(backend.get());
            auto backend_type = ggml_backend_dev_type(ggml_backend_get_device(backend.get()));

            if (backend_type == GGML_BACKEND_DEVICE_TYPE_CPU && !model.devices.empty()) {
                // use the host buffer of the first device CPU for faster transfer of the intermediate state
                auto * dev = model.devices[0];
                auto * host_buft = ggml_backend_dev_host_buffer_type(dev);
                if (host_buft) {
                    buft = host_buft;
                }
            }

            backend_buft.push_back(buft);
            backend_ptrs.push_back(backend.get());
            backend_buf_exp_size.push_back(0);
        }

        LFG_LOG_DEBUG("%s: backend_ptrs.size() = %zu\n", __func__, backend_ptrs.size());

        // TODO: move these checks to ggml_backend_sched
        // enabling pipeline parallelism in the scheduler increases memory usage, so it is only done when necessary
        bool pipeline_parallel =
            model.n_devices() > 1 &&
            model.n_gpu_layers() > model.hparams.n_layer &&
            model.split_mode() == LFG_SPLIT_MODE_LAYER &&
            cparams.offload_kqv &&
            !model.has_tensor_overrides();

        // pipeline parallelism requires support for async compute and events in all devices
        if (pipeline_parallel) {
            for (auto & backend : backends) {
                auto dev_type = ggml_backend_dev_type(ggml_backend_get_device(backend.get()));
                if (dev_type == GGML_BACKEND_DEVICE_TYPE_CPU) {
                    // ignore CPU backend
                    continue;
                }
                auto * dev = ggml_backend_get_device(backend.get());
                ggml_backend_dev_props props;
                ggml_backend_dev_get_props(dev, &props);
                if (!props.caps.async || !props.caps.events) {
                    // device does not support async compute or events
                    pipeline_parallel = false;
                    break;
                }
            }
        }

        cparams.pipeline_parallel = pipeline_parallel;

        if (cparams.pipeline_parallel) {
            LFG_LOG_INFO("%s: pipeline parallelism enabled\n", __func__);
        }


        sched_reserve();

        sched_reserve();

        if (!cparams.flash_attn) {
            if (ggml_is_quantized(params.type_v)) {
                throw std::runtime_error("quantized V cache was requested, but this requires Flash Attention");
            }
        }
    }

    // Initialize the full vocabulary token ids for backend samplers.
    {
        const int n_vocab = model.vocab.n_tokens();

        sampling.token_ids_full_vocab.resize(n_vocab);
        for (int i = 0; i < n_vocab; ++i) {
            sampling.token_ids_full_vocab[i] = i;
        }
    }
}

lfg_context::~lfg_context() {
    if (!model.hparams.no_alloc) {
        for (size_t i = 0; i < backend_ptrs.size(); ++i) {
            ggml_backend_t             backend = backend_ptrs[i];
            ggml_backend_buffer_type_t buft    = backend_buft[i];

            const size_t size_exp = backend_buf_exp_size[i];
            const size_t size_act = ggml_backend_sched_get_buffer_size(sched.get(), backend);
            if (size_exp == size_act) {
                LFG_LOG_DEBUG("%s: %10s compute buffer size is %8.4f MiB, matches expectation of %8.4f MiB\n",
                    __func__, ggml_backend_buft_name(buft), size_act / (1024.0*1024.0), size_exp / (1024.0*1024.0));
            } else {
                LFG_LOG_WARN("%s: %10s compute buffer size of %8.4f MiB, does not match expectation of %8.4f MiB\n",
                    __func__, ggml_backend_buft_name(buft), size_act / (1024.0*1024.0), size_exp / (1024.0*1024.0));
            }
        }
    }
    ggml_opt_free(opt_ctx);
}

void lfg_context::sched_reserve() {
    if (!sched_need_reserve) {
        return;
    }

    sched_need_reserve = false;

    LFG_LOG_INFO("%s: reserving ...\n", __func__);

    synchronize();

    const int64_t t_start_us = ggml_time_us();

    const uint32_t n_seqs = cparams.n_seq_max;
    const uint32_t n_tokens = std::min(cparams.n_ctx, cparams.n_ubatch);

    const size_t max_nodes = this->graph_max_nodes(n_tokens);

    LFG_LOG_DEBUG("%s: max_nodes = %zu\n", __func__, max_nodes);

    gf_res_prev.reset(new lfg_graph_result(max_nodes));
    gf_res_reserve.reset(new lfg_graph_result(max_nodes));

    sched.reset(ggml_backend_sched_new(backend_ptrs.data(), backend_buft.data(), backend_ptrs.size(), max_nodes, cparams.pipeline_parallel, cparams.op_offload));

    std::unique_ptr<lfg_memory_context_i> mctx;
    if (memory) {
        LFG_LOG_DEBUG("%s: reserving full memory module\n", __func__);
        mctx = memory->init_full();
        if (!mctx) {
            throw std::runtime_error("failed to initialize memory module");
        }
    }

    // avoid reserving graphs with zero outputs - assume one output per sequence
    const int n_outputs = n_seqs;

    LFG_LOG_DEBUG("%s: worst-case: n_tokens = %d, n_seqs = %d, n_outputs = %d\n", __func__, n_tokens, n_seqs, n_outputs);

    // resolve automatic Flash Attention use
    if (cparams.auto_fa) {
        auto * gf = graph_reserve(1, n_seqs, n_outputs, mctx.get(), true);
        if (!gf) {
            throw std::runtime_error("failed to split graph for Flash Attention check");
        }

        const size_t prefix_len = strlen(LFG_TENSOR_NAME_FATTN) + 1;
        bool fa_device_mismatch = false;
        for (int i = 0; i < ggml_graph_n_nodes(gf); i++) {
            ggml_tensor * n = ggml_graph_node(gf, i);
            if (n->op != GGML_OP_FLASH_ATTN_EXT) {
                continue;
            }
            ggml_backend_dev_t device_fa = ggml_backend_get_device(
                    ggml_backend_sched_get_tensor_backend(sched.get(), n));

            // TODO: instead of the tensor names, use a map to keep track of which (FA) tensors belong to which layer
            GGML_ASSERT(strncmp(n->name, LFG_TENSOR_NAME_FATTN "-", prefix_len) == 0);
            const int il = std::stoi(n->name + prefix_len);
            ggml_backend_dev_t device_kv = model.dev_layer(il);
            if (device_fa != device_kv) {
                LFG_LOG_WARN("%s: layer %d is assigned to device %s but the Flash Attention tensor "
                        "is assigned to device %s (usually due to missing support)\n",
                        __func__, il, ggml_backend_dev_name(device_kv), ggml_backend_dev_name(device_fa));
                // FIXME: fa_device_mismatch logic is wrong for --no-kv-offload, but this is broken anyways
                fa_device_mismatch = true;
                break;
            }
        }
        if (fa_device_mismatch) {
            cparams.flash_attn = false;
            LFG_LOG_WARN("%s: Flash Attention was auto, set to disabled\n", __func__);
        } else {
            cparams.flash_attn = true;
            LFG_LOG_INFO("%s: Flash Attention was auto, set to enabled\n", __func__);
        }


        cparams.auto_fa = false;
    }

    // reserve worst-case graph
    int n_splits_pp = -1;
    int n_nodes_pp  = -1;

    int n_splits_tg = -1;
    int n_nodes_tg  = -1;

    // reserve pp (prompt processing) graph first so that buffers are only allocated once
    {
        auto * gf = graph_reserve(n_tokens, n_seqs, n_tokens, mctx.get(),
                model.hparams.no_alloc, model.hparams.no_alloc ? backend_buf_exp_size.data() : nullptr);
        if (!gf) {
            if (cparams.pipeline_parallel) {
                LFG_LOG_WARN("%s: compute buffer allocation failed, retrying without pipeline parallelism\n", __func__);
                cparams.pipeline_parallel = false;
                sched.reset(ggml_backend_sched_new(backend_ptrs.data(), backend_buft.data(), backend_ptrs.size(), max_nodes, false, cparams.op_offload));
                gf = graph_reserve(n_tokens, n_seqs, n_tokens, mctx.get());
            }
            if (!gf) {
                throw std::runtime_error("failed to allocate compute pp buffers");
            }
        }

        n_splits_pp = ggml_backend_sched_get_n_splits(sched.get());
        n_nodes_pp  = ggml_graph_n_nodes(gf);
    }

    // reserve with tg (token generation) graph to get the number of splits and nodes
    {
        auto * gf = graph_reserve(n_seqs, n_seqs, n_seqs, mctx.get(), model.hparams.no_alloc);
        if (!gf) {
            throw std::runtime_error("failed to allocate compute tg buffers");
        }

        n_splits_tg = ggml_backend_sched_get_n_splits(sched.get());
        n_nodes_tg  = ggml_graph_n_nodes(gf);
    }

    // reserve again with pp graph to avoid ggml-alloc reallocations during inference
    {
        // TODO: not sure if the following graph would be worster case for multi-stream KV caches:
        //
        // auto * gf = graph_reserve(n_tokens, 1, n_tokens, mctx.get());
        //
        auto * gf = graph_reserve(n_tokens, n_seqs, n_tokens, mctx.get(), model.hparams.no_alloc);
        if (!gf) {
            throw std::runtime_error("failed to allocate compute pp buffers");
        }
    }

    for (size_t i = 0; i < backend_ptrs.size(); ++i) {
        ggml_backend_t             backend = backend_ptrs[i];
        ggml_backend_buffer_type_t buft    = backend_buft[i];
        if (!model.hparams.no_alloc) {
            backend_buf_exp_size[i] = ggml_backend_sched_get_buffer_size(sched.get(), backend);
        }
        if (backend_buf_exp_size[i] > 1) {
            LFG_LOG_INFO("%s: %10s compute buffer size = %8.2f MiB\n", __func__,
                    ggml_backend_buft_name(buft),
                    backend_buf_exp_size[i] / 1024.0 / 1024.0);
        }
    }

    if (n_nodes_pp == n_nodes_tg) {
        LFG_LOG_INFO("%s: graph nodes  = %d\n", __func__, n_nodes_pp);
    } else {
        LFG_LOG_INFO("%s: graph nodes  = %d (with bs=%d), %d (with bs=1)\n", __func__, n_nodes_pp, n_tokens, n_nodes_tg);
    }

    if (n_splits_pp == n_splits_tg) {
        LFG_LOG_INFO("%s: graph splits = %d\n", __func__, n_splits_pp);
    } else {
        LFG_LOG_INFO("%s: graph splits = %d (with bs=%d), %d (with bs=1)\n", __func__, n_splits_pp, n_tokens, n_splits_tg);
    }

    const int64_t t_end_us = ggml_time_us();

    LFG_LOG_INFO("%s: reserve took %.2f ms, sched copies = %d\n",
            __func__, (t_end_us - t_start_us)/1000.0, ggml_backend_sched_get_n_copies(sched.get()));
}

void lfg_context::synchronize() {
    if (!sched) {
        return;
    }

    ggml_backend_sched_synchronize(sched.get());

    // FIXME: if multiple single tokens are evaluated without a synchronization,
    // the stats will be added to the prompt evaluation stats
    // this should only happen when using batch size 1 to evaluate a batch

    // add the evaluation to the stats
    if (n_queued_tokens == 1) {
        if (!cparams.no_perf) {
            t_eval_us += ggml_time_us() - t_compute_start_us;
        }
        n_eval++;
    } else if (n_queued_tokens > 1) {
        if (!cparams.no_perf) {
            t_p_eval_us += ggml_time_us() - t_compute_start_us;
        }
        n_p_eval += n_queued_tokens;
    }

    // get a more accurate load time, upon first eval
    if (n_queued_tokens > 0 && !has_evaluated_once) {
        t_load_us = ggml_time_us() - t_start_us;
        has_evaluated_once = true;
    }

    n_queued_tokens = 0;
    t_compute_start_us = 0;
}

const lfg_model & lfg_context::get_model() const {
    return model;
}

const lfg_cparams & lfg_context::get_cparams() const {
    return cparams;
}

ggml_backend_sched_t lfg_context::get_sched() const {
    return sched.get();
}

uint32_t lfg_context::n_ctx() const {
    return cparams.n_ctx;
}

uint32_t lfg_context::n_ctx_seq() const {
    return cparams.n_ctx_seq;
}

uint32_t lfg_context::n_batch() const {
    return cparams.n_batch;
}

uint32_t lfg_context::n_ubatch() const {
    return cparams.n_ubatch;
}

uint32_t lfg_context::n_seq_max() const {
    return cparams.n_seq_max;
}

uint32_t lfg_context::n_threads() const {
    return cparams.n_threads;
}

uint32_t lfg_context::n_threads_batch() const {
    return cparams.n_threads_batch;
}

lfg_memory_t lfg_context::get_memory() const {
    return memory.get();
}

bool lfg_context::memory_update(bool optimize) {
    if (!memory) {
        return false;
    }

    {
        const auto mctx = memory->init_update(this, optimize);
        switch (mctx->get_status()) {
            case LFG_MEMORY_STATUS_SUCCESS:
                {
                    // noop
                } break;
            case LFG_MEMORY_STATUS_NO_UPDATE:
                {
                    // no updates need to be performed
                    return false;
                }
            case LFG_MEMORY_STATUS_FAILED_PREPARE:
            case LFG_MEMORY_STATUS_FAILED_COMPUTE:
                {
                    LFG_LOG_ERROR("%s: failed to prepare memory update\n", __func__);
                    return false;
                }
        }

        // reset the previous graph result to make sure that it won't be reused
        // TODO: change the mctx->apply() to return information if a graph reserve is needed
        //       reset the graph result only if the memory module did reset the scheduler
        gf_res_prev->reset();

        if (!mctx->apply()) {
            LFG_LOG_ERROR("%s: failed to apply memory update\n", __func__);
        }
    }

    // if the memory module did any computation, we have to reserve a new worst-case graph
    {
        const auto mctx = memory->init_full();
        if (!mctx) {
            throw std::runtime_error("failed to initialize memory context");
        }

        const uint32_t n_seqs = cparams.n_seq_max;
        const uint32_t n_tokens = std::min(cparams.n_ctx, cparams.n_ubatch);

        auto * gf = graph_reserve(n_tokens, n_seqs, n_tokens, mctx.get());
        if (!gf) {
            LFG_LOG_ERROR("%s: failed to reserve graph after the memory update\n", __func__);
        }
    }

    return true;
}

enum lfg_pooling_type lfg_context::pooling_type() const {
    return cparams.pooling_type;
}

float * lfg_context::get_logits() {
    output_reorder();

    return logits;
}

int64_t lfg_context::output_resolve_row(int32_t i) const {
    int64_t j = -1;

    // support negative indices (last output row)
    if (i < 0) {
        j = n_outputs + i;
        if (j < 0) {
            throw std::runtime_error(lfg_format("negative index out of range [0, %d)", n_outputs));
        }
    } else if ((size_t) i >= output_ids.size()) {
        throw std::runtime_error(lfg_format("out of range [0, %zu)", output_ids.size()));
    } else {
        // use output_ids to translate the batch token index into a row number
        // that holds this token's data.
        j = output_ids[i];
    }

    if (j < 0) {
        // the batch token was not configured to output anything
        throw std::runtime_error(lfg_format("batch.logits[%d] != true", i));
    }

    if (j >= n_outputs) {
        throw std::runtime_error(lfg_format("corrupt output buffer (j=%" PRId64 ", n_outputs=%d)", j, n_outputs));
    }

    return j;
}

float * lfg_context::get_logits_ith(int32_t i) {
    int64_t j = -1;

    output_reorder();

    try {
        if (logits == nullptr) {
            throw std::runtime_error("no logits");
        }

        // TODO: use output_resolve_row()
        if (i < 0) {
            j = n_outputs + i;
            if (j < 0) {
                throw std::runtime_error(lfg_format("negative index out of range [0, %d)", n_outputs));
            }
        } else if ((size_t) i >= output_ids.size()) {
            throw std::runtime_error(lfg_format("out of range [0, %zu)", output_ids.size()));
        } else {
            j = output_ids[i];
        }

        if (j < 0) {
            throw std::runtime_error(lfg_format("batch.logits[%d] != true", i));
        }
        if (j >= n_outputs) {
            // This should not happen
            throw std::runtime_error(lfg_format("corrupt output buffer (j=%" PRId64 ", n_outputs=%d)", j, n_outputs));
        }

        return logits + j*model.vocab.n_tokens();
    } catch (const std::exception & err) {
        LFG_LOG_ERROR("%s: invalid logits id %d, reason: %s\n", __func__, i, err.what());
        lfg_set_last_error(LFG_ERROR_INVALID_ARGUMENT, "%s: invalid logits id %d: %s", __func__, i, err.what());
        return nullptr;
    }
}

float * lfg_context::get_embeddings() {
    output_reorder();

    return embd;
}

lfg_token * lfg_context::get_sampled_tokens()  const{
    return sampling.sampled;
}

float * lfg_context::get_embeddings_ith(int32_t i) {
    int64_t j = -1;

    output_reorder();

    try {
        if (embd == nullptr) {
            throw std::runtime_error("no embeddings");
        }

        // TODO: use output_resolve_row()
        if (i < 0) {
            j = n_outputs + i;
            if (j < 0) {
                throw std::runtime_error(lfg_format("negative index out of range [0, %d)", n_outputs));
            }
        } else if ((size_t) i >= output_ids.size()) {
            throw std::runtime_error(lfg_format("out of range [0, %zu)", output_ids.size()));
        } else {
            j = output_ids[i];
        }

        if (j < 0) {
            throw std::runtime_error(lfg_format("batch.logits[%d] != true", i));
        }
        if (j >= n_outputs) {
            // This should not happen
            throw std::runtime_error(lfg_format("corrupt output buffer (j=%" PRId64 ", n_outputs=%d)", j, n_outputs));
        }

        const uint32_t n_embd_out = model.hparams.get_n_embd_out();
        return embd + j*n_embd_out;
    } catch (const std::exception & err) {
        LFG_LOG_ERROR("%s: invalid embeddings id %d, reason: %s\n", __func__, i, err.what());
        lfg_set_last_error(LFG_ERROR_INVALID_ARGUMENT, "%s: invalid embeddings id %d: %s", __func__, i, err.what());
        return nullptr;
    }
}

float * lfg_context::get_embeddings_seq(lfg_seq_id seq_id) {
    auto it = embd_seq.find(seq_id);
    if (it == embd_seq.end()) {
        return nullptr;
    }

    return it->second.data();
}

lfg_token lfg_context::get_sampled_token_ith(int32_t idx) {
    output_reorder();

    if (sampling.sampled == nullptr) {
        return LFG_TOKEN_NULL;
    }

    try {
        const int64_t row = output_resolve_row(idx);
        GGML_ASSERT(row < (int64_t) sampling.sampled_size);
        return sampling.sampled[row];
    } catch (const std::exception & err) {
        LFG_LOG_ERROR("%s: invalid backend sampled token id %d, reason: %s\n", __func__, idx, err.what());
        lfg_set_last_error(LFG_ERROR_INVALID_ARGUMENT, "%s: invalid sampled token id %d: %s", __func__, idx, err.what());
        return LFG_TOKEN_NULL;
    }
}

float * lfg_context::get_sampled_probs_ith(int32_t idx) {
    output_reorder();

    if (sampling.probs == nullptr) {
        return nullptr;
    }

    try {
        const int64_t row = output_resolve_row(idx);
        if ((size_t) row >= sampling.probs_count.size() || sampling.probs_count[row] == 0) {
            return nullptr;
        }
        return sampling.probs + row*model.vocab.n_tokens();
    } catch (const std::exception & err) {
        LFG_LOG_ERROR("%s: invalid backend sampled probs id %d, reason: %s\n", __func__, idx, err.what());
        lfg_set_last_error(LFG_ERROR_INVALID_ARGUMENT, "%s: invalid sampled probs id %d: %s", __func__, idx, err.what());
        return nullptr;
    }
}

float * lfg_context::get_sampled_logits_ith(int32_t idx) {
    output_reorder();

    if (sampling.logits == nullptr) {
        return nullptr;
    }

    try {
        const int64_t row = output_resolve_row(idx);
        if ((size_t) row >= sampling.logits_count.size() || sampling.logits_count[row] == 0) {
            return nullptr;
        }
        return sampling.logits + row*model.vocab.n_tokens();
    } catch (const std::exception & err) {
        LFG_LOG_ERROR("%s: invalid backend sampled logits id %d, reason: %s\n", __func__, idx, err.what());
        lfg_set_last_error(LFG_ERROR_INVALID_ARGUMENT, "%s: invalid sampled logits id %d: %s", __func__, idx, err.what());
        return nullptr;
    }
}

const lfg_token * lfg_context::get_sampled_candidates_ith(int32_t idx) {
    output_reorder();

    try {
        const int64_t row = output_resolve_row(idx);
        if (sampling.candidates != nullptr &&
            (size_t) row < sampling.candidates_count.size() &&
            sampling.candidates_count[row] > 0) {
            return sampling.candidates + row*model.vocab.n_tokens();
        }
    } catch (const std::exception & err) {
        // fallback to full vocab list
    }

    return sampling.token_ids_full_vocab.data();
}

size_t lfg_context::get_sampled_candidates_count(int32_t idx) {
    output_reorder();

    if (sampling.candidates == nullptr) {
        return 0;
    }

    try {
        const int64_t row = output_resolve_row(idx);
        if ((size_t) row >= sampling.candidates_count.size()) {
            return 0;
        }
        return sampling.candidates_count[row];
    } catch (const std::exception & err) {
        LFG_LOG_ERROR("%s: invalid backend sampled candidates count id %d, reason: %s\n", __func__, idx, err.what());
        return 0;
    }
}

size_t lfg_context::get_sampled_logits_count(int32_t idx) {
    output_reorder();

    if (sampling.logits == nullptr) {
        return model.vocab.n_tokens();
    }

    try {
        const int64_t row = output_resolve_row(idx);
        if ((size_t) row >= sampling.logits_count.size()) {
            return 0;
        }
        return sampling.logits_count[row];
    } catch (const std::exception & err) {
        LFG_LOG_ERROR("%s: invalid backend sampled logits count id %d, reason: %s\n", __func__, idx, err.what());
        return 0;
    }
}

size_t lfg_context::get_sampled_probs_count(int32_t idx) {
    output_reorder();

    if (sampling.probs == nullptr) {
        return 0;
    }

    try {
        const int64_t row = output_resolve_row(idx);
        if ((size_t) row >= sampling.probs_count.size()) {
            return 0;
        }
        return sampling.probs_count[row];
    } catch (const std::exception & err) {
        LFG_LOG_ERROR("%s: invalid backend sampled probs count id %d, reason: %s\n", __func__, idx, err.what());
        return 0;
    }
}


void lfg_context::attach_threadpool(
           ggml_threadpool_t threadpool,
           ggml_threadpool_t threadpool_batch) {
    LFG_LOG_DEBUG("%s: call\n", __func__);

    this->threadpool       = threadpool;
    this->threadpool_batch = threadpool_batch ? threadpool_batch : threadpool;
}

void lfg_context::detach_threadpool() {
    LFG_LOG_DEBUG("%s: call\n", __func__);

    this->threadpool       = nullptr;
    this->threadpool_batch = nullptr;
}

void lfg_context::set_n_threads(int32_t n_threads, int32_t n_threads_batch) {
    LFG_LOG_DEBUG("%s: n_threads = %d, n_threads_batch = %d\n", __func__, n_threads, n_threads_batch);

    cparams.n_threads       = n_threads;
    cparams.n_threads_batch = n_threads_batch;
}

void lfg_context::set_abort_callback(bool (*abort_callback)(void * data), void * abort_callback_data) {
    LFG_LOG_DEBUG("%s: call\n", __func__);

    this->abort_callback      = abort_callback;
    this->abort_callback_data = abort_callback_data;

    for (auto & backend : backends) {
        auto * reg = ggml_backend_dev_backend_reg(ggml_backend_get_device(backend.get()));
        auto * set_abort_callback_fn = (ggml_backend_set_abort_callback_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_set_abort_callback");
        if (set_abort_callback_fn) {
            set_abort_callback_fn(backend.get(), this->abort_callback, this->abort_callback_data);
        }
    }
}

void lfg_context::set_embeddings(bool value) {
    LFG_LOG_DEBUG("%s: value = %d\n", __func__, value);

    cparams.embeddings = value;

    // TODO: not sure yet if we want to reserve here
    //sched_need_reserve = true;
}

void lfg_context::set_causal_attn(bool value) {
    LFG_LOG_DEBUG("%s: value = %d\n", __func__, value);

    if (cparams.causal_attn == value) {
        return;
    }

    cparams.causal_attn = value;

    sched_need_reserve = true;
}

void lfg_context::set_warmup(bool value) {
    LFG_LOG_DEBUG("%s: value = %d\n", __func__, value);

    if (cparams.warmup == value) {
        return;
    }

    cparams.warmup = value;

    // warmups are usually with small batches, so no need to reserve
    //sched_need_reserve = true;
}

bool lfg_context::set_sampler(lfg_seq_id seq_id, lfg_sampler * sampler) {
    if (!sampler && sampling.samplers.count(seq_id) == 0) {
        return true;
    }

    LFG_LOG_DEBUG("%s: seq_id = %d, sampler = %p\n", __func__, (int) seq_id, (void *) sampler);

    const bool can_offload =
        sampler &&
        sampler->iface->backend_init &&
        sampler->iface->backend_apply &&
        lfg_sampler_chain_n(sampler) > 0;

    if (sampler && can_offload) {
        ggml_backend_buffer_type_t buft = ggml_backend_dev_buffer_type(model.dev_output());
        auto * host_buft = ggml_backend_dev_host_buffer_type(model.dev_output());
        if (host_buft) {
            buft = host_buft;
        }

        sampler->iface->backend_init(sampler, buft);

        sampling.samplers[seq_id] = sampler;

        sched_need_reserve = true;

        return true;
    }

    if (sampler && !can_offload) {
        LFG_LOG_WARN("%s: sampler '%s' for seq_id = %d, cannot be offloaded to the backend\n", __func__, lfg_sampler_name(sampler), seq_id);

        if (sampling.samplers.count(seq_id) > 0) {
            sched_need_reserve = true;
        }

        sampling.samplers.erase(seq_id);

        return false;
    }

    sampling.samplers.erase(seq_id);

    sched_need_reserve = true;

    return true;
}

void lfg_context::set_adapter_lora(
            lfg_adapter_lora * adapter,
            float scale) {
    LFG_LOG_DEBUG("%s: adapter = %p, scale = %f\n", __func__, (void *) adapter, scale);

    if (auto it = loras.find(adapter); it != loras.end()) {
        if (it->second == scale) {
            return;
        }
    }

    loras[adapter] = scale;

    sched_need_reserve = true;
}

bool lfg_context::rm_adapter_lora(
            lfg_adapter_lora * adapter) {
    LFG_LOG_DEBUG("%s: adapter = %p\n", __func__, (void *) adapter);

    auto it = loras.find(adapter);
    if (it != loras.end()) {
        loras.erase(it);

        sched_need_reserve = true;

        return true;
    }

    return false;
}

void lfg_context::clear_adapter_lora() {
    LFG_LOG_DEBUG("%s: call\n", __func__);

    if (loras.empty()) {
        return;
    }

    loras.clear();

    sched_need_reserve = true;
}

bool lfg_context::apply_adapter_cvec(
            const float * data,
                 size_t   len,
                int32_t   n_embd,
                int32_t   il_start,
                int32_t   il_end) {
    LFG_LOG_DEBUG("%s: il_start = %d, il_end = %d\n", __func__, il_start, il_end);

    // TODO: should we reserve?

    return cvec.apply(model, data, len, n_embd, il_start, il_end);
}

lfg_graph_result * lfg_context::process_ubatch(const lfg_ubatch & ubatch, lfg_graph_type gtype, lfg_memory_context_i * mctx, ggml_status & ret) {
    if (mctx && !mctx->apply()) {
        LFG_LOG_ERROR("%s: failed to apply memory context\n", __func__);
        ret = GGML_STATUS_FAILED;
        return nullptr;
    }

    auto * res = gf_res_prev.get();
    auto * gf  = res->get_gf();

    // the new graph parameters
    // in order to correctly reuse a graph, it's full topology has to be uniquely determined by these parameters
    const auto gparams = graph_params(res, ubatch, mctx, gtype);

    if (!graph_reuse_disable && res->can_reuse(gparams)) {
        //LFG_LOG_DEBUG("%s: reusing previous graph\n", __func__);

        n_reused++;
    } else {
        res->reset();

        ggml_backend_sched_reset(sched.get());
        ggml_backend_sched_set_eval_callback(sched.get(), cparams.cb_eval, cparams.cb_eval_user_data);

        //const auto t_start_us = ggml_time_us();

        gf = model.build_graph(gparams);

        //LFG_LOG_INFO("graph build time: %.3f ms\n", (ggml_time_us() - t_start_us)/1000.0);

        if (!gf) {
            LFG_LOG_ERROR("%s: failed to initialize graph\n", __func__);
            ret = GGML_STATUS_FAILED;
            return nullptr;
        }

        if (!ggml_backend_sched_alloc_graph(sched.get(), gf)) {
            LFG_LOG_ERROR("%s: failed to allocate graph\n", __func__);
            ret = GGML_STATUS_ALLOC_FAILED;
            return nullptr;
        }
    }

    // set the input data for the input tensors
    {
        //const auto t_start_us = ggml_time_us();

        res->set_inputs(&ubatch);

        //LFG_LOG_INFO("graph set inputs time: %.3f ms\n", (ggml_time_us() - t_start_us)/1000.0);
    }

    const auto status = graph_compute(res->get_gf(), ubatch.n_tokens > 1);
    if (status != GGML_STATUS_SUCCESS) {
        LFG_LOG_ERROR("%s: failed to compute graph, compute status: %d\n", __func__, status);
        ret = status;
        return nullptr;
    }

    ret = GGML_STATUS_SUCCESS;

    return res;
}

int lfg_context::encode(const lfg_batch & batch_inp) {
    GGML_ASSERT((!batch_inp.token && batch_inp.embd) || (batch_inp.token && !batch_inp.embd)); // NOLINT
    
    LFG_LOG_INFO("%s: start\n", __func__);

    if (batch_inp.n_tokens == 0) {
        LFG_LOG_ERROR("%s: n_tokens == 0\n", __func__);
        return -1;
    }

    const auto & hparams = model.hparams;

    const int64_t n_embd  = hparams.n_embd_inp();
    const int64_t n_vocab = model.vocab.n_tokens();

    // note: during encode, we always pass the full sequence starting from pos = 0
    if (!balloc->init(batch_inp, model.vocab, nullptr, n_embd, cparams.kv_unified ? LFG_MAX_SEQ : cparams.n_seq_max, true)) {
        LFG_LOG_ERROR("%s: failed to initialize batch\n", __func__);
        return -1;
    }
    
    LFG_LOG_INFO("%s: balloc init done\n", __func__);

    const uint32_t n_tokens = balloc->get_n_tokens();

    // [TAG_NO_CACHE_PAD]
    // TODO: add new split mode where we pad the input sequences so that ubatch.equal_seqs == true
    const lfg_ubatch ubatch = balloc->split_simple(n_tokens);

    // micro-batching is not possible for non-causal encoding, so we process the batch in a single shot
    GGML_ASSERT(cparams.n_ubatch >= n_tokens && "encoder requires n_ubatch >= n_tokens");

    if (t_compute_start_us == 0) {
        t_compute_start_us = ggml_time_us();
    }

    // TODO: this clear of the buffer can easily be forgotten - need something better
    embd_seq.clear();

    LFG_LOG_INFO("%s: calling sched_reserve\n", __func__);
    sched_reserve();
    LFG_LOG_INFO("%s: sched_reserve done\n", __func__);

    n_queued_tokens += n_tokens;

    // reserve output buffer
    if (output_reserve(n_tokens, batch_inp) < n_tokens) {
        LFG_LOG_ERROR("%s: could not reserve space for batch with %u outputs\n", __func__, n_tokens);
        return -2;
    };

    for (uint32_t i = 0; i < n_tokens; ++i) {
        output_ids[i] = i;
    }

    n_outputs = n_tokens;

    const auto causal_attn_org = cparams.causal_attn;

    // TODO: this is a tmp solution until we have a proper way to support enc-dec models
    //       ref: https://github.com/ggml-org/liquid.cpp/pull/12181#issuecomment-2730451223
    
    bool is_enc_dec = false || false;
    
    if (is_enc_dec) {
        // always use non-causal attention for encoder graphs
        cparams.causal_attn = false;
    }

    ggml_status status;
    const auto * res = process_ubatch(ubatch, is_enc_dec ? LFG_GRAPH_TYPE_ENCODER : LFG_GRAPH_TYPE_DECODER, nullptr, status);

    if (is_enc_dec) {
        cparams.causal_attn = causal_attn_org;
    }

    if (!res) {
        switch (status) {
            case GGML_STATUS_ABORTED:      return  2;
            case GGML_STATUS_ALLOC_FAILED: return -2;
            case GGML_STATUS_FAILED:       return -3;
            case GGML_STATUS_SUCCESS:
                lfg_set_last_error(LFG_ERROR_INTERNAL, "%s: graph build failed with success status", __func__);
                return -3;
        }
    }

    auto * t_logits = res->get_logits();
    auto * t_embd = res->get_embd_pooled() ? res->get_embd_pooled() : res->get_embd();

    // extract logits
    if (logits && t_logits) {
        ggml_backend_t backend_res = ggml_backend_sched_get_tensor_backend(sched.get(), t_logits);
        GGML_ASSERT(backend_res != nullptr);
        GGML_ASSERT(logits != nullptr);

        ggml_backend_tensor_get_async(backend_res, t_logits, logits, 0, n_tokens*n_vocab*sizeof(float));
    }

    // extract embeddings
    if (embd && t_embd) {
        ggml_backend_t backend_embd = ggml_backend_sched_get_tensor_backend(sched.get(), t_embd);
        GGML_ASSERT(backend_embd != nullptr);

        switch (cparams.pooling_type) {
            case LFG_POOLING_TYPE_NONE:
                {
                    // extract token embeddings
                    GGML_ASSERT(embd != nullptr);
                    const uint32_t n_embd_out = hparams.get_n_embd_out();

                    GGML_ASSERT(n_tokens*n_embd_out <= (int64_t) embd_size);
                    ggml_backend_tensor_get_async(backend_embd, t_embd, embd, 0, n_tokens*n_embd_out*sizeof(float));
                } break;
            case LFG_POOLING_TYPE_MEAN:
            case LFG_POOLING_TYPE_CLS:
            case LFG_POOLING_TYPE_LAST:
                {
                    // extract sequence embeddings
                    auto & embd_seq_out = embd_seq;

                    for (uint32_t s = 0; s < ubatch.n_seqs_unq; ++s) {
                        const lfg_seq_id seq_id  = ubatch.seq_id_unq[s];
                        const int32_t      seq_idx = ubatch.seq_idx[seq_id];

                        embd_seq_out[seq_id].resize(n_embd);
                        ggml_backend_tensor_get_async(backend_embd, t_embd, embd_seq_out[seq_id].data(), (n_embd*seq_idx)*sizeof(float), n_embd*sizeof(float));
                    }
                } break;
            case LFG_POOLING_TYPE_RANK:
                {
                    // extract the rerank score - n_cls_out floats per sequence
                    auto & embd_seq_out = embd_seq;

                    const uint32_t n_cls_out = hparams.n_cls_out;

                    for (uint32_t s = 0; s < ubatch.n_seqs_unq; ++s) {
                        const lfg_seq_id seq_id  = ubatch.seq_id_unq[s];
                        const int32_t      seq_idx = ubatch.seq_idx[seq_id];

                        embd_seq_out[seq_id].resize(n_cls_out);
                        ggml_backend_tensor_get_async(backend_embd, t_embd, embd_seq_out[seq_id].data(), (n_cls_out*seq_idx)*sizeof(float), n_cls_out*sizeof(float));
                    }
                } break;
            case LFG_POOLING_TYPE_UNSPECIFIED:
                {
                    lfg_set_last_error(LFG_ERROR_UNSUPPORTED, "%s: unknown pooling type", __func__);
                    return -3;
                }
        }
    }

    // TODO: hacky solution
    if (false && t_embd) {
        //cross.t_embd = t_embd;

        synchronize();

        cross.n_embd = t_embd->ne[0];
        cross.n_enc  = t_embd->ne[1];
        cross.v_embd.resize(cross.n_embd*cross.n_enc);
        memcpy(cross.v_embd.data(), embd, ggml_nbytes(t_embd));

        const auto & batch = balloc->get_batch();

        // remember the sequence ids used during the encoding - needed for cross attention later
        cross.seq_ids_enc.resize(n_tokens);
        for (uint32_t i = 0; i < n_tokens; i++) {
            cross.seq_ids_enc[i].clear();

            for (int s = 0; s < batch.n_seq_id[i]; s++) {
                const lfg_seq_id seq_id = batch.seq_id[i][s];

                cross.seq_ids_enc[i].insert(seq_id);
            }
        }
    }

    return 0;
}

static std::map<lfg_seq_id, uint32_t> build_seq_to_output_row(const lfg_ubatch & ubatch, uint32_t row_offset) {
    std::map<lfg_seq_id, uint32_t> seq_to_row;
    // how many output tokens we have seen so far for this ubatch.
    uint32_t local = 0;
    for (uint32_t i = 0; i < ubatch.n_tokens; ++i) {
        // skip tokens that are not output.
        if (!ubatch.output[i]) {
            continue;
        }

        const lfg_seq_id seq_id = ubatch.seq_id[i][0];
        // row_offset is the number of output tokens before this ubatch.
        seq_to_row[seq_id] = row_offset + local;
        ++local;
    }
    return seq_to_row;
}

static void copy_tensor_async_ints(
    const std::map<lfg_seq_id, ggml_tensor*> & tensor_map,
    lfg_token * sampled,
    size_t sampled_size,
    const std::map<lfg_seq_id, uint32_t> & seq_to_row,
    ggml_backend_sched_t sched) {
    if (sampled == nullptr) {
        return;
    }

    for (const auto & [seq_id, tensor] : tensor_map) {
        auto it = seq_to_row.find(seq_id);
        if (it == seq_to_row.end()) {
            continue;
        }

        const uint32_t row = it->second;
        GGML_ASSERT(row < sampled_size);

        GGML_ASSERT(ggml_is_contiguous(tensor) && "sampled tokens tensor must be contiguous for async copy");

        ggml_backend_t backend = ggml_backend_sched_get_tensor_backend(sched, tensor);
        ggml_backend_tensor_get_async(backend, tensor, sampled + row, 0, sizeof(sampled[row]));
    }
}

static void copy_tensor_async_floats(
    const std::map<lfg_seq_id, ggml_tensor*> & tensor_map,
    float * dst,
    size_t stride,
    std::vector<uint32_t> & counts,
    const std::map<lfg_seq_id, uint32_t> & seq_to_row,
    ggml_backend_sched_t sched) {
    if (dst == nullptr) {
        return;
    }

    for (const auto & [seq_id, tensor] : tensor_map) {
        auto it = seq_to_row.find(seq_id);
        if (it == seq_to_row.end()) {
            continue;
        }

        const uint32_t row = it->second;
        GGML_ASSERT(row < counts.size());

        GGML_ASSERT(ggml_is_contiguous(tensor) && "logits/probs tensor must be contiguous for async copy");

        ggml_backend_t backend = ggml_backend_sched_get_tensor_backend(sched, tensor);
        float * row_ptr = dst + (size_t) row * stride;
        ggml_backend_tensor_get_async(backend, tensor, row_ptr, 0, ggml_nbytes(tensor));

        // Update the actual number of logits/probabilities that were written for this row.
        counts[row] = ggml_nelements(tensor);
    }
}

static void copy_tensor_async_candidates(
    const std::map<lfg_seq_id, ggml_tensor*> & tensor_map,
    lfg_token * dst,
    size_t stride,
    std::vector<uint32_t> & counts,
    const std::map<lfg_seq_id, uint32_t> & seq_to_row,
    ggml_backend_sched_t sched) {
    if (dst == nullptr) {
        return;
    }

    for (const auto & [seq_id, tensor] : tensor_map) {
        auto it = seq_to_row.find(seq_id);
        if (it == seq_to_row.end()) {
            continue;
        }

        const uint32_t row = it->second;
        GGML_ASSERT(row < counts.size());

        GGML_ASSERT(ggml_is_contiguous(tensor) && "candidates tensor must be contiguous for async copy");

        ggml_backend_t backend = ggml_backend_sched_get_tensor_backend(sched, tensor);
        lfg_token * row_ptr = dst + (size_t) row * stride;
        ggml_backend_tensor_get_async(backend, tensor, row_ptr, 0, ggml_nbytes(tensor));

        // Update the actual number of candidates that were written.
        counts[row] = ggml_nelements(tensor);
    }
}

int lfg_context::decode(const lfg_batch & batch_inp) {
    GGML_ASSERT((!batch_inp.token && batch_inp.embd) || (batch_inp.token && !batch_inp.embd)); // NOLINT

    if (!memory) {
        LFG_LOG_DEBUG("%s: cannot decode batches with this context (calling encode() instead)\n", __func__);
        return encode(batch_inp);
    }

    if (batch_inp.n_tokens == 0) {
        LFG_LOG_ERROR("%s: n_tokens == 0\n", __func__);
        return -1;
    }

    const auto & vocab   = model.vocab;
    const auto & hparams = model.hparams;

    const int64_t n_vocab = vocab.n_tokens();
    const int64_t n_embd  = hparams.n_embd_inp();

    // when computing embeddings, all tokens are output
    const bool output_all   = cparams.embeddings;
    const bool has_samplers = !sampling.samplers.empty();

    const uint32_t n_seq_max = cparams.kv_unified ? LFG_MAX_SEQ : cparams.n_seq_max;

    // TODO: avoid this workaround in the future
    if (has_samplers && batch_inp.logits) {
        std::vector<int32_t> seq_output_count(n_seq_max, 0);

        for (int32_t i = 0; i < batch_inp.n_tokens; ++i) {
            if (batch_inp.logits[i] == 0) {
                continue;
            }

            const int ns = batch_inp.n_seq_id ? batch_inp.n_seq_id[i] : 1;

            for (int32_t s = 0; s < ns; ++s) {
                const lfg_seq_id seq_id = batch_inp.seq_id ? batch_inp.seq_id[i][s] : 0;

                seq_output_count[seq_id]++;
                if (seq_output_count[seq_id] > 1) {
                    LFG_LOG_ERROR("%s: backend sampling requires at most one output token per sequence (seq_id %d had %d)\n",
                            __func__, seq_id, seq_output_count[seq_id]);
                    return -1;
                }
            }
        }
    }

    if (!balloc->init(batch_inp, vocab, memory.get(), n_embd, n_seq_max, output_all)) {
        LFG_LOG_ERROR("%s: failed to initialize batch\n", __func__);
        return -1;
    }

    const uint32_t n_tokens_all  = balloc->get_n_tokens();
    const uint32_t n_outputs_all = balloc->get_n_outputs();

    if (output_all) {
        // require that all tokens are output
        if (n_outputs_all != n_tokens_all) {
            LFG_LOG_ERROR("%s: pooled embedding requires that all tokens are output (n_outputs_all = %d, n_tokens_all = %d)\n",
                    __func__, n_outputs_all, n_tokens_all);
            return -1;
        }
    }

    GGML_ASSERT(n_tokens_all <= cparams.n_batch);

    GGML_ASSERT((cparams.causal_attn || cparams.n_ubatch >= n_tokens_all) && "non-causal attention requires n_ubatch >= n_tokens");

    if (t_compute_start_us == 0) {
        t_compute_start_us = ggml_time_us();
    }
    n_queued_tokens += n_tokens_all;

    // TODO: this clear of the buffer can easily be forgotten - need something better
    embd_seq.clear();
    output_swaps.clear();

    sched_reserve();

    bool did_optimize = false;

    // handle any pending shifts/copies
    memory_update(false);

    lfg_memory_context_ptr mctx;

    while (true) {
        mctx = memory->init_batch(*balloc, cparams.n_ubatch, output_all);
        if (!mctx) {
            return -2;
        }

        switch (mctx->get_status()) {
            case LFG_MEMORY_STATUS_SUCCESS:
                {
                } break;
            case LFG_MEMORY_STATUS_NO_UPDATE:
                {
                    LFG_LOG_ERROR("%s: unexpected memory context status: %d\n", __func__, mctx->get_status());

                    return -2;
                }
            case LFG_MEMORY_STATUS_FAILED_PREPARE:
                {
                    if (!did_optimize) {
                        did_optimize = true;

                        if (memory_update(true)) {
                            LFG_LOG_DEBUG("%s: retrying batch size %d after cache optimization\n", __func__, balloc->get_n_tokens());

                            continue;
                        }
                    }

                    LFG_LOG_WARN("%s: failed to find a memory slot for batch of size %d\n", __func__, balloc->get_n_tokens());

                    return 1;
                }
            case LFG_MEMORY_STATUS_FAILED_COMPUTE:
                {
                    LFG_LOG_ERROR("%s: compute failed while preparing batch of size %d\n", __func__, balloc->get_n_tokens());

                    return -2;
                }
        }

        break;
    }

    // reserve output buffer
    if (output_reserve(n_outputs_all, balloc->get_batch()) < n_outputs_all) {
        LFG_LOG_ERROR("%s: could not reserve space for batch with %d outputs\n", __func__, n_outputs_all);
        return -2;
    };

    int64_t n_outputs_prev = 0;

    do {
        const auto & ubatch = mctx->get_ubatch();

        // count the outputs in this ubatch
        {
            int32_t n_outputs_new = 0;

            if (n_outputs_all == n_tokens_all) {
                n_outputs_new = ubatch.n_tokens;
            } else {
                for (uint32_t i = 0; i < ubatch.n_tokens; i++) {
                    n_outputs_new += (int32_t) (ubatch.output[i] != 0);
                }
            }

            // needs to happen before the graph is built
            n_outputs = n_outputs_new;
        }

        ggml_status status;
        const auto * res = process_ubatch(ubatch, LFG_GRAPH_TYPE_DECODER, mctx.get(), status);

        if (!res) {
            // the last ubatch failed or was aborted -> remove all positions of that ubatch from the memory module
            lfg_pos pos_min[LFG_MAX_SEQ];
            for (int s = 0; s < LFG_MAX_SEQ; ++s) {
                pos_min[s] = std::numeric_limits<lfg_pos>::max();
            }

            for (uint32_t i = 0; i < ubatch.n_tokens; ++i) {
                const auto & seq_id = ubatch.seq_id[i][0];

                pos_min[seq_id] = std::min(pos_min[seq_id], ubatch.pos[i]);
            }

            for (int s = 0; s < LFG_MAX_SEQ; ++s) {
                if (pos_min[s] == std::numeric_limits<lfg_pos>::max()) {
                    continue;
                }

                LFG_LOG_WARN("%s: removing memory module entries for seq_id = %d, pos = [%d, +inf)\n", __func__, s, pos_min[s]);

                memory->seq_rm(s, pos_min[s], -1);
            }

            switch (status) {
                case GGML_STATUS_ABORTED:      return  2;
                case GGML_STATUS_ALLOC_FAILED: return -2;
                case GGML_STATUS_FAILED:       return -3;
                case GGML_STATUS_SUCCESS:
                    lfg_set_last_error(LFG_ERROR_INTERNAL, "%s: graph build failed with success status", __func__);
                    return -3;
            }
        }

        // plot the computation graph in dot format (for debugging purposes)
        //if (n_past%100 == 0) {
        //    ggml_graph_dump_dot(gf, NULL, "liquid.dot");
        //}

        auto * t_logits = res->get_logits();
        auto * t_embd   = cparams.embeddings ? res->get_embd() : nullptr;

        if (t_embd && res->get_embd_pooled()) {
            t_embd = res->get_embd_pooled();
        }

        // extract logits
        // For multi-sequence batches that mix backend samplers and CPU sampler
        // this is currently inefficient as we copy all logits even for the
        // backend sampled tokens.
        if (logits && t_logits && n_outputs > 0) {
            ggml_backend_t backend_res = ggml_backend_sched_get_tensor_backend(sched.get(), t_logits);
            GGML_ASSERT(backend_res != nullptr);
            GGML_ASSERT(logits != nullptr);

            float * logits_out = logits + n_outputs_prev*n_vocab;

            if (n_outputs) {
                GGML_ASSERT( n_outputs_prev + n_outputs <= n_outputs_all);
                GGML_ASSERT((n_outputs_prev + n_outputs)*n_vocab <= (int64_t) logits_size);
                ggml_backend_tensor_get_async(backend_res, t_logits, logits_out, 0, n_outputs*n_vocab*sizeof(float));
            }
        }

        // extract embeddings
        if (embd && t_embd && n_outputs > 0) {
            ggml_backend_t backend_embd = ggml_backend_sched_get_tensor_backend(sched.get(), t_embd);
            GGML_ASSERT(backend_embd != nullptr);

            switch (cparams.pooling_type) {
                case LFG_POOLING_TYPE_NONE:
                    {
                        // extract token embeddings
                        GGML_ASSERT(embd != nullptr);
                        const uint32_t n_embd_out = hparams.get_n_embd_out();
                        float * embd_out = embd + n_outputs_prev*n_embd_out;

                        if (n_outputs) {
                            GGML_ASSERT( n_outputs_prev + n_outputs <= n_outputs_all);
                            GGML_ASSERT((n_outputs_prev + n_outputs)*n_embd_out <= (int64_t) embd_size);
                            ggml_backend_tensor_get_async(backend_embd, t_embd, embd_out, 0, n_outputs*n_embd_out*sizeof(float));
                        }
                    } break;
                case LFG_POOLING_TYPE_MEAN:
                case LFG_POOLING_TYPE_CLS:
                case LFG_POOLING_TYPE_LAST:
                    {
                        // extract sequence embeddings (cleared before processing each batch)
                        auto & embd_seq_out = embd_seq;

                        for (uint32_t s = 0; s < ubatch.n_seqs_unq; ++s) {
                            const lfg_seq_id seq_id  = ubatch.seq_id_unq[s];
                            const int32_t      seq_idx = ubatch.seq_idx[seq_id];

                            embd_seq_out[seq_id].resize(n_embd);
                            ggml_backend_tensor_get_async(backend_embd, t_embd, embd_seq_out[seq_id].data(), (n_embd*seq_idx)*sizeof(float), n_embd*sizeof(float));
                        }
                    } break;
                case LFG_POOLING_TYPE_RANK:
                    {
                        // extract the rerank score - n_cls_out floats per sequence
                        auto & embd_seq_out = embd_seq;

                        const uint32_t n_cls_out = hparams.n_cls_out;

                        for (uint32_t s = 0; s < ubatch.n_seqs_unq; ++s) {
                            const lfg_seq_id seq_id  = ubatch.seq_id_unq[s];
                            const int32_t      seq_idx = ubatch.seq_idx[seq_id];

                            embd_seq_out[seq_id].resize(n_cls_out);
                            ggml_backend_tensor_get_async(backend_embd, t_embd, embd_seq_out[seq_id].data(), (n_cls_out*seq_idx)*sizeof(float), n_cls_out*sizeof(float));
                        }
                    } break;
                case LFG_POOLING_TYPE_UNSPECIFIED:
                    {
                        lfg_set_last_error(LFG_ERROR_UNSUPPORTED, "%s: unknown pooling type", __func__);
                        return -3;
                    }
            }
        }

        // This flag indicates whether a backend sampler has actually sampled a specific
        // token, or if it has produced probabilites. If true, we can skip the normal copying of logits and embeddings.
        const bool has_sampled = !res->t_sampled.empty() || !res->t_sampled_probs.empty() || !res->t_sampled_logits.empty();

        if (has_samplers && has_sampled) {
            const auto seq_to_output_row = build_seq_to_output_row(ubatch, n_outputs_prev);
            const auto stride = n_vocab;

            // async copy the sampling data from the backend to the host
            copy_tensor_async_ints(res->t_sampled, sampling.sampled, sampling.sampled_size, seq_to_output_row, sched.get());

            copy_tensor_async_floats    (res->t_sampled_logits, sampling.logits,     stride, sampling.logits_count,     seq_to_output_row, sched.get());
            copy_tensor_async_floats    (res->t_sampled_probs,  sampling.probs,      stride, sampling.probs_count,      seq_to_output_row, sched.get());
            copy_tensor_async_candidates(res->t_candidates,     sampling.candidates, stride, sampling.candidates_count, seq_to_output_row, sched.get());
        }

        n_outputs_prev += n_outputs;
    } while (mctx->next());

    // set to total number of outputs in the batch, for use in lfg_get_logits_ith
    n_outputs = n_outputs_all;

    // set output mappings
    if (n_outputs > 0) {
        bool sorted_output = true;

        auto & out_ids = balloc->get_out_ids();

        GGML_ASSERT(out_ids.size() == (size_t) n_outputs);

        for (int64_t i = 0; i < n_outputs; ++i) {
            int64_t out_id = out_ids[i];
            output_ids[out_id] = i;
            if (out_id != i) {
                sorted_output = false;
            }
        }

        // make the outputs have the same order they had in the user-provided batch
        // note: this is mostly relevant for recurrent models atm
        if (!sorted_output && n_outputs > 1) {
            GGML_ASSERT((size_t) n_outputs == out_ids.size());

            // TODO: is there something more efficient which also minimizes swaps?
            // selection sort, to minimize swaps (from https://en.wikipedia.org/wiki/Selection_sort)
            for (uint32_t i = 0; i < n_outputs - 1; ++i) {
                uint32_t j_min = i;
                for (uint32_t j = i + 1; j < n_outputs; ++j) {
                    if (out_ids[j] < out_ids[j_min]) {
                        j_min = j;
                    }
                }
                if (j_min == i) {
                    continue;
                }
                std::swap(out_ids[i], out_ids[j_min]);

                // remember the swaps and apply them lazily upon logits/embeddings access
                output_swaps.push_back({ i, j_min });
            }

            std::fill(output_ids.begin(), output_ids.end(), -1);

            for (uint32_t i = 0; i < n_outputs; ++i) {
                output_ids[out_ids[i]] = i;
            }
        }
    }

    // wait for the computation to finish (automatically done when obtaining the model output)
    //synchronize();

    return 0;
}

//
// output
//

uint32_t lfg_context::output_reserve(int32_t n_outputs, const lfg_batch & batch) {
    const auto & hparams = model.hparams;
    const auto & vocab   = model.vocab;

    const int64_t n_outputs_max = std::max<int64_t>(n_outputs, n_seq_max());

    const auto n_batch    = cparams.n_batch;
    const auto n_vocab    = vocab.n_tokens();
    const auto n_embd_out = hparams.get_n_embd_out();

    bool has_logits = true;
    bool has_embd   = cparams.embeddings;

    // TODO: hacky enc-dec support
    if (false) {
        has_logits = true;
        has_embd   = true;
    }

    // Check which sampling modes are needed for the current batch.
    // TODO: avoid this branching by working with the worst-case
    bool has_sampling = false;
    bool cpu_logits   = false;

    if (batch.logits) {
        for (int32_t i = 0; i < batch.n_tokens; i++) {
            if (!batch.logits[i]) {
                continue;
            }
            for (int32_t j = 0; j < batch.n_seq_id[i]; j++) {
                lfg_seq_id seq_id = batch.seq_id[i][j];
                if (sampling.samplers.find(seq_id) != sampling.samplers.end()) {
                    has_sampling = true;
                } else {
                    cpu_logits = true;
                }
            }
        }
    } else {
        // When batch.logits is nullptr (when loading state with a dummy batch),
        // allocate CPU logits.
        cpu_logits = true;
    }

    size_t backend_float_count = 0;
    size_t backend_token_count = 0;

    // Allocate CPU logits buffer only if needed by sequences in this batch
    logits_size = (has_logits && cpu_logits) ? n_vocab*n_outputs_max : 0;
    embd_size   = has_embd ? n_embd_out*n_outputs_max : 0;

    // TODO: avoid this branching by working with the worst-case
    if (!has_sampling) {
        sampling.logits_size     = 0;
        sampling.probs_size      = 0;
        sampling.sampled_size    = 0;
        sampling.candidates_size = 0;
    } else {
        sampling.logits_size     = n_vocab*n_outputs_max;
        sampling.probs_size      = n_vocab*n_outputs_max;
        sampling.sampled_size    =         n_outputs_max;
        sampling.candidates_size = n_vocab*n_outputs_max;

        backend_float_count = sampling.logits_size  + sampling.probs_size;
        backend_token_count = sampling.sampled_size + sampling.candidates_size;
    }

    if (output_ids.empty()) {
        // init, never resized afterwards
        output_ids.resize(n_batch);
    }

    const size_t prev_size = buf_output ? ggml_backend_buffer_get_size(buf_output.get()) : 0;
    const size_t new_size  =
        (logits_size + embd_size + backend_float_count) * sizeof(float) +
        (                          backend_token_count) * sizeof(lfg_token);

    // alloc only when more than the current capacity is required
    // TODO: also consider shrinking the buffer
    if (!buf_output || prev_size < new_size) {
        if (buf_output) {
#ifndef NDEBUG
            // This doesn't happen often, but may be annoying in some cases (like the HellaSwag benchmark)
            LFG_LOG_DEBUG("%s: reallocating output buffer from size %.02f MiB to %.02f MiB\n", __func__, prev_size / 1024.0 / 1024.0, new_size / 1024.0 / 1024.0);
#endif
            synchronize();

            // TODO: not needed?
            buf_output = nullptr;
            logits = nullptr;
            embd = nullptr;
        }

        auto * buft = ggml_backend_cpu_buffer_type();
        // try to use the host buffer of the device where the output tensor is allocated for faster transfer to system memory
        auto * output_dev = model.dev_output();
        auto * output_dev_host_buft = output_dev ? ggml_backend_dev_host_buffer_type(output_dev) : nullptr;
        if (output_dev_host_buft) {
            buft = output_dev_host_buft;
        }
        buf_output.reset(ggml_backend_buft_alloc_buffer(buft, new_size));
        if (buf_output == nullptr) {
            LFG_LOG_ERROR("%s: failed to allocate output buffer of size %.2f MiB\n", __func__, new_size / (1024.0 * 1024.0));
            return 0;
        }
    }

    float * output_base = (float *) ggml_backend_buffer_get_base(buf_output.get());

    logits = nullptr;
    embd   = nullptr;

    size_t offset = 0;
    uint8_t * base = (uint8_t *) output_base;

    logits = (has_logits && cpu_logits) ? output_base : nullptr;
    offset += logits_size * sizeof(float);

    embd = has_embd ? (float *) (base + offset) : nullptr;
    offset += embd_size * sizeof(float);

    sampling.logits     = nullptr;
    sampling.probs      = nullptr;
    sampling.sampled    = nullptr;
    sampling.candidates = nullptr;

    if (has_sampling) {
        sampling.logits = (float *) (base + offset);
        offset += sampling.logits_size * sizeof(float);

        sampling.probs = (float *) (base + offset);
        offset += sampling.probs_size * sizeof(float);

        sampling.sampled = (lfg_token *) (base + offset);
        offset += sampling.sampled_size * sizeof(lfg_token);

        sampling.candidates = (lfg_token *) (base + offset);
        offset += sampling.candidates_size * sizeof(lfg_token);

        // The count vectors keep track of the actual number of logits/probs/candidates
        // copied from the backend for each output row.

        sampling.logits_count.resize(n_outputs_max);
        sampling.probs_count.resize(n_outputs_max);
        sampling.candidates_count.resize(n_outputs_max);

        std::fill(sampling.logits_count.begin(),     sampling.logits_count.end(),     0);
        std::fill(sampling.probs_count.begin(),      sampling.probs_count.end(),      0);
        std::fill(sampling.candidates_count.begin(), sampling.candidates_count.end(), 0);

        std::fill_n(sampling.sampled, sampling.sampled_size, LFG_TOKEN_NULL);
    }

    // set all ids as invalid (negative)
    std::fill(output_ids.begin(), output_ids.end(), -1);

    this->n_outputs = 0;

    return n_outputs_max;
}

void lfg_context::output_reorder() {
    const uint64_t n_vocab = model.vocab.n_tokens();
    const uint64_t n_embd  = model.hparams.n_embd;

    for (size_t s = 0; s < output_swaps.size(); ++s) {
        const uint64_t i0 = output_swaps[s].i0;
        const uint64_t i1 = output_swaps[s].i1;

        if (logits_size > 0) {
            for (uint64_t k = 0; k < n_vocab; k++) {
                std::swap(logits[i0*n_vocab + k], logits[i1*n_vocab + k]);
            }
        }

        if (embd_size > 0) {
            for (uint64_t k = 0; k < n_embd; k++) {
                std::swap(embd[i0*n_embd + k], embd[i1*n_embd + k]);
            }
        }

        if (sampling.logits && sampling.logits_size > 0) {
            for (uint64_t k = 0; k < n_vocab; ++k) {
                std::swap(sampling.logits[i0*n_vocab + k], sampling.logits[i1*n_vocab + k]);
            }
        }

        if (sampling.probs && sampling.probs_size > 0) {
            for (uint64_t k = 0; k < n_vocab; ++k) {
                std::swap(sampling.probs[i0*n_vocab + k], sampling.probs[i1*n_vocab + k]);
            }
        }

        if (sampling.candidates && sampling.candidates_size > 0) {
            for (uint64_t k = 0; k < n_vocab; ++k) {
                std::swap(sampling.candidates[i0*n_vocab + k], sampling.candidates[i1*n_vocab + k]);
            }
        }

        if (sampling.sampled && sampling.sampled_size > 0) {
            std::swap(sampling.sampled[i0], sampling.sampled[i1]);
        }

        if (!sampling.logits_count.empty()) {
            std::swap(sampling.logits_count[i0], sampling.logits_count[i1]);
        }

        if (!sampling.probs_count.empty()) {
            std::swap(sampling.probs_count[i0], sampling.probs_count[i1]);
        }

        if (!sampling.candidates_count.empty()) {
            std::swap(sampling.candidates_count[i0], sampling.candidates_count[i1]);
        }
    }

    output_swaps.clear();
}

//
// graph
//

uint32_t lfg_context::graph_max_nodes(uint32_t n_tokens) const {
    if (false) {
        return std::max<uint32_t>(n_tokens * 40, 32u * model.n_tensors());
    }
    uint32_t res = std::max<uint32_t>(1024u, 8u*model.n_tensors());
    for (const auto & lora : model.loras) {
        res += lora->get_n_nodes();
    }
    return res;
}

lfg_graph_result * lfg_context::get_gf_res_reserve() const {
    return static_cast<lfg_graph_result *>(gf_res_reserve.get());
}

ggml_cgraph * lfg_context::graph_reserve(
        uint32_t n_tokens, uint32_t n_seqs, uint32_t n_outputs, const lfg_memory_context_i * mctx, bool split_only, size_t * sizes) {
    LFG_LOG_DEBUG("%s: reserving a graph for ubatch with n_tokens = %4u, n_seqs = %2u, n_outputs = %4u\n", __func__, n_tokens, n_seqs, n_outputs);
    GGML_ASSERT(n_outputs >= 1);

    if (n_tokens % n_seqs != 0) {
        n_tokens = ((n_tokens + (n_seqs - 1)) / n_seqs) * n_seqs; // round to next multiple of n_seqs
        n_outputs = std::max(n_outputs, n_tokens);

        LFG_LOG_DEBUG("%s: making n_tokens a multiple of n_seqs - n_tokens = %u, n_seqs = %u, n_outputs = %u\n", __func__, n_tokens, n_seqs, n_outputs);
    }

    ggml_backend_sched_reset(sched.get());

    // when the scheduler is reset, we cannnot reuse the old graph, so we reset the previous graph result to prevent that
    gf_res_prev->reset();

    // store the n_outputs as it is, and restore it afterwards
    // TODO: not sure if needed, might simplify in the future by removing this
    const auto save_n_outputs = this->n_outputs;

    this->n_outputs = n_outputs;

    lfg_batch_allocr balloc(model.hparams.n_pos_per_embd());
    lfg_ubatch ubatch = balloc.ubatch_reserve(n_tokens/n_seqs, n_seqs);

    // set one output token per sequence in order to activate all backend samplers
    std::vector<lfg_seq_id> seq_ids(n_seqs);
    for (uint32_t i = 0; i < n_seqs; ++i) {
        seq_ids[i] = i;
        ubatch.n_seq_id[i] = 1;
        ubatch.seq_id[i] = &seq_ids[i];
        ubatch.output[i] = true;
    }

    auto * res = gf_res_reserve.get();

    const auto gparams = graph_params(res, ubatch, mctx, LFG_GRAPH_TYPE_DEFAULT);

    res->reset();

    auto * gf = model.build_graph(gparams);

    this->n_outputs = save_n_outputs;

    // initialize scheduler with the specified graph
    if (split_only) {
        if (sizes) {
            ggml_backend_sched_reserve_size(sched.get(), gf, sizes);
        } else {
            ggml_backend_sched_split_graph(sched.get(), gf);
        }
    } else if (!ggml_backend_sched_reserve(sched.get(), gf)) {
        GGML_ASSERT(!sizes);
        LFG_LOG_ERROR("%s: failed to allocate compute buffers\n", __func__);
        return nullptr;
    }

    return gf;
}

lfg_graph_params lfg_context::graph_params(
                        lfg_graph_result * res,
                      const lfg_ubatch & ubatch,
            const lfg_memory_context_i * mctx,
                          lfg_graph_type   gtype) const {
    return {
        /*.arch        =*/ model.arch,
        /*.hparams     =*/ model.hparams,
        /*.cparams     =*/ cparams,
        /*.ubatch      =*/ ubatch,
        /*.gtype       =*/ gtype,
        /*.sched       =*/ sched.get(),
        /*.backend_cpu =*/ backend_cpu,
        /*.cvec        =*/ &cvec,
        /*.loras       =*/ &loras,
        /*.mctx        =*/ mctx,
        /*.cross       =*/ &cross,
        /*.samplers    =*/ sampling.samplers,
        /*.n_outputs   =*/ n_outputs,
        /*.cb          =*/ graph_get_cb(),
        /*.res         =*/ res,
    };
}

ggml_status lfg_context::graph_compute(
            ggml_cgraph * gf,
                   bool   batched) {
    int n_threads        = batched ? cparams.n_threads_batch : cparams.n_threads;
    ggml_threadpool_t tp = batched ? threadpool_batch        : threadpool;

    if (backend_cpu != nullptr) {
        auto * reg = ggml_backend_dev_backend_reg(ggml_backend_get_device(backend_cpu));
        auto * set_threadpool_fn = (decltype(ggml_backend_cpu_set_threadpool) *) ggml_backend_reg_get_proc_address(reg, "ggml_backend_cpu_set_threadpool");
        if (set_threadpool_fn) {
            set_threadpool_fn(backend_cpu, tp);
        }
    }

    // set the number of threads for all the backends
    for (const auto & set_n_threads_fn : set_n_threads_fns) {
        set_n_threads_fn.second(set_n_threads_fn.first, n_threads);
    }

    auto status = ggml_backend_sched_graph_compute_async(sched.get(), gf);
    if (status != GGML_STATUS_SUCCESS) {
        LFG_LOG_ERROR("%s: ggml_backend_sched_graph_compute_async failed with error %d\n", __func__, status);
    }

    // fprintf(stderr, "splits: %d\n", ggml_backend_sched_get_n_splits(sched));

    return status;
}

lfg_graph_cb lfg_context::graph_get_cb() const {
    return [&](const lfg_ubatch & ubatch, ggml_tensor * cur, const char * name, int il) {
        if (il >= 0) {
            ggml_format_name(cur, "%s-%d", name, il);
        } else {
            ggml_set_name(cur, name);
        }

        if (!cparams.offload_kqv) {
            if (strcmp(name, "kqv_merged_cont") == 0) {
                // all nodes between the KV store and the attention output are run on the CPU
                ggml_backend_sched_set_tensor_backend(sched.get(), cur, backend_cpu);
            }
        }

        // norm may be automatically assigned to the backend of the previous layer, increasing data transfer between backends
        // FIXME: fix in ggml_backend_sched
        const bool full_offload = model.n_gpu_layers() > model.hparams.n_layer;
        if (ubatch.n_tokens < 32 || full_offload) {
            if (il != -1 && strcmp(name, "norm") == 0) {
                const auto & dev_layer = model.dev_layer(il);
                for (const auto & backend : backends) {
                    if (ggml_backend_get_device(backend.get()) == dev_layer) {
                        if (ggml_backend_supports_op(backend.get(), cur)) {
                            ggml_backend_sched_set_tensor_backend(sched.get(), cur, backend.get());
                        }
                    }
                }
            }
        }
    };
}

//
// state save/load
//

class lfg_io_write_dummy : public lfg_io_write_i {
public:
    lfg_io_write_dummy() = default;

    void write(const void * /* src */, size_t size) override {
        size_written += size;
    }

    void write_tensor(const ggml_tensor * /* tensor */, size_t /* offset */, size_t size) override {
        size_written += size;
    }

    size_t n_bytes() override {
        return size_written;
    }

private:
    size_t size_written = 0;
};

class lfg_io_write_buffer : public lfg_io_write_i {
public:
    lfg_io_write_buffer(
            uint8_t * p, size_t len) : ptr(p), buf_size(len) {}

    void write(const void * src, size_t size) override {
        if (size > buf_size) {
            throw std::runtime_error("unexpectedly reached end of buffer");
        }
        memcpy(ptr, src, size);
        ptr += size;
        size_written += size;
        buf_size -= size;
    }

    void write_tensor(const ggml_tensor * tensor, size_t offset, size_t size) override {
        if (size > buf_size) {
            throw std::runtime_error("unexpectedly reached end of buffer");
        }
        ggml_backend_tensor_get(tensor, ptr, offset, size);
        ptr += size;
        size_written += size;
        buf_size -= size;
    }

    size_t n_bytes() override {
        return size_written;
    }

private:
    uint8_t * ptr;
    size_t buf_size = 0;
    size_t size_written = 0;
};

class lfg_io_read_buffer : public lfg_io_read_i {
public:
    lfg_io_read_buffer(const uint8_t * p, size_t len) : ptr(p), buf_size(len) {}

    const uint8_t * read(size_t size) override {
        const uint8_t * base_ptr = ptr;
        if (size > buf_size) {
            throw std::runtime_error("unexpectedly reached end of buffer");
        }
        ptr += size;
        size_read += size;
        buf_size -= size;
        return base_ptr;
    }

    void read_to(void * dst, size_t size) override {
        memcpy(dst, read(size), size);
    }

    size_t n_bytes() override {
        return size_read;
    }

private:
    const uint8_t * ptr;
    size_t buf_size = 0;
    size_t size_read = 0;
};

class lfg_io_write_file : public lfg_io_write_i {
public:
    lfg_io_write_file(lfg_file * f) : file(f) {}

    void write(const void * src, size_t size) override {
        file->write_raw(src, size);
        size_written += size;
    }

    void write_tensor(const ggml_tensor * tensor, size_t offset, size_t size) override {
        temp_buffer.resize(size);
        ggml_backend_tensor_get(tensor, temp_buffer.data(), offset, size);
        write(temp_buffer.data(), temp_buffer.size());
    }

    size_t n_bytes() override {
        return size_written;
    }

private:
    lfg_file * file;
    size_t size_written = 0;
    std::vector<uint8_t> temp_buffer;
};

class lfg_io_read_file : public lfg_io_read_i {
public:
    lfg_io_read_file(lfg_file * f) : file(f) {}

    void read_to(void * dst, size_t size) override {
        file->read_raw(dst, size);
        size_read += size;
    }

    const uint8_t * read(size_t size) override {
        temp_buffer.resize(size);
        read_to(temp_buffer.data(), size);
        return temp_buffer.data();
    }

    size_t n_bytes() override {
        return size_read;
    }

private:
    lfg_file * file;
    size_t size_read = 0;
    std::vector<uint8_t> temp_buffer;
};

size_t lfg_context::state_get_size() {
    lfg_io_write_dummy io;
    try {
        return state_write_data(io);
    } catch (const std::exception & err) {
        LFG_LOG_ERROR("%s: error getting state size: %s\n", __func__, err.what());
        return 0;
    }
}

size_t lfg_context::state_get_data(uint8_t * dst, size_t size) {
    lfg_io_write_buffer io(dst, size);
    try {
        return state_write_data(io);
    } catch (const std::exception & err) {
        LFG_LOG_ERROR("%s: error saving state: %s\n", __func__, err.what());
        return 0;
    }
}

size_t lfg_context::state_set_data(const uint8_t * src, size_t size) {
    lfg_io_read_buffer io(src, size);
    try {
        return state_read_data(io);
    } catch (const std::exception & err) {
        LFG_LOG_ERROR("%s: error loading state: %s\n", __func__, err.what());
        return 0;
    }
}

size_t lfg_context::state_seq_get_size(lfg_seq_id seq_id, lfg_state_seq_flags flags) {
    lfg_io_write_dummy io;
    try {
        return state_seq_write_data(io, seq_id, flags);
    } catch (const std::exception & err) {
        LFG_LOG_ERROR("%s: error getting state size: %s\n", __func__, err.what());
        return 0;
    }
}

size_t lfg_context::state_seq_get_data(lfg_seq_id seq_id, uint8_t * dst, size_t size, lfg_state_seq_flags flags) {
    lfg_io_write_buffer io(dst, size);
    try {
        return state_seq_write_data(io, seq_id, flags);
    } catch (const std::exception & err) {
        LFG_LOG_ERROR("%s: error saving state: %s\n", __func__, err.what());
        return 0;
    }
}

size_t lfg_context::state_seq_set_data(lfg_seq_id seq_id, const uint8_t * src, size_t size, lfg_state_seq_flags flags) {
    lfg_io_read_buffer io(src, size);
    try {
        return state_seq_read_data(io, seq_id, flags);
    } catch (const std::exception & err) {
        LFG_LOG_ERROR("%s: error loading state: %s\n", __func__, err.what());
        return 0;
    }
}

bool lfg_context::state_load_file(const char * filepath, lfg_token * tokens_out, size_t n_token_capacity, size_t * n_token_count_out) {
    lfg_file file(filepath, "rb");

    // sanity checks
    {
        const uint32_t magic   = file.read_u32();
        const uint32_t version = file.read_u32();

        if (magic != LFG_SESSION_MAGIC || version != LFG_SESSION_VERSION) {
            LFG_LOG_ERROR("%s: unknown (magic, version) for session file: %08x, %08x\n", __func__, magic, version);
            return false;
        }
    }

    // load the prompt
    {
        const uint32_t n_token_count = file.read_u32();

        if (n_token_count > n_token_capacity) {
            LFG_LOG_ERROR("%s: token count in session file exceeded capacity! %u > %zu\n", __func__, n_token_count, n_token_capacity);
            return false;
        }

        file.read_raw(tokens_out, sizeof(lfg_token) * n_token_count);
        *n_token_count_out = n_token_count;
    }

    // restore the context state
    {
        const size_t n_state_size_cur = file.size() - file.tell();

        lfg_io_read_file io( &file);
        const size_t n_read = state_read_data(io);

        if (n_read != n_state_size_cur) {
            LFG_LOG_ERROR("%s: did not read all of the session file data! size %zu, got %zu\n", __func__, n_state_size_cur, n_read);
            return false;
        }
    }

    return true;
}

bool lfg_context::state_save_file(const char * filepath, const lfg_token * tokens, size_t n_token_count) {
    lfg_file file(filepath, "wb");

    file.write_u32(LFG_SESSION_MAGIC);
    file.write_u32(LFG_SESSION_VERSION);

    // save the prompt
    file.write_u32((uint32_t) n_token_count);
    file.write_raw(tokens, sizeof(lfg_token) * n_token_count);

    // save the context state using stream saving
    lfg_io_write_file io(&file);
    state_write_data(io);

    return true;
}

size_t lfg_context::state_seq_load_file(lfg_seq_id seq_id, const char * filepath, lfg_token * tokens_out, size_t n_token_capacity, size_t * n_token_count_out) {
    lfg_file file(filepath, "rb");

    // version checks
    {
        const uint32_t magic   = file.read_u32();
        const uint32_t version = file.read_u32();

        if (magic != LFG_STATE_SEQ_MAGIC || version != LFG_STATE_SEQ_VERSION) {
            LFG_LOG_ERROR("%s: unknown (magic, version) for sequence state file: %08x, %08x\n", __func__, magic, version);
            return 0;
        }
    }

    // load the prompt
    {
        const uint32_t n_token_count = file.read_u32();

        if (n_token_count > n_token_capacity) {
            LFG_LOG_ERROR("%s: token count in sequence state file exceeded capacity! %u > %zu\n", __func__, n_token_count, n_token_capacity);
            return 0;
        }

        file.read_raw(tokens_out, sizeof(lfg_token) * n_token_count);
        *n_token_count_out = n_token_count;
    }

    // restore the context state
    {
        const size_t state_size = file.size() - file.tell();
        lfg_io_read_file io(&file);
        const size_t nread = state_seq_read_data(io, seq_id, 0);
        if (!nread) {
            LFG_LOG_ERROR("%s: failed to restore sequence state\n", __func__);
            return 0;
        }
        GGML_ASSERT(nread <= state_size);
        GGML_ASSERT(nread + sizeof(uint32_t) * 3 + sizeof(lfg_token) * *n_token_count_out == file.tell());
    }

    return file.tell();
}

size_t lfg_context::state_seq_save_file(lfg_seq_id seq_id, const char * filepath, const lfg_token * tokens, size_t n_token_count) {
    lfg_file file(filepath, "wb");

    file.write_u32(LFG_STATE_SEQ_MAGIC);
    file.write_u32(LFG_STATE_SEQ_VERSION);

    // save the prompt
    file.write_u32((uint32_t) n_token_count);
    file.write_raw(tokens, sizeof(lfg_token) * n_token_count);

    // save the context state using stream saving
    lfg_io_write_file io(&file);
    state_seq_write_data(io, seq_id, 0);

    const size_t res = file.tell();
    GGML_ASSERT(res == sizeof(uint32_t) * 3 + sizeof(lfg_token) * n_token_count + io.n_bytes());

    return res;
}

size_t lfg_context::state_write_data(lfg_io_write_i & io) {
    const int64_t n_vocab = model.vocab.n_tokens();
    const int64_t n_embd  = model.hparams.n_embd;

    // write model info
    {
        const std::string arch_str = lfg_arch_name(model.arch);
        io.write_string(arch_str);
        // TODO: add more model-specific info which should prevent loading the session file if not identical
    }

    // write output ids
    {
        const auto n_outputs    = this->n_outputs;
        const auto & output_ids = this->output_ids;

        std::vector<int32_t> w_output_pos;

        w_output_pos.resize(n_outputs);

        // build a more compact representation of the output ids
        for (size_t i = 0; i < n_batch(); ++i) {
            // map an output id to a position in the batch
            int64_t pos = output_ids[i];
            if (pos >= 0) {
                GGML_ASSERT(pos < n_outputs);
                w_output_pos[pos] = i;
            }
        }

        io.write(&n_outputs, sizeof(n_outputs));

        if (n_outputs) {
            io.write(w_output_pos.data(), n_outputs * sizeof(int32_t));
        }
    }

    // write logits
    {
        const uint64_t logits_size = logits ? n_outputs * n_vocab : 0;
        io.write(&logits_size, sizeof(logits_size));

        if (logits) {
            io.write(logits, logits_size * sizeof(float));
        }
    }

    // write embeddings
    {
        const uint64_t embd_size = embd ? n_outputs * n_embd : 0;
        io.write(&embd_size, sizeof(embd_size));

        if (embd) {
            io.write(embd, embd_size * sizeof(float));
        }
    }

    // write memory module
    {
        memory->state_write(io);
    }

    return io.n_bytes();
}

size_t lfg_context::state_read_data(lfg_io_read_i & io) {
    // read model info
    {
        const std::string cur_arch_str = lfg_arch_name(model.arch);

        std::string arch_str;
        io.read_string(arch_str);
        if (cur_arch_str != arch_str) {
            throw std::runtime_error(lfg_format("wrong model arch: '%s' instead of '%s'", arch_str.c_str(), cur_arch_str.c_str()));
        }
        // TODO: add more info which needs to be identical but which is not verified otherwise
    }

    // read output ids
    {
        auto n_outputs = this->n_outputs;
        io.read_to(&n_outputs, sizeof(n_outputs));

        // Create a dummy batch for state loading.
        lfg_batch dummy_batch = {};
        dummy_batch.n_tokens = 0;
        if (n_outputs > output_reserve(n_outputs, dummy_batch)) {
            throw std::runtime_error("could not reserve outputs");
        }

        std::vector<int32_t> output_pos;

        if (n_outputs) {
            output_pos.resize(n_outputs);
            io.read_to(output_pos.data(), n_outputs * sizeof(int32_t));

            for (int32_t i = 0; i < (int32_t) output_pos.size(); ++i) {
                int32_t id = output_pos[i];
                if ((uint32_t) id >= n_batch()) {
                    throw std::runtime_error(lfg_format("invalid output id, %d does not fit in batch size of %u", id, n_batch()));
                }
                this->output_ids[id] = i;
            }

            this->n_outputs = n_outputs;
        }
    }

    // read logits
    {
        uint64_t logits_size;
        io.read_to(&logits_size, sizeof(logits_size));

        if (this->logits_size < logits_size) {
            throw std::runtime_error("logits buffer too small");
        }

        if (logits_size) {
            io.read_to(this->logits, logits_size * sizeof(float));
        }
    }

    // read embeddings
    {
        uint64_t embd_size;
        io.read_to(&embd_size, sizeof(embd_size));

        if (this->embd_size < embd_size) {
            throw std::runtime_error("embeddings buffer too small");
        }

        if (embd_size) {
            io.read_to(this->embd, embd_size * sizeof(float));
        }
    }

    // TODO: handle sampling buffers and samplers state ?
    //       https://github.com/ggml-org/liquid.cpp/pull/17004

    if (memory) {
        memory->state_read(io);
    }

    return io.n_bytes();
}

size_t lfg_context::state_seq_write_data(lfg_io_write_i & io, lfg_seq_id seq_id, lfg_state_seq_flags flags) {
    GGML_UNUSED(seq_id);

    if (memory) {
        memory->state_write(io, seq_id, flags);
    }

    return io.n_bytes();
}

size_t lfg_context::state_seq_read_data(lfg_io_read_i & io, lfg_seq_id seq_id, lfg_state_seq_flags flags) {
    GGML_UNUSED(seq_id);

    if (memory) {
        memory->state_read(io, seq_id, flags);
    }

    return io.n_bytes();
}

//
// perf
//

lfg_perf_context_data lfg_context::perf_get_data() const {
    lfg_perf_context_data data = {};

    data.t_start_ms  = 1e-3 * t_start_us;
    data.t_load_ms   = 1e-3 * t_load_us;
    data.t_p_eval_ms = 1e-3 * t_p_eval_us;
    data.t_eval_ms   = 1e-3 * t_eval_us;
    data.n_p_eval    = std::max(1, n_p_eval);
    data.n_eval      = std::max(1, n_eval);
    data.n_reused    = std::max(0, n_reused);

    return data;
}

void lfg_context::perf_reset() {
    t_start_us  = ggml_time_us();
    t_eval_us   = n_eval = 0;
    t_p_eval_us = n_p_eval = 0;
    n_reused    = 0;
}

std::map<ggml_backend_buffer_type_t, lfg_memory_breakdown_data> lfg_context::memory_breakdown() const {
    std::map<ggml_backend_buffer_type_t, lfg_memory_breakdown_data> ret;
    for (const auto & [buft, size] : model.memory_breakdown()) {
        ret[buft].model += size;
    }
    if (memory) {
        for (const auto & [buft, size] : memory->memory_breakdown()) {
            ret[buft].context += size;
        }
    }
    if (model.hparams.no_alloc) {
        for (size_t i = 0; i < backends.size(); ++i) {
            ggml_backend_t             backend = backends[i].get();
            ggml_backend_buffer_type_t buft    = ggml_backend_sched_get_buffer_type(sched.get(), backend);
            ret[buft].compute += backend_buf_exp_size[i];
        }
    } else {
        for (const auto & backend_ptr : backends) {
            ggml_backend_t             backend = backend_ptr.get();
            ggml_backend_buffer_type_t buft    = ggml_backend_sched_get_buffer_type(sched.get(), backend);
            ret[buft].compute += ggml_backend_sched_get_buffer_size(sched.get(), backend);
        }
    }
    return ret;
}

//
// training
//

static void lfg_set_param(struct ggml_tensor * tensor, lfg_opt_param_filter param_filter, void * userdata) {
    if (!tensor || tensor->type != GGML_TYPE_F32) {
        return;
    }
    if (!param_filter(tensor, userdata)) {
        return;
    }
    if (strcmp(tensor->name, "token_embd.weight") == 0) {
        return; // FIXME
    }
    if (strcmp(tensor->name, "rope_freqs.weight") == 0) {
        return; // FIXME
    }
    ggml_set_param(tensor);
}

void lfg_context::opt_init(struct lfg_model * model_ptr, struct lfg_opt_params lopt_params) {
    GGML_ASSERT(!opt_ctx);
    model_ptr->hparams.n_ctx_train = lopt_params.n_ctx_train > 0 ? lopt_params.n_ctx_train : n_ctx();
    const uint32_t n_batch     = std::min(this->n_batch(),  model_ptr->hparams.n_ctx_train);
    const uint32_t n_ubatch    = std::min(this->n_ubatch(), n_batch);
    GGML_ASSERT(model_ptr->hparams.n_ctx_train % n_batch  == 0);
    GGML_ASSERT(n_batch                    % n_ubatch == 0);

    ggml_opt_params opt_params = ggml_opt_default_params(sched.get(), GGML_OPT_LOSS_TYPE_CROSS_ENTROPY);
    opt_params.opt_period      = (int32_t)(n_batch / n_ubatch);
    opt_params.get_opt_pars    = lopt_params.get_opt_pars;
    opt_params.get_opt_pars_ud = lopt_params.get_opt_pars_ud;
    opt_params.optimizer       = lopt_params.optimizer_type;
    opt_ctx = ggml_opt_init(opt_params);

    lfg_opt_param_filter param_filter = lopt_params.param_filter;
    void * param_filter_ud              = lopt_params.param_filter_ud;

  //lfg_set_param(model->tok_embd,        param_filter, param_filter_ud); // FIXME
    lfg_set_param(model_ptr->type_embd,       param_filter, param_filter_ud);
    lfg_set_param(model_ptr->pos_embd,        param_filter, param_filter_ud);
    lfg_set_param(model_ptr->tok_norm,        param_filter, param_filter_ud);
    lfg_set_param(model_ptr->tok_norm_b,      param_filter, param_filter_ud);
    lfg_set_param(model_ptr->output_norm,     param_filter, param_filter_ud);
    lfg_set_param(model_ptr->output_norm_b,   param_filter, param_filter_ud);
    lfg_set_param(model_ptr->output,          param_filter, param_filter_ud);
    lfg_set_param(model_ptr->output_b,        param_filter, param_filter_ud);
    lfg_set_param(model_ptr->output_norm_enc, param_filter, param_filter_ud);
    lfg_set_param(model_ptr->cls,             param_filter, param_filter_ud);
    lfg_set_param(model_ptr->cls_b,           param_filter, param_filter_ud);
    lfg_set_param(model_ptr->cls_out,         param_filter, param_filter_ud);
    lfg_set_param(model_ptr->cls_out_b,       param_filter, param_filter_ud);

    for (struct lfg_layer & layer : model_ptr->layers) {
        for (size_t i = 0; i < sizeof(layer)/sizeof(struct ggml_tensor *); ++i) {
            lfg_set_param(reinterpret_cast<struct ggml_tensor **>(&layer)[i], param_filter, param_filter_ud);
        }
    }
}

void lfg_context::opt_epoch_iter(
        ggml_opt_dataset_t               dataset,
        ggml_opt_result_t                result,
        const std::vector<lfg_token> & tokens,
        const std::vector<lfg_token> & labels_sparse,
        lfg_batch                    & batch,
        ggml_opt_epoch_callback          callback,
        bool                             train,
        int64_t                          idata_in_loop,
        int64_t                          ndata_in_loop,
        int64_t                          t_loop_start) {
    GGML_ASSERT(opt_ctx);
    const uint32_t n_ctx    = (uint32_t)lfg_model_n_ctx_train(&model);
    const uint32_t n_batch  = std::min(this->n_batch(),  n_ctx);
    const uint32_t n_ubatch = std::min(this->n_ubatch(), n_batch);

    memory->clear(true);

    for (uint32_t pos_ctx = 0; pos_ctx < n_ctx; pos_ctx += n_batch) {
        batch.n_tokens = (int32_t)n_batch;
        for (uint32_t pos_batch = 0; pos_batch < n_batch; ++pos_batch) {
            batch.token   [pos_batch]    = tokens[pos_ctx + pos_batch];
            batch.pos     [pos_batch]    = static_cast<lfg_pos>(pos_ctx + pos_batch);
            batch.n_seq_id[pos_batch]    = 1;
            batch.seq_id  [pos_batch][0] = 0;
            batch.logits  [pos_batch]    = true;
        }

        if (!balloc->init(batch, model.vocab, nullptr, model.hparams.n_embd_inp(), cparams.kv_unified ? LFG_MAX_SEQ : cparams.n_seq_max, true)) {
            LFG_LOG_ERROR("%s: failed to initialize batch\n", __func__);
            return;
        }

        const uint32_t n_tokens_all = balloc->get_n_tokens();

        n_queued_tokens += n_tokens_all;

        embd_seq.clear();

        uint32_t n_outputs_all = n_tokens_all;

        auto mctx = memory->init_batch(*balloc, cparams.n_ubatch, true);
        if (!mctx || mctx->get_status() != LFG_MEMORY_STATUS_SUCCESS) {
            LFG_LOG_ERROR("%s: could not initialize batch\n", __func__);
            break;
        }

        // reserve output buffer
        if (output_reserve(n_outputs_all, balloc->get_batch()) < n_outputs_all) {
            LFG_LOG_ERROR("%s: could not reserve space for batch with %d outputs\n", __func__, n_outputs_all);
            lfg_set_last_error(LFG_ERROR_OUT_OF_MEMORY, "%s: could not reserve space for batch outputs", __func__);
            return;
        };

        uint32_t pos_batch = 0;
        do {
            const auto & ubatch = mctx->get_ubatch();

            n_outputs = ubatch.n_tokens;

            if (!mctx->apply()) {
                LFG_LOG_ERROR("%s: failed to update the memory context\n", __func__);
                break;
            }

            auto * res = gf_res_prev.get();

            const auto gparams = graph_params(res, ubatch, mctx.get(), LFG_GRAPH_TYPE_DEFAULT);

            res->reset();

            auto * gf = model.build_graph(gparams);

            struct ggml_context * ctx_compute_opt;
            {
                const size_t size_gf = (size_t)ggml_graph_size(gf);
                const size_t size_meta = 4*size_gf*ggml_tensor_overhead() + 2*ggml_graph_overhead_custom(size_gf, /*grads = */ true);
                struct ggml_init_params params = {
                    /*.mem_size   =*/ size_meta,
                    /*.mem_buffer =*/ nullptr,
                    /*.no_alloc   =*/ true,
                };
                ctx_compute_opt = ggml_init(params);
            }
            ggml_opt_prepare_alloc(opt_ctx, ctx_compute_opt, gf, res->get_tokens(), res->get_logits());
            ggml_opt_alloc(opt_ctx, train);

            res->set_inputs(&ubatch);
            {
                struct ggml_tensor * labels = ggml_opt_labels(opt_ctx);
                GGML_ASSERT(labels->ne[1] == n_ubatch);
                ggml_set_zero(labels);
                const float onef = 1.0f;
                for (uint32_t pos_ubatch = 0; pos_ubatch < n_ubatch; ++pos_ubatch) {
                    const uint32_t ilabel = pos_ctx + pos_batch + pos_ubatch;
                    GGML_ASSERT(labels_sparse[ilabel] < labels->ne[0]);
                    ggml_backend_tensor_set(labels, &onef, (pos_ubatch*labels->ne[0] + labels_sparse[ilabel])*sizeof(float), sizeof(float));
                }
            }
            ggml_opt_eval(opt_ctx, result);
            if (callback) {
                callback(train, opt_ctx, dataset, result, idata_in_loop + (pos_ctx + pos_batch)/n_ubatch + 1, ndata_in_loop, t_loop_start);
            }
            ggml_free(ctx_compute_opt);

            pos_batch += ubatch.n_tokens;
        } while (mctx->next());
    }
}

void lfg_context::opt_epoch(
        ggml_opt_dataset_t        dataset,
        ggml_opt_result_t         result_train,
        ggml_opt_result_t         result_eval,
        int64_t                   idata_split,
        ggml_opt_epoch_callback   callback_train,
        ggml_opt_epoch_callback   callback_eval) {
    const uint32_t n_ctx    = this->n_ctx();
    const uint32_t n_batch  = std::min(cparams.n_batch,  n_ctx);
    const uint32_t n_ubatch = std::min(cparams.n_ubatch, n_batch);
    const  int64_t ndata    = ggml_opt_dataset_ndata(dataset);

    GGML_ASSERT(idata_split >= 0);
    GGML_ASSERT(idata_split <= ndata);

    const uint32_t ubatch_per_ctx = n_ctx / n_ubatch;

    struct lfg_batch batch = lfg_batch_init((int32_t)n_batch, 0, 1);
    std::vector<lfg_token>        tokens(n_ctx);
    std::vector<lfg_token> labels_sparse(n_ctx);

    int64_t idata = 0;

    int64_t t_loop_start = ggml_time_us();
    int64_t ndata_in_loop = idata_split*ubatch_per_ctx;
    for (; idata < idata_split; ++idata) {
        constexpr bool train = true;
        const int64_t idata_in_loop = idata*ubatch_per_ctx;

        ggml_opt_dataset_get_batch_host(dataset, tokens.data(), n_ctx*sizeof(lfg_token), labels_sparse.data(), idata);
        opt_epoch_iter(dataset, result_train, tokens, labels_sparse, batch,
            callback_train, train, idata_in_loop, ndata_in_loop, t_loop_start);
    }

    t_loop_start = ggml_time_us();
    ndata_in_loop = (ndata - idata_split)*ubatch_per_ctx;
    for (; idata < ndata; ++idata) {
        constexpr bool train = false;
        const int64_t idata_in_loop = (idata - idata_split)*ubatch_per_ctx;

        ggml_opt_dataset_get_batch_host(dataset, tokens.data(), n_ctx*sizeof(lfg_token), labels_sparse.data(), idata);
        opt_epoch_iter(dataset, result_eval, tokens, labels_sparse, batch,
            callback_eval, train, idata_in_loop, ndata_in_loop, t_loop_start);
    }

    lfg_batch_free(batch);
}

//
// interface implementation
//

lfg_context_params lfg_context_default_params() {
    lfg_context_params result = {
        /*.n_ctx                       =*/ 512,
        /*.n_batch                     =*/ 2048,
        /*.n_ubatch                    =*/ 512,
        /*.n_seq_max                   =*/ 1,
        /*.n_threads                   =*/ GGML_DEFAULT_N_THREADS, // TODO: better default
        /*.n_threads_batch             =*/ GGML_DEFAULT_N_THREADS,
        /*.rope_scaling_type           =*/ LFG_ROPE_SCALING_TYPE_UNSPECIFIED,
        /*.pooling_type                =*/ LFG_POOLING_TYPE_UNSPECIFIED,
        /*.attention_type              =*/ LFG_ATTENTION_TYPE_UNSPECIFIED,
        /*.flash_attn_type             =*/ LFG_FLASH_ATTN_TYPE_AUTO,
        /*.rope_freq_base              =*/ 0.0f,
        /*.rope_freq_scale             =*/ 0.0f,
        /*.yarn_ext_factor             =*/ -1.0f,
        /*.yarn_attn_factor            =*/ -1.0f,
        /*.yarn_beta_fast              =*/ -1.0f,
        /*.yarn_beta_slow              =*/ -1.0f,
        /*.yarn_orig_ctx               =*/ 0,
        /*.defrag_thold                =*/ -1.0f,
        /*.cb_eval                     =*/ nullptr,
        /*.cb_eval_user_data           =*/ nullptr,
        /*.type_k                      =*/ GGML_TYPE_F16,
        /*.type_v                      =*/ GGML_TYPE_F16,
        /*.abort_callback              =*/ nullptr,
        /*.abort_callback_data         =*/ nullptr,
        /*.embeddings                  =*/ false,
        /*.offload_kqv                 =*/ true,
        /*.no_perf                     =*/ true,
        /*.op_offload                  =*/ true,
        /*.swa_full                    =*/ true,
        /*.kv_unified                  =*/ false,
        /*.sampler                     =*/ nullptr,
        /*.n_sampler                   =*/ 0,
    };

    return result;
}

lfg_context * lfg_init_from_model(
                 lfg_model * model,
        lfg_context_params   params) {
    if (!model) {
        LFG_LOG_ERROR("%s: model cannot be NULL\n", __func__);
        lfg_set_last_error(LFG_ERROR_INVALID_ARGUMENT, "%s: model is NULL", __func__);
        return nullptr;
    }

    if (params.n_batch == 0 && params.n_ubatch == 0) {
        LFG_LOG_ERROR("%s: n_batch and n_ubatch cannot both be zero\n", __func__);
        lfg_set_last_error(LFG_ERROR_INVALID_ARGUMENT, "%s: n_batch and n_ubatch cannot both be zero", __func__);
        return nullptr;
    }

    if (params.n_ctx == 0 && model->hparams.n_ctx_train == 0) {
        LFG_LOG_ERROR("%s: n_ctx and model->hparams.n_ctx_train cannot both be zero\n", __func__);
        lfg_set_last_error(LFG_ERROR_INVALID_ARGUMENT, "%s: n_ctx and model->hparams.n_ctx_train cannot both be zero", __func__);
        return nullptr;
    }

    if (params.flash_attn_type == LFG_FLASH_ATTN_TYPE_AUTO && ggml_is_quantized(params.type_k)) {
        const uint32_t blck_size = (uint32_t)ggml_blck_size(params.type_k);
        if (model->hparams.n_embd_head_k % blck_size != 0) {
            LFG_LOG_ERROR("%s: K cache type %s with block size %u does not divide n_embd_head_k=%u\n",
                __func__, ggml_type_name(params.type_k), blck_size, model->hparams.n_embd_head_k);
            lfg_set_last_error(LFG_ERROR_INVALID_ARGUMENT, "%s: invalid K cache type %s for n_embd_head_k=%u",
                __func__, ggml_type_name(params.type_k), model->hparams.n_embd_head_k);
            return nullptr;
        }
    }

    if (params.flash_attn_type == LFG_FLASH_ATTN_TYPE_AUTO && ggml_is_quantized(params.type_v)) {
        const uint32_t blck_size = (uint32_t)ggml_blck_size(params.type_v);
        if (model->hparams.n_embd_head_v % blck_size != 0) {
            LFG_LOG_ERROR("%s: V cache type %s with block size %u does not divide n_embd_head_k=%u\n",
                __func__, ggml_type_name(params.type_v), blck_size, model->hparams.n_embd_head_v);
            lfg_set_last_error(LFG_ERROR_INVALID_ARGUMENT, "%s: invalid V cache type %s for n_embd_head_v=%u",
                __func__, ggml_type_name(params.type_v), model->hparams.n_embd_head_v);
            return nullptr;
        }
    }

    if (ggml_is_quantized(params.type_v) && params.flash_attn_type == LFG_FLASH_ATTN_TYPE_DISABLED) {
        LFG_LOG_ERROR("%s: V cache quantization requires flash_attn\n", __func__);
        lfg_set_last_error(LFG_ERROR_INVALID_ARGUMENT, "%s: V cache quantization requires flash_attn", __func__);
        return nullptr;
    }

    if (params.pooling_type != LFG_POOLING_TYPE_UNSPECIFIED &&
        params.pooling_type != model->hparams.pooling_type) {
        //user-specified pooling-type is different from the model default
        LFG_LOG_WARN("%s: model default pooling_type is [%d], but [%d] was specified\n", __func__,
                       model->hparams.pooling_type, params.pooling_type);
    }

    try {
        auto * ctx = new lfg_context(*model, params);
        return ctx;
    } catch (const std::exception & err) {
        LFG_LOG_ERROR("%s: failed to initialize the context: %s\n", __func__, err.what());
        lfg_set_last_error(LFG_ERROR_INTERNAL, "%s: failed to initialize the context: %s", __func__, err.what());
    }

    return nullptr;
}

// deprecated
lfg_context * lfg_new_context_with_model(
                 lfg_model * model,
        lfg_context_params   params) {
    return lfg_init_from_model(model, params);
}

void lfg_free(lfg_context * ctx) {
    if (!ctx) {
        return;
    }
    delete ctx;
}

uint32_t lfg_n_ctx(const lfg_context * ctx) {
    if (!lfg_require_ctx(ctx, __func__)) {
        return 0;
    }
    return ctx->n_ctx();
}

uint32_t lfg_n_ctx_seq(const lfg_context * ctx) {
    if (!lfg_require_ctx(ctx, __func__)) {
        return 0;
    }
    return ctx->n_ctx_seq();
}

uint32_t lfg_n_batch(const lfg_context * ctx) {
    if (!lfg_require_ctx(ctx, __func__)) {
        return 0;
    }
    return ctx->n_batch();
}

uint32_t lfg_n_ubatch(const lfg_context * ctx) {
    if (!lfg_require_ctx(ctx, __func__)) {
        return 0;
    }
    return ctx->n_ubatch();
}

uint32_t lfg_n_seq_max(const lfg_context * ctx) {
    if (!lfg_require_ctx(ctx, __func__)) {
        return 0;
    }
    return ctx->n_seq_max();
}

const lfg_model * lfg_get_model(const lfg_context * ctx) {
    if (!lfg_require_ctx(ctx, __func__)) {
        return nullptr;
    }
    return &ctx->get_model();
}

enum lfg_pooling_type lfg_pooling_type(const lfg_context * ctx) {
    if (!lfg_require_ctx(ctx, __func__)) {
        return LFG_POOLING_TYPE_UNSPECIFIED;
    }
    return ctx->pooling_type();
}

void lfg_attach_threadpool(
            lfg_context * ctx,
        ggml_threadpool_t   threadpool,
        ggml_threadpool_t   threadpool_batch) {
    if (!lfg_require_ctx(ctx, __func__)) {
        return;
    }
    ctx->attach_threadpool(threadpool, threadpool_batch);
}

void lfg_detach_threadpool(lfg_context * ctx) {
    if (!lfg_require_ctx(ctx, __func__)) {
        return;
    }
    ctx->detach_threadpool();
}

void lfg_set_n_threads(lfg_context * ctx, int32_t n_threads, int32_t n_threads_batch) {
    if (!lfg_require_ctx(ctx, __func__)) {
        return;
    }
    ctx->set_n_threads(n_threads, n_threads_batch);
}

int32_t lfg_n_threads(lfg_context * ctx) {
    if (!lfg_require_ctx(ctx, __func__)) {
        return 0;
    }
    return ctx->n_threads();
}

int32_t lfg_n_threads_batch(lfg_context * ctx) {
    if (!lfg_require_ctx(ctx, __func__)) {
        return 0;
    }
    return ctx->n_threads_batch();
}

void lfg_set_abort_callback(lfg_context * ctx, bool (*abort_callback)(void * data), void * abort_callback_data) {
    if (!lfg_require_ctx(ctx, __func__)) {
        return;
    }
    ctx->set_abort_callback(abort_callback, abort_callback_data);
}

void lfg_set_embeddings(lfg_context * ctx, bool embeddings) {
    if (!lfg_require_ctx(ctx, __func__)) {
        return;
    }
    ctx->set_embeddings(embeddings);
}

void lfg_set_causal_attn(lfg_context * ctx, bool causal_attn) {
    if (!lfg_require_ctx(ctx, __func__)) {
        return;
    }
    ctx->set_causal_attn(causal_attn);
}

void lfg_set_warmup(lfg_context * ctx, bool warmup) {
    if (!lfg_require_ctx(ctx, __func__)) {
        return;
    }
    ctx->set_warmup(warmup);
}

void lfg_synchronize(lfg_context * ctx) {
    if (!lfg_require_ctx(ctx, __func__)) {
        return;
    }
    ctx->synchronize();
}

float * lfg_get_logits(lfg_context * ctx) {
    if (!lfg_require_ctx(ctx, __func__)) {
        return nullptr;
    }
    ctx->synchronize();

    return ctx->get_logits();
}

float * lfg_get_logits_ith(lfg_context * ctx, int32_t i) {
    if (!lfg_require_ctx(ctx, __func__)) {
        return nullptr;
    }
    ctx->synchronize();

    float * res = nullptr;

    res = ctx->get_sampled_logits_ith(i);

    if (!res) {
        res = ctx->get_logits_ith(i);
    }

    return res;
}

float * lfg_get_embeddings(lfg_context * ctx) {
    if (!lfg_require_ctx(ctx, __func__)) {
        return nullptr;
    }
    ctx->synchronize();

    return ctx->get_embeddings();
}

float * lfg_get_embeddings_ith(lfg_context * ctx, int32_t i) {
    if (!lfg_require_ctx(ctx, __func__)) {
        return nullptr;
    }
    ctx->synchronize();

    return ctx->get_embeddings_ith(i);
}

float * lfg_get_embeddings_seq(lfg_context * ctx, lfg_seq_id seq_id) {
    if (!lfg_require_ctx(ctx, __func__)) {
        return nullptr;
    }
    ctx->synchronize();

    return ctx->get_embeddings_seq(seq_id);
}

bool lfg_set_sampler(lfg_context * ctx, lfg_seq_id seq_id, lfg_sampler * smpl) {
    if (!lfg_require_ctx(ctx, __func__)) {
        return false;
    }
    return ctx->set_sampler(seq_id, smpl);
}

lfg_token lfg_get_sampled_token_ith(lfg_context * ctx, int32_t i) {
    if (!lfg_require_ctx(ctx, __func__)) {
        return LFG_TOKEN_NULL;
    }
    ctx->synchronize();

    return ctx->get_sampled_token_ith(i);
}

float * lfg_get_sampled_probs_ith(lfg_context * ctx, int32_t i) {
    if (!lfg_require_ctx(ctx, __func__)) {
        return nullptr;
    }
    ctx->synchronize();

    return ctx->get_sampled_probs_ith(i);
}

float * lfg_get_sampled_logits_ith(lfg_context * ctx, int32_t i) {
    if (!lfg_require_ctx(ctx, __func__)) {
        return nullptr;
    }
    ctx->synchronize();

    return ctx->get_sampled_logits_ith(i);
}

lfg_token * lfg_get_sampled_candidates_ith(lfg_context * ctx, int32_t i) {
    if (!lfg_require_ctx(ctx, __func__)) {
        return nullptr;
    }
    ctx->synchronize();

    return const_cast<lfg_token *>(ctx->get_sampled_candidates_ith(i));
}

uint32_t lfg_get_sampled_candidates_count_ith(lfg_context * ctx, int32_t i) {
    if (!lfg_require_ctx(ctx, __func__)) {
        return 0;
    }
    ctx->synchronize();

    return static_cast<uint32_t>(ctx->get_sampled_candidates_count(i));
}

uint32_t lfg_get_sampled_logits_count_ith(lfg_context * ctx, int32_t i) {
    if (!lfg_require_ctx(ctx, __func__)) {
        return 0;
    }
    ctx->synchronize();

    return static_cast<uint32_t>(ctx->get_sampled_logits_count(i));
}

uint32_t lfg_get_sampled_probs_count_ith(lfg_context * ctx, int32_t i) {
    if (!lfg_require_ctx(ctx, __func__)) {
        return 0;
    }
    ctx->synchronize();

    return static_cast<uint32_t>(ctx->get_sampled_probs_count(i));
}

// liquid adapter API

int32_t lfg_set_adapter_lora(
            lfg_context * ctx,
            lfg_adapter_lora * adapter,
            float scale) {
    if (!lfg_require_ctx(ctx, __func__)) {
        return -1;
    }
    if (!adapter) {
        lfg_set_last_error(LFG_ERROR_INVALID_ARGUMENT, "%s: adapter is NULL", __func__);
        return -1;
    }
    ctx->set_adapter_lora(adapter, scale);

    return 0;
}

int32_t lfg_rm_adapter_lora(
            lfg_context * ctx,
            lfg_adapter_lora * adapter) {
    if (!lfg_require_ctx(ctx, __func__)) {
        return -1;
    }
    if (!adapter) {
        lfg_set_last_error(LFG_ERROR_INVALID_ARGUMENT, "%s: adapter is NULL", __func__);
        return -1;
    }
    bool res = ctx->rm_adapter_lora(adapter);

    return res ? 0 : -1;
}

void lfg_clear_adapter_lora(lfg_context * ctx) {
    if (!lfg_require_ctx(ctx, __func__)) {
        return;
    }
    ctx->clear_adapter_lora();
}

int32_t lfg_apply_adapter_cvec(
        lfg_context * ctx,
                 const float * data,
                      size_t   len,
                     int32_t   n_embd,
                     int32_t   il_start,
                     int32_t   il_end) {
    if (!lfg_require_ctx(ctx, __func__)) {
        return -1;
    }
    if (!data && len > 0) {
        lfg_set_last_error(LFG_ERROR_INVALID_ARGUMENT, "%s: data is NULL", __func__);
        return -1;
    }
    bool res = ctx->apply_adapter_cvec(data, len, n_embd, il_start, il_end);

    return res ? 0 : -1;
}

//
// memory
//

lfg_memory_t lfg_get_memory(const struct lfg_context * ctx) {
    if (!lfg_require_ctx(ctx, __func__)) {
        return nullptr;
    }
    return ctx->get_memory();
}

void lfg_memory_clear(lfg_memory_t mem, bool data) {
    if (!mem) {
        return;
    }

    mem->clear(data);
}

bool lfg_memory_seq_rm(
        lfg_memory_t mem,
          lfg_seq_id seq_id,
             lfg_pos p0,
             lfg_pos p1) {
    if (!mem) {
        return true;
    }

    return mem->seq_rm(seq_id, p0, p1);
}

void lfg_memory_seq_cp(
        lfg_memory_t mem,
          lfg_seq_id seq_id_src,
          lfg_seq_id seq_id_dst,
             lfg_pos p0,
             lfg_pos p1) {
    if (!mem) {
        return;
    }

    mem->seq_cp(seq_id_src, seq_id_dst, p0, p1);
}

void lfg_memory_seq_keep(
        lfg_memory_t mem,
          lfg_seq_id seq_id) {
    if (!mem) {
        return;
    }

    mem->seq_keep(seq_id);
}

void lfg_memory_seq_add(
        lfg_memory_t mem,
          lfg_seq_id seq_id,
             lfg_pos p0,
             lfg_pos p1,
             lfg_pos delta) {
    if (!mem) {
        return;
    }

    mem->seq_add(seq_id, p0, p1, delta);
}

void lfg_memory_seq_div(
        lfg_memory_t mem,
          lfg_seq_id seq_id,
             lfg_pos p0,
             lfg_pos p1,
                   int d) {
    if (!mem) {
        return;
    }

    mem->seq_div(seq_id, p0, p1, d);
}

lfg_pos lfg_memory_seq_pos_min(
        lfg_memory_t mem,
          lfg_seq_id seq_id) {
    if (!mem) {
        return -1;
    }

    return mem->seq_pos_min(seq_id);
}

lfg_pos lfg_memory_seq_pos_max(
        lfg_memory_t mem,
          lfg_seq_id seq_id) {
    if (!mem) {
        return -1;
    }

    return mem->seq_pos_max(seq_id);
}

bool lfg_memory_can_shift(lfg_memory_t mem) {
    if (!mem) {
        return false;
    }

    return mem->get_can_shift();
}

// liquid state API

// deprecated
size_t lfg_get_state_size(lfg_context * ctx) {
    return lfg_state_get_size(ctx);
}

// deprecated
size_t lfg_copy_state_data(lfg_context * ctx, uint8_t * dst) {
    return lfg_state_get_data(ctx, dst, (size_t)-1);
}

// deprecated
size_t lfg_set_state_data(lfg_context * ctx, const uint8_t * src) {
    return lfg_state_set_data(ctx, src, (size_t)-1);
}

// deprecated
bool lfg_load_session_file(lfg_context * ctx, const char * path_session, lfg_token * tokens_out, size_t n_token_capacity, size_t * n_token_count_out) {
    return lfg_state_load_file(ctx, path_session, tokens_out, n_token_capacity, n_token_count_out);
}

// deprecated
bool lfg_save_session_file(lfg_context * ctx, const char * path_session, const lfg_token * tokens, size_t n_token_count) {
    return lfg_state_save_file(ctx, path_session, tokens, n_token_count);
}

// Returns the *actual* size of the state.
// Intended to be used when saving to state to a buffer.
size_t lfg_state_get_size(lfg_context * ctx) {
    if (!lfg_require_ctx(ctx, __func__)) {
        return 0;
    }
    return ctx->state_get_size();
}

size_t lfg_state_get_data(lfg_context * ctx, uint8_t * dst, size_t size) {
    if (!lfg_require_ctx(ctx, __func__)) {
        return 0;
    }
    if (!dst && size > 0) {
        lfg_set_last_error(LFG_ERROR_INVALID_ARGUMENT, "%s: dst is NULL", __func__);
        return 0;
    }
    ctx->synchronize();

    return ctx->state_get_data(dst, size);
}

// Sets the state reading from the specified source address
size_t lfg_state_set_data(lfg_context * ctx, const uint8_t * src, size_t size) {
    if (!lfg_require_ctx(ctx, __func__)) {
        return 0;
    }
    if (!src && size > 0) {
        lfg_set_last_error(LFG_ERROR_INVALID_ARGUMENT, "%s: src is NULL", __func__);
        return 0;
    }
    ctx->synchronize();

    return ctx->state_set_data(src, size);
}

bool lfg_state_load_file(lfg_context * ctx, const char * path_session, lfg_token * tokens_out, size_t n_token_capacity, size_t * n_token_count_out) {
    if (!lfg_require_ctx(ctx, __func__)) {
        return false;
    }
    if (!path_session) {
        lfg_set_last_error(LFG_ERROR_INVALID_ARGUMENT, "%s: path_session is NULL", __func__);
        return false;
    }
    if (!tokens_out && n_token_capacity > 0) {
        lfg_set_last_error(LFG_ERROR_INVALID_ARGUMENT, "%s: tokens_out is NULL", __func__);
        return false;
    }
    ctx->synchronize();

    try {
        return ctx->state_load_file(path_session, tokens_out, n_token_capacity, n_token_count_out);
    } catch (const std::exception & err) {
        LFG_LOG_ERROR("%s: error loading session file: %s\n", __func__, err.what());
        lfg_set_last_error(LFG_ERROR_IO, "%s: error loading session file: %s", __func__, err.what());
        return false;
    }
}

bool lfg_state_save_file(lfg_context * ctx, const char * path_session, const lfg_token * tokens, size_t n_token_count) {
    if (!lfg_require_ctx(ctx, __func__)) {
        return false;
    }
    if (!path_session) {
        lfg_set_last_error(LFG_ERROR_INVALID_ARGUMENT, "%s: path_session is NULL", __func__);
        return false;
    }
    if (!tokens && n_token_count > 0) {
        lfg_set_last_error(LFG_ERROR_INVALID_ARGUMENT, "%s: tokens is NULL", __func__);
        return false;
    }
    ctx->synchronize();

    try {
        return ctx->state_save_file(path_session, tokens, n_token_count);
    } catch (const std::exception & err) {
        LFG_LOG_ERROR("%s: error saving session file: %s\n", __func__, err.what());
        lfg_set_last_error(LFG_ERROR_IO, "%s: error saving session file: %s", __func__, err.what());
        return false;
    }
}

size_t lfg_state_seq_get_size(lfg_context * ctx, lfg_seq_id seq_id) {
    return lfg_state_seq_get_size_ext(ctx, seq_id, 0);
}

size_t lfg_state_seq_get_data(lfg_context * ctx, uint8_t * dst, size_t size, lfg_seq_id seq_id) {
    return lfg_state_seq_get_data_ext(ctx, dst, size, seq_id, 0);
}

size_t lfg_state_seq_set_data(lfg_context * ctx, const uint8_t * src, size_t size, lfg_seq_id seq_id) {
    return lfg_state_seq_set_data_ext(ctx, src, size, seq_id, 0);
}

size_t lfg_state_seq_get_size_ext(lfg_context * ctx, lfg_seq_id seq_id, lfg_state_seq_flags flags) {
    if (!lfg_require_ctx(ctx, __func__)) {
        return 0;
    }
    return ctx->state_seq_get_size(seq_id, flags);
}

size_t lfg_state_seq_get_data_ext(lfg_context * ctx, uint8_t * dst, size_t size, lfg_seq_id seq_id, lfg_state_seq_flags flags) {
    if (!lfg_require_ctx(ctx, __func__)) {
        return 0;
    }
    if (!dst && size > 0) {
        lfg_set_last_error(LFG_ERROR_INVALID_ARGUMENT, "%s: dst is NULL", __func__);
        return 0;
    }
    ctx->synchronize();

    return ctx->state_seq_get_data(seq_id, dst, size, flags);
}

size_t lfg_state_seq_set_data_ext(lfg_context * ctx, const uint8_t * src, size_t size, lfg_seq_id seq_id, lfg_state_seq_flags flags) {
    if (!lfg_require_ctx(ctx, __func__)) {
        return 0;
    }
    if (!src && size > 0) {
        lfg_set_last_error(LFG_ERROR_INVALID_ARGUMENT, "%s: src is NULL", __func__);
        return 0;
    }
    ctx->synchronize();

    return ctx->state_seq_set_data(seq_id, src, size, flags);
}

size_t lfg_state_seq_save_file(lfg_context * ctx, const char * filepath, lfg_seq_id seq_id, const lfg_token * tokens, size_t n_token_count) {
    if (!lfg_require_ctx(ctx, __func__)) {
        return 0;
    }
    if (!filepath) {
        lfg_set_last_error(LFG_ERROR_INVALID_ARGUMENT, "%s: filepath is NULL", __func__);
        return 0;
    }
    if (!tokens && n_token_count > 0) {
        lfg_set_last_error(LFG_ERROR_INVALID_ARGUMENT, "%s: tokens is NULL", __func__);
        return 0;
    }
    ctx->synchronize();

    try {
        return ctx->state_seq_save_file(seq_id, filepath, tokens, n_token_count);
    } catch (const std::exception & err) {
        LFG_LOG_ERROR("%s: error saving sequence state file: %s\n", __func__, err.what());
        lfg_set_last_error(LFG_ERROR_IO, "%s: error saving sequence state file: %s", __func__, err.what());
        return 0;
    }
}

size_t lfg_state_seq_load_file(lfg_context * ctx, const char * filepath, lfg_seq_id dest_seq_id, lfg_token * tokens_out, size_t n_token_capacity, size_t * n_token_count_out) {
    if (!lfg_require_ctx(ctx, __func__)) {
        return 0;
    }
    if (!filepath) {
        lfg_set_last_error(LFG_ERROR_INVALID_ARGUMENT, "%s: filepath is NULL", __func__);
        return 0;
    }
    if (!tokens_out && n_token_capacity > 0) {
        lfg_set_last_error(LFG_ERROR_INVALID_ARGUMENT, "%s: tokens_out is NULL", __func__);
        return 0;
    }
    ctx->synchronize();

    try {
        return ctx->state_seq_load_file(dest_seq_id, filepath, tokens_out, n_token_capacity, n_token_count_out);
    } catch (const std::exception & err) {
        LFG_LOG_ERROR("%s: error loading sequence state file: %s\n", __func__, err.what());
        lfg_set_last_error(LFG_ERROR_IO, "%s: error loading sequence state file: %s", __func__, err.what());
        return 0;
    }
}

///

int32_t lfg_encode(
        lfg_context * ctx,
          lfg_batch   batch) {
    if (!lfg_require_ctx(ctx, __func__)) {
        return -1;
    }
    if (batch.n_tokens < 0) {
        lfg_set_last_error(LFG_ERROR_INVALID_ARGUMENT, "%s: n_tokens is negative", __func__);
        return -1;
    }
    if (batch.n_tokens > 0 && batch.token == nullptr && batch.embd == nullptr) {
        lfg_set_last_error(LFG_ERROR_INVALID_ARGUMENT, "%s: token and embd are NULL", __func__);
        return -1;
    }
    int ret = -1;
    try {
        ret = ctx->encode(batch);
    } catch (const std::exception & err) {
        LFG_LOG_ERROR("%s: failed to encode: %s\n", __func__, err.what());
        lfg_set_last_error(LFG_ERROR_INTERNAL, "%s: failed to encode: %s", __func__, err.what());
        return -1;
    }
    if (ret != 0) {
        LFG_LOG_ERROR("%s: failed to encode, ret = %d\n", __func__, ret);
        lfg_set_last_error(LFG_ERROR_INTERNAL, "%s: failed to encode, ret=%d", __func__, ret);
    }

    return ret;
}

int32_t lfg_decode(
        lfg_context * ctx,
          lfg_batch   batch) {
    if (!lfg_require_ctx(ctx, __func__)) {
        return -1;
    }
    if (batch.n_tokens < 0) {
        lfg_set_last_error(LFG_ERROR_INVALID_ARGUMENT, "%s: n_tokens is negative", __func__);
        return -1;
    }
    if (batch.n_tokens > 0 && batch.token == nullptr && batch.embd == nullptr) {
        lfg_set_last_error(LFG_ERROR_INVALID_ARGUMENT, "%s: token and embd are NULL", __func__);
        return -1;
    }
    int ret = -1;
    try {
        ret = ctx->decode(batch);
    } catch (const std::exception & err) {
        LFG_LOG_ERROR("%s: failed to decode: %s\n", __func__, err.what());
        lfg_set_last_error(LFG_ERROR_INTERNAL, "%s: failed to decode: %s", __func__, err.what());
        return -1;
    }
    if (ret != 0 && ret != 1) {
        LFG_LOG_ERROR("%s: failed to decode, ret = %d\n", __func__, ret);
        lfg_set_last_error(LFG_ERROR_INTERNAL, "%s: failed to decode, ret=%d", __func__, ret);
    }

    return ret;
}

//
// perf
//

lfg_perf_context_data lfg_perf_context(const lfg_context * ctx) {
    lfg_perf_context_data data = {};

    if (!lfg_require_ctx(ctx, __func__)) {
        return data;
    }

    data = ctx->perf_get_data();

    return data;
}

void lfg_perf_context_print(const lfg_context * ctx) {
    if (!lfg_require_ctx(ctx, __func__)) {
        return;
    }
    const auto data = lfg_perf_context(ctx);

    const double t_end_ms = 1e-3 * (double)ggml_time_us();

    LFG_LOG_INFO("%s:        load time = %10.2f ms\n", __func__, data.t_load_ms);
    LFG_LOG_INFO("%s: prompt eval time = %10.2f ms / %5d tokens (%8.2f ms per token, %8.2f tokens per second)\n",
            __func__, data.t_p_eval_ms, data.n_p_eval, data.t_p_eval_ms / data.n_p_eval, 1e3 / data.t_p_eval_ms * data.n_p_eval);
    LFG_LOG_INFO("%s:        eval time = %10.2f ms / %5d runs   (%8.2f ms per token, %8.2f tokens per second)\n",
            __func__, data.t_eval_ms, data.n_eval, data.t_eval_ms / data.n_eval, 1e3 / data.t_eval_ms * data.n_eval);
    LFG_LOG_INFO("%s:       total time = %10.2f ms / %5d tokens\n", __func__, (t_end_ms - data.t_start_ms), (data.n_p_eval + data.n_eval));
    LFG_LOG_INFO("%s:    graphs reused = %10d\n", __func__, data.n_reused);
}

void lfg_perf_context_reset(lfg_context * ctx) {
    if (!lfg_require_ctx(ctx, __func__)) {
        return;
    }
    ctx->perf_reset();
}

void lfg_memory_breakdown_print(const struct lfg_context * ctx) {
    if (!lfg_require_ctx(ctx, __func__)) {
        return;
    }
    const std::vector<ggml_backend_dev_t> & devices = ctx->get_model().devices;

    std::map<ggml_backend_buffer_type_t, lfg_memory_breakdown_data> memory_breakdown = ctx->memory_breakdown();

    std::vector<std::array<std::string, 9>> table_data;
    table_data.reserve(devices.size());
    const std::string template_header = "%s: | %s | %s   %s    %s   %s   %s   %s    %s |\n";
    const std::string template_gpu    = "%s: | %s | %s = %s + (%s = %s + %s + %s) + %s |\n";
    const std::string template_other  = "%s: | %s | %s   %s    %s = %s + %s + %s    %s |\n";

    table_data.push_back({template_header, "memory breakdown [MiB]", "total", "free", "self", "model", "context", "compute", "unaccounted"});

    constexpr size_t MiB = 1024 * 1024;
    const std::vector<std::string> desc_prefixes_strip = {"NVIDIA ", "GeForce ", "Tesla ", "AMD ", "Radeon ", "Instinct "};

    // track seen buffer types to avoid double counting:
    std::set<ggml_backend_buffer_type_t> seen_buffer_types;

    // accumulative memory breakdown for each device and for host:
    std::vector<lfg_memory_breakdown_data> mb_dev(devices.size());
    lfg_memory_breakdown_data              mb_host;

    for (const auto & buft_mb : memory_breakdown) {
        ggml_backend_buffer_type_t          buft = buft_mb.first;
        const lfg_memory_breakdown_data & mb   = buft_mb.second;
        if (ggml_backend_buft_is_host(buft)) {
            mb_host.model   += mb.model;
            mb_host.context += mb.context;
            mb_host.compute += mb.compute;
            seen_buffer_types.insert(buft);
            continue;
        }
        ggml_backend_dev_t dev = ggml_backend_buft_get_device(buft);
        if (dev) {
            int i_dev = -1;
            for (size_t i = 0; i < devices.size(); i++) {
                if (devices[i] == dev) {
                    i_dev = (int)i;
                    break;
                }
            }
            if (i_dev != -1) {
                mb_dev[i_dev].model   += (size_t)mb.model;
                mb_dev[i_dev].context += (size_t)mb.context;
                mb_dev[i_dev].compute += (size_t)mb.compute;
                seen_buffer_types.insert(buft);
                continue;
            }
        }
    }

    // print memory breakdown for each device:
    for (size_t i = 0; i < devices.size(); i++) {
        ggml_backend_dev_t          dev = devices[i];
        lfg_memory_breakdown_data mb  = mb_dev[i];

        const std::string name = ggml_backend_dev_name(dev);
        std::string desc = ggml_backend_dev_description(dev);
        for (const std::string & prefix : desc_prefixes_strip) {
            if (desc.length() >= prefix.length() && desc.substr(0, prefix.length()) == prefix) {
                desc = desc.substr(prefix.length());
            }
        }

        size_t free, total;
        ggml_backend_dev_memory(dev, &free, &total);

        const size_t self = mb.model + mb.context + mb.compute;
        const size_t unaccounted = total - self - free;

        table_data.push_back({
            template_gpu,
            "  - " + name + " (" + desc + ")",
            std::to_string(total / MiB),
            std::to_string(free / MiB),
            std::to_string(self / MiB),
            std::to_string(mb.model / MiB),
            std::to_string(mb.context / MiB),
            std::to_string(mb.compute / MiB),
            std::to_string(unaccounted / MiB)});
    }

    // print memory breakdown for host:
    {
        const size_t self = mb_host.model + mb_host.context + mb_host.compute;
        table_data.push_back({
            template_other,
            "  - Host",
            "", // total
            "", // free
            std::to_string(self / MiB),
            std::to_string(mb_host.model / MiB),
            std::to_string(mb_host.context / MiB),
            std::to_string(mb_host.compute / MiB),
            ""}); // unaccounted
    }

    // print memory breakdown for all remaining buffer types:
    for (const auto & buft_mb : memory_breakdown) {
        ggml_backend_buffer_type_t          buft = buft_mb.first;
        const lfg_memory_breakdown_data & mb   = buft_mb.second;
        if (seen_buffer_types.count(buft) == 1) {
            continue;
        }
        const std::string name = ggml_backend_buft_name(buft);
        const size_t self = mb.model + mb.context + mb.compute;
        table_data.push_back({
            template_other,
            "  - " + name,
            "", // total
            "", // free
            std::to_string(self / MiB),
            std::to_string(mb.model / MiB),
            std::to_string(mb.context / MiB),
            std::to_string(mb.compute / MiB),
            ""}); // unaccounted
        seen_buffer_types.insert(buft);
    }

    for (size_t j = 1; j < table_data[0].size(); j++) {
        size_t max_len = 0;
        for (const auto & td : table_data) {
            max_len = std::max(max_len, td[j].length());
        }
        for (auto & td : table_data) {
            td[j].insert(j == 1 ? td[j].length() : 0, max_len - td[j].length(), ' ');
        }
    }
    for (const auto & td : table_data) {
        LFG_LOG_INFO(td[0].c_str(),
            __func__, td[1].c_str(), td[2].c_str(), td[3].c_str(), td[4].c_str(), td[5].c_str(),
            td[6].c_str(), td[7].c_str(), td[8].c_str());
    }
}

//
// training
//

bool lfg_opt_param_filter_all(const struct ggml_tensor * tensor, void * userdata) {
    GGML_UNUSED(tensor);
    GGML_UNUSED(userdata);
    return true;
}

void lfg_opt_init(struct lfg_context * ctx, struct lfg_model * model, struct lfg_opt_params lopt_params) {
    if (!lfg_require_ctx(ctx, __func__)) {
        return;
    }
    if (!model) {
        lfg_set_last_error(LFG_ERROR_INVALID_ARGUMENT, "%s: model is NULL", __func__);
        return;
    }
    ctx->opt_init(model, lopt_params);
}

void lfg_opt_epoch(
        struct lfg_context    * ctx,
        ggml_opt_dataset_t        dataset,
        ggml_opt_result_t         result_train,
        ggml_opt_result_t         result_eval,
        int64_t                   idata_split,
        ggml_opt_epoch_callback   callback_train,
        ggml_opt_epoch_callback   callback_eval) {
    if (!lfg_require_ctx(ctx, __func__)) {
        return;
    }
    ctx->opt_epoch(
        dataset,
        result_train,
        result_eval,
        idata_split,
        callback_train,
        callback_eval);
}
