#include "lfg_model.h"

#include "lfg_impl.h"
#include "lfg_mmap.h"
#include "lfg_cparams.h"
#include "lfg_model_loader.h"

#include "lfg_kv_cache.h"
#include "lfg_kv_cache_iswa.h"
#include "lfg_memory_hybrid.h"
#include "lfg_memory_hybrid_iswa.h"
#include "lfg_memory_recurrent.h"

#ifdef __cplusplus
#include "ggml-cpp.h"
#endif

#include "models/lfg_lfm2.h"
#include "models/lfg_bert.h"
#include "models/lfg_neo_bert.h"
#include "models/lfg_modern_bert.h"

#include <algorithm>
#include <cassert>
#include <fstream>
#include <cfloat>
#include <cstring>
#include <cmath>
#include <functional>
#include <map>
#include <regex>
#include <sstream>
#include <stdexcept>

const char * lfg_type_name(lfg_type_enum type) {
    switch (type) {
        case LFG_TYPE_14M:           return "14M";
        case LFG_TYPE_17M:           return "17M";
        case LFG_TYPE_22M:           return "22M";
        case LFG_TYPE_33M:           return "33M";
        case LFG_TYPE_47M:           return "47M";
        case LFG_TYPE_60M:           return "60M";
        case LFG_TYPE_70M:           return "70M";
        case LFG_TYPE_80M:           return "80M";
        case LFG_TYPE_109M:          return "109M";
        case LFG_TYPE_137M:          return "137M";
        case LFG_TYPE_140M:          return "140M";
        case LFG_TYPE_149M:          return "149M";
        case LFG_TYPE_160M:          return "160M";
        case LFG_TYPE_190M:          return "190M";
        case LFG_TYPE_220M:          return "220M";
        case LFG_TYPE_250M:          return "250M";
        case LFG_TYPE_256M:          return "256M";
        case LFG_TYPE_270M:          return "270M";
        case LFG_TYPE_335M:          return "335M";
        case LFG_TYPE_350M:          return "350M";
        case LFG_TYPE_360M:          return "360M";
        case LFG_TYPE_395M:          return "395M";
        case LFG_TYPE_410M:          return "410M";
        case LFG_TYPE_450M:          return "450M";
        case LFG_TYPE_475M:          return "475M";
        case LFG_TYPE_558M:          return "558M";
        case LFG_TYPE_700M:          return "700M";
        case LFG_TYPE_770M:          return "770M";
        case LFG_TYPE_780M:          return "780M";
        case LFG_TYPE_950M:          return "950M";
        case LFG_TYPE_0_3B:          return "0.3B";
        case LFG_TYPE_0_5B:          return "0.5B";
        case LFG_TYPE_0_6B:          return "0.6B";
        case LFG_TYPE_1B:            return "1B";
        case LFG_TYPE_1_2B:          return "1.2B";
        case LFG_TYPE_1_3B:          return "1.3B";
        case LFG_TYPE_1_4B:          return "1.4B";
        case LFG_TYPE_1_5B:          return "1.5B";
        case LFG_TYPE_1_6B:          return "1.6B";
        case LFG_TYPE_1_7B:          return "1.7B";
        case LFG_TYPE_1_8B:          return "1.8B";
        case LFG_TYPE_2B:            return "2B";
        case LFG_TYPE_2_6B:          return "2.6B";
        case LFG_TYPE_2_8B:          return "2.8B";
        case LFG_TYPE_2_9B:          return "2.9B";
        case LFG_TYPE_3B:            return "3B";
        case LFG_TYPE_4B:            return "4B";
        case LFG_TYPE_6B:            return "6B";
        case LFG_TYPE_6_9B:          return "6.9B";
        case LFG_TYPE_7B:            return "7B";
        case LFG_TYPE_8B:            return "8B";
        case LFG_TYPE_9B:            return "9B";
        case LFG_TYPE_11B:           return "11B";
        case LFG_TYPE_12B:           return "12B";
        case LFG_TYPE_13B:           return "13B";
        case LFG_TYPE_14B:           return "14B";
        case LFG_TYPE_15B:           return "15B";
        case LFG_TYPE_16B:           return "16B";
        case LFG_TYPE_20B:           return "20B";
        case LFG_TYPE_26B:           return "26B";
        case LFG_TYPE_27B:           return "27B";
        case LFG_TYPE_30B:           return "30B";
        case LFG_TYPE_32B:           return "32B";
        case LFG_TYPE_34B:           return "34B";
        case LFG_TYPE_35B:           return "35B";
        case LFG_TYPE_36B:           return "36B";
        case LFG_TYPE_40B:           return "40B";
        case LFG_TYPE_65B:           return "65B";
        case LFG_TYPE_70B:           return "70B";
        case LFG_TYPE_120B:          return "120B";
        case LFG_TYPE_142B:          return "142B";
        case LFG_TYPE_236B:          return "236B";
        case LFG_TYPE_290B:          return "290B";
        case LFG_TYPE_314B:          return "314B";
        case LFG_TYPE_405B:          return "405B";
        case LFG_TYPE_671B:          return "671B";
        case LFG_TYPE_SMALL:         return "0.1B";
        case LFG_TYPE_MEDIUM:        return "0.4B";
        case LFG_TYPE_LARGE:         return "0.8B";
        case LFG_TYPE_XL:            return "1.5B";
        case LFG_TYPE_A1_7B:         return "A1.7B";
        case LFG_TYPE_A2_7B:         return "A2.7B";
        case LFG_TYPE_8x7B:          return "8x7B";
        case LFG_TYPE_8x22B:         return "8x22B";
        case LFG_TYPE_16x12B:        return "16x12B";
        case LFG_TYPE_16x3_8B:       return "16x3.8B";
        case LFG_TYPE_10B_128x3_66B: return "10B+128x3.66B";
        case LFG_TYPE_57B_A14B:      return "57B.A14B";
        case LFG_TYPE_17B_16E:       return "17Bx16E (Scout)";
        case LFG_TYPE_17B_128E:      return "17Bx128E (Maverick)";
        case LFG_TYPE_A13B:          return "A13B";
        case LFG_TYPE_7B_A1B:        return "7B.A1B";
        case LFG_TYPE_8B_A1B:        return "8B.A1B";
        case LFG_TYPE_16B_A1B:       return "16B.A1B";
        case LFG_TYPE_21B_A3B:       return "21B.A3B";
        case LFG_TYPE_30B_A3B:       return "30B.A3B";
        case LFG_TYPE_31B_A3_5B:     return "31B.A3.5B";
        case LFG_TYPE_80B_A3B:       return "80B.A3B";
        case LFG_TYPE_100B_A6B:      return "100B.A6B";
        case LFG_TYPE_102B_A12B:     return "102B.A12B";
        case LFG_TYPE_106B_A12B:     return "106B.A12B";
        case LFG_TYPE_230B_A10B:     return "230B.A10B";
        case LFG_TYPE_235B_A22B:     return "235B.A22B";
        case LFG_TYPE_300B_A47B:     return "300B.A47B";
        case LFG_TYPE_310B_A15B:     return "310B.A15B";
        case LFG_TYPE_355B_A32B:     return "355B.A32B";
        case LFG_TYPE_E2B:           return "E2B";
        case LFG_TYPE_E4B:           return "E4B";
        default:                     return "?B";
    }
}

static const char * lfg_expert_gating_func_name(lfg_expert_gating_func_type type) {
    switch (type) {
        case LFG_EXPERT_GATING_FUNC_TYPE_SOFTMAX: return "softmax";
        case LFG_EXPERT_GATING_FUNC_TYPE_SIGMOID: return "sigmoid";
        default:                                    return "unknown";
    }
}

static const std::map<lfg_rope_scaling_type, const char *> LFG_ROPE_SCALING_TYPES = {
    { LFG_ROPE_SCALING_TYPE_NONE,       "none"       },
    { LFG_ROPE_SCALING_TYPE_LINEAR,     "linear"     },
    { LFG_ROPE_SCALING_TYPE_YARN,       "yarn"       },
    { LFG_ROPE_SCALING_TYPE_LONGROPE,   "longrope"   },
};

std::string lfg_rope_scaling_type_name(lfg_rope_scaling_type rope_scaling_type) {
    return LFG_ROPE_SCALING_TYPES.at(rope_scaling_type);
}

static lfg_rope_scaling_type lfg_rope_scaling_type_from_string(const std::string & name) {
    for (const auto & kv : LFG_ROPE_SCALING_TYPES) {
        if (kv.second == name) {
            return (lfg_rope_scaling_type) kv.first;
        }
    }

    return LFG_ROPE_SCALING_TYPE_UNSPECIFIED;
}

// checks if the weight tensor can be used with the specified buffer type and device
static bool weight_buft_supported(const lfg_hparams & hparams, ggml_tensor * w, ggml_op op, ggml_backend_buffer_type_t buft, ggml_backend_dev_t dev) {
    GGML_ASSERT(w != nullptr);

    if (op == GGML_OP_NONE) {
        return true;
    }

    ggml_init_params params = {
        /*.mem_size   =*/ ggml_tensor_overhead()*8,
        /*.mem_buffer =*/ NULL,
        /*.no_alloc   =*/ true,
    };
    ggml_context_ptr ctx_ptr { ggml_init(params) };
    if (!ctx_ptr) {
        throw std::runtime_error(lfg_format("failed to create ggml context"));
    }
    ggml_context * ctx = ctx_ptr.get();

    ggml_tensor * op_tensor = nullptr;

    switch (op) {
        case GGML_OP_GET_ROWS:
            {
                ggml_tensor * b = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 512);
                op_tensor = ggml_get_rows(ctx, w, b);
            } break;
        case GGML_OP_MUL_MAT:
            {
                ggml_tensor * b = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, w->ne[0], 512, w->ne[2], w->ne[3]);
                op_tensor = ggml_mul_mat(ctx, w, b);
            } break;
        case GGML_OP_MUL_MAT_ID:
            {
                int n_expert_used = hparams.n_expert_used;
                ggml_tensor * b = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, w->ne[0], n_expert_used, 512);
                ggml_tensor * ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, n_expert_used, 512);
                op_tensor = ggml_mul_mat_id(ctx, w, b, ids);
            } break;
        case GGML_OP_ADD:
            {
                ggml_tensor * a = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, w->ne[0], w->ne[1], w->ne[2], w->ne[3]);
                op_tensor = ggml_add(ctx, a, w);
            } break;
        case GGML_OP_ADD_ID:
            {
                int n_expert_used = hparams.n_expert_used;
                ggml_tensor * a = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, w->ne[0], n_expert_used, 512);
                ggml_tensor * c = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, n_expert_used, 512);
                op_tensor = ggml_add_id(ctx, a, w, c);
            } break;
        case GGML_OP_MUL:
            {
                ggml_tensor * a = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, w->ne[0], w->ne[1], w->ne[2], w->ne[3]);
                op_tensor = ggml_mul(ctx, a, w);
            } break;
        case GGML_OP_DIV:
            {
                ggml_tensor * a = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, w->ne[0]);
                op_tensor = ggml_div(ctx, a, w);
            } break;
        case GGML_OP_ROPE:
            {
                int n_embd_head = hparams.n_embd_head_v;
                int n_head = hparams.n_head();
                ggml_tensor * a = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, n_embd_head, n_head, 512);
                ggml_tensor * b = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 512);
                op_tensor = ggml_rope_ext(
                    ctx, a, b, w,
                    0, 0, 0, 0, 0,
                    0, 0, 0, 0
                );

            } break;
        case GGML_OP_SSM_CONV:
            {
                const int64_t n_seq_tokens = 512;
                const int64_t n_seqs       = 3;
                ggml_tensor * conv_x = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, w->ne[0] - 1 + n_seq_tokens, w->ne[1], n_seqs);
                op_tensor = ggml_ssm_conv(ctx, conv_x, w);
            } break;
        case GGML_OP_SSM_SCAN:
            {
                // w is ssm_a, which is used to distinguish Mamba-1 and Mamba-2
                const int64_t d_state      = w->ne[0] == 1 ? hparams.ssm_d_state : w->ne[0];
                const int64_t n_head       = w->ne[1];
                const int64_t head_dim     = hparams.ssm_d_inner / n_head;
                const int64_t n_group      = hparams.ssm_n_group ? hparams.ssm_n_group : 1;
                const int64_t n_seq_tokens = 512;
                const int64_t n_seqs       = 3;
                ggml_tensor * s   = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, d_state, head_dim, n_head, n_seqs);
                ggml_tensor * x   = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, head_dim, n_head, n_seq_tokens, n_seqs);
                ggml_tensor * dt  = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, n_head, n_seq_tokens, n_seqs);
                ggml_tensor * B   = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, d_state, n_group, n_seq_tokens, n_seqs);
                ggml_tensor * C   = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, d_state, n_group, n_seq_tokens, n_seqs);
                ggml_tensor * ids = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_seqs);
                op_tensor = ggml_ssm_scan(ctx, s, x, dt, w, B, C, ids);
            } break;
        case GGML_OP_RWKV_WKV6:
            {
                // FIXME
                const int64_t S = 123;
                const int64_t H = 123;
                const int64_t n_tokens = 123;
                const int64_t n_seqs = 123;
                ggml_tensor  * k = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, S, H, n_tokens);
                ggml_tensor  * v = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, S, H, n_tokens);
                ggml_tensor  * r = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, S, H, n_tokens);
                ggml_tensor  * tf = w;
                ggml_tensor  * td = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, S, H, n_tokens);
                ggml_tensor  * state = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, S, n_seqs, S, H);
                op_tensor = ggml_rwkv_wkv6(ctx, k, v, r, tf, td, state);
            } break;
        case GGML_OP_IM2COL:
            {
                const int n_embd_inp = hparams.n_embd_inp();
                ggml_tensor * b = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, n_embd_inp, w->ne[1], 1, 1);
                op_tensor = ggml_im2col(ctx, w, b, 1, 0, 0, 0, 1, 0, false, GGML_TYPE_F16);
            } break;
        case GGML_OP_SCALE:
            {
                op_tensor = ggml_scale(ctx, w, 1.0f);
            } break;
        default:
            lfg_set_last_error(LFG_ERROR_UNSUPPORTED, "%s: missing test for op %s for tensor %s", __func__, ggml_op_name(op), w->name);
            return false;
    }

    // create a temporary dummy buffer for the weight so that supports_op can check the buffer type
    GGML_ASSERT(w->buffer == nullptr);
    w->buffer = ggml_backend_buft_alloc_buffer(buft, 0);
    bool op_supported = ggml_backend_dev_supports_op(dev, op_tensor);
    ggml_backend_buffer_free(w->buffer);
    w->buffer = nullptr;

    return op_supported;
}

// lists of buffer types used for each layer
using buft_list_t = std::vector<std::pair<ggml_backend_dev_t, ggml_backend_buffer_type_t>>;

// find the first buffer type in the list that can use the tensor
static ggml_backend_buffer_type_t select_weight_buft(const lfg_hparams & hparams, ggml_tensor * tensor, ggml_op op, const buft_list_t & buft_list) {
    GGML_ASSERT(!buft_list.empty());
    for (const auto & cur : buft_list) {
        ggml_backend_dev_t cur_dev = cur.first;
        ggml_backend_buffer_type_t cur_buft = cur.second;
        if (weight_buft_supported(hparams, tensor, op, cur_buft, cur_dev)) {
            return cur_buft;
        }
    }

    return nullptr;
}

// CPU: ACCEL -> GPU host -> CPU extra -> CPU
static buft_list_t make_cpu_buft_list(const std::vector<ggml_backend_dev_t> & devices, bool use_extra_bufts, bool no_host) {
    buft_list_t buft_list;

    // add ACCEL buffer types
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        if (ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_ACCEL) {
            auto * buft = ggml_backend_dev_buffer_type(dev);
            // skip
            if (buft != ggml_backend_cpu_buffer_type()) {
                buft_list.emplace_back(dev, buft);
            }
        }
    }

    // add a host buffer type
    // storing the tensors in a host buffer is useful when the processing of large batches
    // is offloaded to a GPU device, since it reduces the time spent on data transfers
    // generally, this will be done using the first device in the list
    // a better approach would be to handle this on a weight-by-weight basis using the offload_op
    // function of the device to determine if it would benefit from being stored in a host buffer
    if (!no_host) {
        for (auto * dev : devices) {
            ggml_backend_buffer_type_t buft = ggml_backend_dev_host_buffer_type(dev);
            if (buft) {
                buft_list.emplace_back(dev, buft);
                break;
            }
        }
    }

    // add extra buffer types
    if (use_extra_bufts) {
        auto * cpu_dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
        if (cpu_dev == nullptr) {
            throw std::runtime_error(lfg_format("%s: no CPU backend found", __func__));
        }

        auto * cpu_reg = ggml_backend_dev_backend_reg(cpu_dev);
        auto ggml_backend_dev_get_extra_bufts_fn = (ggml_backend_dev_get_extra_bufts_t)
            ggml_backend_reg_get_proc_address(cpu_reg, "ggml_backend_dev_get_extra_bufts");
        if (ggml_backend_dev_get_extra_bufts_fn) {
            ggml_backend_buffer_type_t * extra_bufts = ggml_backend_dev_get_extra_bufts_fn(cpu_dev);
            while (extra_bufts && *extra_bufts) {
                buft_list.emplace_back(cpu_dev, *extra_bufts);
                ++extra_bufts;
            }
        }
    }

    // add the CPU buffer type
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        if (ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_CPU) {
            buft_list.emplace_back(dev, ggml_backend_dev_buffer_type(dev));
        }
    }

    return buft_list;
}

// GPU: split if LFG_SPLIT_MODE_ROW -> GPU
static buft_list_t make_gpu_buft_list(ggml_backend_dev_t dev, lfg_split_mode split_mode, const float * tensor_split) {
    buft_list_t buft_list;

    // add the device split buffer type if requested and available
    if (split_mode == LFG_SPLIT_MODE_ROW) {
        ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(dev);
        auto ggml_backend_split_buffer_type_fn = (ggml_backend_split_buffer_type_t)
            ggml_backend_reg_get_proc_address(reg, "ggml_backend_split_buffer_type");
        if (ggml_backend_split_buffer_type_fn) {
            size_t dev_index = [&]() {
                auto * reg = ggml_backend_dev_backend_reg(dev);
                for (size_t i = 0; i < ggml_backend_reg_dev_count(reg); ++i) {
                    if (ggml_backend_reg_dev_get(reg, i) == dev) {
                        return i;
                    }
                }
                throw std::runtime_error(lfg_format("device %s not found in its backend reg", ggml_backend_dev_name(dev)));
            }();
            auto * buft = ggml_backend_split_buffer_type_fn(dev_index, tensor_split);
            if (buft != nullptr) {
                buft_list.emplace_back(dev, buft);
            }
        }
    }

    // add the device default buffer type
    buft_list.emplace_back(dev, ggml_backend_dev_buffer_type(dev));

    // add the device extra buffer type (if any)
    ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(dev);
    auto ggml_backend_dev_get_extra_bufts_fn = (ggml_backend_dev_get_extra_bufts_t)
        ggml_backend_reg_get_proc_address(reg, "ggml_backend_dev_get_extra_bufts");

    if (ggml_backend_dev_get_extra_bufts_fn) {
        ggml_backend_buffer_type_t * extra_bufts = ggml_backend_dev_get_extra_bufts_fn(dev);
        while (extra_bufts && *extra_bufts) {
            buft_list.emplace_back(dev, *extra_bufts);
            ++extra_bufts;
        }
    }

    return buft_list;
}

struct lfg_model::impl {
    impl() = default;
    ~impl() = default;

    uint64_t n_elements = 0;

    size_t n_bytes = 0;

    std::string desc_str;

    // model memory mapped files
    lfg_mmaps mappings;

    // objects representing data potentially being locked in memory
    lfg_mlocks mlock_bufs;
    lfg_mlocks mlock_mmaps;

    // contexts where the model tensors metadata is stored as well as the corresponding buffers:
    std::vector<std::pair<ggml_context_ptr, std::vector<ggml_backend_buffer_ptr>>> ctxs_bufs;

    buft_list_t cpu_buft_list;
    std::map<ggml_backend_dev_t, buft_list_t> gpu_buft_list;

    struct layer_dev {
        ggml_backend_dev_t dev;
        buft_list_t * buft_list;
    };

    layer_dev dev_input = {};
    layer_dev dev_output = {};
    std::vector<layer_dev> dev_layer;

    bool has_tensor_overrides;
};

lfg_model::lfg_model(const lfg_model_params & params) : params(params), pimpl(std::make_unique<impl>()) {
    pimpl->has_tensor_overrides = params.tensor_buft_overrides && params.tensor_buft_overrides[0].pattern;
}

lfg_model::~lfg_model() {
    for (auto * lora : loras) {
        delete lora;
    }
}

void lfg_model::load_stats(lfg_model_loader & ml) {
    pimpl->n_elements = ml.n_elements;
    pimpl->n_bytes = ml.n_bytes;
}

void lfg_model::load_arch(lfg_model_loader & ml) {
    arch = ml.get_arch();
    if (arch == LFG_ARCH_UNKNOWN) {
        throw std::runtime_error("unknown model architecture: '" + ml.get_arch_name() + "'");
    }
}

void lfg_model::load_hparams(lfg_model_loader & ml) {
    const gguf_context * ctx = ml.meta.get();

    // get metadata as string
    for (int i = 0; i < gguf_get_n_kv(ctx); i++) {
        gguf_type type = gguf_get_kv_type(ctx, i);
        if (type == GGUF_TYPE_ARRAY) {
            continue;
        }
        const char * name = gguf_get_key(ctx, i);
        const std::string value = lfg_gguf_kv_to_str(ctx, i);
        gguf_kv.emplace(name, value);
    }

    // get general kv
    ml.get_key(LFG_KV_GENERAL_NAME, name, false);

    // everything past this point is not vocab-related
    // for CLIP models, we only need to load tensors, no hparams
    if (hparams.vocab_only) {
        return;
    }

    ml.get_key(LFG_KV_CONTEXT_LENGTH,          hparams.n_ctx_train);
    ml.get_key(LFG_KV_EMBEDDING_LENGTH,        hparams.n_embd);
    ml.get_key(LFG_KV_EMBEDDING_LENGTH_OUT,    hparams.n_embd_out, false);
    ml.get_key(LFG_KV_BLOCK_COUNT,             hparams.n_layer);
    ml.get_key(LFG_KV_EXPERT_COUNT,            hparams.n_expert,        false);
    ml.get_key(LFG_KV_EXPERT_USED_COUNT,       hparams.n_expert_used,   false);
    ml.get_key(LFG_KV_EXPERT_GROUP_COUNT,      hparams.n_expert_groups, false);
    ml.get_key(LFG_KV_EXPERT_GROUP_USED_COUNT, hparams.n_group_used,    false);

    if (false) {
        ml.get_key(LFG_KV_FEATURES_LENGTH, hparams.n_embd_features);

        ml.get_key(LFG_KV_POSNET_EMBEDDING_LENGTH, hparams.posnet.n_embd);
        ml.get_key(LFG_KV_POSNET_BLOCK_COUNT,      hparams.posnet.n_layer);

        ml.get_key(LFG_KV_CONVNEXT_EMBEDDING_LENGTH, hparams.convnext.n_embd);
        ml.get_key(LFG_KV_CONVNEXT_BLOCK_COUNT,      hparams.convnext.n_layer);
    }

    GGML_ASSERT(hparams.n_expert <= LFG_MAX_EXPERTS);
    GGML_ASSERT(hparams.n_expert_used <= hparams.n_expert);
    if (hparams.n_expert > 0) {
        GGML_ASSERT(hparams.n_expert_used > 0);
        GGML_ASSERT(hparams.n_expert_groups < hparams.n_expert);
        if (hparams.n_expert_groups > 1) {
            GGML_ASSERT(hparams.n_expert % hparams.n_expert_groups == 0);
            GGML_ASSERT(hparams.n_group_used > 0);
            GGML_ASSERT(hparams.n_group_used < hparams.n_expert_groups);
        }
    } else {
        GGML_ASSERT(hparams.n_expert_used == 0);
        GGML_ASSERT(hparams.n_expert_groups == 0);
    }

    std::fill(hparams.n_head_arr.begin(),    hparams.n_head_arr.end(),    0);
    std::fill(hparams.n_head_kv_arr.begin(), hparams.n_head_kv_arr.end(), 0);
    std::fill(hparams.n_ff_arr.begin(),      hparams.n_ff_arr.end(),      0);
    std::fill(
        hparams.recurrent_layer_arr.begin(),
        hparams.recurrent_layer_arr.end(),
        lfg_arch_is_recurrent(ml.get_arch()));

    std::fill(hparams.rope_sections.begin(), hparams.rope_sections.end(), 0);
    std::fill(hparams.swa_layers.begin(), hparams.swa_layers.end(), 0);

    std::fill(hparams.xielu_alpha_n.begin(), hparams.xielu_alpha_n.end(), 0.0f);
    std::fill(hparams.xielu_alpha_p.begin(), hparams.xielu_alpha_p.end(), 0.0f);
    std::fill(hparams.xielu_beta.begin(), hparams.xielu_beta.end(), 0.0f);
    std::fill(hparams.xielu_eps.begin(), hparams.xielu_eps.end(), 0.0f);

    ml.get_key_or_arr(LFG_KV_FEED_FORWARD_LENGTH,  hparams.n_ff_arr,   hparams.n_layer, false);
    ml.get_key_or_arr(LFG_KV_ATTENTION_HEAD_COUNT, hparams.n_head_arr, hparams.n_layer, false);

    // n_head_kv is optional, default to n_head
    hparams.n_head_kv_arr = hparams.n_head_arr;

    ml.get_key_or_arr(LFG_KV_ATTENTION_HEAD_COUNT_KV, hparams.n_head_kv_arr, hparams.n_layer, false);

    bool rope_finetuned = false;
    ml.get_key(LFG_KV_ROPE_SCALING_FINETUNED, rope_finetuned, false);
    hparams.rope_finetuned = rope_finetuned;

    hparams.n_ctx_orig_yarn = hparams.n_ctx_train;
    ml.get_key(LFG_KV_ROPE_SCALING_ORIG_CTX_LEN, hparams.n_ctx_orig_yarn, false);

    // rope_freq_base (optional)
    hparams.rope_freq_base_train = 10000.0f;
    ml.get_key(LFG_KV_ROPE_FREQ_BASE, hparams.rope_freq_base_train, false);

    std::string rope_scaling("linear");
    ml.get_key(LFG_KV_ROPE_SCALING_TYPE, rope_scaling, false);
    hparams.rope_scaling_type_train = lfg_rope_scaling_type_from_string(rope_scaling);
    GGML_ASSERT(hparams.rope_scaling_type_train != LFG_ROPE_SCALING_TYPE_UNSPECIFIED);

    // TODO: Handle SWA metadata similarly when models start implementing it
    // rope_freq_scale (inverse of the kv) is optional
    float ropescale = 0.0f;
    if (!ml.get_key(LFG_KV_ROPE_SCALING_FACTOR, ropescale, false)) {
        // try the old key name
        ml.get_key(LFG_KV_ROPE_SCALE_LINEAR, ropescale, false);
    }
    hparams.rope_freq_scale_train = ropescale == 0.0f ? 1.0f : 1.0f/ropescale;

    ml.get_key(LFG_KV_ROPE_SCALING_ATTN_FACTOR, hparams.rope_attn_factor, false);

    // non-transformer models do not have attention heads
    if (hparams.n_head() > 0) {
        // gpt-neox n_rot = rotary_pct * (n_embd / n_head)
        // gpt-j n_rot = rotary_dim

        hparams.n_embd_head_k = hparams.n_embd / hparams.n_head();
        ml.get_key(LFG_KV_ATTENTION_KEY_LENGTH, hparams.n_embd_head_k, false);

        hparams.n_embd_head_v = hparams.n_embd / hparams.n_head();
        ml.get_key(LFG_KV_ATTENTION_VALUE_LENGTH, hparams.n_embd_head_v, false);

        // sanity check for n_rot (optional)
        hparams.n_rot = hparams.n_embd_head_k;

        ml.get_key(LFG_KV_ROPE_DIMENSION_COUNT, hparams.n_rot, false);

        if (false) {
            if (hparams.n_rot != hparams.n_embd_head_k) {
                throw std::runtime_error(lfg_format("invalid n_rot: %u, expected %u", hparams.n_rot, hparams.n_embd_head_k));
            }
        }
    } else {
        hparams.n_rot = 0;
        hparams.n_embd_head_k = 0;
        hparams.n_embd_head_v = 0;
    }

    // for differentiating model types
    uint32_t n_vocab = 0;
    ml.get_key(LFG_KV_VOCAB_SIZE, n_vocab, false) || ml.get_arr_n(LFG_KV_TOKENIZER_LIST, n_vocab, false);

    // for classifier models
    ml.get_arr(LFG_KV_CLASSIFIER_OUTPUT_LABELS, classifier_labels, false);
    if (!classifier_labels.empty()) {
        hparams.n_cls_out = classifier_labels.size();
    }

    // arch-specific KVs
    switch (arch) {
        case LFG_ARCH_LFM2:
            {
                ml.get_key(LFG_KV_SHORTCONV_L_CACHE,           hparams.n_shortconv_l_cache);
                ml.get_key(LFG_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);
                for (uint32_t il = 0; il < hparams.n_layer; ++il) {
                    hparams.recurrent_layer_arr[il] = hparams.n_head_kv(il) == 0;
                }
                hparams.n_layer_dense_lead = hparams.n_layer;
                switch (hparams.n_ff()) {
                    case  4608: type = LFG_TYPE_350M; break;
                    case  6912: type = LFG_TYPE_700M; break;
                    case  8192: type = LFG_TYPE_1_2B; break;
                    case 10752: type = LFG_TYPE_2_6B; break;
                    default:    type = LFG_TYPE_UNKNOWN;
                }

                if (const auto is_swa = ml.get_key(LFG_KV_ATTENTION_SLIDING_WINDOW, hparams.n_swa, false); is_swa) {
                    hparams.swa_type = LFG_SWA_TYPE_STANDARD;
                    for (uint32_t il = 0; il < hparams.n_layer; ++il) {
                        hparams.swa_layers[il] = !hparams.recurrent_layer_arr[il];
                    }
                }
            } break;
        case LFG_ARCH_LFM2MOE:
            {
                ml.get_key(LFG_KV_SHORTCONV_L_CACHE,           hparams.n_shortconv_l_cache);
                ml.get_key(LFG_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);
                ml.get_key(LFG_KV_LEADING_DENSE_BLOCK_COUNT,   hparams.n_layer_dense_lead);
                ml.get_key(LFG_KV_EXPERT_FEED_FORWARD_LENGTH,  hparams.n_ff_exp);
                ml.get_key(LFG_KV_EXPERT_GATING_FUNC,          hparams.expert_gating_func);

                for (uint32_t il = 0; il < hparams.n_layer; ++il) {
                    hparams.recurrent_layer_arr[il] = hparams.n_head_kv(il) == 0;
                }

                if (const auto is_swa = ml.get_key(LFG_KV_ATTENTION_SLIDING_WINDOW, hparams.n_swa, false); is_swa) {
                    hparams.swa_type = LFG_SWA_TYPE_STANDARD;
                    for (uint32_t il = 0; il < hparams.n_layer; ++il) {
                        hparams.swa_layers[il] = !hparams.recurrent_layer_arr[il];
                    }
                }

                if (hparams.n_embd == 2048 && hparams.n_ff_exp == 1792) {
                    type = LFG_TYPE_8B_A1B;
                } else if (hparams.n_embd == 2048 && hparams.n_ff_exp == 1536) {
                    type = LFG_TYPE_7B_A1B;
                } else if (hparams.n_embd == 3072 && hparams.n_ff_exp == 2048) {
                    type = LFG_TYPE_16B_A1B;
                } else {
                    type = LFG_TYPE_UNKNOWN;
                }
            } break;
        case LFG_ARCH_BERT:
            {
                ml.get_key(LFG_KV_ATTENTION_LAYERNORM_EPS,    hparams.f_norm_eps);
                ml.get_key(LFG_KV_ATTENTION_CAUSAL,           hparams.causal_attn);
                ml.get_key(LFG_KV_POOLING_TYPE,               hparams.pooling_type, false);

                switch (hparams.n_layer) {
                    case 3:
                        type = LFG_TYPE_17M; break; // bge-micro
                    case 6:
                        type = LFG_TYPE_22M; break; // MiniLM-L6
                    case 12:
                        switch (hparams.n_embd) {
                            case 384: type = LFG_TYPE_33M; break; // MiniLM-L12, bge-small
                            case 768: type = LFG_TYPE_109M; break; // bge-base
                            default: type = LFG_TYPE_UNKNOWN;
                        } break;
                    case 24:
                        type = LFG_TYPE_335M; break; // bge-large
                    default: type = LFG_TYPE_UNKNOWN;
                }
            } break;
        case LFG_ARCH_MODERN_BERT:
            {
                const bool found_swa = ml.get_key(LFG_KV_ATTENTION_SLIDING_WINDOW, hparams.n_swa, false);
                if (found_swa && hparams.n_swa > 0) {
                    uint32_t swa_period = 3;
                    hparams.swa_type = LFG_SWA_TYPE_SYMMETRIC;

                    ml.get_key(LFG_KV_ROPE_FREQ_BASE_SWA, hparams.rope_freq_base_train_swa);
                    ml.get_key_or_arr(LFG_KV_ATTENTION_SLIDING_WINDOW_PATTERN, swa_period, false);
                    hparams.set_swa_pattern(swa_period);
                } else {
                    hparams.swa_type = LFG_SWA_TYPE_NONE;
                }

                ml.get_key(LFG_KV_ATTENTION_LAYERNORM_EPS, hparams.f_norm_eps);
                ml.get_key(LFG_KV_ATTENTION_CAUSAL,        hparams.causal_attn);
                ml.get_key(LFG_KV_POOLING_TYPE,            hparams.pooling_type, false);

                switch (hparams.n_layer) {
                    case 12:
                        type = LFG_TYPE_47M; break; // granite-embedding-small
                    case 22:
                        type = LFG_TYPE_149M; break; // modern-bert-base
                    case 28:
                        type = LFG_TYPE_395M; break; // modern-bert-large
                    default: type = LFG_TYPE_UNKNOWN;
                }
            } break;
        case LFG_ARCH_JINA_BERT_V2:
            {
                ml.get_key(LFG_KV_ATTENTION_LAYERNORM_EPS,    hparams.f_norm_eps);
                ml.get_key(LFG_KV_ATTENTION_CAUSAL,           hparams.causal_attn);
                ml.get_key(LFG_KV_POOLING_TYPE,               hparams.pooling_type, false);
                hparams.f_max_alibi_bias = 8.0f;

                switch (hparams.n_layer) {
                    case 4:  type = LFG_TYPE_33M;  break; // jina-embeddings-small
                    case 12: type = LFG_TYPE_137M; break; // jina-embeddings-base
                    default: type = LFG_TYPE_UNKNOWN;
                }
            } break;
        case LFG_ARCH_JINA_BERT_V3:
            {
                ml.get_key(LFG_KV_ATTENTION_LAYERNORM_EPS,    hparams.f_norm_eps);
                ml.get_key(LFG_KV_ATTENTION_CAUSAL,           hparams.causal_attn);
                ml.get_key(LFG_KV_POOLING_TYPE,               hparams.pooling_type, false);

                switch (hparams.n_layer) {
                    case 24:
                        type = LFG_TYPE_558M; break;
                    default: type = LFG_TYPE_UNKNOWN;
                }
            } break;
        case LFG_ARCH_NOMIC_BERT:
        case LFG_ARCH_NOMIC_BERT_MOE:
            {
                ml.get_key(LFG_KV_ATTENTION_LAYERNORM_EPS,    hparams.f_norm_eps);
                ml.get_key(LFG_KV_ATTENTION_CAUSAL,           hparams.causal_attn);
                ml.get_key(LFG_KV_POOLING_TYPE,               hparams.pooling_type);
                ml.get_key(LFG_KV_MOE_EVERY_N_LAYERS,         hparams.moe_every_n_layers, 0);

                if (hparams.n_layer == 12 && hparams.n_embd == 768) {
                    if (arch == LFG_ARCH_NOMIC_BERT) {
                        type = LFG_TYPE_137M;
                    } else if (arch == LFG_ARCH_NOMIC_BERT_MOE && hparams.moe_every_n_layers == 2) {
                        type = LFG_TYPE_475M;
                    }
                }
            } break;
        case LFG_ARCH_NEO_BERT:
            {
                ml.get_key(LFG_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);
                ml.get_key(LFG_KV_ATTENTION_CAUSAL,            hparams.causal_attn);
                ml.get_key(LFG_KV_POOLING_TYPE,                hparams.pooling_type);

                if (hparams.n_layer == 28) {
                    type = LFG_TYPE_250M;
                }
            } break;
        default: (void)0;
    }
    pimpl->n_bytes = ml.n_bytes;

    pimpl->desc_str = arch_name() + " " + type_name() + " " + ml.ftype_name();

    if (hparams.f_max_alibi_bias > 0.0f) {
        hparams.use_alibi = true;
    }

    hparams.rope_type = lfg_model_rope_type(this);
}

void lfg_model::load_vocab(lfg_model_loader & ml) {
    const auto kv = LFG_KV(arch);

    vocab.load(ml, kv);
}

bool lfg_model::load_tensors(lfg_model_loader & ml) {
    const auto & split_mode   = params.split_mode;
    const auto & use_mlock    = params.use_mlock;
    const auto & tensor_split = params.tensor_split;
    
    const uint32_t n_layer = hparams.n_layer;
    const int n_gpu_layers = this->n_gpu_layers();

    const bool use_mmap_buffer = true;

    LFG_LOG_INFO("%s: loading model tensors, this can take a while... (mmap = %s, direct_io = %s)\n",
        __func__, ml.use_mmap ? "true" : "false", ml.use_direct_io ? "true" : "false");

    // build a list of buffer types for the CPU and GPU devices
    pimpl->cpu_buft_list = make_cpu_buft_list(devices, params.use_extra_bufts, params.no_host);
    for (auto * dev : devices) {
        buft_list_t buft_list = make_gpu_buft_list(dev, split_mode, tensor_split);
        // add CPU buffer types as a fallback
        buft_list.insert(buft_list.end(), pimpl->cpu_buft_list.begin(), pimpl->cpu_buft_list.end());
        pimpl->gpu_buft_list.emplace(dev, std::move(buft_list));
    }

    ggml_backend_dev_t cpu_dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    if (cpu_dev == nullptr) {
        throw std::runtime_error(lfg_format("%s: no CPU backend found", __func__));
    }

    // calculate the split points
    bool all_zero = tensor_split == nullptr || std::all_of(tensor_split, tensor_split + n_devices(), [](float x) { return x == 0.0f; });
    std::vector<float> splits(n_devices());
    if (all_zero) {
        // default split, by free memory
        for (size_t i = 0; i < n_devices(); ++i) {
            ggml_backend_dev_t dev = devices[i];
            size_t total;
            size_t free;
            ggml_backend_dev_memory(dev, &free, &total);

            // devices can return 0 bytes for free and total memory if they do not
            // have any to report. in this case, we will use the host memory as a fallback
            // fixes: https://github.com/ggml-org/liquid.cpp/issues/18577
            if (free == 0 && total == 0) {
                ggml_backend_dev_memory(cpu_dev, &free, &total);
            }
            splits[i] = free;
        }
    } else {
        std::copy(tensor_split, tensor_split + n_devices(), splits.begin());
    }

    // sum and normalize the splits to get the split points
    float split_sum = 0.0f;
    for (size_t i = 0; i < n_devices(); ++i) {
        split_sum += splits[i];
        splits[i] = split_sum;
    }
    for (size_t i = 0; i < n_devices(); ++i) {
        splits[i] /= split_sum;
    }

    const int i_gpu_start = std::max(int(hparams.n_layer) + 1 - n_gpu_layers, 0);
    const int act_gpu_layers = devices.empty() ? 0 : std::min(n_gpu_layers, int(n_layer) + 1);
    auto get_layer_buft_list = [&](int il) -> lfg_model::impl::layer_dev {
        const bool is_swa = il < int(hparams.n_layer) && hparams.is_swa(il);
        GGML_UNUSED(is_swa);
        if (il < i_gpu_start || (il - i_gpu_start) >= act_gpu_layers) {
            LFG_LOG_DEBUG("load_tensors: layer %3d assigned to device %s, is_swa = %d\n", il, ggml_backend_dev_name(cpu_dev), is_swa);
            return {cpu_dev, &pimpl->cpu_buft_list};
        }
        const int layer_gpu = std::upper_bound(splits.begin(), splits.begin() + n_devices(), float(il - i_gpu_start)/act_gpu_layers) - splits.begin();
        auto * dev = devices.at(layer_gpu);
        LFG_LOG_DEBUG("load_tensors: layer %3d assigned to device %s, is_swa = %d\n", il, ggml_backend_dev_name(dev), is_swa);
        return {dev, &pimpl->gpu_buft_list.at(dev)};
    };

    // assign the input layer
    // there is very little benefit to offloading the input layer, so always keep it on the CPU
    pimpl->dev_input = { cpu_dev, &pimpl->cpu_buft_list };

    // assign the repeating layers to the devices according to the splits
    pimpl->dev_layer.resize(n_layer);
    for (uint32_t il = 0; il < n_layer; ++il) {
        pimpl->dev_layer[il] = get_layer_buft_list(il);
    }

    // assign the output layer
    pimpl->dev_output = get_layer_buft_list(n_layer);

    // one ggml context per buffer type
    int max_n_tensors = ml.n_tensors;
    max_n_tensors += 1;         // duplicated output tensor
    max_n_tensors += n_layer*2; // duplicated rope freq tensors
    const size_t ctx_size = ggml_tensor_overhead()*max_n_tensors;

    // define a comparator for the buft -> ctx map to ensure that the order is well-defined:
    struct ggml_backend_buft_comparator {
        bool operator()(const ggml_backend_buffer_type_t & lhs, const ggml_backend_buffer_type_t & rhs) const {
            return strcmp(ggml_backend_buft_name(lhs), ggml_backend_buft_name(rhs)) < 0;
        }
    };
    std::map<ggml_backend_buffer_type_t, ggml_context_ptr, ggml_backend_buft_comparator> ctx_map;

    auto ctx_for_buft = [&](ggml_backend_buffer_type_t buft) -> ggml_context * {
        auto it = ctx_map.find(buft);
        if (it == ctx_map.end()) {
            ggml_init_params params = {
                /*.mem_size   =*/ ctx_size,
                /*.mem_buffer =*/ NULL,
                /*.no_alloc   =*/ true,
            };

            ggml_context * ctx = ggml_init(params);
            if (!ctx) {
                throw std::runtime_error(lfg_format("failed to create ggml context"));
            }

            ctx_map.emplace(buft, ctx);

            return ctx;
        }
        return it->second.get();
    };

    const auto TENSOR_DUPLICATED   = lfg_model_loader::TENSOR_DUPLICATED;
    const auto TENSOR_NOT_REQUIRED = lfg_model_loader::TENSOR_NOT_REQUIRED;
    const auto TENSOR_SKIP         = lfg_model_loader::TENSOR_SKIP;

    // create tensors for the weights
    {
        // note: cast to int64_t since we will use these for the tensor dimensions
        // const int64_t n_head        = hparams.n_head();
        // const int64_t n_head_kv     = hparams.n_head_kv();
        const int64_t n_embd        = hparams.n_embd;
        const int64_t n_embd_k_gqa  = hparams.n_embd_k_gqa();
        const int64_t n_embd_v_gqa  = hparams.n_embd_v_gqa();
        const int64_t n_embd_head_k = hparams.n_embd_head_k;
        // const int64_t n_embd_head_v = hparams.n_embd_head_v;
        const int64_t n_ff          = hparams.n_ff();
        const int64_t n_embd_gqa    = n_embd_v_gqa;
        const int64_t n_vocab       = vocab.n_tokens();
        const int64_t n_token_types = vocab.n_token_types();
        // const int64_t n_rot         = hparams.n_rot;
        const int64_t n_expert      = hparams.n_expert;
        const int64_t n_expert_used = hparams.n_expert_used;
        const int64_t n_ctx_train   = hparams.n_ctx_train;

        if (hparams.n_expert > 0 && hparams.n_expert_used == 0) {
            throw std::runtime_error("model has expert layers but no expert layers are used");
        }

        int n_moved_tensors = 0;
        ggml_tensor * first_moved_tensor = nullptr;
        ggml_backend_buffer_type_t first_moved_from_buft = nullptr;
        ggml_backend_buffer_type_t first_moved_to_buft = nullptr;

        auto create_tensor = [&](const LFG_TN_IMPL & tn, const std::initializer_list<int64_t> & ne, int flags) -> ggml_tensor * {
            ggml_tensor * t_meta = ml.get_tensor_meta(tn.str().c_str());

            if (!t_meta) {
                if (flags & TENSOR_NOT_REQUIRED) {
                    return nullptr;
                }
                throw std::runtime_error(lfg_format("missing tensor '%s'", tn.str().c_str()));
            }

            // some models use the token embedding tensor as the output, but since these are used in different layers and with different ops
            // the tensor is duplicated
            // to handle this, we check if the tensor is duplicated, and if so, we assume that it is being loaded as the output tensor
            lfg_tensor_enum tn_tensor = tn.tensor;
            if (tn.tensor == LFG_TENSOR_TOKEN_EMBD && flags & TENSOR_DUPLICATED) {
                tn_tensor = LFG_TENSOR_OUTPUT;
            }

            lfg_tensor_info info;
            try {
                info = lfg_tensor_info_for(tn_tensor);
            } catch (const std::out_of_range & e) {
                throw std::runtime_error(lfg_format("missing tensor info mapping for %s", tn.str().c_str()));
            }

            // skip unused tensors
            if (info.op == GGML_OP_NONE || flags & TENSOR_SKIP) {
                const size_t nbytes = ggml_nbytes(t_meta);
                LFG_LOG_WARN("model has unused tensor %s (size = %zu bytes) -- ignoring\n", tn.str().c_str(), nbytes);

                ml.size_data -= nbytes;
                ml.n_created++;

                return nullptr;
            }

            // tensors with "bias" suffix are always used with GGML_OP_ADD or GGML_OP_ADD_ID
            ggml_op op;
            bool bias = tn.suffix != nullptr && strcmp(tn.suffix, "bias") == 0;
            if (bias) {
                if (info.op == GGML_OP_MUL_MAT_ID) {
                    op = GGML_OP_ADD_ID;
                } else {
                    op = GGML_OP_ADD;
                }
            } else {
                op = info.op;
            }

            // sanity checks
            if (info.layer == LFG_TENSOR_LAYER_INPUT || info.layer == LFG_TENSOR_LAYER_OUTPUT) {
                if (tn.bid != -1) {
                    lfg_set_last_error(LFG_ERROR_INVALID_ARGUMENT, "%s: input/output tensor %s used with layer number", __func__, tn.str().c_str());
                    throw std::runtime_error("input/output layer tensor used with a layer number");
                }
            } else {
                if (tn.bid == -1) {
                    lfg_set_last_error(LFG_ERROR_INVALID_ARGUMENT, "%s: repeating tensor %s used without layer number", __func__, tn.str().c_str());
                    throw std::runtime_error("repeating layer tensor used without a layer number");
                }
            }

            // select the buffer type for this tensor
            buft_list_t * buft_list;
            switch (info.layer) {
                case LFG_TENSOR_LAYER_INPUT:
                    buft_list = pimpl->dev_input.buft_list;
                    break;
                case LFG_TENSOR_LAYER_OUTPUT:
                    buft_list = pimpl->dev_output.buft_list;
                    break;
                case LFG_TENSOR_LAYER_REPEATING:
                    buft_list = pimpl->dev_layer.at(tn.bid).buft_list;
                    break;
                default:
                    lfg_set_last_error(LFG_ERROR_INVALID_ARGUMENT, "%s: invalid layer %d for tensor %s", __func__, info.layer, tn.str().c_str());
                    throw std::runtime_error("invalid layer for tensor");
            }

            ggml_backend_buffer_type_t buft = nullptr;

            // check overrides
            if (ml.tensor_buft_overrides) {
                std::string tensor_name = tn.str();
                for (const auto * overrides = ml.tensor_buft_overrides; overrides->pattern != nullptr; ++overrides) {
                    std::regex pattern(overrides->pattern);
                    if (std::regex_search(tensor_name, pattern)) {
                        if (overrides->buft == ggml_backend_cpu_buffer_type()) {
                            // when overriding to a CPU buffer, consider the extra buffer types
                            buft = select_weight_buft(hparams, t_meta, op, pimpl->cpu_buft_list);
                        } else {
                            buft = overrides->buft;
                        }

                        LFG_LOG_DEBUG("tensor %s (%zu MiB %s) buffer type overridden to %s\n",
                                tensor_name.c_str(),
                                ggml_nbytes(t_meta) / 1024 / 1024, ggml_type_name(t_meta->type),
                                ggml_backend_buft_name(buft));
                        break;
                    }
                }
            }

            if (!buft) {
                buft = select_weight_buft(hparams, t_meta, op, *buft_list);
                if (!buft) {
                    throw std::runtime_error(lfg_format("failed to find a compatible buffer type for tensor %s", tn.str().c_str()));
                }
            }

            // avoid using a host buffer when using mmap
            auto * buft_dev = ggml_backend_buft_get_device(buft);
            if (ml.use_mmap && buft_dev && buft == ggml_backend_dev_host_buffer_type(buft_dev)) {
                auto * cpu_dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
                if (!cpu_dev) {
                    throw std::runtime_error("no CPU backend found");
                }
                buft = ggml_backend_dev_buffer_type(cpu_dev);
            }

            if (buft != buft_list->front().second) {
                n_moved_tensors++;
                if (!first_moved_tensor) {
                    first_moved_tensor = t_meta;
                    first_moved_from_buft = buft_list->front().second;
                    first_moved_to_buft   = buft;
                }
            }

            ggml_context * ctx = ctx_for_buft(buft);

            // if duplicated, check if the original tensor was allocated in the same buffer type context and avoid creating a new one
            if (flags & TENSOR_DUPLICATED) {
                ggml_tensor * t = ggml_get_tensor(ctx, tn.str().c_str());
                if (t) {
                    return t;
                }
            }
            return ml.create_tensor(ctx, tn, ne, flags);
        };

        layers.resize(n_layer);

        // TODO: move to a separate function
        const auto tn = LFG_TN(arch);
        switch (arch) {
            case LFG_ARCH_LFM2:
            case LFG_ARCH_LFM2MOE:
                {
                    tok_embd = create_tensor(tn(LFG_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, 0);

                    output_norm = create_tensor(tn(LFG_TENSOR_OUTPUT_NORM_LFM2, "weight"), {n_embd}, 0);
                    output      = create_tensor(tn(LFG_TENSOR_OUTPUT,           "weight"), {n_embd, n_vocab}, TENSOR_NOT_REQUIRED);

                    if (output == NULL) {
                        output = create_tensor(tn(LFG_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, TENSOR_DUPLICATED);
                    }

                    for (uint32_t i = 0; i < n_layer; ++i) {
                        auto & layer = layers[i];

                        const bool is_moe_layer = i >= hparams.n_layer_dense_lead;

                        // ffn/moe is same for transformer and conv layers
                        layer.ffn_norm = create_tensor(tn(LFG_TENSOR_FFN_NORM, "weight", i), {n_embd}, 0);
                        if (is_moe_layer) {
                            GGML_ASSERT(n_expert && n_expert_used);
                            layer.ffn_gate_inp    = create_tensor(tn(LFG_TENSOR_FFN_GATE_INP, "weight", i),  {n_embd, n_expert}, 0);
                            layer.ffn_gate_exps   = create_tensor(tn(LFG_TENSOR_FFN_GATE_EXPS, "weight", i), {n_embd, hparams.n_ff_exp, n_expert}, 0);
                            layer.ffn_down_exps   = create_tensor(tn(LFG_TENSOR_FFN_DOWN_EXPS, "weight", i), {hparams.n_ff_exp,   n_embd, n_expert}, 0);
                            layer.ffn_up_exps     = create_tensor(tn(LFG_TENSOR_FFN_UP_EXPS, "weight", i),   {n_embd, hparams.n_ff_exp, n_expert}, 0);
                            layer.ffn_exp_probs_b = create_tensor(tn(LFG_TENSOR_FFN_EXP_PROBS_B, "bias", i), {n_expert}, 0);
                        } else {  // dense
                            layer.ffn_gate = create_tensor(tn(LFG_TENSOR_FFN_GATE, "weight", i), {n_embd,   n_ff}, 0);
                            layer.ffn_down = create_tensor(tn(LFG_TENSOR_FFN_DOWN, "weight", i), {  n_ff, n_embd}, 0);
                            layer.ffn_up   = create_tensor(tn(LFG_TENSOR_FFN_UP,   "weight", i), {n_embd,   n_ff}, 0);
                        }

                        // for operator_norm
                        layer.attn_norm = create_tensor(tn(LFG_TENSOR_ATTN_NORM, "weight", i), {n_embd}, 0);

                        if (!hparams.is_recurrent(i)) {
                            layer.attn_q_norm = create_tensor(tn(LFG_TENSOR_ATTN_Q_NORM, "weight", i), {n_embd_head_k}, 0);
                            layer.attn_k_norm = create_tensor(tn(LFG_TENSOR_ATTN_K_NORM, "weight", i), {n_embd_head_k}, 0);
                            GGML_ASSERT(n_embd_v_gqa == n_embd_k_gqa);

                            layer.wq = create_tensor(tn(LFG_TENSOR_ATTN_Q, "weight", i), {n_embd, n_embd}, 0);
                            layer.wk = create_tensor(tn(LFG_TENSOR_ATTN_K, "weight", i), {n_embd, hparams.n_embd_k_gqa(i)}, 0);
                            layer.wv = create_tensor(tn(LFG_TENSOR_ATTN_V, "weight", i), {n_embd, hparams.n_embd_v_gqa(i)}, 0);

                            layer.wo = create_tensor(tn(LFG_TENSOR_ATTN_OUT, "weight", i), {n_embd, n_embd}, 0);
                        } else {
                            layer.shortconv.conv     = create_tensor(tn(LFG_TENSOR_SHORTCONV_CONV,    "weight", i), {hparams.n_shortconv_l_cache, n_embd}, 0);
                            layer.shortconv.in_proj  = create_tensor(tn(LFG_TENSOR_SHORTCONV_INPROJ,  "weight", i), {n_embd, 3 * n_embd}, 0);
                            layer.shortconv.out_proj = create_tensor(tn(LFG_TENSOR_SHORTCONV_OUTPROJ, "weight", i), {n_embd, n_embd}, 0);
                        }
                    }

                    // for LFM2-ColBert-350M
                    dense_2_out_layers   = create_tensor(tn(LFG_TENSOR_DENSE_2_OUT, "weight"), {n_embd, hparams.get_n_embd_out()}, TENSOR_NOT_REQUIRED);
                    dense_2_out_layers_b = create_tensor(tn(LFG_TENSOR_DENSE_2_OUT, "bias"),   {hparams.get_n_embd_out()}, TENSOR_NOT_REQUIRED);
                } break;
            case LFG_ARCH_BERT:
            case LFG_ARCH_NOMIC_BERT:
            case LFG_ARCH_NOMIC_BERT_MOE:
            case LFG_ARCH_JINA_BERT_V3:
                {
                    tok_embd     = create_tensor(tn(LFG_TENSOR_TOKEN_EMBD,  "weight"), {n_embd, n_vocab}, 0);
                    type_embd    = create_tensor(tn(LFG_TENSOR_TOKEN_TYPES, "weight"), {n_embd, n_token_types}, TENSOR_NOT_REQUIRED);

                    if (arch == LFG_ARCH_BERT) {
                        pos_embd = create_tensor(tn(LFG_TENSOR_POS_EMBD,    "weight"), {n_embd, n_ctx_train}, 0);

                        cls   = create_tensor(tn(LFG_TENSOR_CLS, "weight"), {n_embd, n_embd}, TENSOR_NOT_REQUIRED);
                        cls_b = create_tensor(tn(LFG_TENSOR_CLS, "bias"),   {n_embd},         TENSOR_NOT_REQUIRED);

                        cls_out   = create_tensor(tn(LFG_TENSOR_CLS_OUT, "weight"), {n_embd, hparams.n_cls_out}, TENSOR_NOT_REQUIRED);
                        cls_out_b = create_tensor(tn(LFG_TENSOR_CLS_OUT, "bias"),   {hparams.n_cls_out},         TENSOR_NOT_REQUIRED);
                    }

                    tok_norm   = create_tensor(tn(LFG_TENSOR_TOKEN_EMBD_NORM, "weight"), {n_embd}, 0);
                    tok_norm_b = create_tensor(tn(LFG_TENSOR_TOKEN_EMBD_NORM, "bias"),   {n_embd}, 0);

                    for (uint32_t i = 0; i < n_layer; ++i) {
                        auto & layer = layers[i];

                        layer.wqkv = create_tensor(tn(LFG_TENSOR_ATTN_QKV, "weight", i), {n_embd, n_embd + 2*n_embd_gqa}, TENSOR_NOT_REQUIRED);
                        layer.bqkv = create_tensor(tn(LFG_TENSOR_ATTN_QKV, "bias", i), {n_embd + 2*n_embd_gqa}, TENSOR_NOT_REQUIRED);

                        if (!layer.wqkv) {
                            layer.wq = create_tensor(tn(LFG_TENSOR_ATTN_Q,   "weight", i), {n_embd, n_embd}, 0);
                            layer.bq = create_tensor(tn(LFG_TENSOR_ATTN_Q,   "bias", i),   {n_embd}, 0);

                            layer.wk = create_tensor(tn(LFG_TENSOR_ATTN_K,   "weight", i), {n_embd, n_embd_gqa}, 0);
                            layer.bk = create_tensor(tn(LFG_TENSOR_ATTN_K,   "bias", i),   {n_embd_gqa}, 0);

                            layer.wv = create_tensor(tn(LFG_TENSOR_ATTN_V,   "weight", i), {n_embd, n_embd_gqa}, 0);
                            layer.bv = create_tensor(tn(LFG_TENSOR_ATTN_V,   "bias", i),   {n_embd_gqa}, 0);
                        }

                        layer.wo = create_tensor(tn(LFG_TENSOR_ATTN_OUT,      "weight", i), {n_embd, n_embd}, 0);
                        layer.bo = create_tensor(tn(LFG_TENSOR_ATTN_OUT,      "bias", i),   {n_embd}, TENSOR_NOT_REQUIRED);

                        layer.attn_out_norm   = create_tensor(tn(LFG_TENSOR_ATTN_OUT_NORM, "weight", i), {n_embd}, 0);
                        layer.attn_out_norm_b = create_tensor(tn(LFG_TENSOR_ATTN_OUT_NORM, "bias", i),   {n_embd}, 0);

                        if (hparams.moe_every_n_layers > 0 && i % hparams.moe_every_n_layers == 1) {
                            layer.ffn_up_exps   = create_tensor(tn(LFG_TENSOR_FFN_UP_EXPS,   "weight", i), {  n_embd, n_ff,   n_expert}, 0);
                            layer.ffn_down_exps = create_tensor(tn(LFG_TENSOR_FFN_DOWN_EXPS, "weight", i), {  n_ff,   n_embd, n_expert}, 0);
                            layer.ffn_gate_inp  = create_tensor(tn(LFG_TENSOR_FFN_GATE_INP,  "weight", i), {n_embd, n_expert}, 0);
                        } else {
                            layer.ffn_up     = create_tensor(tn(LFG_TENSOR_FFN_UP,   "weight", i), {n_embd, n_ff}, 0);
                            layer.ffn_up_b   = create_tensor(tn(LFG_TENSOR_FFN_UP,   "bias", i),   {n_ff}, TENSOR_NOT_REQUIRED);
                            layer.ffn_down   = create_tensor(tn(LFG_TENSOR_FFN_DOWN, "weight", i), {n_ff, n_embd}, 0);
                            layer.ffn_down_b = create_tensor(tn(LFG_TENSOR_FFN_DOWN, "bias", i),   {n_embd}, TENSOR_NOT_REQUIRED);

                            if (arch == LFG_ARCH_NOMIC_BERT) {
                                layer.ffn_gate = create_tensor(tn(LFG_TENSOR_FFN_GATE, "weight", i), {n_embd, n_ff}, 0);
                            }
                        }

                        layer.layer_out_norm   = create_tensor(tn(LFG_TENSOR_LAYER_OUT_NORM, "weight", i), {n_embd}, 0);
                        layer.layer_out_norm_b = create_tensor(tn(LFG_TENSOR_LAYER_OUT_NORM, "bias", i),   {n_embd}, 0);
                    }
                } break;
            case LFG_ARCH_MODERN_BERT:
                {
                    tok_embd = create_tensor(tn(LFG_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, 0);
                    tok_norm = create_tensor(tn(LFG_TENSOR_TOKEN_EMBD_NORM, "weight"), {n_embd}, 0);

                    output_norm = create_tensor(tn(LFG_TENSOR_OUTPUT_NORM, "weight"), {n_embd}, 0);

                    for (uint32_t i = 0; i < n_layer; ++i) {
                        auto & layer = layers[i];

                        if (i != 0) {
                            layer.attn_norm = create_tensor(tn(LFG_TENSOR_ATTN_NORM, "weight", i), {n_embd}, 0);
                        } else {
                            // layer 0 uses identity
                            layer.attn_norm = create_tensor(tn(LFG_TENSOR_ATTN_NORM, "weight", i), {n_embd}, TENSOR_NOT_REQUIRED);
                        }

                        layer.wqkv = create_tensor(tn(LFG_TENSOR_ATTN_QKV, "weight", i), {n_embd, 3 * n_embd}, 0);
                        layer.wo   = create_tensor(tn(LFG_TENSOR_ATTN_OUT, "weight", i), {n_embd, n_embd}, 0);

                        layer.ffn_up   = create_tensor(tn(LFG_TENSOR_FFN_UP,   "weight", i), {n_embd, 2 * n_ff}, 0);
                        layer.ffn_down = create_tensor(tn(LFG_TENSOR_FFN_DOWN, "weight", i), {n_ff, n_embd}, 0);
                        layer.ffn_norm = create_tensor(tn(LFG_TENSOR_FFN_NORM, "weight", i), {n_embd}, 0);
                    }

                    cls       = create_tensor(tn(LFG_TENSOR_CLS,     "weight"), {n_embd, n_embd}, TENSOR_NOT_REQUIRED);
                    cls_out   = create_tensor(tn(LFG_TENSOR_CLS_OUT, "weight"), {n_embd, hparams.n_cls_out}, TENSOR_NOT_REQUIRED);
                    cls_out_b = create_tensor(tn(LFG_TENSOR_CLS_OUT, "bias"),   {hparams.n_cls_out},         TENSOR_NOT_REQUIRED);
                } break;
            case LFG_ARCH_NEO_BERT:
                {
                    tok_embd = create_tensor(tn(LFG_TENSOR_TOKEN_EMBD,  "weight"), {n_embd, n_vocab}, 0);

                    cls   = create_tensor(tn(LFG_TENSOR_CLS, "weight"), {n_embd, n_embd}, TENSOR_NOT_REQUIRED);
                    cls_b = create_tensor(tn(LFG_TENSOR_CLS, "bias"),   {n_embd},         TENSOR_NOT_REQUIRED);

                    cls_out   = create_tensor(tn(LFG_TENSOR_CLS_OUT, "weight"), {n_embd, hparams.n_cls_out}, TENSOR_NOT_REQUIRED);
                    cls_out_b = create_tensor(tn(LFG_TENSOR_CLS_OUT, "bias"),   {hparams.n_cls_out},         TENSOR_NOT_REQUIRED);

                    output_norm_enc = create_tensor(tn(LFG_TENSOR_ENC_OUTPUT_NORM, "weight"), {n_embd}, 0);

                    for (uint32_t i = 0; i < n_layer; ++i) {
                        auto & layer = layers[i];

                        layer.attn_norm = create_tensor(tn(LFG_TENSOR_ATTN_NORM, "weight", i), {n_embd}, 0);

                        layer.wqkv = create_tensor(tn(LFG_TENSOR_ATTN_QKV, "weight", i), {n_embd, n_embd + 2*n_embd_gqa}, 0);
                        layer.wo   = create_tensor(tn(LFG_TENSOR_ATTN_OUT, "weight", i), {n_embd, n_embd}, 0);

                        layer.ffn_norm = create_tensor(tn(LFG_TENSOR_FFN_NORM, "weight", i), {n_embd}, 0);

                        layer.ffn_up   = create_tensor(tn(LFG_TENSOR_FFN_UP,   "weight", i), {n_embd, n_ff*2}, 0);
                        layer.ffn_down = create_tensor(tn(LFG_TENSOR_FFN_DOWN, "weight", i), {n_ff, n_embd}, 0);
                    }
                } break;
            case LFG_ARCH_JINA_BERT_V2:
                {
                    tok_embd  = create_tensor(tn(LFG_TENSOR_TOKEN_EMBD,  "weight"), {n_embd, n_vocab}, 0);
                    type_embd = create_tensor(tn(LFG_TENSOR_TOKEN_TYPES, "weight"), {n_embd, n_token_types}, 0);

                    tok_norm   = create_tensor(tn(LFG_TENSOR_TOKEN_EMBD_NORM, "weight"), {n_embd}, 0);
                    tok_norm_b = create_tensor(tn(LFG_TENSOR_TOKEN_EMBD_NORM, "bias"),   {n_embd}, 0);

                    cls   = create_tensor(tn(LFG_TENSOR_CLS, "weight"), {n_embd, 1}, TENSOR_NOT_REQUIRED);
                    cls_b = create_tensor(tn(LFG_TENSOR_CLS, "bias"),   {1},         TENSOR_NOT_REQUIRED);

                    for (uint32_t i = 0; i < n_layer; ++i) {
                        auto & layer = layers[i];

                        layer.wq = create_tensor(tn(LFG_TENSOR_ATTN_Q, "weight", i), {n_embd, n_embd}, 0);
                        layer.bq = create_tensor(tn(LFG_TENSOR_ATTN_Q, "bias", i),   {n_embd}, 0);

                        layer.attn_q_norm   = create_tensor(tn(LFG_TENSOR_ATTN_Q_NORM, "weight", i), {n_embd}, TENSOR_NOT_REQUIRED);
                        layer.attn_q_norm_b = create_tensor(tn(LFG_TENSOR_ATTN_Q_NORM, "bias",   i), {n_embd}, TENSOR_NOT_REQUIRED);

                        layer.wk = create_tensor(tn(LFG_TENSOR_ATTN_K, "weight", i), {n_embd, n_embd_gqa}, 0);
                        layer.bk = create_tensor(tn(LFG_TENSOR_ATTN_K, "bias",   i), {n_embd_gqa}, 0);

                        layer.attn_k_norm   = create_tensor(tn(LFG_TENSOR_ATTN_K_NORM, "weight", i), {n_embd}, TENSOR_NOT_REQUIRED);
                        layer.attn_k_norm_b = create_tensor(tn(LFG_TENSOR_ATTN_K_NORM, "bias",   i), {n_embd}, TENSOR_NOT_REQUIRED);

                        layer.wv = create_tensor(tn(LFG_TENSOR_ATTN_V, "weight", i), {n_embd, n_embd_gqa}, 0);
                        layer.bv = create_tensor(tn(LFG_TENSOR_ATTN_V, "bias",   i), {n_embd_gqa}, 0);

                        layer.wo = create_tensor(tn(LFG_TENSOR_ATTN_OUT, "weight", i), {n_embd, n_embd}, 0);
                        layer.bo = create_tensor(tn(LFG_TENSOR_ATTN_OUT, "bias",   i), {n_embd}, 0);

                        layer.attn_out_norm   = create_tensor(tn(LFG_TENSOR_ATTN_OUT_NORM, "weight", i), {n_embd}, 0);
                        layer.attn_out_norm_b = create_tensor(tn(LFG_TENSOR_ATTN_OUT_NORM, "bias",   i), {n_embd}, 0);

                        layer.attn_norm_2   = create_tensor(tn(LFG_TENSOR_ATTN_NORM_2, "weight", i), {n_embd}, TENSOR_NOT_REQUIRED);
                        layer.attn_norm_2_b = create_tensor(tn(LFG_TENSOR_ATTN_NORM_2, "bias",   i), {n_embd}, TENSOR_NOT_REQUIRED);

                        layer.ffn_gate = create_tensor(tn(LFG_TENSOR_FFN_GATE, "weight", i), {n_embd, n_ff}, TENSOR_NOT_REQUIRED);

                        const auto tn_ffn_up_weight = tn(LFG_TENSOR_FFN_UP, "weight", i);
                        ggml_tensor * t_ffn_up = ml.get_tensor_meta(tn_ffn_up_weight.str().c_str());
                        const int64_t n_ffn_up = t_ffn_up ? t_ffn_up->ne[1] : n_ff;

                        GGML_ASSERT(n_ffn_up == n_ff || n_ffn_up == n_ff * 2);
                        layer.ffn_up   = create_tensor(tn_ffn_up_weight, {n_embd, n_ffn_up}, 0);
                        layer.ffn_up_b = create_tensor(tn(LFG_TENSOR_FFN_UP, "bias", i), {n_ffn_up}, TENSOR_NOT_REQUIRED);

                        layer.ffn_down   = create_tensor(tn(LFG_TENSOR_FFN_DOWN, "weight", i), {n_ff, n_embd}, 0);
                        layer.ffn_down_b = create_tensor(tn(LFG_TENSOR_FFN_DOWN, "bias",   i), {n_embd}, 0);

                        layer.layer_out_norm   = create_tensor(tn(LFG_TENSOR_LAYER_OUT_NORM, "weight", i), {n_embd}, 0);
                        layer.layer_out_norm_b = create_tensor(tn(LFG_TENSOR_LAYER_OUT_NORM, "bias", i),   {n_embd}, 0);
                    }
                } break;
            default:
                throw std::runtime_error("unknown architecture");
        }

        if (n_moved_tensors > 0) {
            LFG_LOG_DEBUG("%s: tensor '%s' (%s) (and %d others) cannot be used with preferred buffer type %s, using %s instead\n",
                __func__, first_moved_tensor->name, ggml_type_name(first_moved_tensor->type), n_moved_tensors - 1,
                ggml_backend_buft_name(first_moved_from_buft), ggml_backend_buft_name(first_moved_to_buft));
        }
    }

    ml.done_getting_tensors();

    ml.init_mappings(true, use_mlock ? &pimpl->mlock_mmaps : nullptr);
    pimpl->mappings.reserve(ml.mappings.size());

    // create the backend buffers
    std::vector<std::pair<ggml_context *, lfg_buf_map>> ctx_buf_maps;
    ctx_buf_maps.reserve(ctx_map.size());

    // Ensure we have enough capacity for the maximum backend buffeliquidwill potentially create
    const size_t n_max_backend_buffer = ctx_map.size() * ml.files.size();
    pimpl->ctxs_bufs.reserve(n_max_backend_buffer);

    for (auto & [buft, ctx_ptr] : ctx_map) {
        ggml_context * ctx = ctx_ptr.get();

        // skip contexts without tensors
        if (ggml_get_first_tensor(ctx) == nullptr) {
            continue;
        }

        lfg_buf_map buf_map;
        buf_map.reserve(n_max_backend_buffer);

        // check if it is possible to use buffer_from_host_ptr with this buffer type
        ggml_backend_dev_t dev = ggml_backend_buft_get_device(buft);
        if (!dev) {
            // FIXME: workaround for CPU backend buft having a NULL device
            dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
            if (!dev) {
                throw std::runtime_error(lfg_format("%s: no CPU backend found", __func__));
            }
        }
        ggml_backend_dev_props props;
        ggml_backend_dev_get_props(dev, &props);
        bool buffer_from_host_ptr_supported = props.caps.buffer_from_host_ptr;
        bool is_default_buft = buft == ggml_backend_dev_buffer_type(dev);

        std::vector<ggml_backend_buffer_ptr> bufs;
        if (ml.use_mmap && use_mmap_buffer && buffer_from_host_ptr_supported && is_default_buft) {
            GGML_ASSERT(!ml.no_alloc);
            for (uint32_t idx = 0; idx < ml.files.size(); idx++) {
                // only the mmap region containing the tensors in the model is mapped to the backend buffer
                // this is important for metal with apple silicon: if the entire model could be mapped to a metal buffer,
                //     then we could just use metal for all layers
                // this allows using partial offloading when the model size exceeds the metal buffer size, but not the RAM size
                void * addr = nullptr;
                size_t first, last; // NOLINT
                ml.get_mapping_range(&first, &last, &addr, idx, ctx);
                if (first >= last) {
                    continue;
                }
                const size_t max_size = ggml_get_max_tensor_size(ctx);
                ggml_backend_buffer_t buf = ggml_backend_dev_buffer_from_host_ptr(dev, (char *) addr + first, last - first, max_size);
                if (buf == nullptr) {
                    throw std::runtime_error(lfg_format("unable to allocate %s buffer", ggml_backend_buft_name(buft)));
                }
                bufs.emplace_back(buf);
                buf_map.emplace(idx, buf);
            }
        } else {
            ggml_backend_buffer_t buf;
            if (ml.no_alloc) {
                buf = ggml_backend_buft_alloc_buffer(buft, /*size =*/ 0); // dummy buffer
                for (ggml_tensor * t = ggml_get_first_tensor(ctx); t != nullptr; t = ggml_get_next_tensor(ctx, t)) {
                    t->buffer = buf; // set dummy buffer for weights so that the backend scheduler won't try to allocate them
                }
            } else {
                buf = ggml_backend_alloc_ctx_tensors_from_buft(ctx, buft); // real buffer
            }
            if (buf == nullptr) {
                throw std::runtime_error(lfg_format("unable to allocate %s buffer", ggml_backend_buft_name(buft)));
            }
            if (use_mlock && ggml_backend_buffer_is_host(buf)) {
                pimpl->mlock_bufs.emplace_back(new lfg_mlock);
                auto & mlock_buf = pimpl->mlock_bufs.back();
                mlock_buf->init   (ggml_backend_buffer_get_base(buf));
                mlock_buf->grow_to(ggml_backend_buffer_get_size(buf));
            }
            bufs.emplace_back(buf);
            for (uint32_t idx = 0; idx < ml.files.size(); idx++) {
                buf_map.emplace(idx, buf);
            }
        }
        pimpl->ctxs_bufs.emplace_back(std::move(ctx_ptr), std::move(bufs));

        for (auto & buf : buf_map) {
            // indicate that this buffer contains weights
            // this is used by ggml_backend_sched to improve op scheduling: ops that use a weight are preferably scheduled to the backend that contains the weight
            ggml_backend_buffer_set_usage(buf.second, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        }

        ctx_buf_maps.emplace_back(ctx, buf_map);
    }

    if (lfg_supports_gpu_offload()) {
        const int n_gpu = std::min(n_gpu_layers, (int)hparams.n_layer);

        int n_repeating = n_gpu;
        if (n_repeating > 0) {
            LFG_LOG_INFO("%s: offloading output layer to GPU\n", __func__);
            n_repeating--;
        }
        LFG_LOG_INFO("%s: offloading %d repeating layers to GPU\n", __func__, n_repeating);

        const int max_backend_supported_layers = hparams.n_layer + 1;
        const int max_offloadable_layers       = hparams.n_layer + 1;

        LFG_LOG_INFO("%s: offloaded %d/%d layers to GPU\n", __func__, std::min(n_gpu_layers, max_offloadable_layers), max_backend_supported_layers);
    }

    // print memory requirements per buffer type
    for (auto & [_, bufs] : pimpl->ctxs_bufs) {
        for (auto & buf: bufs) {
            LFG_LOG_INFO("%s: %12s model buffer size = %8.2f MiB\n",
                __func__, ggml_backend_buffer_name(buf.get()), ggml_backend_buffer_get_size(buf.get()) / 1024.0 / 1024.0);
        }
    }

    // populate tensors_by_name
    for (auto & [ctx, _] : pimpl->ctxs_bufs) {
        for (auto * cur = ggml_get_first_tensor(ctx.get()); cur != NULL; cur = ggml_get_next_tensor(ctx.get(), cur)) {
            tensors_by_name.emplace_back(ggml_get_name(cur), cur);
        }
    }

    if (ml.no_alloc) {
        return true;
    }

    // load tensor data
    for (auto & [ctx, buf_map] : ctx_buf_maps) {
        if (!ml.load_all_data(ctx, buf_map, use_mlock ? &pimpl->mlock_mmaps : NULL, params.progress_callback, params.progress_callback_user_data)) {
            return false;
        }
    }

    if (use_mmap_buffer) {
        for (auto & mapping : ml.mappings) {
            pimpl->mappings.emplace_back(std::move(mapping));
        }
    }

    return true;
}

std::string lfg_model::arch_name() const {
    return lfg_arch_name(arch);
}

std::string lfg_model::type_name() const {
    return lfg_type_name(type);
}

std::string lfg_model::desc() const {
    return pimpl->desc_str;
}

size_t lfg_model::size() const {
    return pimpl->n_bytes;
}

size_t lfg_model::n_tensors() const {
    return tensors_by_name.size();
}

size_t lfg_model::n_devices() const {
    return devices.size();
}

uint32_t lfg_model::n_gpu_layers() const {
    return params.n_gpu_layers >= 0 ? (uint32_t)params.n_gpu_layers : hparams.n_layer + 1;
}

lfg_split_mode lfg_model::split_mode() const {
    return params.split_mode;
}

std::map<ggml_backend_buffer_type_t, size_t> lfg_model::memory_breakdown() const {
    std::map<ggml_backend_buffer_type_t, size_t> ret;
    for (const auto & [ctx, bufs] : pimpl->ctxs_bufs) {
        if (hparams.no_alloc) {
            GGML_ASSERT(bufs.size() == 1);
            ggml_backend_buffer_t buf = bufs[0].get();
            GGML_ASSERT(ggml_backend_buffer_get_base(buf) == nullptr);
            ggml_backend_buffer_type_t buft = ggml_backend_buffer_get_type(buf);
            ret[buft] += ggml_backend_alloc_ctx_tensors_from_buft_size(ctx.get(), buft);
        } else {
            for (const auto & buf : bufs) {
                // GGML_ASSERT(ggml_backend_buffer_get_base(buf.get()) != nullptr); // multi_buffer does not have a defined base
                ret[ggml_backend_buffer_get_type(buf.get())] += ggml_backend_buffer_get_size(buf.get());
            }
        }
    }
    return ret;
}

uint64_t lfg_model::n_elements() const {
    return pimpl->n_elements;
}

void lfg_model::print_info() const {
    const std::string rope_scaling_type = lfg_rope_scaling_type_name(hparams.rope_scaling_type_train);

    auto print_f = [](const std::function<uint32_t(uint32_t)> & f, uint32_t n) {
        bool is_var = false;

        std::vector<uint32_t> v;
        for (uint32_t i = 0; i < n; ++i) {
            v.push_back(f(i));
            if (v[i] != v[0]) {
                is_var = true;
            }
        }

        std::stringstream ss;

        if (is_var) {
            ss << "[";
            for (uint32_t i = 0; i < n; ++i) {
                ss << v[i];
                if (i < n - 1) {
                    ss << ", ";
                }
            }
            ss << "]";
        } else {
            ss << v[0];
        }

        return ss.str();
    };

    // hparams
    LFG_LOG_INFO("{}: arch                  = {}\n",     __func__, arch_name().c_str());
    LFG_LOG_INFO("{}: vocab_only            = {}\n",     __func__, hparams.vocab_only);
    LFG_LOG_INFO("{}: no_alloc              = {}\n",     __func__, hparams.no_alloc);

    if (!hparams.vocab_only) {
        LFG_LOG_INFO("{}: n_ctx_train           = {}\n",     __func__, hparams.n_ctx_train);
        LFG_LOG_INFO("{}: n_embd                = {}\n",     __func__, hparams.n_embd);
        LFG_LOG_INFO("{}: n_embd_inp            = {}\n",     __func__, hparams.n_embd_inp());
        LFG_LOG_INFO("{}: n_layer               = {}\n",     __func__, hparams.n_layer);
        LFG_LOG_INFO("{}: n_head                = {}\n",     __func__, print_f([&](uint32_t il) { return hparams.n_head(il);    }, hparams.n_layer).c_str());
        LFG_LOG_INFO("{}: n_head_kv             = {}\n",     __func__, print_f([&](uint32_t il) { return hparams.n_head_kv(il); }, hparams.n_layer).c_str());
        LFG_LOG_INFO("{}: n_rot                 = {}\n",     __func__, hparams.n_rot);
        LFG_LOG_INFO("{}: n_swa                 = {}\n",     __func__, hparams.n_swa);
        LFG_LOG_INFO("{}: is_swa_any            = {}\n",     __func__, hparams.is_swa_any());
        LFG_LOG_INFO("{}: n_embd_head_k         = {}\n",     __func__, hparams.n_embd_head_k);
        LFG_LOG_INFO("{}: n_embd_head_v         = {}\n",     __func__, hparams.n_embd_head_v);
        LFG_LOG_INFO("{}: n_gqa                 = {}\n",     __func__, print_f([&](uint32_t il) { return hparams.n_gqa(il);        }, hparams.n_layer).c_str());
        LFG_LOG_INFO("{}: n_embd_k_gqa          = {}\n",     __func__, print_f([&](uint32_t il) { return hparams.n_embd_k_gqa(il); }, hparams.n_layer).c_str());
        LFG_LOG_INFO("{}: n_embd_v_gqa          = {}\n",     __func__, print_f([&](uint32_t il) { return hparams.n_embd_v_gqa(il); }, hparams.n_layer).c_str());
        LFG_LOG_INFO("{}: f_norm_eps            = {:.1e}\n",   __func__, hparams.f_norm_eps);
        LFG_LOG_INFO("{}: f_norm_rms_eps        = {:.1e}\n",   __func__, hparams.f_norm_rms_eps);
        LFG_LOG_INFO("{}: f_clamp_kqv           = {:.1e}\n",   __func__, hparams.f_clamp_kqv);
        LFG_LOG_INFO("{}: f_max_alibi_bias      = {:.1e}\n",   __func__, hparams.f_max_alibi_bias);
        LFG_LOG_INFO("{}: f_logit_scale         = {:.1e}\n",   __func__, hparams.f_logit_scale);
        LFG_LOG_INFO("{}: f_attn_scale          = {:.1e}\n",   __func__, hparams.f_attention_scale);
        LFG_LOG_INFO("{}: n_ff                  = {}\n",     __func__, print_f([&](uint32_t il) { return hparams.n_ff(il); }, hparams.n_layer).c_str());
        LFG_LOG_INFO("{}: n_expert              = {}\n",     __func__, hparams.n_expert);
        LFG_LOG_INFO("{}: n_expert_used         = {}\n",     __func__, hparams.n_expert_used);
        LFG_LOG_INFO("{}: n_expert_groups       = {}\n",     __func__, hparams.n_expert_groups);
        LFG_LOG_INFO("{}: n_group_used          = {}\n",     __func__, hparams.n_group_used);
        LFG_LOG_INFO("{}: causal attn           = {}\n",     __func__, hparams.causal_attn);
        LFG_LOG_INFO("{}: pooling type          = {}\n",     __func__, static_cast<int>(hparams.pooling_type));
        LFG_LOG_INFO("{}: rope type             = {}\n",     __func__, static_cast<int>(hparams.rope_type));
        LFG_LOG_INFO("{}: rope scaling          = {}\n",     __func__, rope_scaling_type.c_str());
        LFG_LOG_INFO("{}: freq_base_train       = {:.1f}\n",   __func__, hparams.rope_freq_base_train);
        LFG_LOG_INFO("{}: freq_scale_train      = {:g}\n",     __func__, hparams.rope_freq_scale_train);
        if (hparams.swa_type != LFG_SWA_TYPE_NONE) {
            LFG_LOG_INFO("{}: freq_base_swa         = {:.1f}\n",   __func__, hparams.rope_freq_base_train_swa);
            LFG_LOG_INFO("{}: freq_scale_swa        = {:g}\n",     __func__, hparams.rope_freq_scale_train_swa);
        }
        LFG_LOG_INFO("{}: n_ctx_orig_yarn       = {}\n",     __func__, hparams.n_ctx_orig_yarn);
        LFG_LOG_INFO("{}: rope_yarn_log_mul     = {:.4f}\n",   __func__, hparams.rope_yarn_log_mul);
        LFG_LOG_INFO("{}: rope_finetuned        = {}\n",     __func__, hparams.rope_finetuned ? "yes" : "unknown");
        // MRoPE (Multi-axis Rotary Position Embedding) sections
        if (const auto & s = hparams.rope_sections; s[0] || s[1] || s[2] || s[3]) {
            LFG_LOG_INFO("{}: mrope sections        = [{}, {}, {}, {}]\n", __func__, s[0], s[1], s[2], s[3]);
        }
        if (!classifier_labels.empty()) {
            LFG_LOG_INFO("{}: n_cls_out             = {}\n", __func__, hparams.n_cls_out);

            size_t i = 0;
            for (auto label : classifier_labels) {
                LFG_LOG_INFO("{}: cls_label[{:2zu}]         = {}\n", __func__, i++, label.c_str());
            }
        }
    }



    LFG_LOG_INFO("{}: model type            = {}\n",     __func__, type_name().c_str());
    if (pimpl->n_elements >= 1e12) {
        LFG_LOG_INFO("{}: model params          = {:.2f} T\n", __func__, pimpl->n_elements*1e-12);
    } else if (pimpl->n_elements >= 1e9) {
        LFG_LOG_INFO("{}: model params          = {:.2f} B\n", __func__, pimpl->n_elements*1e-9);
    } else if (pimpl->n_elements >= 1e6) {
        LFG_LOG_INFO("{}: model params          = {:.2f} M\n", __func__, pimpl->n_elements*1e-6);
    } else {
        LFG_LOG_INFO("{}: model params          = {:.2f} K\n", __func__, pimpl->n_elements*1e-3);
    }

    // general kv
    LFG_LOG_INFO("{}: general.name          = {}\n",    __func__, name.c_str());

    if (false) {
        LFG_LOG_INFO("{}: n_layer_dense_lead    = {}\n",     __func__, hparams.n_layer_dense_lead);
        LFG_LOG_INFO("{}: n_ff_exp              = {}\n",     __func__, hparams.n_ff_exp);
        LFG_LOG_INFO("{}: n_expert_shared       = {}\n",     __func__, hparams.n_expert_shared);
        LFG_LOG_INFO("{}: expert_weights_scale  = {:.1f}\n",   __func__, hparams.expert_weights_scale);
    }

    if (false) {
        LFG_LOG_INFO("{}: n_layer_dense_lead    = {}\n",     __func__, hparams.n_layer_dense_lead);
        LFG_LOG_INFO("{}: n_lora_q              = {}\n",     __func__, hparams.n_lora_q);
        LFG_LOG_INFO("{}: n_lora_kv             = {}\n",     __func__, hparams.n_lora_kv);
        LFG_LOG_INFO("{}: n_embd_head_k_mla     = {}\n",     __func__, hparams.n_embd_head_k_mla);
        LFG_LOG_INFO("{}: n_embd_head_v_mla     = {}\n",     __func__, hparams.n_embd_head_v_mla);
        LFG_LOG_INFO("{}: n_ff_exp              = {}\n",     __func__, hparams.n_ff_exp);
        LFG_LOG_INFO("{}: n_expert_shared       = {}\n",     __func__, hparams.n_expert_shared);
        LFG_LOG_INFO("{}: expert_weights_scale  = {:.1f}\n",   __func__, hparams.expert_weights_scale);
        LFG_LOG_INFO("{}: expert_weights_norm   = {}\n",     __func__, hparams.expert_weights_norm);
        LFG_LOG_INFO("{}: expert_gating_func    = {}\n",     __func__, lfg_expert_gating_func_name((lfg_expert_gating_func_type) hparams.expert_gating_func));
    }

    if (false) {
        LFG_LOG_INFO("{}: n_ff_exp              = {}\n",     __func__, hparams.n_ff_exp);
        LFG_LOG_INFO("{}: n_ff_shexp            = {}\n",     __func__, hparams.n_ff_shexp);
    }

    if (false) {
        LFG_LOG_INFO("{}: n_ff_exp              = {}\n",     __func__, hparams.n_ff_exp);
    }



    if (false) {
        LFG_LOG_INFO("{}: n_layer_dense_lead    = {}\n",     __func__, hparams.n_layer_dense_lead);
        LFG_LOG_INFO("{}: n_ff_exp              = {}\n",     __func__, hparams.n_ff_exp);
        LFG_LOG_INFO("{}: n_expert_shared       = {}\n",     __func__, hparams.n_expert_shared);
        LFG_LOG_INFO("{}: expert_weights_scale  = {:.1f}\n",   __func__, hparams.expert_weights_scale);
        LFG_LOG_INFO("{}: expert_weights_norm   = {}\n",     __func__, hparams.expert_weights_norm);
    }

    if (false) {
        LFG_LOG_INFO("{}: n_layer_dense_lead    = {}\n",     __func__, hparams.n_layer_dense_lead);
        LFG_LOG_INFO("{}: n_ff_exp              = {}\n",     __func__, hparams.n_ff_exp);
        LFG_LOG_INFO("{}: n_ff_shexp            = {}\n",     __func__, hparams.n_ff_shexp);
        LFG_LOG_INFO("{}: n_expert_shared       = {}\n",     __func__, hparams.n_expert_shared);
        LFG_LOG_INFO("{}: expert_weights_scale  = {:.1f}\n",   __func__, hparams.expert_weights_scale);
        LFG_LOG_INFO("{}: expert_weights_norm   = {}\n",     __func__, hparams.expert_weights_norm);
        LFG_LOG_INFO("{}: expert_gating_func    = {}\n",     __func__, lfg_expert_gating_func_name((lfg_expert_gating_func_type) hparams.expert_gating_func));
        LFG_LOG_INFO("{}: nextn_predict_layers  = {}\n",     __func__, hparams.nextn_predict_layers);
    }

    if (arch == LFG_ARCH_LFM2MOE) {
        LFG_LOG_INFO("{}: n_ff_exp              = {}\n",     __func__, hparams.n_ff_exp);
        LFG_LOG_INFO("{}: expert_gating_func    = {}\n",     __func__, lfg_expert_gating_func_name((lfg_expert_gating_func_type) hparams.expert_gating_func));
    }

    if (false) {
        LFG_LOG_INFO("{}: n_ff_exp              = {}\n",     __func__, hparams.n_ff_exp);
        LFG_LOG_INFO("{}: n_ff_chexp            = {}\n",     __func__, hparams.n_ff_chexp);
        LFG_LOG_INFO("{}: n_group_experts       = {}\n",     __func__, hparams.n_group_experts);
        LFG_LOG_INFO("{}: expert_group_scale    = {:.2f}\n",   __func__, hparams.expert_group_scale);
    }

    vocab.print_info();
}

ggml_backend_dev_t lfg_model::dev_layer(uint32_t il) const {
    if (pimpl->dev_layer.empty()) {
        return pimpl->dev_input.dev;
    }
    return pimpl->dev_layer.at(il).dev;
}

ggml_backend_dev_t lfg_model::dev_output() const {
    return pimpl->dev_output.dev;
}

template<typename F>
static bool buft_supported(ggml_backend_buffer_type_t buft, ggml_backend_dev_t dev, F & fn) {
    ggml_init_params init_params = {
        /*.mem_size   =*/ ggml_tensor_overhead()*8,
        /*.mem_buffer =*/ NULL,
        /*.no_alloc   =*/ true,
    };

    ggml_context_ptr ctx { ggml_init(init_params) };
    if (!ctx) {
        throw std::runtime_error(lfg_format("failed to create ggml context"));
    }

    ggml_backend_buffer_ptr buf { ggml_backend_buft_alloc_buffer(buft, 0) };
    ggml_tensor * op_tensor = fn(ctx.get());
    for (int i = 0; i < GGML_MAX_SRC; i++) {
        if (op_tensor->src[i] != nullptr) {
            assert(op_tensor->src[i]->buffer == nullptr);
            op_tensor->src[i]->buffer = buf.get();
        }
    }

    bool op_supported = ggml_backend_dev_supports_op(dev, op_tensor);

    return op_supported;
}

template<typename F>
static ggml_backend_buffer_type_t select_buft(const buft_list_t & buft_list, const F & fn) {
    for (const auto & cur : buft_list) {
        ggml_backend_dev_t cur_dev = cur.first;
        ggml_backend_buffer_type_t cur_buft = cur.second;
        if (buft_supported(cur_buft, cur_dev, fn)) {
            return cur_buft;
        }
    }

    throw std::runtime_error(lfg_format("no suitable buffer type found"));
}

ggml_backend_buffer_type_t lfg_model::select_buft(uint32_t il) const {
    if (pimpl->has_tensor_overrides) {
        // TODO: implement overrides logic
    }

    if (pimpl->dev_layer.empty()) {
        return select_weight_buft(hparams, nullptr, GGML_OP_NONE, *pimpl->dev_input.buft_list);
    }
    return select_weight_buft(hparams, nullptr, GGML_OP_NONE,
            *pimpl->dev_layer.at(il).buft_list);
}

bool lfg_model::has_tensor_overrides() const {
    return pimpl->has_tensor_overrides;
}

const ggml_tensor * lfg_model::get_tensor(const char * name) const {
    auto it = std::find_if(tensors_by_name.begin(), tensors_by_name.end(),
            [name](const std::pair<std::string, ggml_tensor *> & it) {
                return it.first == name;
            });
    if (it == tensors_by_name.end()) {
        return nullptr;
    }

    return it->second;
}

float lfg_model::get_rope_freq_base (const lfg_cparams & cparams, uint32_t il) const {
    return hparams.is_swa(il) ? hparams.rope_freq_base_train_swa : cparams.rope_freq_base;
}

float lfg_model::get_rope_freq_scale(const lfg_cparams & cparams, uint32_t il) const {
    return hparams.is_swa(il) ? hparams.rope_freq_scale_train_swa : cparams.rope_freq_scale;
}

ggml_tensor * lfg_model::get_rope_factors(const lfg_cparams & cparams, uint32_t il) const {
    const uint32_t n_ctx_seq = cparams.n_ctx_seq;

    // choose long/short freq factors based on the context size
    if (layers[il].rope_freqs != nullptr) {
        return layers[il].rope_freqs;
    }

    if (n_ctx_seq > hparams.n_ctx_orig_yarn) {
        return layers[il].rope_long;
    }

    return layers[il].rope_short;
}

lfg_memory_i * lfg_model::create_memory(const lfg_memory_params & mem_params, const lfg_cparams & cparams) const {
    lfg_memory_i * res = nullptr;

    if (!mem_params.kv_cache_path.empty()) {
        // check if file exists
        std::ifstream f(mem_params.kv_cache_path);
        if (f.good()) {
            LFG_LOG_INFO("%s: loading KV cache from '%s'...\n", __func__, mem_params.kv_cache_path.c_str());
        } else {
            LFG_LOG_INFO("%s: saving KV cache to '%s'...\n", __func__, mem_params.kv_cache_path.c_str());
        }
    }

    switch (arch) {
        case LFG_ARCH_BERT:
        case LFG_ARCH_NOMIC_BERT:
        case LFG_ARCH_NOMIC_BERT_MOE:
        case LFG_ARCH_JINA_BERT_V2:
        case LFG_ARCH_JINA_BERT_V3:
        case LFG_ARCH_NEO_BERT:
        case LFG_ARCH_MODERN_BERT:
            { res = nullptr; } break;
        // Models that need specific instantiation should be handled in the
        // switch statement
        default:
            {
                if (lfg_arch_is_recurrent(arch)) {
                    res = new lfg_memory_recurrent(
                            *this,
                            GGML_TYPE_F32,
                            GGML_TYPE_F32,
                            cparams.offload_kqv,
                            std::max((uint32_t) 1, cparams.n_seq_max),
                            cparams.n_seq_max,
                            nullptr);
                } else if (lfg_arch_is_hybrid(arch)) {

                    // The main difference between hybrid architectures is the
                    // layer filters, so pick the LIQUID one here
                    lfg_memory_hybrid::layer_filter_cb filter_attn = nullptr;
                    lfg_memory_hybrid::layer_filter_cb filter_recr = nullptr;
                    if (false) {
                        filter_attn = [&](int32_t) { return true; };
                        filter_recr = [&](int32_t) { return true; };
                    } else if (false) {
                        filter_attn = [&](int32_t il) {
                            return !hparams.is_recurrent(il) && hparams.n_ff(il) == 0;
                        };
                        filter_recr = [&](int32_t il) {
                            return hparams.is_recurrent(il) && hparams.n_ff(il) == 0;
                        };
                    }

                    if (hparams.swa_type != LFG_SWA_TYPE_NONE) {
                        // Use hybrid-iswa for hybrid models with SWA
                        res = new lfg_memory_hybrid_iswa(
                            /* model             */ *this,
                            /* attn_type_k       */ params.type_k,
                            /* attn_type_v       */ params.type_v,
                            /* attn_v_trans      */ !cparams.flash_attn,
                            /* attn_swa_full     */ params.swa_full,
                            /* attn_kv_size      */ cparams.n_ctx,
                            /* attn_n_ubatch     */ cparams.n_ubatch,
                            /* attn_n_pad        */ 1,
                            /* recurrent_type_r  */ GGML_TYPE_F32,
                            /* recurrent_type_s  */ GGML_TYPE_F32,
                            /* recurrent_rs_size */ std::max((uint32_t) 1, cparams.n_seq_max),
                            /* n_seq_max         */ cparams.n_seq_max,
                            /* offload           */ cparams.offload_kqv,
                            /* unified           */ cparams.kv_unified,
                            /* filter_attn       */ std::move(filter_attn),
                            /* filter_recr       */ std::move(filter_recr));
                    } else {
                        res = new lfg_memory_hybrid(
                            /* model             */ *this,
                            /* attn_type_k       */ params.type_k,
                            /* attn_type_v       */ params.type_v,
                            /* attn_v_trans      */ !cparams.flash_attn,
                            /* attn_kv_size      */ cparams.n_ctx,
                            /* attn_n_pad        */ 1,
                            /* attn_n_swa        */ hparams.n_swa,
                            /* attn_swa_type     */ hparams.swa_type,
                            /* recurrent_type_k  */ GGML_TYPE_F32,
                            /* recurrent_type_v  */ GGML_TYPE_F32,
                            /* recurrent_kv_size */ std::max((uint32_t) 1, cparams.n_seq_max),
                            /* n_seq_max         */ cparams.n_seq_max,
                            /* offload           */ cparams.offload_kqv,
                            /* unified           */ cparams.kv_unified,
                            /* filter_attn       */ std::move(filter_attn),
                            /* filter_recr       */ std::move(filter_recr));
                    }
                } else {
                    lfg_memory_i::layer_reuse_cb reuse = nullptr;

                    if (false) {
                        reuse = [&](int32_t il) {
                            if (il >= (int32_t) hparams.n_layer_kv_from_start) {
                                return (int32_t) hparams.n_layer_kv_from_start - (hparams.is_swa(il) ? 2 : 1);
                            }

                            return -1;
                        };
                    }

                    if (hparams.swa_type != LFG_SWA_TYPE_NONE) {
                        GGML_ASSERT(hparams.is_swa_any());

                        res = new lfg_kv_cache_iswa(
                                *this,
                                params.type_k,
                                params.type_v,
                                !cparams.flash_attn,
                                cparams.offload_kqv,
                                params.swa_full,
                                cparams.kv_unified,
                                cparams.n_ctx_seq,
                                cparams.n_seq_max,
                                cparams.n_ubatch,
                                1,
                                nullptr,
                                reuse);
                    } else {
                        GGML_ASSERT(!hparams.is_swa_any());

                        res = new lfg_kv_cache(
                                *this,
                                params.type_k,
                                params.type_v,
                                !cparams.flash_attn,
                                cparams.offload_kqv,
                                cparams.kv_unified,
                                cparams.n_ctx_seq,
                                cparams.n_seq_max,
                                1,
                                hparams.n_swa,
                                hparams.swa_type,
                                nullptr,
                                nullptr);
                    }
                }
            }
    }

    return res;
}

ggml_cgraph * lfg_model::build_graph(const lfg_graph_params & graph_params) const {
    std::unique_ptr<lfg_graph_context> llm;

    switch (arch) {
        case LFG_ARCH_LFM2:
        case LFG_ARCH_LFM2MOE:
            {
                if (hparams.swa_type == LFG_SWA_TYPE_STANDARD) {
                    llm = std::make_unique<lfg_build_lfm2<true>>(*this, graph_params);
                } else {
                    llm = std::make_unique<lfg_build_lfm2<false>>(*this, graph_params);
                }
            } break;
        case LFG_ARCH_BERT:
        case LFG_ARCH_NOMIC_BERT:
        case LFG_ARCH_NOMIC_BERT_MOE:
        case LFG_ARCH_JINA_BERT_V2:
        case LFG_ARCH_JINA_BERT_V3:
            {
                llm = std::make_unique<lfg_build_bert>(*this, graph_params);
            } break;
        case LFG_ARCH_NEO_BERT:
            {
                llm = std::make_unique<lfg_build_neo_bert>(*this, graph_params);
            } break;
        case LFG_ARCH_MODERN_BERT:
            {
                llm = std::make_unique<lfg_build_modern_bert>(*this, graph_params);
            } break;
        default:
            lfg_set_last_error(LFG_ERROR_UNSUPPORTED, "%s: unsupported architecture", __func__);
            throw std::runtime_error("unsupported architecture");
    }

    // add on pooling layer
    llm->build_pooling(cls, cls_b, cls_out, cls_out_b);

    // add backend sampling layers (if any)
    llm->build_sampling();

    // if the gguf model was converted with --sentence-transformers-dense-modules
    // there will be two additional dense projection layers
    // dense linear projections are applied after pooling
    // TODO: move reranking logic here and generalize
    llm->build_dense_out(dense_2_out_layers, dense_2_out_layers_b, dense_3_out_layers);

    llm->res->set_outputs();

    return llm->res->get_gf();
}


//
// interface implementation
//

lfg_model_params lfg_model_default_params() {
    lfg_model_params result = {
        /*.devices                     =*/ nullptr,
        /*.tensor_buft_overrides       =*/ nullptr,
        /*.n_gpu_layers                =*/ -1,
        /*.split_mode                  =*/ LFG_SPLIT_MODE_LAYER,
        /*.main_gpu                    =*/ 0,
        /*.tensor_split                =*/ nullptr,
        /*.progress_callback           =*/ nullptr,
        /*.progress_callback_user_data =*/ nullptr,
        /*.kv_overrides                =*/ nullptr,
        /*.vocab_only                  =*/ false,
        /*.use_mmap                    =*/ true,
        /*.use_direct_io               =*/ true,
        /*.use_mlock                   =*/ false,
        /*.check_tensors               =*/ false,
        /*.use_extra_bufts             =*/ true,
        /*.no_host                     =*/ false,
        /*.no_alloc                    =*/ false,
        /*.type_k                      =*/ GGML_TYPE_F16,
        /*.type_v                      =*/ GGML_TYPE_F16,
        /*.swa_full                    =*/ false,
    };

    return result;
}

const lfg_vocab * lfg_model_get_vocab(const lfg_model * model) {
    if (!model) {
        lfg_set_last_error(LFG_ERROR_INVALID_ARGUMENT, "%s: model is NULL", __func__);
        return nullptr;
    }
    return &model->vocab;
}

void lfg_free_model(lfg_model * model) {
    lfg_model_free(model);
}

void lfg_model_free(lfg_model * model) {
    if (!model) {
        return;
    }
    delete model;
}

int32_t lfg_model_n_ctx_train(const lfg_model * model) {
    if (!model) {
        lfg_set_last_error(LFG_ERROR_INVALID_ARGUMENT, "%s: model is NULL", __func__);
        return 0;
    }
    return static_cast<int32_t>(model->hparams.n_ctx_train);
}

int32_t lfg_model_n_embd(const lfg_model * model) {
    if (!model) {
        lfg_set_last_error(LFG_ERROR_INVALID_ARGUMENT, "%s: model is NULL", __func__);
        return 0;
    }
    return static_cast<int32_t>(model->hparams.n_embd);
}

int32_t lfg_model_n_embd_inp(const lfg_model * model) {
    if (!model) {
        lfg_set_last_error(LFG_ERROR_INVALID_ARGUMENT, "%s: model is NULL", __func__);
        return 0;
    }
    return static_cast<int32_t>(model->hparams.n_embd_inp());
}

int32_t lfg_model_n_embd_out(const lfg_model * model) {
    if (!model) {
        lfg_set_last_error(LFG_ERROR_INVALID_ARGUMENT, "%s: model is NULL", __func__);
        return 0;
    }
    return static_cast<int32_t>(model->hparams.get_n_embd_out());
}

int32_t lfg_model_n_layer(const lfg_model * model) {
    if (!model) {
        lfg_set_last_error(LFG_ERROR_INVALID_ARGUMENT, "%s: model is NULL", __func__);
        return 0;
    }
    return static_cast<int32_t>(model->hparams.n_layer);
}

int32_t lfg_model_n_head(const lfg_model * model) {
    if (!model) {
        lfg_set_last_error(LFG_ERROR_INVALID_ARGUMENT, "%s: model is NULL", __func__);
        return 0;
    }
    return static_cast<int32_t>(model->hparams.n_head());
}

int32_t lfg_model_n_head_kv(const lfg_model * model) {
    if (!model) {
        lfg_set_last_error(LFG_ERROR_INVALID_ARGUMENT, "%s: model is NULL", __func__);
        return 0;
    }
    return static_cast<int32_t>(model->hparams.n_head_kv());
}

int32_t lfg_model_n_swa(const lfg_model * model) {
    if (!model) {
        lfg_set_last_error(LFG_ERROR_INVALID_ARGUMENT, "%s: model is NULL", __func__);
        return 0;
    }
    return static_cast<int32_t>(model->hparams.n_swa);
}

uint32_t lfg_model_n_cls_out(const struct lfg_model * model) {
    if (!model) {
        lfg_set_last_error(LFG_ERROR_INVALID_ARGUMENT, "%s: model is NULL", __func__);
        return 0;
    }
    return model->hparams.n_cls_out;
}

const char * lfg_model_cls_label(const struct lfg_model * model, uint32_t i) {
    if (!model) {
        lfg_set_last_error(LFG_ERROR_INVALID_ARGUMENT, "%s: model is NULL", __func__);
        return nullptr;
    }
    if (i < model->classifier_labels.size()) {
        return model->classifier_labels[i].c_str();
    }

    return nullptr;
}

// deprecated
int32_t lfg_n_ctx_train(const lfg_model * model) {
    return lfg_model_n_ctx_train(model);
}

// deprecated
int32_t lfg_n_embd(const lfg_model * model) {
    return lfg_model_n_embd(model);
}

// deprecated
int32_t lfg_n_layer(const lfg_model * model) {
    return lfg_model_n_layer(model);
}

// deprecated
int32_t lfg_n_head(const lfg_model * model) {
    return lfg_model_n_head(model);
}

lfg_rope_type lfg_model_rope_type(const lfg_model * model) {
    if (!model) {
        lfg_set_last_error(LFG_ERROR_INVALID_ARGUMENT, "%s: model is NULL", __func__);
        return LFG_ROPE_TYPE_NONE;
    }
    switch (model->arch) {
        // LFM2 uses NEOX-style RoPE
        case LFG_ARCH_LFM2:
        case LFG_ARCH_LFM2MOE:
             return LFG_ROPE_TYPE_NEOX;

        // BERT: absolute position embeddings (no RoPE)
        case LFG_ARCH_BERT:
        case LFG_ARCH_JINA_BERT_V2:
            return LFG_ROPE_TYPE_NONE;

        // BERT variants with RoPE (NeoX-style)
        case LFG_ARCH_NOMIC_BERT:
        case LFG_ARCH_NOMIC_BERT_MOE:
        case LFG_ARCH_JINA_BERT_V3:
        case LFG_ARCH_NEO_BERT:
        case LFG_ARCH_MODERN_BERT:
            return LFG_ROPE_TYPE_NEOX;

        case LFG_ARCH_UNKNOWN:
            lfg_set_last_error(LFG_ERROR_UNSUPPORTED, "%s: unknown architecture", __func__);
            return LFG_ROPE_TYPE_NONE;
    }

    return LFG_ROPE_TYPE_NONE;
}

float lfg_model_rope_freq_scale_train(const lfg_model * model) {
    if (!model) {
        lfg_set_last_error(LFG_ERROR_INVALID_ARGUMENT, "%s: model is NULL", __func__);
        return 0.0f;
    }
    return model->hparams.rope_freq_scale_train;
}

int32_t lfg_model_meta_val_str(const lfg_model * model, const char * key, char * buf, size_t buf_size) {
    if (!model) {
        lfg_set_last_error(LFG_ERROR_INVALID_ARGUMENT, "%s: model is NULL", __func__);
        return -1;
    }
    if (!key) {
        lfg_set_last_error(LFG_ERROR_INVALID_ARGUMENT, "%s: key is NULL", __func__);
        return -1;
    }
    if (!buf && buf_size > 0) {
        lfg_set_last_error(LFG_ERROR_INVALID_ARGUMENT, "%s: buf is NULL", __func__);
        return -1;
    }
    const auto & it = model->gguf_kv.find(key);
    if (it == model->gguf_kv.end()) {
        if (buf_size > 0) {
            buf[0] = '\0';
        }
        return -1;
    }
    return snprintf(buf, buf_size, "%s", it->second.c_str());
}

int32_t lfg_model_meta_count(const lfg_model * model) {
    if (!model) {
        lfg_set_last_error(LFG_ERROR_INVALID_ARGUMENT, "%s: model is NULL", __func__);
        return 0;
    }
    return (int)model->gguf_kv.size();
}

const char * lfg_model_meta_key_str(lfg_model_meta_key key) {
    switch (key) {
        case LFG_MODEL_META_KEY_SAMPLING_SEQUENCE:        return "general.sampling.sequence";
        case LFG_MODEL_META_KEY_SAMPLING_TOP_K:           return "general.sampling.top_k";
        case LFG_MODEL_META_KEY_SAMPLING_TOP_P:           return "general.sampling.top_p";
        case LFG_MODEL_META_KEY_SAMPLING_MIN_P:           return "general.sampling.min_p";
        case LFG_MODEL_META_KEY_SAMPLING_XTC_PROBABILITY: return "general.sampling.xtc_probability";
        case LFG_MODEL_META_KEY_SAMPLING_XTC_THRESHOLD:   return "general.sampling.xtc_threshold";
        case LFG_MODEL_META_KEY_SAMPLING_TEMP:            return "general.sampling.temp";
        case LFG_MODEL_META_KEY_SAMPLING_PENALTY_LAST_N:  return "general.sampling.penalty_last_n";
        case LFG_MODEL_META_KEY_SAMPLING_PENALTY_REPEAT:  return "general.sampling.penalty_repeat";
        case LFG_MODEL_META_KEY_SAMPLING_MIROSTAT:        return "general.sampling.mirostat";
        case LFG_MODEL_META_KEY_SAMPLING_MIROSTAT_TAU:    return "general.sampling.mirostat_tau";
        case LFG_MODEL_META_KEY_SAMPLING_MIROSTAT_ETA:    return "general.sampling.mirostat_eta";
        default:                                            return nullptr;
    }
}

int32_t lfg_model_meta_key_by_index(const lfg_model * model, int i, char * buf, size_t buf_size) {
    if (!model) {
        lfg_set_last_error(LFG_ERROR_INVALID_ARGUMENT, "%s: model is NULL", __func__);
        return -1;
    }
    if (!buf && buf_size > 0) {
        lfg_set_last_error(LFG_ERROR_INVALID_ARGUMENT, "%s: buf is NULL", __func__);
        return -1;
    }
    if (i < 0 || i >= (int)model->gguf_kv.size()) {
        lfg_set_last_error(LFG_ERROR_INVALID_ARGUMENT, "%s: index out of range", __func__);
        if (buf_size > 0) {
            buf[0] = '\0';
        }
        return -1;
    }
    auto it = model->gguf_kv.begin();
    std::advance(it, i);
    return snprintf(buf, buf_size, "%s", it->first.c_str());
}

int32_t lfg_model_meta_val_str_by_index(const lfg_model * model, int32_t i, char * buf, size_t buf_size) {
    if (!model) {
        lfg_set_last_error(LFG_ERROR_INVALID_ARGUMENT, "%s: model is NULL", __func__);
        return -1;
    }
    if (!buf && buf_size > 0) {
        lfg_set_last_error(LFG_ERROR_INVALID_ARGUMENT, "%s: buf is NULL", __func__);
        return -1;
    }
    if (i < 0 || i >= (int)model->gguf_kv.size()) {
        lfg_set_last_error(LFG_ERROR_INVALID_ARGUMENT, "%s: index out of range", __func__);
        if (buf_size > 0) {
            buf[0] = '\0';
        }
        return -1;
    }
    auto it = model->gguf_kv.begin();
    std::advance(it, i);
    return snprintf(buf, buf_size, "%s", it->second.c_str());
}

int32_t lfg_model_desc(const lfg_model * model, char * buf, size_t buf_size) {
    if (!model) {
        lfg_set_last_error(LFG_ERROR_INVALID_ARGUMENT, "%s: model is NULL", __func__);
        return -1;
    }
    if (!buf && buf_size > 0) {
        lfg_set_last_error(LFG_ERROR_INVALID_ARGUMENT, "%s: buf is NULL", __func__);
        return -1;
    }
    return snprintf(buf, buf_size, "%s", model->desc().c_str());
}

uint64_t lfg_model_size(const lfg_model * model) {
    if (!model) {
        lfg_set_last_error(LFG_ERROR_INVALID_ARGUMENT, "%s: model is NULL", __func__);
        return 0;
    }
    return model->size();
}

const char * lfg_model_chat_template(const lfg_model * model, const char * name) {
    if (!model) {
        lfg_set_last_error(LFG_ERROR_INVALID_ARGUMENT, "%s: model is NULL", __func__);
        return nullptr;
    }
    const auto key = name ? LFG_KV(model->arch, name)(LFG_KV_TOKENIZER_CHAT_TEMPLATE)
        : LFG_KV(model->arch)(LFG_KV_TOKENIZER_CHAT_TEMPLATE);
    const auto & it = model->gguf_kv.find(key);
    if (it == model->gguf_kv.end()) {
        // one-off fix for very popular models (so we are not flooded with issues)
        // do not extend this list unless absolutely necessary
        // Mistral-Small-2503 does not have built-in chat template
        lfg_vocab_pre_type pre_type = model->vocab.get_pre_type();
        if (!name && pre_type == LFG_VOCAB_PRE_TYPE_TEKKEN && model->layers.size() == 40) {
            return "mistral-v7-tekken";
        }

        return nullptr;
    }

    return it->second.c_str();
}

uint64_t lfg_model_n_params(const lfg_model * model) {
    if (!model) {
        lfg_set_last_error(LFG_ERROR_INVALID_ARGUMENT, "%s: model is NULL", __func__);
        return 0;
    }
    return model->n_elements();
}

bool lfg_model_has_encoder(const lfg_model * model) {
    if (!model) {
        lfg_set_last_error(LFG_ERROR_INVALID_ARGUMENT, "%s: model is NULL", __func__);
        return false;
    }
    switch (model->arch) {
        default:                 return false;
    }
}

bool lfg_model_has_decoder(const lfg_model * model) {
    if (!model) {
        lfg_set_last_error(LFG_ERROR_INVALID_ARGUMENT, "%s: model is NULL", __func__);
        return false;
    }
    switch (model->arch) {
        default:                 return true;
    }
}

lfg_token lfg_model_decoder_start_token(const lfg_model * model) {
    if (!model) {
        lfg_set_last_error(LFG_ERROR_INVALID_ARGUMENT, "%s: model is NULL", __func__);
        return LFG_TOKEN_NULL;
    }
    return model->hparams.dec_start_token_id;
}

bool lfg_model_is_recurrent(const lfg_model * model) {
    if (!model) {
        lfg_set_last_error(LFG_ERROR_INVALID_ARGUMENT, "%s: model is NULL", __func__);
        return false;
    }
    return lfg_arch_is_recurrent(model->arch);
}

bool lfg_model_is_hybrid(const lfg_model * model) {
    if (!model) {
        lfg_set_last_error(LFG_ERROR_INVALID_ARGUMENT, "%s: model is NULL", __func__);
        return false;
    }
    return lfg_arch_is_hybrid(model->arch);
}

bool lfg_model_is_diffusion(const lfg_model * model) {
    if (!model) {
        lfg_set_last_error(LFG_ERROR_INVALID_ARGUMENT, "%s: model is NULL", __func__);
        return false;
    }
    return lfg_arch_is_diffusion(model->arch);
}

const std::vector<std::pair<std::string, ggml_tensor *>> & lfg_internal_get_tensor_map(const lfg_model * model) {
    return model->tensors_by_name;
}
