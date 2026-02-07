#pragma once

#ifdef __cplusplus
#include "lfg_arch.h"
#include "lfg_batch.h"
#include "lfg_hparams.h"
#include "lfg_adapter.h"

#include <cstdint>
#include <vector>
#include <memory>
#include <set>
#include <functional>
#include <map>

struct ggml_cgraph;
struct ggml_context;
struct ggml_tensor;

struct lfg_cparams;

struct lfg_memory_context_i;

class lfg_kv_cache_context;
class lfg_kv_cache_iswa_context;
class lfg_memory_recurrent_context;
class lfg_memory_hybrid_context;
class lfg_memory_hybrid_iswa_context;


// certain models (typically multi-modal) can produce different types of graphs
enum lfg_graph_type {
    LFG_GRAPH_TYPE_DEFAULT,
    LFG_GRAPH_TYPE_ENCODER,
    LFG_GRAPH_TYPE_DECODER,
};

enum lfg_ffn_op_type {
    LFG_FFN_SILU,
    LFG_FFN_GELU,
    LFG_FFN_RELU,
    LFG_FFN_RELU_SQR,
    LFG_FFN_SWIGLU,
    LFG_FFN_GEGLU,
    LFG_FFN_REGLU,
    LFG_FFN_SWIGLU_OAI_MOE,
};

enum lfg_ffn_gate_type {
    LFG_FFN_SEQ,
    LFG_FFN_PAR, // ffn_gate is parallel to ffn_up
};

enum lfg_norm_type {
    LFG_NORM,
    LFG_NORM_RMS,
    LFG_NORM_GROUP,
};

// TODO: tmp - need something better to pass the data from the encoder to the decoder
struct lfg_cross {
    // the output embeddings from the encoder as a ggml tensor
    // TODO: this needs more work to be correct, for now copy the embeddings data to host memory
    //       ref: https://github.com/ggml-org/liquid.cpp/pull/11213#discussion_r1969892524
    //ggml_tensor * t_embd = nullptr;

    int64_t n_embd = 0;
    int64_t n_enc  = 0;

    // embeddings data copied to host memory (tmp)
    std::vector<float> v_embd;

    // needed to construct the cross-attention mask in the decoder
    std::vector<std::set<lfg_seq_id>> seq_ids_enc;
};

struct lfg_graph_params;

//
// lfg_graph_input
//

class lfg_graph_input_i {
public:
    lfg_graph_input_i() {
        const char * LFG_GRAPH_INPUT_DEBUG = getenv("LFG_GRAPH_INPUT_DEBUG");
        debug = LFG_GRAPH_INPUT_DEBUG ? atoi(LFG_GRAPH_INPUT_DEBUG) : 0;
    }

    virtual ~lfg_graph_input_i() = default;

    virtual void set_input(const lfg_ubatch * ubatch) = 0;

    // return true if the resulting input tensors using the provided graph parameters would be
    //   the same as the previous input tensors that we have currently stored in the object
    virtual bool can_reuse(const lfg_graph_params & params) {
        // returning false here by default will prevent from reusing the graph if the check
        //   for the input type has not been implemented yet
        GGML_UNUSED(params);
        return false;
    }
protected:
    // env: LFG_GRAPH_INPUT_DEBUG
    int debug = 0;
};

using lfg_graph_input_ptr = std::unique_ptr<lfg_graph_input_i>;

class lfg_graph_input_embd : public lfg_graph_input_i {
public:
    lfg_graph_input_embd()          = default;
    virtual ~lfg_graph_input_embd() = default;

    void set_input(const lfg_ubatch * ubatch) override;

    bool can_reuse(const lfg_graph_params & params) override;

    ggml_tensor * tokens = nullptr; // I32 [n_batch]
    ggml_tensor * embd   = nullptr; // F32 [n_embd, n_batch]
};

class lfg_graph_input_pos : public lfg_graph_input_i {
public:
    lfg_graph_input_pos(uint32_t n_pos_per_embd) : n_pos_per_embd(n_pos_per_embd) {}
    virtual ~lfg_graph_input_pos() = default;

    void set_input(const lfg_ubatch * ubatch) override;

    bool can_reuse(const lfg_graph_params & params) override;

    ggml_tensor * pos = nullptr; // I32 [n_batch]

    const uint32_t n_pos_per_embd = 1;
};

// temperature tuning, used by liquid4
class lfg_graph_input_attn_temp : public lfg_graph_input_i {
public:
    lfg_graph_input_attn_temp(uint32_t n_attn_temp_floor_scale, float f_attn_temp_scale, float f_attn_temp_offset)
        : n_attn_temp_floor_scale(n_attn_temp_floor_scale), f_attn_temp_scale(f_attn_temp_scale), f_attn_temp_offset(f_attn_temp_offset) {}
    virtual ~lfg_graph_input_attn_temp() = default;

    void set_input(const lfg_ubatch * ubatch) override;

    ggml_tensor * attn_scale = nullptr; // F32 [n_batch]

    const uint32_t n_attn_temp_floor_scale;
    const float    f_attn_temp_scale;
    const float    f_attn_temp_offset;
};

class lfg_graph_input_pos_bucket : public lfg_graph_input_i {
public:
    lfg_graph_input_pos_bucket(const lfg_hparams & hparams) : hparams(hparams) {}
    virtual ~lfg_graph_input_pos_bucket() = default;

    void set_input(const lfg_ubatch * ubatch) override;

    ggml_tensor * pos_bucket = nullptr; // I32 [n_batch, n_batch]

    const lfg_hparams hparams;
};

class lfg_graph_input_pos_bucket_kv : public lfg_graph_input_i {
public:
    lfg_graph_input_pos_bucket_kv(
            const lfg_hparams & hparams,
            const lfg_kv_cache_context * mctx) : hparams(hparams), mctx(mctx) {}
    virtual ~lfg_graph_input_pos_bucket_kv() = default;

    void set_input(const lfg_ubatch * ubatch) override;

    ggml_tensor * pos_bucket = nullptr; // I32 [n_kv, n_batch]

    const lfg_hparams hparams;

    const lfg_kv_cache_context * mctx;
};

class lfg_graph_input_out_ids : public lfg_graph_input_i {
public:
    lfg_graph_input_out_ids(
            const lfg_hparams & hparams,
            const lfg_cparams & cparams,
            uint32_t n_outputs) : hparams(hparams), cparams(cparams), n_outputs(n_outputs) {}
    virtual ~lfg_graph_input_out_ids() = default;

    void set_input(const lfg_ubatch * ubatch) override;

    bool can_reuse(const lfg_graph_params & params) override;

    ggml_tensor * out_ids; // I32 [n_outputs]

    const lfg_hparams hparams;
    const lfg_cparams cparams;

    const uint32_t n_outputs;
};

class lfg_graph_input_mean : public lfg_graph_input_i {
public:
    lfg_graph_input_mean(const lfg_cparams & cparams) : cparams(cparams) {}
    virtual ~lfg_graph_input_mean() = default;

    void set_input(const lfg_ubatch * ubatch) override;

    ggml_tensor * mean; // F32 [n_batch, n_batch]

    const lfg_cparams cparams;
};

class lfg_graph_input_cls : public lfg_graph_input_i {
public:
    lfg_graph_input_cls(const lfg_cparams & cparams, const lfg_arch_enum arch) : cparams(cparams), arch(arch) {}
    virtual ~lfg_graph_input_cls() = default;

    void set_input(const lfg_ubatch * ubatch) override;

    ggml_tensor * cls; // I32 [n_batch]

    const lfg_cparams cparams;
    const lfg_arch_enum arch;
};

class lfg_graph_input_rs : public lfg_graph_input_i {
public:
    lfg_graph_input_rs(const lfg_memory_recurrent_context * mctx) : mctx(mctx) {}
    virtual ~lfg_graph_input_rs() = default;

    void set_input(const lfg_ubatch * ubatch) override;

    bool can_reuse(const lfg_graph_params & params) override;

    ggml_tensor * s_copy;  // I32 [n_rs]

    // views of s_copy, computed once per graph
    // and shared across layers which use build_rs
    ggml_tensor * s_copy_main;   // I32 [n_seqs]
    ggml_tensor * s_copy_extra;  // I32 [n_rs - n_seqs]

    const lfg_memory_recurrent_context * mctx;

    // used in view offsets, need to match for valid graph reuse
    uint32_t head;
    int32_t rs_z;
};

class lfg_graph_input_cross_embd : public lfg_graph_input_i {
public:
    lfg_graph_input_cross_embd(
            const lfg_cross * cross) : cross(cross) {}
    virtual ~lfg_graph_input_cross_embd() = default;

    void set_input(const lfg_ubatch * ubatch) override;

    ggml_tensor * cross_embd; // F32 [n_embd, n_outputs_enc]

    const lfg_cross * cross;
};

class lfg_graph_input_attn_no_cache : public lfg_graph_input_i {
public:
    lfg_graph_input_attn_no_cache(const lfg_hparams & hparams, const lfg_cparams & cparams) :
        hparams(hparams),
        cparams(cparams) {
    }
    ~lfg_graph_input_attn_no_cache() = default;

    void set_input(const lfg_ubatch * ubatch) override;

    ggml_tensor * get_kq_mask()     const { return self_kq_mask_cnv; }
    ggml_tensor * get_kq_mask_swa() const { return self_kq_mask_swa_cnv; }

    // n_tokens == n_batch
    ggml_tensor * self_kq_mask         = nullptr; // F32 [n_tokens, n_batch/n_stream, 1, n_stream]
    ggml_tensor * self_kq_mask_cnv     = nullptr; //     [n_tokens, n_batch/n_stream, 1, n_stream]
    ggml_tensor * self_kq_mask_swa     = nullptr; // F32 [n_tokens, n_batch/n_stream, 1, n_stream]
    ggml_tensor * self_kq_mask_swa_cnv = nullptr; //     [n_tokens, n_batch/n_stream, 1, n_stream]

    const lfg_hparams hparams;
    const lfg_cparams cparams;
};

class lfg_graph_input_attn_kv : public lfg_graph_input_i {
public:
    lfg_graph_input_attn_kv(
            const lfg_hparams & hparams,
            const lfg_cparams & cparams,
            const lfg_kv_cache_context * mctx) :
        hparams(hparams),
        cparams(cparams),
        mctx(mctx) {
    }
    ~lfg_graph_input_attn_kv() = default;

    void set_input(const lfg_ubatch * ubatch) override;

    bool can_reuse(const lfg_graph_params & params) override;

    ggml_tensor * get_k_idxs() const { return self_k_idxs; }
    ggml_tensor * get_v_idxs() const { return self_v_idxs; }

    ggml_tensor * get_kq_mask() const { return self_kq_mask_cnv; }

    ggml_tensor * self_k_idxs = nullptr; // I64 [n_batch]
    ggml_tensor * self_v_idxs = nullptr; // I64 [n_batch] or [n_batch*n_embd_v_gqa]

    ggml_tensor * self_kq_mask     = nullptr; // F32 [n_kv, n_batch/n_stream, 1, n_stream]
    ggml_tensor * self_kq_mask_cnv = nullptr; //     [n_kv, n_batch/n_stream, 1, n_stream]

    // note: these have to be copies because in order to be able to reuse a graph, its inputs
    //       need to carry these parameters with them. otherwise, they can point to freed
    //       lfg_graph_params from a previous batch, causing stack-use-after-return
    const lfg_hparams hparams;
    const lfg_cparams cparams;

    const lfg_kv_cache_context * mctx;
};

class lfg_graph_input_attn_kv_iswa : public lfg_graph_input_i {
public:
    lfg_graph_input_attn_kv_iswa(
            const lfg_hparams & hparams,
            const lfg_cparams & cparams,
            const lfg_kv_cache_iswa_context * mctx) :
        hparams(hparams),
        cparams(cparams),
        mctx(mctx) {
    }
    ~lfg_graph_input_attn_kv_iswa() = default;

    void set_input(const lfg_ubatch * ubatch) override;

    bool can_reuse(const lfg_graph_params & params) override;

    ggml_tensor * get_k_idxs()     const { return self_k_idxs; }
    ggml_tensor * get_v_idxs()     const { return self_v_idxs; }
    ggml_tensor * get_k_idxs_swa() const { return self_k_idxs_swa; }
    ggml_tensor * get_v_idxs_swa() const { return self_v_idxs_swa; }

    ggml_tensor * get_kq_mask()     const { return self_kq_mask_cnv; }
    ggml_tensor * get_kq_mask_swa() const { return self_kq_mask_swa_cnv; }

    ggml_tensor * self_k_idxs     = nullptr; // I64 [n_batch]
    ggml_tensor * self_v_idxs     = nullptr; // I64 [n_batch] or [n_batch*n_embd_v_gqa]
    ggml_tensor * self_k_idxs_swa = nullptr; // I64 [n_batch]
    ggml_tensor * self_v_idxs_swa = nullptr; // I64 [n_batch] or [n_batch*n_embd_v_gqa]

    ggml_tensor * self_kq_mask         = nullptr; // F32 [n_kv, n_batch/n_stream, 1, n_stream]
    ggml_tensor * self_kq_mask_cnv     = nullptr; //     [n_kv, n_batch/n_stream, 1, n_stream]
    ggml_tensor * self_kq_mask_swa     = nullptr; // F32 [n_kv, n_batch/n_stream, 1, n_stream]
    ggml_tensor * self_kq_mask_swa_cnv = nullptr; //     [n_kv, n_batch/n_stream, 1, n_stream]

    const lfg_hparams hparams;
    const lfg_cparams cparams;

    const lfg_kv_cache_iswa_context * mctx;
};

class lfg_graph_input_attn_cross : public lfg_graph_input_i {
public:
    lfg_graph_input_attn_cross(const lfg_cross * cross) : cross(cross) {}
    ~lfg_graph_input_attn_cross() = default;

    void set_input(const lfg_ubatch * ubatch) override;

    ggml_tensor * get_kq_mask_cross() const { return cross_kq_mask_cnv; }

    ggml_tensor * cross_kq_mask     = nullptr; // F32 [n_outputs_enc, n_batch, 1, 1]
    ggml_tensor * cross_kq_mask_cnv = nullptr; // F32 [n_outputs_enc, n_batch, 1, 1]

    const lfg_cross * cross = nullptr;
};

class lfg_graph_input_mem_hybrid : public lfg_graph_input_i {
public:
    lfg_graph_input_mem_hybrid(
            const lfg_cparams & cparams,
            std::unique_ptr<lfg_graph_input_attn_kv> inp_attn,
            std::unique_ptr<lfg_graph_input_rs>      inp_rs,
            const lfg_memory_hybrid_context *      mctx) :
        inp_attn(std::move(inp_attn)),
        inp_rs(std::move(inp_rs)),
        cparams(cparams),
        mctx(mctx) { }
    virtual ~lfg_graph_input_mem_hybrid() = default;

    void set_input(const lfg_ubatch * ubatch) override;

    bool can_reuse(const lfg_graph_params & params) override;

    std::unique_ptr<lfg_graph_input_attn_kv> inp_attn;
    std::unique_ptr<lfg_graph_input_rs>      inp_rs;

    lfg_graph_input_attn_kv * get_attn() const { return inp_attn.get(); }
    lfg_graph_input_rs      * get_recr() const { return inp_rs.get(); }

    const lfg_cparams cparams;

    const lfg_memory_hybrid_context * mctx;
};

class lfg_graph_input_mem_hybrid_iswa : public lfg_graph_input_i {
public:
    lfg_graph_input_mem_hybrid_iswa(
            const lfg_cparams & cparams,
            std::unique_ptr<lfg_graph_input_attn_kv_iswa> inp_attn,
            std::unique_ptr<lfg_graph_input_rs>          inp_rs,
            const lfg_memory_hybrid_iswa_context *     mctx) :
        inp_attn(std::move(inp_attn)),
        inp_rs(std::move(inp_rs)),
        cparams(cparams),
        mctx(mctx) { }
    virtual ~lfg_graph_input_mem_hybrid_iswa() = default;

    void set_input(const lfg_ubatch * ubatch) override;

    bool can_reuse(const lfg_graph_params & params) override;

    std::unique_ptr<lfg_graph_input_attn_kv_iswa> inp_attn;
    std::unique_ptr<lfg_graph_input_rs>          inp_rs;

    lfg_graph_input_attn_kv_iswa * get_attn() const { return inp_attn.get(); }
    lfg_graph_input_rs           * get_recr() const { return inp_rs.get(); }

    const lfg_cparams cparams;

    const lfg_memory_hybrid_iswa_context * mctx;
};

class lfg_graph_input_sampling : public lfg_graph_input_i {
public:
    lfg_graph_input_sampling(std::map<lfg_seq_id, lfg_sampler *> samplers) :
        samplers(std::move(samplers)) { }
    virtual ~lfg_graph_input_sampling() = default;

    void set_input(const lfg_ubatch * ubatch) override;
    bool can_reuse(const lfg_graph_params & params) override;

    std::map<lfg_seq_id, lfg_sampler *> samplers;
};

//
// lfg_graph_result
//

// these objects deliver the result from the graph build process back to the lfg_context
// note that the input tensors created for the graph are referenced here - the goal is to be able to populate their
//   specific data, by calling the set_inputs() method
// along with the input tensors, the object also provides commonly used outputs tensors, such as logits, embeddings, etc.
//   these are used by the lfg_context to extact the relevant data, based on the compute parameters

// callback that allows us to apply custom logic to each tensor (e.g. ggml-alloc, offloading, etc.)
using lfg_graph_cb = std::function<void(const lfg_ubatch & ubatch, ggml_tensor * cur, const char * name, int il)>;

class lfg_graph_result;

struct lfg_graph_params {
    lfg_arch_enum arch = LFG_ARCH_UNKNOWN;

    lfg_hparams hparams;
    lfg_cparams cparams;

    lfg_ubatch ubatch; // note: intentionally make a copy

    lfg_graph_type gtype;

    ggml_backend_sched_t sched;
    ggml_backend_t backend_cpu;

    const lfg_adapter_cvec     * cvec;
    const lfg_adapter_loras    * loras;
    const lfg_memory_context_i * mctx;
    const lfg_cross            * cross;

    std::map<lfg_seq_id, lfg_sampler *> samplers;

    static bool samplers_equal(
          const std::map<lfg_seq_id, lfg_sampler *> & lhs,
          const std::map<lfg_seq_id, lfg_sampler *> & rhs) {
        if (lhs.size() != rhs.size()) {
            return false;
        }
        for (const auto & [seq_id, sampler] : lhs) {
            auto it = rhs.find(seq_id);
            if (it == rhs.end() || it->second != sampler) {
                return false;
            }
        }
        return true;
    }

    uint32_t n_outputs;

    lfg_graph_cb cb;

    lfg_graph_result * res;

    // return true if the "other" params would result in a graph with the same topology as with the current params
    //   having the same topology allows us to reuse the graph in some cases
    bool allow_reuse(const lfg_graph_params & other) const {
        // first check the ubatch
        bool can_reuse_ubatch =
            ubatch.equal_seqs() == other.ubatch.equal_seqs() &&
            ubatch.n_tokens     == other.ubatch.n_tokens &&
            ubatch.n_seq_tokens == other.ubatch.n_seq_tokens &&
            ubatch.n_seqs       == other.ubatch.n_seqs &&
            ubatch.n_seqs_unq   == other.ubatch.n_seqs_unq &&
            (
                (!ubatch.token && !other.ubatch.token) ||
                (!ubatch.embd  && !other.ubatch.embd)
            );

        // when we split the batch using "equal_seqs" we have to verify that the participating sequences are the same
        //   the reason is because the set of attention streams would be different for different sequences
        if (can_reuse_ubatch && ubatch.equal_seqs()) {
            if (!ubatch.data) {
                // if the old ubatch does not own it's data, then we cannot guarantee that it is still alive, and
                //   therefore we cannot perform the sequence id check. normally should never happen
                can_reuse_ubatch = false;
            } else {
                for (uint32_t s = 0; s < ubatch.n_seqs_unq; ++s) {
                    can_reuse_ubatch &= ubatch.seq_id_unq[s] == other.ubatch.seq_id_unq[s];
                }
            }
        }

        if (!can_reuse_ubatch) {
            return false;
        }

        if (n_outputs != other.n_outputs) {
            return false;
        }

        if (!samplers_equal(samplers, other.samplers)) {
            return false;
        }

        if (samplers.size() > 0) {
            if (!ubatch.data || !other.ubatch.data) {
                return false;
            }

            // check that the outputs are the same for all samplers
            for (uint32_t i = 0; i < ubatch.n_tokens; ++i) {
                if (ubatch.output[i]    != other.ubatch.output[i] ||
                    ubatch.seq_id[i][0] != other.ubatch.seq_id[i][0]) {
                    return false;
                }
            }
        }

        return
            cparams.embeddings  == other.cparams.embeddings  &&
            cparams.causal_attn == other.cparams.causal_attn &&
            arch  == other.arch  &&
            gtype == other.gtype &&
            cvec  == other.cvec  &&
            loras == other.loras &&
            cross == other.cross;
    }
};

class lfg_graph_result {
public:
    lfg_graph_result(int64_t max_nodes);

    virtual ~lfg_graph_result() = default;

    ggml_tensor * get_tokens()      const { return t_tokens; }
    ggml_tensor * get_logits()      const { return t_logits; }
    ggml_tensor * get_embd()        const { return t_embd; }
    ggml_tensor * get_embd_pooled() const { return t_embd_pooled; }

    ggml_cgraph  * get_gf()  const { return gf; }
    ggml_context * get_ctx() const { return ctx_compute.get(); }

    int64_t get_max_nodes() const;

    void reset();

    void set_inputs(const lfg_ubatch * ubatch);
    void set_outputs();

    // try to update the existing graph result using the new graph parameters in order to reuse it
    // this can only be done if we determine that the resulting graph using the new graph parameters
    //   would be identical to the existing graph. in that case, we simply have to update the memory
    //   contexts of the input tensors of the graph and we can reuse it for another computation
    // return true if the graph was updated and can be reused
    bool can_reuse(const lfg_graph_params & params);

    lfg_graph_input_i * add_input(lfg_graph_input_ptr input);

    void set_params(const lfg_graph_params & params);

    // important graph nodes
    ggml_tensor * t_tokens      = nullptr;
    ggml_tensor * t_logits      = nullptr;
    ggml_tensor * t_embd        = nullptr;
    ggml_tensor * t_embd_pooled = nullptr;

    std::map<lfg_seq_id, ggml_tensor*> t_sampled_logits;
    std::map<lfg_seq_id, ggml_tensor*> t_candidates;
    std::map<lfg_seq_id, ggml_tensor*> t_sampled;
    std::map<lfg_seq_id, ggml_tensor*> t_sampled_probs;

    std::vector<lfg_graph_input_ptr> inputs;

    ggml_context_ptr ctx_compute;

    // memory buffers used to evaluate the model
    std::vector<uint8_t> buf_compute_meta;

    ggml_cgraph * gf;

    int64_t max_nodes;

private:
    // keep a copy of the previous graph parameters
    // we will use this to determine whether the graph can be reused by comparing them with the new parameters
    // note: these are updated after constructing the new graph
    lfg_graph_params params;

    // env: LFG_GRAPH_RESULT_DEBUG
    int debug = 0;
};

using lfg_graph_result_ptr = std::unique_ptr<lfg_graph_result>;

//
// lfg_graph_context
//

// used in build_rs to properly order writes and avoid unnecessary copies
using lfg_graph_get_rows_fn = std::function<ggml_tensor * (ggml_context *, ggml_tensor * states, ggml_tensor * ids)>;

struct lfg_graph_context {
    const lfg_arch_enum arch;

    const lfg_hparams & hparams;
    const lfg_cparams & cparams;
    const lfg_ubatch  & ubatch;

    const int64_t n_embd;
    const int64_t n_layer;
    const int64_t n_rot;
    const int64_t n_ctx;       // user-specified context size (can be different from n_ctx_train)
    const int64_t n_head;
    const int64_t n_head_kv;
    const int64_t n_embd_head_k;
    const int64_t n_embd_k_gqa;
    const int64_t n_embd_head_v;
    const int64_t n_embd_v_gqa;
    const int64_t n_expert;
    const int64_t n_expert_used;

    const float freq_base;
    const float freq_scale;
    const float ext_factor;
    const float attn_factor;
    const float beta_fast;
    const float beta_slow;
    const float norm_eps;
    const float norm_rms_eps;

    const int64_t n_tokens;
    const int64_t n_outputs;
    const int32_t n_ctx_orig; // yarn

    const enum lfg_pooling_type pooling_type;
    const enum lfg_rope_type    rope_type;

    ggml_backend_sched_t sched;

    ggml_backend_t backend_cpu; // TODO: needed by build_attn_mha, figure out a way to remove?

    const lfg_adapter_cvec     * cvec;
    const lfg_adapter_loras    * loras;
    const lfg_memory_context_i * mctx;
    const lfg_cross            * cross;

    std::map<lfg_seq_id, lfg_sampler *> samplers;

    const lfg_graph_cb & cb_func;

    lfg_graph_result * res;

    ggml_context * ctx0 = nullptr;
    ggml_cgraph  * gf   = nullptr;

    lfg_graph_context(const lfg_graph_params & params);
    virtual ~lfg_graph_context() = default;

    void cb(ggml_tensor * cur, const char * name, int il) const;

    //
    // common
    //

    ggml_tensor * build_cvec(
             ggml_tensor * cur,
                     int   il) const;

    // do mat_mul, while optionally apply lora
    ggml_tensor * build_lora_mm(
              ggml_tensor * w,
              ggml_tensor * cur) const;

    // do mat_mul_id, while optionally apply lora
    ggml_tensor * build_lora_mm_id(
              ggml_tensor * w,   // ggml_tensor * as
              ggml_tensor * cur, // ggml_tensor * b
              ggml_tensor * ids) const;

    ggml_tensor * build_norm(
             ggml_tensor * cur,
             ggml_tensor * mw,
             ggml_tensor * mb,
           lfg_norm_type   type,
                     int   il) const;

    ggml_tensor * build_ffn(
             ggml_tensor * cur,
             ggml_tensor * up,
             ggml_tensor * up_b,
             ggml_tensor * up_s,
             ggml_tensor * gate,
             ggml_tensor * gate_b,
             ggml_tensor * gate_s,
             ggml_tensor * down,
             ggml_tensor * down_b,
             ggml_tensor * down_s,
             ggml_tensor * act_scales,
         lfg_ffn_op_type   type_op,
       lfg_ffn_gate_type   type_gate,
                     int   il) const;

    // build MoE FFN without bias tensors
    ggml_tensor * build_moe_ffn(
             ggml_tensor * cur,
             ggml_tensor * gate_inp,
             ggml_tensor * up_exps,
             ggml_tensor * gate_exps,
             ggml_tensor * down_exps,
             ggml_tensor * exp_probs_b,
                 int64_t   n_expert,
                 int64_t   n_expert_used,
         lfg_ffn_op_type   type_op,
                    bool   norm_w,
                    bool   scale_w,
                   float   w_scale,
            lfg_expert_gating_func_type gating_op,
                     int   il,
             ggml_tensor * probs_in = nullptr) const;

    ggml_tensor * build_moe_ffn(
             ggml_tensor * cur,
             ggml_tensor * gate_inp,
             ggml_tensor * gate_inp_b,
             ggml_tensor * up_exps,
             ggml_tensor * up_exps_b,
             ggml_tensor * gate_exps,
             ggml_tensor * gate_exps_b,
             ggml_tensor * down_exps,
             ggml_tensor * down_exps_b,
             ggml_tensor * exp_probs_b,
                 int64_t   n_expert,
                 int64_t   n_expert_used,
         lfg_ffn_op_type   type_op,
                    bool   norm_w,
                    bool   scale_w,
                   float   w_scale,
            lfg_expert_gating_func_type gating_op,
                     int   il,
             ggml_tensor * probs_in = nullptr) const;

    //
    // inputs
    //

    ggml_tensor * build_inp_embd(ggml_tensor * tok_embd) const;
    ggml_tensor * build_inp_pos() const;
    ggml_tensor * build_inp_attn_scale() const;
    ggml_tensor * build_inp_out_ids() const;
    ggml_tensor * build_inp_mean() const;
    ggml_tensor * build_inp_cls() const;

    ggml_tensor * build_inp_cross_embd() const;
    ggml_tensor * build_inp_pos_bucket_enc() const;
    ggml_tensor * build_inp_pos_bucket_dec() const;
    ggml_tensor * build_pos_bias(ggml_tensor * pos_bucket, ggml_tensor * attn_rel_b) const;

    //
    // attention
    //

    ggml_tensor * build_attn_mha(
            ggml_tensor * q,       // [n_embd_head_q, n_head_q, n_tokens]
            ggml_tensor * k,       // [n_embd_head_k, n_head_k, n_tokens]
            ggml_tensor * v,       // [n_embd_head_v, n_head_v, n_tokens] (v_trans == false)
            ggml_tensor * kq_b,
            ggml_tensor * kq_mask,
            ggml_tensor * sinks,   // [n_head_q]
            ggml_tensor * v_mla,   // [n_embd_head_v_mla, n_embd_head_v, n_head_v]
                  float   kq_scale,
                    int   il,
            ggml_tensor * inp_pos = nullptr,
                   bool   use_rope = false) const;

    lfg_graph_input_attn_no_cache * build_attn_inp_no_cache() const;

    ggml_tensor * build_attn(
            lfg_graph_input_attn_no_cache * inp,
            ggml_tensor * wo,
            ggml_tensor * wo_b,
            ggml_tensor * q_cur, // [n_embd_head_q, n_head_q, n_tokens]
            ggml_tensor * k_cur, // [n_embd_head_k, n_head_k, n_tokens]
            ggml_tensor * v_cur, // [n_embd_head_v, n_head_v, n_tokens]
            ggml_tensor * kq_b,
            ggml_tensor * sinks, // [n_head_q]
            ggml_tensor * v_mla, // [n_embd_head_v_mla, n_embd_head_v, n_head_v]
                  float   kq_scale,
                    int   il,
            ggml_tensor * inp_pos = nullptr,
                   bool   use_rope = false) const;

    lfg_graph_input_attn_kv * build_attn_inp_kv() const;

    ggml_tensor * build_attn(
            lfg_graph_input_attn_kv * inp,
            ggml_tensor * wo,
            ggml_tensor * wo_b,
            ggml_tensor * q_cur, // [n_embd_head_q, n_head_q, n_tokens]
            ggml_tensor * k_cur, // [n_embd_head_k, n_head_k, n_tokens]
            ggml_tensor * v_cur, // [n_embd_head_v, n_head_v, n_tokens]
            ggml_tensor * kq_b,
            ggml_tensor * sinks, // [n_head_q]
            ggml_tensor * v_mla, // [n_embd_head_v_mla, n_embd_head_v, n_head_v]
                  float   kq_scale,
                    int   il,
            ggml_tensor * inp_pos = nullptr,
                   bool   use_rope = false) const;

    lfg_graph_input_attn_kv_iswa * build_attn_inp_kv_iswa() const;

    // note: if k_cur or v_cur are not provided, they will not be stored in the memory
    ggml_tensor * build_attn(
            lfg_graph_input_attn_kv_iswa * inp,
            ggml_tensor * wo,
            ggml_tensor * wo_b,
            ggml_tensor * q_cur, // [n_embd_head_q, n_head_q, n_tokens]
            ggml_tensor * k_cur, // [n_embd_head_k, n_head_k, n_tokens] optional
            ggml_tensor * v_cur, // [n_embd_head_v, n_head_v, n_tokens] optional
            ggml_tensor * kq_b,
            ggml_tensor * sinks, // [n_head_q]
            ggml_tensor * v_mla, // [n_embd_head_v_mla, n_embd_head_v, n_head_v]
                  float   kq_scale,
                    int   il,
            ggml_tensor * inp_pos = nullptr,
                   bool   use_rope = false) const;

    lfg_graph_input_attn_cross * build_attn_inp_cross() const;

    ggml_tensor * build_attn(
            lfg_graph_input_attn_cross * inp,
            ggml_tensor * wo,
            ggml_tensor * wo_b,
            ggml_tensor * q_cur, // [n_embd_head_q, n_head_q, n_tokens]
            ggml_tensor * k_cur, // [n_embd_head_k, n_head_k, n_tokens]
            ggml_tensor * v_cur, // [n_embd_head_v, n_head_v, n_tokens]
            ggml_tensor * kq_b,
            ggml_tensor * sinks, // [n_head_q]
            ggml_tensor * v_mla, // [n_embd_head_v_mla, n_embd_head_v, n_head_v]
                  float   kq_scale,
                    int   il) const;

    //
    // recurrent
    //

    // TODO: move this implementation to lfg_memory_recurrent.
    //       this is analogous to lfg_kv_cache::cpy_k / cpy_v
    //       when moving, avoid passing `ggml_cgraph` - only pass `ggml_context`. would likely need to split the
    //         implementation in 2 separate methods. the goal is to avoid calling `ggml_build_forward_expand` in
    //         `lfg_memory_recurrent`
    ggml_tensor * build_rs(
            ggml_tensor * s,
            ggml_tensor * state_copy_main,
            ggml_tensor * state_copy_extra,
                int32_t   state_size,
                int32_t   n_seqs,
               uint32_t   n_rs,
               uint32_t   rs_head,
               uint32_t   rs_size,
                int32_t   rs_zero,
            const lfg_graph_get_rows_fn & get_state_rows = ggml_get_rows) const;

    lfg_graph_input_rs * build_rs_inp() const;

    ggml_tensor * build_rs(
            lfg_graph_input_rs * inp,
            ggml_tensor * s,
                int32_t   state_size,
                int32_t   n_seqs,
            const lfg_graph_get_rows_fn & get_state_rows = ggml_get_rows) const;

    ggml_tensor * build_rwkv_token_shift_load(
        lfg_graph_input_rs * inp,
        const lfg_ubatch & ubatch,
                       int   il) const;

    ggml_tensor * build_rwkv_token_shift_store(
             ggml_tensor * token_shift,
      const lfg_ubatch & ubatch,
                     int   il) const;
    //
    // hybrid
    //

    lfg_graph_input_mem_hybrid * build_inp_mem_hybrid() const;

    lfg_graph_input_mem_hybrid_iswa * build_inp_mem_hybrid_iswa() const;

    //
    // pooling
    //

    void build_pooling(
            ggml_tensor * cls,
            ggml_tensor * cls_b,
            ggml_tensor * cls_out,
            ggml_tensor * cls_out_b) const;

    //
    // sampling (backend sampling)
    //

    void build_sampling() const;

    //
    // dense (out)
    //

    void build_dense_out(
            ggml_tensor * dense_2,
            ggml_tensor * dense_2_b,
            ggml_tensor * dense_3) const;
};

// TODO: better name
int32_t lfg_relative_position_bucket(lfg_pos x, lfg_pos y, uint64_t n_buckets, bool bidirectional);
#endif // __cplusplus
