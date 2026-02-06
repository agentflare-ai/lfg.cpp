#pragma once

#include "lfm_inference.h"
#include "lfm_cparams.h"
#include "lfm_graph.h"
#include "lfm_adapter.h"

#ifdef __cplusplus
#include "ggml-cpp.h"
#include "ggml-opt.h"

#include <map>
#include <vector>

struct lfm_model;
class lfm_batch_allocr;

class lfm_io_read_i;
class lfm_io_write_i;

// "memory" as in abstract memory for the context
struct lfm_memory_i;
struct lfm_memory_context_i;

// "memory" as in physical memory for a buffer type, in bytes
struct lfm_memory_breakdown_data {
    size_t model   = 0; // memory allocated for the model
    size_t context = 0; // memory allocated for the context
    size_t compute = 0; // memory allocated for temporary compute buffers

    size_t total() const {
        return model + context + compute;
    }
};

struct lfm_context {
    // init scheduler and compute buffers, reserve worst-case graphs
    lfm_context(
            const lfm_model & model,
                  lfm_context_params params);

    ~lfm_context();

    // reserve a new backend scheduler (if needed)
    // for example, when:
    //   - changing loras
    //   - changing samplers
    //   - changing attention type
    //   - etc.
    void sched_reserve();

    void synchronize();

    const lfm_model   & get_model()   const;
    const lfm_cparams & get_cparams() const;

    ggml_backend_sched_t get_sched() const;

    uint32_t n_ctx()     const;
    uint32_t n_ctx_seq() const;
    uint32_t n_batch()   const;
    uint32_t n_ubatch()  const;
    uint32_t n_seq_max() const;

    uint32_t n_threads()       const;
    uint32_t n_threads_batch() const;

    lfm_memory_t get_memory() const;

    // return true if the memory was updated
    bool memory_update(bool optimize);

    enum lfm_pooling_type pooling_type() const;

    float * get_logits();
    float * get_logits_ith(int32_t i);

    float * get_embeddings();
    float * get_embeddings_ith(int32_t i);
    float * get_embeddings_seq(lfm_seq_id seq_id);

    lfm_token * get_sampled_tokens() const;
    lfm_token   get_sampled_token_ith(int32_t idx);

    float * get_sampled_logits_ith(int32_t idx);
    size_t  get_sampled_logits_count(int32_t idx);

    float * get_sampled_probs_ith(int32_t idx);
    size_t  get_sampled_probs_count(int32_t idx);

    const lfm_token * get_sampled_candidates_ith(int32_t idx);
    size_t get_sampled_candidates_count(int32_t idx);

    void attach_threadpool(
            ggml_threadpool_t threadpool,
            ggml_threadpool_t threadpool_batch);

    void detach_threadpool();

    void set_n_threads(int32_t n_threads, int32_t n_threads_batch);

    void set_abort_callback(bool (*abort_callback)(void * data), void * abort_callback_data);

    void set_embeddings (bool value);
    void set_causal_attn(bool value);
    void set_warmup(bool value);

    void set_adapter_lora(
            lfm_adapter_lora * adapter,
            float scale);

    bool rm_adapter_lora(
            lfm_adapter_lora * adapter);

    void clear_adapter_lora();

    bool apply_adapter_cvec(
            const float * data,
                 size_t   len,
                int32_t   n_embd,
                int32_t   il_start,
                int32_t   il_end);

    // process a single ubatch with a specific graph type
    // if memory_context is provided, it will be applied first to the context's memory
    // ret contains the status of the graph computation
    // returns nullptr only if ret != GGML_STATUS_SUCCESS
    llm_graph_result * process_ubatch(
                const lfm_ubatch & ubatch,
                    llm_graph_type   gtype,
            lfm_memory_context_i * mctx,
                       ggml_status & ret);

    int encode(const lfm_batch & batch_inp);
    int decode(const lfm_batch & batch_inp);

    //
    // state save/load
    //

    size_t state_get_size();
    size_t state_get_data(      uint8_t * dst, size_t size);
    size_t state_set_data(const uint8_t * src, size_t size);

    size_t state_seq_get_size(lfm_seq_id seq_id, lfm_state_seq_flags flags);
    size_t state_seq_get_data(lfm_seq_id seq_id,       uint8_t * dst, size_t size, lfm_state_seq_flags flags);
    size_t state_seq_set_data(lfm_seq_id seq_id, const uint8_t * src, size_t size, lfm_state_seq_flags flags);

    bool state_load_file(
            const char * filepath,
           lfm_token * tokens_out,
                size_t   n_token_capacity,
                size_t * n_token_count_out);

    bool state_save_file(
            const char * filepath,
     const lfm_token * tokens,
                size_t   n_token_count);

    size_t state_seq_load_file(
          lfm_seq_id   seq_id,
            const char * filepath,
           lfm_token * tokens_out,
                size_t   n_token_capacity,
                size_t * n_token_count_out);

    size_t state_seq_save_file(
          lfm_seq_id   seq_id,
            const char * filepath,
     const lfm_token * tokens,
                size_t   n_token_count);

    //
    // perf
    //

    lfm_perf_context_data perf_get_data() const;
    void perf_reset();

    std::map<ggml_backend_buffer_type_t, lfm_memory_breakdown_data> memory_breakdown() const;

    //
    // training
    //

    void opt_init(struct lfm_model * model, struct lfm_opt_params lopt_params);

    // TODO: more flexible combinations of logical/physical batch size and context size
    void opt_epoch(
            ggml_opt_dataset_t      dataset,
            ggml_opt_result_t       result_train,
            ggml_opt_result_t       result_eval,
            int64_t                 idata_split,
            ggml_opt_epoch_callback callback_train,
            ggml_opt_epoch_callback callback_eval);

    void opt_epoch_iter(
            ggml_opt_dataset_t               dataset,
            ggml_opt_result_t                result,
            const std::vector<lfm_token> & tokens,
            const std::vector<lfm_token> & labels_sparse,
            lfm_batch                    & batch,
            ggml_opt_epoch_callback          callback,
            bool                             train,
            int64_t                          idata_in_loop,
            int64_t                          ndata_in_loop,
            int64_t                          t_loop_start);

private:
    //
    // output
    //

    // Make sure enough space is available for outputs.
    // Returns max number of outputs for which space was reserved.
    uint32_t output_reserve(int32_t n_outputs, const lfm_batch & batch);

    void output_reorder();

    // map the output row index `i` to batch index
    int64_t output_resolve_row(int32_t i) const;

    //
    // graph
    //

public:
    uint32_t graph_max_nodes(uint32_t n_tokens) const;

    // can reuse the llm_graph_result instance of the context (for example to update a memory module)
    llm_graph_result * get_gf_res_reserve() const;

    // returns the result of ggml_backend_sched_graph_compute_async execution
    ggml_status graph_compute(ggml_cgraph * gf, bool batched);

    // reserve a graph with a dummy ubatch of the specified size
    ggml_cgraph * graph_reserve(
        uint32_t n_tokens, uint32_t n_seqs, uint32_t n_outputs, const lfm_memory_context_i * mctx, bool split_only = false, size_t * sizes = nullptr);

    bool set_sampler(lfm_seq_id seq_id, lfm_sampler * sampler);

private:
    llm_graph_params graph_params(
                        llm_graph_result * res,
                      const lfm_ubatch & ubatch,
            const lfm_memory_context_i * mctx,
                          llm_graph_type   gtype) const;

    llm_graph_cb graph_get_cb() const;

    // TODO: read/write lora adapters and cvec
    size_t state_write_data(lfm_io_write_i & io);
    size_t state_read_data (lfm_io_read_i  & io);

    size_t state_seq_write_data(lfm_io_write_i & io, lfm_seq_id seq_id, lfm_state_seq_flags flags);
    size_t state_seq_read_data (lfm_io_read_i  & io, lfm_seq_id seq_id, lfm_state_seq_flags flags);

    //
    // members
    //

    const lfm_model & model;

    lfm_cparams       cparams;
    lfm_adapter_cvec  cvec;
    lfm_adapter_loras loras;

    lfm_cross cross; // TODO: tmp for handling cross-attention - need something better probably

    std::unique_ptr<lfm_memory_i> memory;

    // decode output (2-dimensional array: [n_outputs][n_vocab])
    size_t  logits_size = 0; // capacity (of floats) for logits
    float * logits      = nullptr;

    // embeddings output (2-dimensional array: [n_outputs][n_embd])
    // populated only when pooling_type == LFM_POOLING_TYPE_NONE
    size_t  embd_size = 0; // capacity (of floats) for embeddings
    float * embd      = nullptr;

    // TODO: simplify
    struct sampling_info {
        std::map<lfm_seq_id, lfm_sampler *> samplers;

        float       * logits      = nullptr;
        size_t        logits_size = 0;

        lfm_token * sampled      = nullptr;
        size_t        sampled_size = 0;

        float       * probs        = nullptr;
        size_t        probs_size   = 0;

        lfm_token * candidates   = nullptr;
        size_t        candidates_size = 0;

        std::vector<uint32_t> logits_count;
        std::vector<uint32_t> probs_count;
        std::vector<uint32_t> candidates_count;

        std::vector<lfm_token> token_ids_full_vocab;
    };

    sampling_info sampling;

    // sequence embeddings output (map of [n_embd] vectors)
    // populated only when pooling_type != LFM_POOLING_TYPE_NONE
    std::map<lfm_seq_id, std::vector<float>> embd_seq;

    // reuse the batch_allocr to avoid unnecessary memory allocations
    std::unique_ptr<lfm_batch_allocr> balloc;

    uint32_t n_outputs = 0; // number of actually-used outputs in the current ubatch or last logical batch

    std::vector<int32_t> output_ids; // map batch token positions to ids of the logits and embd buffers

    struct swap_info {
        uint32_t i0;
        uint32_t i1;
    };

    std::vector<swap_info> output_swaps;

    ggml_backend_sched_ptr sched;

    bool sched_need_reserve = true;

    ggml_backend_t backend_cpu = nullptr;
    std::vector<ggml_backend_ptr> backends;

    // training
    ggml_opt_context_t opt_ctx = nullptr;

    ggml_threadpool_t threadpool       = nullptr;
    ggml_threadpool_t threadpool_batch = nullptr;

    ggml_abort_callback abort_callback      = nullptr;
    void *              abort_callback_data = nullptr;

    std::vector<std::pair<ggml_backend_t, ggml_backend_set_n_threads_t>> set_n_threads_fns;

    // pointers and buffer types used for the compute buffer of each backend
    std::vector<ggml_backend_t>             backend_ptrs;
    std::vector<ggml_backend_buffer_type_t> backend_buft;
    std::vector<size_t>                     backend_buf_exp_size; // expected buffer sizes

    llm_graph_result_ptr gf_res_prev;
    llm_graph_result_ptr gf_res_reserve;

    // host buffer for the model output (logits and embeddings)
    ggml_backend_buffer_ptr buf_output;

    bool has_evaluated_once = false;

    // env: LFM_GRAPH_REUSE_DISABLE
    bool graph_reuse_disable = false;

    // perf
    mutable int64_t t_start_us  = 0;
    mutable int64_t t_load_us   = 0;
    mutable int64_t t_p_eval_us = 0;
    mutable int64_t t_eval_us   = 0;

    mutable int64_t t_compute_start_us = 0;
    mutable int64_t n_queued_tokens    = 0;

    mutable int32_t n_p_eval = 0; // number of tokens in eval calls for the prompt (with batch size > 1)
    mutable int32_t n_eval   = 0; // number of eval calls

    mutable int32_t n_reused = 0; // number of times the previous graph was reused
};
#endif
