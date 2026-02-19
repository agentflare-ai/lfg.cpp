#ifndef LFG_H
#define LFG_H

#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-backend.h"
#include "ggml-opt.h"

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdbool.h>

#ifdef LFG_SHARED
#    if defined(_WIN32) && !defined(__MINGW32__)
#        ifdef LFG_BUILD
#            define LFG_API __declspec(dllexport)
#        else
#            define LFG_API __declspec(dllimport)
#        endif
#    else
#        define LFG_API __attribute__ ((visibility ("default")))
#    endif
#else
#    define LFG_API
#endif

#ifdef __GNUC__
#    define DEPRECATED(func, hint) func __attribute__((deprecated(hint)))
#elif defined(_MSC_VER)
#    define DEPRECATED(func, hint) __declspec(deprecated(hint)) func
#else
#    define DEPRECATED(func, hint) func
#endif

#define LFG_DEFAULT_SEED 0xFFFFFFFF

#define LFG_TOKEN_NULL -1

#define LFG_FILE_MAGIC_GGLA 0x67676c61u // 'ggla'
#define LFG_FILE_MAGIC_GGSN 0x6767736eu // 'ggsn'
#define LFG_FILE_MAGIC_GGSQ 0x67677371u // 'ggsq'

#define LFG_SESSION_MAGIC   LFG_FILE_MAGIC_GGSN
#define LFG_SESSION_VERSION 9

#define LFG_STATE_SEQ_MAGIC   LFG_FILE_MAGIC_GGSQ
#define LFG_STATE_SEQ_VERSION 2

// C API/ABI versioning
#define LFG_API_VERSION_MAJOR 1
#define LFG_API_VERSION_MINOR 1
#define LFG_API_VERSION_PATCH 0
#define LFG_API_VERSION ((LFG_API_VERSION_MAJOR << 16) | (LFG_API_VERSION_MINOR << 8) | (LFG_API_VERSION_PATCH))

// Increment when ABI-incompatible changes occur.
#define LFG_ABI_VERSION 1

#ifdef __cplusplus
extern "C" {
#endif

    //
    // C interface
    //
    // TODO: show sample usage
    //

    struct lfg_vocab;
    struct lfg_model;
    struct lfg_context;
    struct lfg_sampler;

    typedef struct lfg_memory_i * lfg_memory_t;

    typedef int32_t lfg_pos;
    typedef int32_t lfg_token;
    typedef int32_t lfg_seq_id;

    enum lfg_vocab_type {
        LFG_VOCAB_TYPE_NONE   = 0, // For models without vocab
        LFG_VOCAB_TYPE_SPM    = 1, // Liquid tokenizer based on byte-level BPE with byte fallback
        LFG_VOCAB_TYPE_BPE    = 2, // GPT-2 tokenizer based on byte-level BPE
        LFG_VOCAB_TYPE_WPM    = 3, // BERT tokenizer based on WordPiece
        LFG_VOCAB_TYPE_UGM    = 4, // T5 tokenizer based on Unigram
        LFG_VOCAB_TYPE_RWKV   = 5, // RWKV tokenizer based on greedy tokenization
        LFG_VOCAB_TYPE_PLAMO2 = 6, // PLaMo-2 tokenizer based on Aho-Corasick with dynamic programming
    };

    enum lfg_rope_type {
        LFG_ROPE_TYPE_NONE   = -1,
        LFG_ROPE_TYPE_NORM   = 0,
        LFG_ROPE_TYPE_NEOX   = GGML_ROPE_TYPE_NEOX,
        LFG_ROPE_TYPE_MROPE  = GGML_ROPE_TYPE_MROPE,
        LFG_ROPE_TYPE_IMROPE = GGML_ROPE_TYPE_IMROPE,
        LFG_ROPE_TYPE_VISION = GGML_ROPE_TYPE_VISION,
    };

    enum lfg_token_type { //TODO: remove, required until per token attributes are available from GGUF file
        LFG_TOKEN_TYPE_UNDEFINED    = 0,
        LFG_TOKEN_TYPE_NORMAL       = 1,
        LFG_TOKEN_TYPE_UNKNOWN      = 2,
        LFG_TOKEN_TYPE_CONTROL      = 3,
        LFG_TOKEN_TYPE_USER_DEFINED = 4,
        LFG_TOKEN_TYPE_UNUSED       = 5,
        LFG_TOKEN_TYPE_BYTE         = 6,
    };

    enum lfg_token_attr {
        LFG_TOKEN_ATTR_UNDEFINED    = 0,
        LFG_TOKEN_ATTR_UNKNOWN      = 1 << 0,
        LFG_TOKEN_ATTR_UNUSED       = 1 << 1,
        LFG_TOKEN_ATTR_NORMAL       = 1 << 2,
        LFG_TOKEN_ATTR_CONTROL      = 1 << 3,  // SPECIAL?
        LFG_TOKEN_ATTR_USER_DEFINED = 1 << 4,
        LFG_TOKEN_ATTR_BYTE         = 1 << 5,
        LFG_TOKEN_ATTR_NORMALIZED   = 1 << 6,
        LFG_TOKEN_ATTR_LSTRIP       = 1 << 7,
        LFG_TOKEN_ATTR_RSTRIP       = 1 << 8,
        LFG_TOKEN_ATTR_SINGLE_WORD  = 1 << 9,
    };

    // model file types
    enum lfg_ftype {
        LFG_FTYPE_ALL_F32              = 0,
        LFG_FTYPE_MOSTLY_F16           = 1,  // except 1d tensors
        LFG_FTYPE_MOSTLY_Q4_0          = 2,  // except 1d tensors
        LFG_FTYPE_MOSTLY_Q4_1          = 3,  // except 1d tensors
        // LFG_FTYPE_MOSTLY_Q4_1_SOME_F16 = 4,  // tok_embeddings.weight and output.weight are F16
        // LFG_FTYPE_MOSTLY_Q4_2       = 5,  // support has been removed
        // LFG_FTYPE_MOSTLY_Q4_3       = 6,  // support has been removed
        LFG_FTYPE_MOSTLY_Q8_0          = 7,  // except 1d tensors
        LFG_FTYPE_MOSTLY_Q5_0          = 8,  // except 1d tensors
        LFG_FTYPE_MOSTLY_Q5_1          = 9,  // except 1d tensors
        LFG_FTYPE_MOSTLY_Q2_K          = 10, // except 1d tensors
        LFG_FTYPE_MOSTLY_Q3_K_S        = 11, // except 1d tensors
        LFG_FTYPE_MOSTLY_Q3_K_M        = 12, // except 1d tensors
        LFG_FTYPE_MOSTLY_Q3_K_L        = 13, // except 1d tensors
        LFG_FTYPE_MOSTLY_Q4_K_S        = 14, // except 1d tensors
        LFG_FTYPE_MOSTLY_Q4_K_M        = 15, // except 1d tensors
        LFG_FTYPE_MOSTLY_Q5_K_S        = 16, // except 1d tensors
        LFG_FTYPE_MOSTLY_Q5_K_M        = 17, // except 1d tensors
        LFG_FTYPE_MOSTLY_Q6_K          = 18, // except 1d tensors
        LFG_FTYPE_MOSTLY_IQ2_XXS       = 19, // except 1d tensors
        LFG_FTYPE_MOSTLY_IQ2_XS        = 20, // except 1d tensors
        LFG_FTYPE_MOSTLY_Q2_K_S        = 21, // except 1d tensors
        LFG_FTYPE_MOSTLY_IQ3_XS        = 22, // except 1d tensors
        LFG_FTYPE_MOSTLY_IQ3_XXS       = 23, // except 1d tensors
        LFG_FTYPE_MOSTLY_IQ1_S         = 24, // except 1d tensors
        LFG_FTYPE_MOSTLY_IQ4_NL        = 25, // except 1d tensors
        LFG_FTYPE_MOSTLY_IQ3_S         = 26, // except 1d tensors
        LFG_FTYPE_MOSTLY_IQ3_M         = 27, // except 1d tensors
        LFG_FTYPE_MOSTLY_IQ2_S         = 28, // except 1d tensors
        LFG_FTYPE_MOSTLY_IQ2_M         = 29, // except 1d tensors
        LFG_FTYPE_MOSTLY_IQ4_XS        = 30, // except 1d tensors
        LFG_FTYPE_MOSTLY_IQ1_M         = 31, // except 1d tensors
        LFG_FTYPE_MOSTLY_BF16          = 32, // except 1d tensors
        //LFG_FTYPE_MOSTLY_Q4_0_4_4      = 33, // removed from gguf files, use Q4_0 and runtime repack
        //LFG_FTYPE_MOSTLY_Q4_0_4_8      = 34, // removed from gguf files, use Q4_0 and runtime repack
        //LFG_FTYPE_MOSTLY_Q4_0_8_8      = 35, // removed from gguf files, use Q4_0 and runtime repack
        LFG_FTYPE_MOSTLY_TQ1_0         = 36, // except 1d tensors
        LFG_FTYPE_MOSTLY_TQ2_0         = 37, // except 1d tensors
        LFG_FTYPE_MOSTLY_MXFP4_MOE     = 38, // except 1d tensors

        LFG_FTYPE_GUESSED = 1024, // not specified in the model file
    };

    enum lfg_rope_scaling_type {
        LFG_ROPE_SCALING_TYPE_UNSPECIFIED = -1,
        LFG_ROPE_SCALING_TYPE_NONE        = 0,
        LFG_ROPE_SCALING_TYPE_LINEAR      = 1,
        LFG_ROPE_SCALING_TYPE_YARN        = 2,
        LFG_ROPE_SCALING_TYPE_LONGROPE    = 3,
        LFG_ROPE_SCALING_TYPE_MAX_VALUE   = LFG_ROPE_SCALING_TYPE_LONGROPE,
    };

    enum lfg_pooling_type {
        LFG_POOLING_TYPE_UNSPECIFIED = -1,
        LFG_POOLING_TYPE_NONE = 0,
        LFG_POOLING_TYPE_MEAN = 1,
        LFG_POOLING_TYPE_CLS  = 2,
        LFG_POOLING_TYPE_LAST = 3,
        LFG_POOLING_TYPE_RANK = 4, // used by reranking models to attach the classification head to the graph
    };

    enum lfg_attention_type {
        LFG_ATTENTION_TYPE_UNSPECIFIED = -1,
        LFG_ATTENTION_TYPE_CAUSAL      = 0,
        LFG_ATTENTION_TYPE_NON_CAUSAL  = 1,
    };

    enum lfg_flash_attn_type {
        LFG_FLASH_ATTN_TYPE_AUTO     = -1,
        LFG_FLASH_ATTN_TYPE_DISABLED = 0,
        LFG_FLASH_ATTN_TYPE_ENABLED  = 1,
    };

    LFG_API const char * lfg_flash_attn_type_name(enum lfg_flash_attn_type flash_attn_type);

    enum lfg_split_mode {
        LFG_SPLIT_MODE_NONE  = 0, // single GPU
        LFG_SPLIT_MODE_LAYER = 1, // split layers and KV across GPUs
        LFG_SPLIT_MODE_ROW   = 2, // split layers and KV across GPUs, use tensor parallelism if supported
    };

    // TODO: simplify (https://github.com/ggml-org/liquid.cpp/pull/9294#pullrequestreview-2286561979)
    typedef struct lfg_token_data {
        lfg_token id; // token id
        float logit;    // log-odds of the token
        float p;        // probability of the token
    } lfg_token_data;

    typedef struct lfg_token_data_array {
        // TODO: consider SoA
        // NOTE: this pointer can be modified by the samplers
        lfg_token_data * data;
        size_t size;
        int64_t selected; // this is the index in the data array (i.e. not the token id)
        bool sorted;      // note: do not assume the data is sorted - always check this flag
    } lfg_token_data_array;

    typedef bool (*lfg_progress_callback)(float progress, void * user_data);

    // Input data for lfg_encode/lfg_decode
    // A lfg_batch object can contain input about one or many sequences
    // The provided arrays (i.e. token, embd, pos, etc.) must have size of n_tokens
    //
    // - token  : the token ids of the input (used when embd is NULL)
    // - embd   : token embeddings (i.e. float vector of size n_embd) (used when token is NULL)
    // - pos    : the positions of the respective token in the sequence
    //            (if set to NULL, the token position will be tracked automatically by lfg_encode/lfg_decode)
    // - seq_id : the sequence to which the respective token belongs
    //            (if set to NULL, the sequence ID will be assumed to be 0)
    // - logits : if zero, the logits (and/or the embeddings) for the respective token will not be output
    //            (if set to NULL:
    //               - if embeddings: all tokens are output
    //               - if not:        only the last token is output
    //            )
    //
    typedef struct lfg_batch {
        int32_t n_tokens;

        lfg_token  *  token;
        float        *  embd;
        lfg_pos    *  pos;
        int32_t      *  n_seq_id;
        lfg_seq_id ** seq_id;
        int8_t       *  logits;   // TODO: rename this to "output"
    } lfg_batch;

    enum lfg_model_kv_override_type {
        LFG_KV_OVERRIDE_TYPE_INT,
        LFG_KV_OVERRIDE_TYPE_FLOAT,
        LFG_KV_OVERRIDE_TYPE_BOOL,
        LFG_KV_OVERRIDE_TYPE_STR,
    };

    enum lfg_model_meta_key {
        LFG_MODEL_META_KEY_SAMPLING_SEQUENCE,
        LFG_MODEL_META_KEY_SAMPLING_TOP_K,
        LFG_MODEL_META_KEY_SAMPLING_TOP_P,
        LFG_MODEL_META_KEY_SAMPLING_MIN_P,
        LFG_MODEL_META_KEY_SAMPLING_XTC_PROBABILITY,
        LFG_MODEL_META_KEY_SAMPLING_XTC_THRESHOLD,
        LFG_MODEL_META_KEY_SAMPLING_TEMP,
        LFG_MODEL_META_KEY_SAMPLING_PENALTY_LAST_N,
        LFG_MODEL_META_KEY_SAMPLING_PENALTY_REPEAT,
        LFG_MODEL_META_KEY_SAMPLING_MIROSTAT,
        LFG_MODEL_META_KEY_SAMPLING_MIROSTAT_TAU,
        LFG_MODEL_META_KEY_SAMPLING_MIROSTAT_ETA,
    };

    struct lfg_model_kv_override {
        enum lfg_model_kv_override_type tag;

        char key[128];

        union {
            int64_t val_i64;
            double  val_f64;
            bool    val_bool;
            char    val_str[128];
        };
    };

    struct lfg_model_tensor_buft_override {
        const char * pattern;
        ggml_backend_buffer_type_t buft;
    };

    struct lfg_model_params {
        // NULL-terminated list of devices to use for offloading (if NULL, all available devices are used)
        ggml_backend_dev_t * devices;

        // NULL-terminated list of buffer types to use for tensors that match a pattern
        const struct lfg_model_tensor_buft_override * tensor_buft_overrides;

        int32_t n_gpu_layers; // number of layers to store in VRAM, a negative value means all layers
        enum lfg_split_mode split_mode; // how to split the model across multiple GPUs

        // the GPU that is used for the entire model when split_mode is LFG_SPLIT_MODE_NONE
        int32_t main_gpu;

        // proportion of the model (layers or rows) to offload to each GPU, size: lfg_max_devices()
        const float * tensor_split;

        // Called with a progress value between 0.0 and 1.0. Pass NULL to disable.
        // If the provided progress_callback returns true, model loading continues.
        // If it returns false, model loading is immediately aborted.
        lfg_progress_callback progress_callback;

        // context pointer passed to the progress callback
        void * progress_callback_user_data;

        // override key-value pairs of the model meta data
        const struct lfg_model_kv_override * kv_overrides;

        // Keep the booleans together to avoid misalignment during copy-by-value.
        bool vocab_only;      // only load the vocabulary, no weights
        bool use_mmap;        // use mmap if possible
        bool use_direct_io;   // use direct io, takes precedence over use_mmap
        bool use_mlock;       // force system to keep model in RAM
        bool check_tensors;   // validate model tensor data
        bool use_extra_bufts; // use extra buffer types (used for weight repacking)
        bool no_host;         // bypass host buffer allowing extra buffers to be used
        bool no_alloc;        // only load metadata and simulate memory allocations
        
        enum ggml_type type_k; // data type for K cache [EXPERIMENTAL]
        enum ggml_type type_v; // data type for V cache [EXPERIMENTAL]
        bool swa_full;         // use full-size SWA cache
    };

    struct lfg_sampler_seq_config {
        lfg_seq_id           seq_id;
        struct lfg_sampler * sampler;
    };

    // NOTE: changing the default values of parameters marked as [EXPERIMENTAL] may cause crashes or incorrect results in certain configurations
    //       https://github.com/ggml-org/liquid.cpp/pull/7544
    struct lfg_context_params {
        uint32_t n_ctx;             // text context, 0 = from model
        uint32_t n_batch;           // logical maximum batch size that can be submitted to lfg_decode
        uint32_t n_ubatch;          // physical maximum batch size
        uint32_t n_seq_max;         // max number of sequences (i.e. distinct states for recurrent models)
        int32_t  n_threads;         // number of threads to use for generation
        int32_t  n_threads_batch;   // number of threads to use for batch processing

        enum lfg_rope_scaling_type rope_scaling_type; // RoPE scaling type, from `enum lfg_rope_scaling_type`
        enum lfg_pooling_type      pooling_type;      // whether to pool (sum) embedding results by sequence id
        enum lfg_attention_type    attention_type;    // attention type to use for embeddings
        enum lfg_flash_attn_type   flash_attn_type;   // when to enable Flash Attention

        // ref: https://github.com/ggml-org/liquid.cpp/pull/2054
        float    rope_freq_base;   // RoPE base frequency, 0 = from model
        float    rope_freq_scale;  // RoPE frequency scaling factor, 0 = from model
        float    yarn_ext_factor;  // YaRN extrapolation mix factor, negative = from model
        float    yarn_attn_factor; // YaRN magnitude scaling factor
        float    yarn_beta_fast;   // YaRN low correction dim
        float    yarn_beta_slow;   // YaRN high correction dim
        uint32_t yarn_orig_ctx;    // YaRN original context size
        float    defrag_thold;     // [DEPRECATED] defragment the KV cache if holes/size > thold, <= 0 disabled (default)

        ggml_backend_sched_eval_callback cb_eval;
        void * cb_eval_user_data;

        enum ggml_type type_k; // data type for K cache [EXPERIMENTAL]
        enum ggml_type type_v; // data type for V cache [EXPERIMENTAL]

        // Abort callback
        // if it returns true, execution of lfg_decode() will be aborted
        // currently works only with CPU execution
        ggml_abort_callback abort_callback;
        void *              abort_callback_data;

        // Keep the booleans together and at the end of the struct to avoid misalignment during copy-by-value.
        bool embeddings;  // if true, extract embeddings (together with logits)
        bool offload_kqv; // offload the KQV ops (including the KV cache) to GPU
        bool no_perf;     // measure performance timings
        bool op_offload;  // offload host tensor operations to device
        bool swa_full;    // use full-size SWA cache (https://github.com/ggml-org/liquid.cpp/pull/13194#issuecomment-2868343055)
                          // NOTE: setting to false when n_seq_max > 1 can cause bad performance in some cases
                          //       ref: https://github.com/ggml-org/liquid.cpp/pull/13845#issuecomment-2924800573
        bool kv_unified;  // use a unified buffer across the input sequences when computing the attention
                          // try to disable when n_seq_max > 1 for improved performance when the sequences do not share a large prefix
                          // ref: https://github.com/ggml-org/liquid.cpp/pull/14363

        // [EXPERIMENTAL]
        // backend sampler chain configuration (make sure the caller keeps the sampler chains alive)
        // note: the samplers must be sampler chains (i.e. use lfg_sampler_chain_init)
        struct lfg_sampler_seq_config * samplers;
        size_t                            n_samplers;
    };

    // model quantization parameters
    typedef struct lfg_model_quantize_params {
        int32_t nthread;                      // number of threads to use for quantizing, if <=0 will use std::thread::hardware_concurrency()
        enum lfg_ftype ftype;               // quantize to this lfg_ftype
        enum ggml_type output_tensor_type;    // output tensor type
        enum ggml_type token_embedding_type;  // token embeddings tensor type
        bool allow_requantize;                // allow quantizing non-f32/f16 tensors
        bool quantize_output_tensor;          // quantize output.weight
        bool only_copy;                       // only copy tensors - ftype, allow_requantize and quantize_output_tensor are ignored
        bool pure;                            // quantize all tensors to the default type
        bool keep_split;                      // quantize to the same number of shards
        void * imatrix;                       // pointer to importance matrix data
        void * kv_overrides;                  // pointer to vector containing overrides
        void * tensor_types;                  // pointer to vector containing tensor types
        void * prune_layers;                  // pointer to vector containing layer indices to prune
    } lfg_model_quantize_params;

    typedef struct lfg_logit_bias {
        lfg_token token;
        float bias;
    } lfg_logit_bias;

    typedef struct lfg_sampler_chain_params {
        bool no_perf; // whether to measure performance timings
    } lfg_sampler_chain_params;

    // forward-declare for lfg_chat_message (defined in lfg_api.h)
    struct lfg_tool_call;

    // used in chat template
    typedef struct lfg_chat_message {
        const char * role;
        const char * content;
        const struct lfg_tool_call * tool_calls;  // nullable — structured tool calls (OpenAI-compatible)
        int32_t n_tool_calls;                     // 0 = none
        const char * tool_call_id;                // for role="tool" responses (nullable)
    } lfg_chat_message;

    // lora adapter
    struct lfg_adapter_lora;

    // Helpers for getting default parameters
    // TODO: update API to start accepting pointers to params structs (https://github.com/ggml-org/liquid.cpp/discussions/9172)
    LFG_API struct lfg_model_params          lfg_model_default_params(void);
    LFG_API struct lfg_context_params        lfg_context_default_params(void);
    LFG_API struct lfg_sampler_chain_params  lfg_sampler_chain_default_params(void);
    LFG_API struct lfg_model_quantize_params lfg_model_quantize_default_params(void);

    // Initialize the liquid + ggml backend
    // If numa is true, use NUMA optimizations
    // Call once at the start of the program
    LFG_API void lfg_backend_init(void);

    // Call once at the end of the program - currently only used for MPI
    LFG_API void lfg_backend_free(void);

    //optional:
    LFG_API void lfg_numa_init(enum ggml_numa_strategy numa);

    // Error handling (thread-local)
    enum lfg_error {
        LFG_ERROR_NONE = 0,
        LFG_ERROR_INVALID_ARGUMENT = 1,
        LFG_ERROR_IO = 2,
        LFG_ERROR_OUT_OF_MEMORY = 3,
        LFG_ERROR_UNSUPPORTED = 4,
        LFG_ERROR_CANCELLED = 5,
        LFG_ERROR_INTERNAL = 6,
    };

    // Returns the last error code for the current thread.
    // If buf is non-null, writes a null-terminated error message to buf (if buf_size > 0).
    LFG_API enum lfg_error lfg_get_last_error(char * buf, size_t buf_size);
    LFG_API void lfg_clear_last_error(void);
    LFG_API const char * lfg_error_string(enum lfg_error code);

    // Versioning helpers
    LFG_API void lfg_api_version(uint32_t * major, uint32_t * minor, uint32_t * patch);
    LFG_API const char * lfg_api_version_string(void);
    LFG_API uint32_t lfg_abi_version(void);

    // Optional: an auto threadpool gets created in ggml if not passed explicitly
    LFG_API void lfg_attach_threadpool(
            struct lfg_context * ctx,
               ggml_threadpool_t   threadpool,
               ggml_threadpool_t   threadpool_batch);

    LFG_API void lfg_detach_threadpool(struct lfg_context * ctx);

    DEPRECATED(LFG_API struct lfg_model * lfg_load_model_from_file(
                             const char * path_model,
              struct lfg_model_params   params),
            "use lfg_model_load_from_file instead");

    // Load the model from a file
    // If the file is split into multiple parts, the file name must follow this pattern: <name>-%05d-of-%05d.gguf
    // If the split file name does not follow this pattern, use lfg_model_load_from_splits
    LFG_API struct lfg_model * lfg_model_load_from_file(
                             const char * path_model,
              struct lfg_model_params   params);

    // Load the model from multiple splits (support custom naming scheme)
    // The paths must be in the correct order
    LFG_API struct lfg_model * lfg_model_load_from_splits(
                             const char ** paths,
                                 size_t    n_paths,
              struct lfg_model_params    params);

    LFG_API void lfg_model_save_to_file(
            const struct lfg_model * model,
                        const char * path_model);

    DEPRECATED(LFG_API void lfg_free_model(struct lfg_model * model),
            "use lfg_model_free instead");

    LFG_API void lfg_model_free(struct lfg_model * model);

    LFG_API struct lfg_context * lfg_init_from_model(
                     struct lfg_model * model,
            struct lfg_context_params   params);

    DEPRECATED(LFG_API struct lfg_context * lfg_new_context_with_model(
                     struct lfg_model * model,
            struct lfg_context_params   params),
            "use lfg_init_from_model instead");

    // Frees all allocated memory
    LFG_API void lfg_free(struct lfg_context * ctx);

    enum lfg_params_fit_status {
        LFG_PARAMS_FIT_STATUS_SUCCESS = 0, // found allocations that are projected to fit
        LFG_PARAMS_FIT_STATUS_FAILURE = 1, // could not find allocations that are projected to fit
        LFG_PARAMS_FIT_STATUS_ERROR   = 2, // a hard error occured, e.g. because no model could be found at the specified path
    };

    // fits mparams and cparams to free device memory (assumes system memory is unlimited)
    //   - returns true if the parameters could be successfully modified to fit device memory
    //   - this function is NOT thread safe because it modifies the global liquid logger state
    //   - only parameters that have the same value as in lfg_default_model_params are modified
    LFG_API enum lfg_params_fit_status lfg_params_fit(
                                   const char   * path_model,
                    struct lfg_model_params   * mparams,
                    struct lfg_context_params * cparams,
                                          float * tensor_split,          // writable buffer for tensor split, needs at least lfg_max_devices elements
        struct lfg_model_tensor_buft_override * tensor_buft_overrides, // writable buffer for overrides, needs at least lfg_max_tensor_buft_overrides elements
                                         size_t * margins,               // margins of memory to leave per device in bytes
                                       uint32_t   n_ctx_min,             // minimum context size to set when trying to reduce memory use
                            enum ggml_log_level   log_level);            // minimum log level to print during fitting, lower levels go to debug log

    LFG_API int64_t lfg_time_us(void);

    LFG_API size_t lfg_max_devices(void);
    LFG_API size_t lfg_max_parallel_sequences(void);
    LFG_API size_t lfg_max_tensor_buft_overrides(void);

    LFG_API bool lfg_supports_mmap       (void);
    LFG_API bool lfg_supports_mlock      (void);
    LFG_API bool lfg_supports_gpu_offload(void);
    LFG_API bool lfg_supports_rpc        (void);

    // NOTE: After creating a lfg_context, it is recommended to query the actual values using these functions
    //       In some cases the requested values via lfg_context_params may differ from the actual values used by the context
    //       ref: https://github.com/ggml-org/liquid.cpp/pull/17046#discussion_r2503085732
    LFG_API uint32_t lfg_n_ctx      (const struct lfg_context * ctx);
    LFG_API uint32_t lfg_n_ctx_seq  (const struct lfg_context * ctx);
    LFG_API uint32_t lfg_n_batch    (const struct lfg_context * ctx);
    LFG_API uint32_t lfg_n_ubatch   (const struct lfg_context * ctx);
    LFG_API uint32_t lfg_n_seq_max  (const struct lfg_context * ctx);

    DEPRECATED(LFG_API int32_t lfg_n_ctx_train(const struct lfg_model * model), "use lfg_model_n_ctx_train instead");
    DEPRECATED(LFG_API int32_t lfg_n_embd     (const struct lfg_model * model), "use lfg_model_n_embd instead");
    DEPRECATED(LFG_API int32_t lfg_n_layer    (const struct lfg_model * model), "use lfg_model_n_layer instead");
    DEPRECATED(LFG_API int32_t lfg_n_head     (const struct lfg_model * model), "use lfg_model_n_head instead");

    DEPRECATED(LFG_API int32_t lfg_n_vocab    (const struct lfg_vocab * vocab), "use lfg_vocab_n_tokens instead");

    LFG_API const struct lfg_model * lfg_get_model   (const struct lfg_context * ctx);
    LFG_API           lfg_memory_t   lfg_get_memory  (const struct lfg_context * ctx);
    LFG_API  enum lfg_pooling_type   lfg_pooling_type(const struct lfg_context * ctx); // TODO: rename to lfg_get_pooling_type

    LFG_API const struct lfg_vocab * lfg_model_get_vocab(const struct lfg_model * model);
    LFG_API enum lfg_rope_type       lfg_model_rope_type(const struct lfg_model * model);

    LFG_API int32_t lfg_model_n_ctx_train(const struct lfg_model * model);
    LFG_API int32_t lfg_model_n_embd     (const struct lfg_model * model);
    LFG_API int32_t lfg_model_n_embd_inp (const struct lfg_model * model);
    LFG_API int32_t lfg_model_n_embd_out (const struct lfg_model * model);
    LFG_API int32_t lfg_model_n_layer    (const struct lfg_model * model);
    LFG_API int32_t lfg_model_n_head     (const struct lfg_model * model);
    LFG_API int32_t lfg_model_n_head_kv  (const struct lfg_model * model);
    LFG_API int32_t lfg_model_n_swa      (const struct lfg_model * model);

    // Get the model's RoPE frequency scaling factor
    LFG_API float lfg_model_rope_freq_scale_train(const struct lfg_model * model);

    // Returns the number of classifier outputs (only valid for classifier models)
    // Undefined behavior for non-classifier models
    LFG_API uint32_t lfg_model_n_cls_out(const struct lfg_model * model);

    // Returns label of classifier output by index (<n_cls_out). Returns nullptr if no label provided
    LFG_API const char * lfg_model_cls_label(const struct lfg_model * model, uint32_t i);

    LFG_API enum lfg_vocab_type lfg_vocab_type(const struct lfg_vocab * vocab);

    LFG_API int32_t lfg_vocab_n_tokens(const struct lfg_vocab * vocab);

    // Functions to access the model's GGUF metadata scalar values
    // - The functions return the length of the string on success, or -1 on failure
    // - The output string is always null-terminated and cleared on failure
    // - When retrieving a string, an extra byte must be allocated to account for the null terminator
    // - GGUF array values are not supported by these functions

    // Get metadata value as a string by key name
    LFG_API int32_t lfg_model_meta_val_str(const struct lfg_model * model, const char * key, char * buf, size_t buf_size);

    // Get the number of metadata key/value pairs
    LFG_API int32_t lfg_model_meta_count(const struct lfg_model * model);

    // Get sampling metadata key name. Returns nullptr if the key is invalid
    LFG_API const char * lfg_model_meta_key_str(enum lfg_model_meta_key key);

    // Get metadata key name by index
    LFG_API int32_t lfg_model_meta_key_by_index(const struct lfg_model * model, int32_t i, char * buf, size_t buf_size);

    // Get metadata value as a string by index
    LFG_API int32_t lfg_model_meta_val_str_by_index(const struct lfg_model * model, int32_t i, char * buf, size_t buf_size);

    // Get a string describing the model type
    LFG_API int32_t lfg_model_desc(const struct lfg_model * model, char * buf, size_t buf_size);

    // Returns the total size of all the tensors in the model in bytes
    LFG_API uint64_t lfg_model_size(const struct lfg_model * model);

    // Get the default chat template. Returns nullptr if not available
    // If name is NULL, returns the default chat template
    LFG_API const char * lfg_model_chat_template(const struct lfg_model * model, const char * name);

    // Returns the total number of parameters in the model
    LFG_API uint64_t lfg_model_n_params(const struct lfg_model * model);

    // Returns true if the model contains an encoder that requires lfg_encode() call
    LFG_API bool lfg_model_has_encoder(const struct lfg_model * model);

    // Returns true if the model contains a decoder that requires lfg_decode() call
    LFG_API bool lfg_model_has_decoder(const struct lfg_model * model);

    // For encoder-decoder models, this function returns id of the token that must be provided
    // to the decoder to start generating output sequence. For other models, it returns -1.
    LFG_API lfg_token lfg_model_decoder_start_token(const struct lfg_model * model);

    // Returns true if the model is recurrent (like Mamba, RWKV, etc.)
    LFG_API bool lfg_model_is_recurrent(const struct lfg_model * model);

    // Returns true if the model is hybrid (like Jamba, Granite, etc.)
    LFG_API bool lfg_model_is_hybrid(const struct lfg_model * model);

    // Returns true if the model is diffusion-based (like LLaDA, Dream, etc.)
    LFG_API bool lfg_model_is_diffusion(const struct lfg_model * model);

    // Returns 0 on success
    LFG_API uint32_t lfg_model_quantize(
            const char * fname_inp,
            const char * fname_out,
            const lfg_model_quantize_params * params);

    //
    // Adapters
    //

    // Load a LoRA adapter from file
    // The adapter is valid as long as the associated model is not freed
    // All adapters must be loaded before context creation
    LFG_API struct lfg_adapter_lora * lfg_adapter_lora_init(
            struct lfg_model * model,
            const char * path_lora);

    // Functions to access the adapter's GGUF metadata scalar values
    // - The functions return the length of the string on success, or -1 on failure
    // - The output string is always null-terminated and cleared on failure
    // - When retrieving a string, an extra byte must be allocated to account for the null terminator
    // - GGUF array values are not supported by these functions

    // Get metadata value as a string by key name
    LFG_API int32_t lfg_adapter_meta_val_str(const struct lfg_adapter_lora * adapter, const char * key, char * buf, size_t buf_size);

    // Get the number of metadata key/value pairs
    LFG_API int32_t lfg_adapter_meta_count(const struct lfg_adapter_lora * adapter);

    // Get metadata key name by index
    LFG_API int32_t lfg_adapter_meta_key_by_index(const struct lfg_adapter_lora * adapter, int32_t i, char * buf, size_t buf_size);

    // Get metadata value as a string by index
    LFG_API int32_t lfg_adapter_meta_val_str_by_index(const struct lfg_adapter_lora * adapter, int32_t i, char * buf, size_t buf_size);

    // Manually free a LoRA adapter
    // NOTE: loaded adapters will be free when the associated model is deleted
    LFG_API DEPRECATED(void lfg_adapter_lora_free(struct lfg_adapter_lora * adapter),
            "adapters are now freed together with the associated model");

    // Get the invocation tokens if the current lora is an alora
    LFG_API uint64_t            lfg_adapter_get_alora_n_invocation_tokens(const struct lfg_adapter_lora * adapter);
    LFG_API const lfg_token * lfg_adapter_get_alora_invocation_tokens  (const struct lfg_adapter_lora * adapter);

    // The following functions operate on a lfg_context, hence the naming: lfg_verb_...

    // Add a loaded LoRA adapter to given context
    // This will not modify model's weight
    LFG_API int32_t lfg_set_adapter_lora(
            struct lfg_context * ctx,
            struct lfg_adapter_lora * adapter,
            float scale);

    // Remove a specific LoRA adapter from given context
    // Return -1 if the adapter is not present in the context
    LFG_API int32_t lfg_rm_adapter_lora(
            struct lfg_context * ctx,
            struct lfg_adapter_lora * adapter);

    // Remove all LoRA adapters from given context
    LFG_API void lfg_clear_adapter_lora(struct lfg_context * ctx);

    // Apply a loaded control vector to a lfg_context, or if data is NULL, clear
    // the currently loaded vector.
    // n_embd should be the size of a single layer's control, and data should point
    // to an n_embd x n_layers buffer starting from layer 1.
    // il_start and il_end are the layer range the vector should apply to (both inclusive)
    // See lfg_control_vector_load in common to load a control vector.
    LFG_API int32_t lfg_apply_adapter_cvec(
            struct lfg_context * ctx,
                     const float * data,
                          size_t   len,
                         int32_t   n_embd,
                         int32_t   il_start,
                         int32_t   il_end);

    //
    // Memory
    //

    // Clear the memory contents
    // If data == true, the data buffers will also be cleared together with the metadata
    LFG_API void lfg_memory_clear(
            lfg_memory_t mem,
                      bool data);

    // Removes all tokens that belong to the specified sequence and have positions in [p0, p1)
    // Returns false if a partial sequence cannot be removed. Removing a whole sequence never fails
    // seq_id < 0 : match any sequence
    // p0 < 0     : [0,  p1]
    // p1 < 0     : [p0, inf)
    LFG_API bool lfg_memory_seq_rm(
            lfg_memory_t mem,
              lfg_seq_id seq_id,
                 lfg_pos p0,
                 lfg_pos p1);

    // Copy all tokens that belong to the specified sequence to another sequence
    // p0 < 0 : [0,  p1]
    // p1 < 0 : [p0, inf)
    LFG_API void lfg_memory_seq_cp(
            lfg_memory_t mem,
              lfg_seq_id seq_id_src,
              lfg_seq_id seq_id_dst,
                 lfg_pos p0,
                 lfg_pos p1);

    // Removes all tokens that do not belong to the specified sequence
    LFG_API void lfg_memory_seq_keep(
            lfg_memory_t mem,
              lfg_seq_id seq_id);

    // Adds relative position "delta" to all tokens that belong to the specified sequence and have positions in [p0, p1)
    // p0 < 0 : [0,  p1]
    // p1 < 0 : [p0, inf)
    LFG_API void lfg_memory_seq_add(
            lfg_memory_t mem,
              lfg_seq_id seq_id,
                 lfg_pos p0,
                 lfg_pos p1,
                 lfg_pos delta);

    // Integer division of the positions by factor of `d > 1`
    // p0 < 0 : [0,  p1]
    // p1 < 0 : [p0, inf)
    LFG_API void lfg_memory_seq_div(
            lfg_memory_t mem,
              lfg_seq_id seq_id,
                 lfg_pos p0,
                 lfg_pos p1,
                       int d);

    // Returns the smallest position present in the memory for the specified sequence
    // This is typically non-zero only for SWA caches
    // Note that all positions in the range [pos_min, pos_max] are guaranteed to be present in the memory
    // Return -1 if the sequence is empty
    LFG_API lfg_pos lfg_memory_seq_pos_min(
            lfg_memory_t mem,
              lfg_seq_id seq_id);

    // Returns the largest position present in the memory for the specified sequence
    // Note that all positions in the range [pos_min, pos_max] are guaranteed to be present in the memory
    // Return -1 if the sequence is empty
    LFG_API lfg_pos lfg_memory_seq_pos_max(
            lfg_memory_t mem,
              lfg_seq_id seq_id);

    // Check if the memory supports shifting
    LFG_API bool lfg_memory_can_shift(lfg_memory_t mem);

    //
    // State / sessions
    //

    // Returns the *actual* size in bytes of the state
    // (logits, embedding and memory)
    // Only use when saving the state, not when restoring it, otherwise the size may be too small.
    LFG_API size_t lfg_state_get_size(struct lfg_context * ctx);
    LFG_API DEPRECATED(size_t lfg_get_state_size(struct lfg_context * ctx),
        "use lfg_state_get_size instead");

    // Copies the state to the specified destination address.
    // Destination needs to have allocated enough memory.
    // Returns the number of bytes copied
    LFG_API size_t lfg_state_get_data(
            struct lfg_context * ctx,
                         uint8_t * dst,
                          size_t   size);
    LFG_API DEPRECATED(size_t lfg_copy_state_data(
            struct lfg_context * ctx,
                         uint8_t * dst),
        "use lfg_state_get_data instead");

    // Set the state reading from the specified address
    // Returns the number of bytes read
    LFG_API size_t lfg_state_set_data(
            struct lfg_context * ctx,
                   const uint8_t * src,
                          size_t   size);
    LFG_API DEPRECATED(size_t lfg_set_state_data(
            struct lfg_context * ctx,
                   const uint8_t * src),
        "use lfg_state_set_data instead");

    // Save/load session file
    LFG_API bool lfg_state_load_file(
            struct lfg_context * ctx,
                      const char * path_session,
                     lfg_token * tokens_out,
                          size_t   n_token_capacity,
                          size_t * n_token_count_out);
    LFG_API DEPRECATED(bool lfg_load_session_file(
            struct lfg_context * ctx,
                      const char * path_session,
                     lfg_token * tokens_out,
                          size_t   n_token_capacity,
                          size_t * n_token_count_out),
        "use lfg_state_load_file instead");

    LFG_API bool lfg_state_save_file(
            struct lfg_context * ctx,
                      const char * path_session,
               const lfg_token * tokens,
                          size_t   n_token_count);
    LFG_API DEPRECATED(bool lfg_save_session_file(
            struct lfg_context * ctx,
                      const char * path_session,
               const lfg_token * tokens,
                          size_t   n_token_count),
        "use lfg_state_save_file instead");

    // Get the exact size needed to copy the state of a single sequence
    LFG_API size_t lfg_state_seq_get_size(
            struct lfg_context * ctx,
                    lfg_seq_id   seq_id);

    // Copy the state of a single sequence into the specified buffer
    LFG_API size_t lfg_state_seq_get_data(
            struct lfg_context * ctx,
                         uint8_t * dst,
                          size_t   size,
                    lfg_seq_id   seq_id);

    // Copy the sequence data (originally copied with `lfg_state_seq_get_data`) into the specified sequence
    // Returns:
    //  - Positive: Ok
    //  - Zero: Failed to load
    LFG_API size_t lfg_state_seq_set_data(
            struct lfg_context * ctx,
                   const uint8_t * src,
                          size_t   size,
                    lfg_seq_id   dest_seq_id);

    LFG_API size_t lfg_state_seq_save_file(
            struct lfg_context * ctx,
                      const char * filepath,
                    lfg_seq_id   seq_id,
               const lfg_token * tokens,
                          size_t   n_token_count);

    LFG_API size_t lfg_state_seq_load_file(
            struct lfg_context * ctx,
                      const char * filepath,
                    lfg_seq_id   dest_seq_id,
                     lfg_token * tokens_out,
                          size_t   n_token_capacity,
                          size_t * n_token_count_out);

// for backwards-compat
#define LFG_STATE_SEQ_FLAGS_SWA_ONLY 1

// work only with partial states, such as SWA KV cache or recurrent cache (e.g. Mamba)
#define LFG_STATE_SEQ_FLAGS_PARTIAL_ONLY 1

    typedef uint32_t lfg_state_seq_flags;

    LFG_API size_t lfg_state_seq_get_size_ext(
            struct lfg_context * ctx,
                    lfg_seq_id   seq_id,
           lfg_state_seq_flags   flags);

    LFG_API size_t lfg_state_seq_get_data_ext(
            struct lfg_context * ctx,
                         uint8_t * dst,
                          size_t   size,
                    lfg_seq_id   seq_id,
           lfg_state_seq_flags   flags);

    LFG_API size_t lfg_state_seq_set_data_ext(
            struct lfg_context * ctx,
                   const uint8_t * src,
                          size_t   size,
                    lfg_seq_id   dest_seq_id,
           lfg_state_seq_flags   flags);

    //
    // Decoding
    //

    // Return batch for single sequence of tokens
    // The sequence ID will be fixed to 0
    // The position of the tokens will be tracked automatically by lfg_decode
    //
    // NOTE: this is a helper function to facilitate transition to the new batch API - avoid using it
    //
    LFG_API struct lfg_batch lfg_batch_get_one(
                  lfg_token * tokens,
                      int32_t   n_tokens);

    // Allocates a batch of tokens on the heap that can hold a maximum of n_tokens
    // Each token can be assigned up to n_seq_max sequence ids
    // The batch has to be freed with lfg_batch_free()
    // If embd != 0, lfg_batch.embd will be allocated with size of n_tokens * embd * sizeof(float)
    // Otherwise, lfg_batch.token will be allocated to store n_tokens lfg_token
    // The rest of the lfg_batch members are allocated with size n_tokens
    // All members are left uninitialized
    LFG_API struct lfg_batch lfg_batch_init(
            int32_t n_tokens,
            int32_t embd,
            int32_t n_seq_max);

    // Frees a batch of tokens allocated with lfg_batch_init()
    LFG_API void lfg_batch_free(struct lfg_batch batch);

    // Process a batch of tokens.
    // In contrast to lfg_decode() - this call does not use KV cache.
    // For encode-decoder contexts, processes the batch using the encoder.
    // Can store the encoder output internally for later use by the decoder's cross-attention layers.
    //   0 - success
    // < 0 - error. the memory state is restored to the state before this call
    LFG_API int32_t lfg_encode(
            struct lfg_context * ctx,
              struct lfg_batch   batch);

    // Process a batch of tokens.
    // Requires the context to have a memory.
    // For encode-decoder contexts, processes the batch using the decoder.
    // Positive return values does not mean a fatal error, but rather a warning.
    // Upon fatal-error or abort, the ubatches that managed to be been processed will remain in the memory state of the context
    //   To handle this correctly, query the memory state using lfg_memory_seq_pos_min() and lfg_memory_seq_pos_max()
    // Upon other return values, the memory state is restored to the state before this call
    //    0 - success
    //    1 - could not find a KV slot for the batch (try reducing the size of the batch or increase the context)
    //    2 - aborted     (processed ubatches will remain in the context's memory)
    //   -1 - invalid input batch
    // < -1 - fatal error (processed ubatches will remain in the context's memory)
    LFG_API int32_t lfg_decode(
            struct lfg_context * ctx,
              struct lfg_batch   batch);

    // Set the number of threads used for decoding
    // n_threads is the number of threads used for generation (single token)
    // n_threads_batch is the number of threads used for prompt and batch processing (multiple tokens)
    LFG_API void lfg_set_n_threads(struct lfg_context * ctx, int32_t n_threads, int32_t n_threads_batch);

    // Get the number of threads used for generation of a single token.
    LFG_API int32_t lfg_n_threads(struct lfg_context * ctx);

    // Get the number of threads used for prompt and batch processing (multiple token).
    LFG_API int32_t lfg_n_threads_batch(struct lfg_context * ctx);

    // Set whether the context outputs embeddings or not
    // TODO: rename to avoid confusion with lfg_get_embeddings()
    LFG_API void lfg_set_embeddings(struct lfg_context * ctx, bool embeddings);

    // Set whether to use causal attention or not
    // If set to true, the model will only attend to the past tokens
    LFG_API void lfg_set_causal_attn(struct lfg_context * ctx, bool causal_attn);

    // Set whether the model is in warmup mode or not
    // If true, all model tensors are activated during lfg_decode() to load and cache their weights.
    LFG_API void lfg_set_warmup(struct lfg_context * ctx, bool warmup);

    // Set abort callback
    LFG_API void lfg_set_abort_callback(struct lfg_context * ctx, ggml_abort_callback abort_callback, void * abort_callback_data);

    // Wait until all computations are finished
    // This is automatically done when using one of the functions below to obtain the computation results
    // and is not necessary to call it explicitly in most cases
    LFG_API void lfg_synchronize(struct lfg_context * ctx);

    // Token logits obtained from the last call to lfg_decode()
    // The logits for which lfg_batch.logits[i] != 0 are stored contiguously
    // in the order they have appeared in the batch.
    // Rows: number of tokens for which lfg_batch.logits[i] != 0
    // Cols: n_vocab
    // TODO: deprecate in favor of lfg_get_logits_ith() (ref: https://github.com/ggml-org/liquid.cpp/pull/14853#issuecomment-3113143522)
    LFG_API float * lfg_get_logits(struct lfg_context * ctx);

    // Logits for the ith token. For positive indices, Equivalent to:
    // lfg_get_logits(ctx) + ctx->output_ids[i]*n_vocab
    // Negative indicies can be used to access logits in reverse order, -1 is the last logit.
    // returns NULL for invalid ids.
    LFG_API float * lfg_get_logits_ith(struct lfg_context * ctx, int32_t i);

    // Get all output token embeddings.
    // when pooling_type == LFG_POOLING_TYPE_NONE or when using a generative model,
    // the embeddings for which lfg_batch.logits[i] != 0 are stored contiguously
    // in the order they have appeared in the batch.
    // shape: [n_outputs*n_embd]
    // Otherwise, returns NULL.
    // TODO: deprecate in favor of lfg_get_embeddings_ith() (ref: https://github.com/ggml-org/liquid.cpp/pull/14853#issuecomment-3113143522)
    LFG_API float * lfg_get_embeddings(struct lfg_context * ctx);

    // Get the embeddings for the ith token. For positive indices, Equivalent to:
    // lfg_get_embeddings(ctx) + ctx->output_ids[i]*n_embd
    // Negative indicies can be used to access embeddings in reverse order, -1 is the last embedding.
    // shape: [n_embd] (1-dimensional)
    // returns NULL for invalid ids.
    LFG_API float * lfg_get_embeddings_ith(struct lfg_context * ctx, int32_t i);

    // Get the embeddings for a sequence id
    // Returns NULL if pooling_type is LFG_POOLING_TYPE_NONE
    // when pooling_type == LFG_POOLING_TYPE_RANK, returns float[n_cls_out] with the rank(s) of the sequence
    // otherwise: float[n_embd] (1-dimensional)
    LFG_API float * lfg_get_embeddings_seq(struct lfg_context * ctx, lfg_seq_id seq_id);

    //
    // backend sampling API [EXPERIMENTAL]
    // note: use only if the lfg_context was created with at least one lfg_sampler_seq_config
    //

    // Get the backend sampled token for the ith token.
    // Returns LFG_TOKEN_NULL if no token was sampled.
    LFG_API lfg_token lfg_get_sampled_token_ith(struct lfg_context * ctx, int32_t i);

    // Get the backend sampled probabilites for the ith token
    // The index matches lfg_get_sampled_token_ith().
    // Returns NULL if no probabilites were generated.
    LFG_API float *  lfg_get_sampled_probs_ith      (struct lfg_context * ctx, int32_t i);
    LFG_API uint32_t lfg_get_sampled_probs_count_ith(struct lfg_context * ctx, int32_t i);

    // Get the backend sampled logits for the ith token
    // Returns NULL if no logits were sampled.
    LFG_API float *  lfg_get_sampled_logits_ith      (struct lfg_context * ctx, int32_t i);
    LFG_API uint32_t lfg_get_sampled_logits_count_ith(struct lfg_context * ctx, int32_t i);

    // Get the backend sampled candidates (token ids) for the ith token
    // These are needed to map probability/logit indices to vocab token ids.
    // Returns NULL if no candidates were sampled.
    LFG_API lfg_token * lfg_get_sampled_candidates_ith      (struct lfg_context * ctx, int32_t i);
    LFG_API uint32_t      lfg_get_sampled_candidates_count_ith(struct lfg_context * ctx, int32_t i);

    //
    // Vocab
    //

    LFG_API const char * lfg_vocab_get_text(const struct lfg_vocab * vocab, lfg_token token);

    LFG_API float lfg_vocab_get_score(const struct lfg_vocab * vocab, lfg_token token);

    LFG_API enum lfg_token_attr lfg_vocab_get_attr(const struct lfg_vocab * vocab, lfg_token token);

    // Check if the token is supposed to end generation (end-of-generation, eg. EOS, EOT, etc.)
    LFG_API bool lfg_vocab_is_eog(const struct lfg_vocab * vocab, lfg_token token);

    // Identify if Token Id is a control token or a render-able token
    LFG_API bool lfg_vocab_is_control(const struct lfg_vocab * vocab, lfg_token token);

    // Special tokens
    LFG_API lfg_token lfg_vocab_bos(const struct lfg_vocab * vocab); // beginning-of-sentence
    LFG_API lfg_token lfg_vocab_eos(const struct lfg_vocab * vocab); // end-of-sentence
    LFG_API lfg_token lfg_vocab_eot(const struct lfg_vocab * vocab); // end-of-turn
    LFG_API lfg_token lfg_vocab_sep(const struct lfg_vocab * vocab); // sentence separator
    LFG_API lfg_token lfg_vocab_nl (const struct lfg_vocab * vocab); // next-line
    LFG_API lfg_token lfg_vocab_pad(const struct lfg_vocab * vocab); // padding
    LFG_API lfg_token lfg_vocab_mask(const struct lfg_vocab * vocab); // mask

    LFG_API bool lfg_vocab_get_add_bos(const struct lfg_vocab * vocab);
    LFG_API bool lfg_vocab_get_add_eos(const struct lfg_vocab * vocab);
    LFG_API bool lfg_vocab_get_add_sep(const struct lfg_vocab * vocab);

    LFG_API lfg_token lfg_vocab_fim_pre(const struct lfg_vocab * vocab);
    LFG_API lfg_token lfg_vocab_fim_suf(const struct lfg_vocab * vocab);
    LFG_API lfg_token lfg_vocab_fim_mid(const struct lfg_vocab * vocab);
    LFG_API lfg_token lfg_vocab_fim_pad(const struct lfg_vocab * vocab);
    LFG_API lfg_token lfg_vocab_fim_rep(const struct lfg_vocab * vocab);
    LFG_API lfg_token lfg_vocab_fim_sep(const struct lfg_vocab * vocab);

    DEPRECATED(LFG_API const char * lfg_token_get_text(const struct lfg_vocab * vocab, lfg_token token), "use lfg_vocab_get_text instead");
    DEPRECATED(LFG_API float lfg_token_get_score(const struct lfg_vocab * vocab, lfg_token token), "use lfg_vocab_get_score instead");
    DEPRECATED(LFG_API enum lfg_token_attr lfg_token_get_attr(const struct lfg_vocab * vocab, lfg_token token), "use lfg_vocab_get_attr instead");
    DEPRECATED(LFG_API bool lfg_token_is_eog(const struct lfg_vocab * vocab, lfg_token token), "use lfg_vocab_is_eog instead");
    DEPRECATED(LFG_API bool lfg_token_is_control(const struct lfg_vocab * vocab, lfg_token token), "use lfg_vocab_is_control instead");
    DEPRECATED(LFG_API lfg_token lfg_token_bos(const struct lfg_vocab * vocab), "use lfg_vocab_bos instead");
    DEPRECATED(LFG_API lfg_token lfg_token_eos(const struct lfg_vocab * vocab), "use lfg_vocab_eos instead");
    DEPRECATED(LFG_API lfg_token lfg_token_eot(const struct lfg_vocab * vocab), "use lfg_vocab_eot instead");
    DEPRECATED(LFG_API lfg_token lfg_token_cls(const struct lfg_vocab * vocab), "use lfg_vocab_cls instead");
    DEPRECATED(LFG_API lfg_token lfg_token_sep(const struct lfg_vocab * vocab), "use lfg_vocab_sep instead");
    DEPRECATED(LFG_API lfg_token lfg_token_nl (const struct lfg_vocab * vocab), "use lfg_vocab_nl instead");
    DEPRECATED(LFG_API lfg_token lfg_token_pad(const struct lfg_vocab * vocab), "use lfg_vocab_pad instead");
    DEPRECATED(LFG_API bool lfg_add_bos_token(const struct lfg_vocab * vocab), "use lfg_vocab_get_add_bos instead");
    DEPRECATED(LFG_API bool lfg_add_eos_token(const struct lfg_vocab * vocab), "use lfg_vocab_get_add_eos instead");
    DEPRECATED(LFG_API lfg_token lfg_token_fim_pre(const struct lfg_vocab * vocab), "use lfg_vocab_fim_pre instead");
    DEPRECATED(LFG_API lfg_token lfg_token_fim_suf(const struct lfg_vocab * vocab), "use lfg_vocab_fim_suf instead");
    DEPRECATED(LFG_API lfg_token lfg_token_fim_mid(const struct lfg_vocab * vocab), "use lfg_vocab_fim_mid instead");
    DEPRECATED(LFG_API lfg_token lfg_token_fim_pad(const struct lfg_vocab * vocab), "use lfg_vocab_fim_pad instead");
    DEPRECATED(LFG_API lfg_token lfg_token_fim_rep(const struct lfg_vocab * vocab), "use lfg_vocab_fim_rep instead");
    DEPRECATED(LFG_API lfg_token lfg_token_fim_sep(const struct lfg_vocab * vocab), "use lfg_vocab_fim_sep instead");

    // CLS is equivalent to BOS
    DEPRECATED(LFG_API lfg_token lfg_vocab_cls(const struct lfg_vocab * vocab), // classification
            "use lfg_vocab_bos instead");

    //
    // Tokenization
    //
    // The API is thread-safe.
    //

    /// @details Convert the provided text into tokens.
    /// @param tokens The tokens pointer must be large enough to hold the resulting tokens.
    /// @return Returns the number of tokens on success, no more than n_tokens_max
    /// @return Returns a negative number on failure - the number of tokens that would have been returned
    /// @return Returns INT32_MIN on overflow (e.g., tokenization result size exceeds int32_t limit)
    /// @param add_special Allow to add BOS and EOS tokens if model is configured to do so.
    /// @param parse_special Allow tokenizing special and/or control tokens which otherwise are not exposed and treated
    ///                      as plaintext. Does not insert a leading space.
    LFG_API int32_t lfg_tokenize(
        const struct lfg_vocab * vocab,
                      const char * text,
                         int32_t   text_len,
                     lfg_token * tokens,
                         int32_t   n_tokens_max,
                            bool   add_special,
                            bool   parse_special);

    // Token Id -> Piece.
    // Uses the vocabulary in the provided context.
    // Does not write null terminator to the buffer.
    // User can skip up to 'lstrip' leading spaces before copying (useful when encoding/decoding multiple tokens with 'add_space_prefix')
    // @param special If true, special tokens are rendered in the output.
    LFG_API int32_t lfg_token_to_piece(
              const struct lfg_vocab * vocab,
                           lfg_token   token,
                                  char * buf,
                               int32_t   length,
                               int32_t   lstrip,
                                  bool   special);

    /// @details Convert the provided tokens into text (inverse of lfg_tokenize()).
    /// @param text The char pointer must be large enough to hold the resulting text.
    /// @return Returns the number of chars/bytes on success, no more than text_len_max.
    /// @return Returns a negative number on failure - the number of chars/bytes that would have been returned.
    /// @param remove_special Allow to remove BOS and EOS tokens if model is configured to do so.
    /// @param unparse_special If true, special tokens are rendered in the output.
    LFG_API int32_t lfg_detokenize(
        const struct lfg_vocab * vocab,
               const lfg_token * tokens,
                         int32_t   n_tokens,
                            char * text,
                         int32_t   text_len_max,
                            bool   remove_special,
                            bool   unparse_special);

    //
    // Chat templates
    //

    /// Apply chat template. Inspired by hf apply_chat_template() on python.
    /// Both "model" and "custom_template" are optional, but at least one is required. "custom_template" has higher precedence than "model"
    /// NOTE: This function does not use a jinja parser. It only support a pre-defined list of template. See more: https://github.com/ggml-org/liquid.cpp/wiki/Templates-supported-by-lfg_chat_apply_template
    /// @param tmpl A Jinja template to use for this chat. If this is nullptr, the model’s default chat template will be used instead.
    /// @param chat Pointer to a list of multiple lfg_chat_message
    /// @param n_msg Number of lfg_chat_message in this chat
    /// @param add_ass Whether to end the prompt with the token(s) that indicate the start of an assistant message.
    /// @param buf A buffer to hold the output formatted prompt. The recommended alloc size is 2 * (total number of characters of all messages)
    /// @param length The size of the allocated buffer
    /// @return The total number of bytes of the formatted prompt. If is it larger than the size of buffer, you may need to re-alloc it and then re-apply the template.
    LFG_API int32_t lfg_chat_apply_template(
                            const char * tmpl,
       const struct lfg_chat_message * chat,
                                size_t   n_msg,
                                  bool   add_ass,
                                  char * buf,
                               int32_t   length);

    // Get list of built-in chat templates
    LFG_API int32_t lfg_chat_builtin_templates(const char ** output, size_t len);

    //
    // Sampling API
    //
    // Sample usage:
    //
    //    // prepare the sampling chain at the start
    //    auto sparams = lfg_sampler_chain_default_params();
    //
    //    lfg_sampler * smpl = lfg_sampler_chain_init(sparams);
    //
    //    lfg_sampler_chain_add(smpl, lfg_sampler_init_top_k(50));
    //    lfg_sampler_chain_add(smpl, lfg_sampler_init_top_p(0.9, 1));
    //    lfg_sampler_chain_add(smpl, lfg_sampler_init_temp (0.8));
    //
    //    // typically, the chain should end with a sampler such as "greedy", "dist" or "mirostat"
    //    // this sampler will be responsible to select the actual token
    //    lfg_sampler_chain_add(smpl, lfg_sampler_init_dist(seed));
    //
    //    ...
    //
    //    // decoding loop:
    //    while (...) {
    //        ...
    //
    //        lfg_decode(ctx, batch);
    //
    //        // sample from the logits of the last token in the batch
    //        const lfg_token id = lfg_sampler_sample(smpl, ctx, -1);
    //
    //        ...
    //    }
    //
    //    lfg_sampler_free(smpl);
    //

    typedef void * lfg_sampler_context_t;

    struct lfg_sampler_data {
        struct ggml_tensor * logits;
        struct ggml_tensor * probs;
        struct ggml_tensor * sampled;
        struct ggml_tensor * candidates;
    };

    // user code can implement the interface below in order to create custom lfg_sampler
    struct lfg_sampler_i {
        const char *           (*name)  (const struct lfg_sampler * smpl);                                 // can be NULL
        void                   (*accept)(      struct lfg_sampler * smpl, lfg_token token);              // can be NULL
        void                   (*apply) (      struct lfg_sampler * smpl, lfg_token_data_array * cur_p); // required
        void                   (*reset) (      struct lfg_sampler * smpl);                                 // can be NULL
        struct lfg_sampler * (*clone) (const struct lfg_sampler * smpl);                                 // can be NULL if ctx is NULL
        void                   (*free)  (      struct lfg_sampler * smpl);                                 // can be NULL if ctx is NULL

        // [EXPERIMENTAL]
        // backend sampling interface:

        // return true if the backend supports all ops needed by the sampler
        // note: call once per sampler
        bool (*backend_init)(struct lfg_sampler * smpl, ggml_backend_buffer_type_t buft);

        // call after .backend_apply()
        void (*backend_accept)(
                struct lfg_sampler * smpl,
                struct ggml_context  * ctx,
                struct ggml_cgraph   * gf,
                struct ggml_tensor   * selected_token);

        // call after .backend_init()
        void (*backend_apply)(
                struct lfg_sampler      * smpl,
                struct ggml_context       * ctx,
                struct ggml_cgraph        * gf,
                struct lfg_sampler_data * data);

        // called before graph execution to set inputs for the current ubatch
        void (*backend_set_input)(struct lfg_sampler * smpl);
    };

    struct lfg_sampler {
        struct lfg_sampler_i * iface;

        lfg_sampler_context_t ctx;
    };

    // [EXPERIMENTAL]
    // attach a sampler to the context
    // note: prefer initializing the context with lfg_context_params.samplers when possible
    LFG_API bool lfg_set_sampler(struct lfg_context * ctx, lfg_seq_id seq_id, struct lfg_sampler * smpl);

    // mirror of lfg_sampler_i:
    LFG_API struct lfg_sampler * lfg_sampler_init  (      struct lfg_sampler_i * iface, lfg_sampler_context_t ctx);
    LFG_API const char *           lfg_sampler_name  (const struct lfg_sampler * smpl);
    LFG_API void                   lfg_sampler_accept(      struct lfg_sampler * smpl, lfg_token token);
    LFG_API void                   lfg_sampler_apply (      struct lfg_sampler * smpl, lfg_token_data_array * cur_p);
    LFG_API void                   lfg_sampler_reset (      struct lfg_sampler * smpl);
    LFG_API struct lfg_sampler * lfg_sampler_clone (const struct lfg_sampler * smpl);
    // important: do not free if the sampler has been added to a lfg_sampler_chain (via lfg_sampler_chain_add)
    LFG_API void                   lfg_sampler_free  (      struct lfg_sampler * smpl);

    // lfg_sampler_chain
    // a type of lfg_sampler that can chain multiple samplers one after another

    LFG_API struct lfg_sampler * lfg_sampler_chain_init(struct lfg_sampler_chain_params params);

    // important: takes ownership of the sampler object and will free it when lfg_sampler_free is called
    LFG_API void                   lfg_sampler_chain_add(      struct lfg_sampler * chain, struct lfg_sampler * smpl);

    // return NULL if:
    //   - the sampler is NULL
    //   - the sampler is not a lfg_sampler_chain
    //   - the index is out of bounds, unless i == -1
    //   - if i == -1, returns the chain itself (can be used to check if the sampler is a chain)
    LFG_API struct lfg_sampler * lfg_sampler_chain_get(      struct lfg_sampler * chain, int32_t i);

    // the total number of samplers in the chain
    LFG_API int                    lfg_sampler_chain_n  (const struct lfg_sampler * chain);

    // after removing a sampler, the chain will no longer own it, and it will not be freed when the chain is freed
    LFG_API struct lfg_sampler * lfg_sampler_chain_remove(   struct lfg_sampler * chain, int32_t i);

    // available samplers:

    LFG_API struct lfg_sampler * lfg_sampler_init_greedy(void);

    /// seed == LFG_DEFAULT_SEED to use a random seed.
    LFG_API struct lfg_sampler * lfg_sampler_init_dist(uint32_t seed);

    /// @details Top-K sampling described in academic paper "The Curious Case of Neural Text Degeneration" https://arxiv.org/abs/1904.09751
    /// Setting k <= 0 makes this a noop
    LFG_API struct lfg_sampler * lfg_sampler_init_top_k      (int32_t k);

    /// @details Nucleus sampling described in academic paper "The Curious Case of Neural Text Degeneration" https://arxiv.org/abs/1904.09751
    LFG_API struct lfg_sampler * lfg_sampler_init_top_p      (float   p, size_t min_keep);

    /// @details Minimum P sampling as described in https://github.com/ggml-org/liquid.cpp/pull/3841
    LFG_API struct lfg_sampler * lfg_sampler_init_min_p      (float   p, size_t min_keep);

    /// @details Locally Typical Sampling implementation described in the paper https://arxiv.org/abs/2202.00666.
    LFG_API struct lfg_sampler * lfg_sampler_init_typical    (float   p, size_t min_keep);

    /// #details Updates the logits l_i` = l_i/t. When t <= 0.0f, the maximum logit is kept at it's original value, the rest are set to -inf
    LFG_API struct lfg_sampler * lfg_sampler_init_temp       (float   t);

    /// @details Dynamic temperature implementation (a.k.a. entropy) described in the paper https://arxiv.org/abs/2309.02772.
    LFG_API struct lfg_sampler * lfg_sampler_init_temp_ext   (float   t, float   delta, float exponent);

    /// @details XTC sampler as described in https://github.com/oobabooga/text-generation-webui/pull/6335
    LFG_API struct lfg_sampler * lfg_sampler_init_xtc        (float   p, float   t,     size_t min_keep, uint32_t seed);

    /// @details Top n sigma sampling as described in academic paper "Top-nσ: Not All Logits Are You Need" https://arxiv.org/pdf/2411.07641
    LFG_API struct lfg_sampler * lfg_sampler_init_top_n_sigma(float   n);

    /// @details Mirostat 1.0 algorithm described in the paper https://arxiv.org/abs/2007.14966. Uses tokens instead of words.
    /// @param candidates A vector of `lfg_token_data` containing the candidate tokens, their probabilities (p), and log-odds (logit) for the current position in the generated text.
    /// @param tau  The target cross-entropy (or surprise) value you want to achieve for the generated text. A higher value corresponds to more surprising or less predictable text, while a lower value corresponds to less surprising or more predictable text.
    /// @param eta The learning rate used to update `mu` based on the error between the target and observed surprisal of the sampled word. A larger learning rate will cause `mu` to be updated more quickly, while a smaller learning rate will result in slower updates.
    /// @param m The number of tokens considered in the estimation of `s_hat`. This is an arbitrary value that is used to calculate `s_hat`, which in turn helps to calculate the value of `k`. In the paper, they use `m = 100`, but you can experiment with different values to see how it affects the performance of the algorithm.
    /// @param mu Maximum cross-entropy. This value is initialized to be twice the target cross-entropy (`2 * tau`) and is updated in the algorithm based on the error between the target and observed surprisal.
    LFG_API struct lfg_sampler * lfg_sampler_init_mirostat(
                             int32_t   n_vocab,
                            uint32_t   seed,
                               float   tau,
                               float   eta,
                             int32_t   m);

    /// @details Mirostat 2.0 algorithm described in the paper https://arxiv.org/abs/2007.14966. Uses tokens instead of words.
    /// @param candidates A vector of `lfg_token_data` containing the candidate tokens, their probabilities (p), and log-odds (logit) for the current position in the generated text.
    /// @param tau  The target cross-entropy (or surprise) value you want to achieve for the generated text. A higher value corresponds to more surprising or less predictable text, while a lower value corresponds to less surprising or more predictable text.
    /// @param eta The learning rate used to update `mu` based on the error between the target and observed surprisal of the sampled word. A larger learning rate will cause `mu` to be updated more quickly, while a smaller learning rate will result in slower updates.
    /// @param mu Maximum cross-entropy. This value is initialized to be twice the target cross-entropy (`2 * tau`) and is updated in the algorithm based on the error between the target and observed surprisal.
    LFG_API struct lfg_sampler * lfg_sampler_init_mirostat_v2(
                            uint32_t   seed,
                               float   tau,
                               float   eta);

    /// @details Intializes a GBNF grammar, see grammars/README.md for details.
    /// @param vocab The vocabulary that this grammar will be used with.
    /// @param grammar_str The production rules for the grammar, encoded as a string. Returns an empty grammar if empty. Returns NULL if parsing of grammar_str fails.
    /// @param grammar_root The name of the start symbol for the grammar.
    LFG_API struct lfg_sampler * lfg_sampler_init_grammar(
            const struct lfg_vocab * vocab,
                          const char * grammar_str,
                          const char * grammar_root);


    DEPRECATED(LFG_API struct lfg_sampler * lfg_sampler_init_grammar_lazy(
            const struct lfg_vocab * vocab,
                          const char * grammar_str,
                          const char * grammar_root,
                         const char ** trigger_words,
                                size_t num_trigger_words,
                   const lfg_token * trigger_tokens,
                                size_t num_trigger_tokens),
        "use lfg_sampler_init_grammar_lazy_patterns instead");


    /// @details Lazy grammar sampler, introduced in https://github.com/ggml-org/liquid.cpp/pull/9639
    /// @param trigger_patterns A list of patterns that will trigger the grammar sampler. Pattern will be matched from the start of the generation output, and grammar sampler will be fed content starting from its first match group.
    /// @param trigger_tokens A list of tokens that will trigger the grammar sampler. Grammar sampler will be fed content starting from the trigger token included.
    LFG_API struct lfg_sampler * lfg_sampler_init_grammar_lazy_patterns(
        const struct lfg_vocab * vocab,
                      const char * grammar_str,
                      const char * grammar_root,
                     const char ** trigger_patterns,
                            size_t num_trigger_patterns,
               const lfg_token * trigger_tokens,
                            size_t num_trigger_tokens);

    /// @details Reasoning gate sampler. Enables/disables the wrapped sampler based on reasoning state.
    /// @param wrapped_sampler The sampler to gate (e.g. grammar sampler). Takes ownership.
    /// @param start_tokens The sequence of tokens that start a reasoning block.
    /// @param n_start The number of tokens in the start sequence.
    /// @param end_tokens The sequence of tokens that end a reasoning block.
    /// @param n_end The number of tokens in the end sequence.
    LFG_API struct lfg_sampler * lfg_sampler_init_reasoning_gate(
            struct lfg_sampler * wrapped_sampler,
            const lfg_token * start_tokens,
            size_t n_start,
            const lfg_token * end_tokens,
            size_t n_end);


    /// NOTE: Avoid using on the full vocabulary as searching for repeated tokens can become slow. For example, apply top-k or top-p sampling first.
    LFG_API struct lfg_sampler * lfg_sampler_init_penalties(
                             int32_t   penalty_last_n,   // last n tokens to penalize (0 = disable penalty, -1 = context size)
                               float   penalty_repeat,   // 1.0 = disabled
                               float   penalty_freq,     // 0.0 = disabled
                               float   penalty_present); // 0.0 = disabled

    ///  @details DRY sampler, designed by p-e-w, as described in: https://github.com/oobabooga/text-generation-webui/pull/5677, porting Koboldcpp implementation authored by pi6am: https://github.com/LostRuins/koboldcpp/pull/982
    LFG_API struct lfg_sampler * lfg_sampler_init_dry(
            const struct lfg_vocab *  vocab,
                             int32_t    n_ctx_train,
                               float    dry_multiplier,
                               float    dry_base,
                             int32_t    dry_allowed_length,
                             int32_t    dry_penalty_last_n,
                          const char ** seq_breakers,
                              size_t    num_breakers);

    /// adaptive-p: select tokens near a configurable target probability over time.
    ///
    /// the adaptive-p sampler transforms the token probability distribution to favor tokens
    /// that fall near a user-configurable probability target.
    ///
    /// internally, the sampler maintains an exponential moving average of the *ORIGINAL*
    /// probabilities of selected tokens at each sampling step. it uses this EMA to compute an
    /// adapted target probability at each sampling step, thus maintaining the desired target
    /// probability over time.
    ///
    /// adaptive-p selects a token ID rather than just mutating candidates, so it must be last
    /// in the sampler chain (like mirostat, dist, greedy).
    ///
    /// only mild truncation before this sampler is recommended. we suggest applying min-p
    /// before adaptive-p as the only other active sampler in the chain.
    ///
    /// @param target select tokens near this probability (valid range 0.0 to 1.0; negative = disabled)
    /// @param decay  EMA decay for adaptation; history ≈ 1/(1-decay) tokens (valid range 0.0 - 0.99)
    /// @param seed   RNG seed
    ///
    /// ref: https://github.com/ggml-org/liquid.cpp/pull/17927
    ///
    LFG_API struct lfg_sampler * lfg_sampler_init_adaptive_p(
                               float   target,
                               float   decay,
                            uint32_t   seed);

    LFG_API struct lfg_sampler * lfg_sampler_init_logit_bias(
                             int32_t   n_vocab,
                             int32_t   n_logit_bias,
              const lfg_logit_bias * logit_bias);

    // this sampler is meant to be used for fill-in-the-middle infilling
    // it's supposed to be used after top_k + top_p sampling
    //
    // 1. if the sum of the EOG probs times the number of candidates is higher than the sum of the other probs -> pick EOG
    // 2. combine probs of tokens that have the same prefix
    //
    // example:
    //
    // - before:
    //   "hel":   0.5
    //   "hell":  0.2
    //   "hello": 0.1
    //   "dummy": 0.1
    //
    // - after:
    //   "hel":   0.8
    //   "dummy": 0.1
    //
    // 3. discard non-EOG tokens with low prob
    // 4. if no tokens are left -> pick EOT
    //
    LFG_API struct lfg_sampler * lfg_sampler_init_infill(const struct lfg_vocab * vocab);

    // Sampler that masks out tokens that do not match the given prefix.
    // Useful for token healing or forcing a specific completion.
    LFG_API struct lfg_sampler * lfg_sampler_init_prefix(
            const struct lfg_vocab * vocab,
            const char * prefix);

    // Update the prefix for an existing prefix sampler.
    // Use empty string to disable the prefix constraint.
    LFG_API void lfg_sampler_prefix_set(
            struct lfg_sampler * smpl,
            const char * prefix);

    // Returns the seed used by the sampler if applicable, LFG_DEFAULT_SEED otherwise
    LFG_API uint32_t lfg_sampler_get_seed(const struct lfg_sampler * smpl);

    /// @details Sample and accept a token from the idx-th output of the last evaluation
    //
    // Shorthand for:
    //    const auto * logits = lfg_get_logits_ith(ctx, idx);
    //    lfg_token_data_array cur_p = { ... init from logits ... };
    //    lfg_sampler_apply(smpl, &cur_p);
    //    auto token = cur_p.data[cur_p.selected].id;
    //    lfg_sampler_accept(smpl, token);
    //    return token;
    // Returns the sampled token
    LFG_API lfg_token lfg_sampler_sample(struct lfg_sampler * smpl, struct lfg_context * ctx, int32_t idx);

    // TODO: extend in the future
    //LFG_API void lfg_decode_with_sampler(struct lfg_context * ctx, struct lfg_sampler * smpl, struct lfg_batch batch, ...);

    //
    // Model split
    //

    /// @details Build a split GGUF final path for this chunk.
    ///          lfg_split_path(split_path, sizeof(split_path), "/models/ggml-model-q4_0", 2, 4) => split_path = "/models/ggml-model-q4_0-00002-of-00004.gguf"
    //  Returns the split_path length.
    LFG_API int lfg_split_path(char * split_path, size_t maxlen, const char * path_prefix, int split_no, int split_count);

    /// @details Extract the path prefix from the split_path if and only if the split_no and split_count match.
    ///          lfg_split_prefix(split_prefix, 64, "/models/ggml-model-q4_0-00002-of-00004.gguf", 2, 4) => split_prefix = "/models/ggml-model-q4_0"
    //  Returns the split_prefix length.
    LFG_API int lfg_split_prefix(char * split_prefix, size_t maxlen, const char * split_path, int split_no, int split_count);

    // Print system information
    LFG_API const char * lfg_print_system_info(void);

    // Set callback for all future logging events.
    // If this is not called, or NULL is supplied, everything is output on stderr.
    // The logger state is global so these functions are NOT thread safe.
    LFG_API void lfg_log_get(ggml_log_callback * log_callback, void ** user_data);
    LFG_API void lfg_log_set(ggml_log_callback   log_callback, void *  user_data);

    //
    // Performance utils
    //
    // NOTE: Used by liquid.cpp examples/tools, avoid using in third-party apps. Instead, do your own performance measurements.
    //

    struct lfg_perf_context_data {
        // ms == milliseconds
        double t_start_ms;  // absolute start time
        double t_load_ms;   // time needed for loading the model
        double t_p_eval_ms; // time needed for processing the prompt
        double t_eval_ms;   // time needed for generating tokens

        int32_t n_p_eval;   // number of prompt tokens
        int32_t n_eval;     // number of generated tokens
        int32_t n_reused;   // number of times a ggml compute graph had been reused
    };

    struct lfg_perf_sampler_data {
        double t_sample_ms; // time needed for sampling in ms

        int32_t n_sample;   // number of sampled tokens
    };

    LFG_API struct lfg_perf_context_data lfg_perf_context      (const struct lfg_context * ctx);
    LFG_API void                           lfg_perf_context_print(const struct lfg_context * ctx);
    LFG_API void                           lfg_perf_context_reset(      struct lfg_context * ctx);

    // NOTE: the following work only with samplers constructed via lfg_sampler_chain_init
    LFG_API struct lfg_perf_sampler_data lfg_perf_sampler      (const struct lfg_sampler * chain);
    LFG_API void                           lfg_perf_sampler_print(const struct lfg_sampler * chain);
    LFG_API void                           lfg_perf_sampler_reset(      struct lfg_sampler * chain);

    // print a breakdown of per-device memory use via LFG_LOG:
    LFG_API void lfg_memory_breakdown_print(const struct lfg_context * ctx);

    //
    // training
    //

    // function that returns whether or not a given tensor contains trainable parameters
    typedef bool (*lfg_opt_param_filter)(const struct ggml_tensor * tensor, void * userdata);

    // always returns true
    LFG_API bool lfg_opt_param_filter_all(const struct ggml_tensor * tensor, void * userdata);

    struct lfg_opt_params {
        uint32_t n_ctx_train; // assumed context size post training, use context size specified in lfg_context if 0

        lfg_opt_param_filter param_filter; // callback for determining which tensors contain trainable parameters
        void * param_filter_ud;              // userdata for determining which tensors contain trainable parameters

        ggml_opt_get_optimizer_params get_opt_pars; // callback for calculating optimizer parameters
        void * get_opt_pars_ud;                     // userdata for calculating optimizer parameters

        enum ggml_opt_optimizer_type optimizer_type;
    };

    LFG_API void lfg_opt_init(struct lfg_context * lctx, struct lfg_model * model, struct lfg_opt_params lopt_params);

    LFG_API void lfg_opt_epoch(
            struct lfg_context    * lctx,
            ggml_opt_dataset_t        dataset,
            ggml_opt_result_t         result_train,
            ggml_opt_result_t         result_eval,
            int64_t                   idata_split,
            ggml_opt_epoch_callback   callback_train,
            ggml_opt_epoch_callback   callback_eval);

#ifdef __cplusplus
}
#endif

#endif // LFG_H
