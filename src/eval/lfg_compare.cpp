// Minimal liquid text generation for comparison
// Mirrors llama_compare.cpp exactly but uses lfm_* API

#include "lfm_inference.h"
#include "ggml-backend.h"

#include <cstdio>
#include <cstring>
#include <string>
#include <chrono>
#include <cstdlib>
#include <vector>

int main(int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <model_path> [n_predict] [seed] [n_threads] [ngl] [--cpu|--gpu|--igpu] [--quiet] [--no-kv-offload] [--no-op-offload] [--no-flash-attn|--force-flash-attn]\n", argv[0]);
        return 1;
    }

    bool quiet = false;
    bool no_kv_offload = false;
    bool no_op_offload = false;
    bool no_flash_attn = false;
    bool force_flash_attn = false;
    enum class device_pref { auto_select, cpu, gpu, igpu };
    device_pref device = device_pref::auto_select;

    std::vector<std::string> positional;
    positional.reserve(argc);
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--quiet") {
            quiet = true;
            continue;
        }
        if (arg == "--no-kv-offload") {
            no_kv_offload = true;
            continue;
        }
        if (arg == "--no-op-offload") {
            no_op_offload = true;
            continue;
        }
        if (arg == "--no-flash-attn") {
            no_flash_attn = true;
            continue;
        }
        if (arg == "--force-flash-attn") {
            force_flash_attn = true;
            continue;
        }
        if (arg == "--cpu") {
            device = device_pref::cpu;
            continue;
        }
        if (arg == "--gpu") {
            device = device_pref::gpu;
            continue;
        }
        if (arg == "--igpu") {
            device = device_pref::igpu;
            continue;
        }
        positional.push_back(arg);
    }

    if (positional.empty()) {
        fprintf(stderr, "Usage: %s <model_path> [n_predict] [seed] [n_threads] [ngl] [--cpu|--gpu|--igpu] [--quiet] [--no-kv-offload] [--no-op-offload] [--no-flash-attn|--force-flash-attn]\n", argv[0]);
        return 1;
    }

    std::string model_path = positional[0];
    int n_predict = positional.size() > 1 ? std::stoi(positional[1]) : 128;
    uint32_t seed = positional.size() > 2 ? std::stoul(positional[2]) : 42;
    int n_threads = positional.size() > 3 ? std::stoi(positional[3]) : 4;
    int ngl = positional.size() > 4 ? std::stoi(positional[4]) : 0;

    fprintf(stderr, "Model: %s\n", model_path.c_str());
    fprintf(stderr, "n_predict: %d\n", n_predict);
    fprintf(stderr, "seed: %u\n", seed);
    fprintf(stderr, "n_threads: %d\n", n_threads);
    fprintf(stderr, "ngl: %d\n", ngl);
    if (no_flash_attn && force_flash_attn) {
        fprintf(stderr, "Error: cannot use --no-flash-attn and --force-flash-attn together\n");
        return 1;
    }

    ggml_backend_load_all();

    // Model params
    lfm_model_params model_params = lfm_model_default_params();
    model_params.n_gpu_layers = ngl;
    std::vector<ggml_backend_dev_t> devices;
    if (device != device_pref::auto_select) {
        ggml_backend_dev_t dev = nullptr;
        if (device == device_pref::cpu) {
            dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
        } else if (device == device_pref::gpu) {
            dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
        } else {
            dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_IGPU);
        }
        if (dev) {
            devices.push_back(dev);
            devices.push_back(nullptr);
            model_params.devices = devices.data();
        } else {
            fprintf(stderr, "Warning: requested device not found, falling back to default selection\n");
        }
    }

    lfm_model* model = lfm_model_load_from_file(model_path.c_str(), model_params);
    if (!model) {
        fprintf(stderr, "Failed to load model\n");
        return 1;
    }

    const lfm_vocab* vocab = lfm_model_get_vocab(model);

    // Context params
    lfm_context_params ctx_params = lfm_context_default_params();
    ctx_params.n_ctx = 2048;
    ctx_params.n_batch = 512;
    ctx_params.n_threads = n_threads;
    ctx_params.no_perf = false;
    ctx_params.offload_kqv = !no_kv_offload;
    ctx_params.op_offload = !no_op_offload;
    if (no_flash_attn) {
        ctx_params.flash_attn_type = LFM_FLASH_ATTN_TYPE_DISABLED;
    } else if (force_flash_attn) {
        ctx_params.flash_attn_type = LFM_FLASH_ATTN_TYPE_ENABLED;
    }

    fprintf(stderr, "offload_kqv: %s\n", ctx_params.offload_kqv ? "true" : "false");
    fprintf(stderr, "op_offload: %s\n", ctx_params.op_offload ? "true" : "false");
    fprintf(stderr, "flash_attn: %s\n",
            ctx_params.flash_attn_type == LFM_FLASH_ATTN_TYPE_DISABLED ? "disabled" :
            (ctx_params.flash_attn_type == LFM_FLASH_ATTN_TYPE_ENABLED ? "enabled" : "auto"));

    lfm_context* ctx = lfm_init_from_model(model, ctx_params);
    if (!ctx) {
        fprintf(stderr, "Failed to create context\n");
        lfm_model_free(model);
        return 1;
    }

    // Sampler - use greedy for deterministic comparison
    auto sparams = lfm_sampler_chain_default_params();
    lfm_sampler* smpl = lfm_sampler_chain_init(sparams);
    lfm_sampler_chain_add(smpl, lfm_sampler_init_greedy());

    // Start with BOS token
    lfm_token bos = lfm_vocab_bos(vocab);
    lfm_batch batch = lfm_batch_get_one(&bos, 1);

    auto start = std::chrono::high_resolution_clock::now();

    // Decode BOS
    if (lfm_decode(ctx, batch)) {
        fprintf(stderr, "Failed to decode BOS\n");
        lfm_sampler_free(smpl);
        lfm_free(ctx);
        lfm_model_free(model);
        return 1;
    }

    std::string output;
    int n_generated = 0;

    // Generation loop
    for (int i = 0; i < n_predict; ++i) {
        // Get logits before sampling
        float* logits = lfm_get_logits(ctx);
        int n_vocab = lfm_vocab_n_tokens(vocab);

        // Find top 3 logits for debugging
        if (!quiet && i >= 45 && i <= 48) {
            float max1 = -1e9, max2 = -1e9, max3 = -1e9;
            int idx1 = -1, idx2 = -1, idx3 = -1;
            for (int j = 0; j < n_vocab; j++) {
                if (logits[j] > max1) { max3 = max2; idx3 = idx2; max2 = max1; idx2 = idx1; max1 = logits[j]; idx1 = j; }
                else if (logits[j] > max2) { max3 = max2; idx3 = idx2; max2 = logits[j]; idx2 = j; }
                else if (logits[j] > max3) { max3 = logits[j]; idx3 = j; }
            }
            fprintf(stderr, "Logits[%d]: top3=(%d:%.4f, %d:%.4f, %d:%.4f)\n", i, idx1, max1, idx2, max2, idx3, max3);
        }

        lfm_token new_token = lfm_sampler_sample(smpl, ctx, -1);

        if (!quiet) {
            // Debug: print token ID
            fprintf(stderr, "Token[%d]: %d\n", i, new_token);
        }

        if (lfm_vocab_is_eog(vocab, new_token)) {
            break;
        }

        if (!quiet) {
            // Convert token to text
            char buf[256];
            int n = lfm_token_to_piece(vocab, new_token, buf, sizeof(buf), 0, true);
            if (n > 0) {
                output.append(buf, n);
            }
        }

        // Prepare next batch
        batch = lfm_batch_get_one(&new_token, 1);
        if (lfm_decode(ctx, batch)) {
            break;
        }

        n_generated++;
    }

    auto end = std::chrono::high_resolution_clock::now();
    double elapsed_ms = std::chrono::duration<double, std::milli>(end - start).count();
    double tps = n_generated / (elapsed_ms / 1000.0);

    // Output results
    fprintf(stderr, "\n--- Results ---\n");
    fprintf(stderr, "Tokens generated: %d\n", n_generated);
    fprintf(stderr, "Time: %.2f ms\n", elapsed_ms);
    fprintf(stderr, "TPS: %.2f\n", tps);
    fprintf(stderr, "--- Output ---\n");
    if (!quiet) {
        // Print generated text to stdout for easy capture/comparison
        printf("%s\n", output.c_str());
    }

    lfm_sampler_free(smpl);
    lfm_free(ctx);
    lfm_model_free(model);

    return 0;
}
