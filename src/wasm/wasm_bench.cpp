#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <vector>
#include "lfm_inference.h"
#include "inference_core.h"

int main(int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: wasm_bench <model_path> [n_predict] [n_threads]\n");
        return 1;
    }
    const char* model_path = argv[1];
    int n_predict = (argc > 2) ? atoi(argv[2]) : 128;
    int n_threads = (argc > 3) ? atoi(argv[3]) : 1;

    lfm_model_params params = lfm_model_default_params();
    params.use_mmap = false;   // WASI has no mmap
    params.use_mlock = false;
    params.n_gpu_layers = 0;

    clock_t t0 = clock();
    lfm_model* model = lfm_model_load_from_file(model_path, params);
    double load_ms = (double)(clock() - t0) / CLOCKS_PER_SEC * 1000.0;
    if (!model) {
        fprintf(stderr, "Failed to load model\n");
        return 1;
    }
    printf("Model loaded in %.1f ms\n", load_ms);

    liquid::InferenceCore::Config config;
    config.n_threads = n_threads;
    config.n_batch = 512;
    config.n_ctx = 2048;
    config.sampling.temp = 0.0f;  // Greedy for reproducibility

    liquid::InferenceCore core(model, config);

    const lfm_vocab* vocab = lfm_model_get_vocab(model);
    lfm_token bos = lfm_vocab_bos(vocab);
    core.IngestTokens({bos});

    clock_t t_decode = clock();
    int generated = 0;
    for (int i = 0; i < n_predict; i++) {
        lfm_token id = core.Sample();
        if (lfm_vocab_is_eog(vocab, id)) break;
        if (!core.IngestTokens({id}, false)) break;
        generated++;
    }
    double decode_ms = (double)(clock() - t_decode) / CLOCKS_PER_SEC * 1000.0;
    double tps = (generated > 0 && decode_ms > 0) ? (double)generated / (decode_ms / 1000.0) : 0.0;

    printf("Generated %d tokens in %.1f ms\n", generated, decode_ms);
    printf("Throughput: %.2f tokens/sec\n", tps);

    lfm_model_free(model);
    return 0;
}
