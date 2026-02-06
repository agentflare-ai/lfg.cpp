#include <spdlog/spdlog.h>
#include <string>
#include <vector>
#include <chrono>
#include <thread>
#include <filesystem>
#include <fstream>
#include <cstdlib>
#include <sstream>
#include <cstdio>
#include <memory>
#include <stdexcept>
#include <array>

#include "../inference/lfg_api.h"
#include "../inference/lfg_inference.h"
#include <nlohmann/json.hpp>
#include "ggml-backend.h"

// MTMD / Vision includes
#include "../vision/clip.h"
#define STB_IMAGE_IMPLEMENTATION
#include "../vision/stb_image.h"

namespace fs = std::filesystem;
using json = nlohmann::json;

struct BenchmarkResult {
    std::string engine;
    double load_time_ms;
    double eval_time_ms;
    double tps;
    int n_generated;
    std::string command; // Executed command (for baseline)
    std::string output_text;
};

// Forward declarations
std::string RunCommand(const std::string& cmd);
BenchmarkResult RunLFG(const std::string& model_path, const std::string& mmproj_path, const std::string& image_path, int n_predict, int n_threads, int seed, bool cpu_only);
BenchmarkResult RunLFGRawDecode(const std::string& model_path, int n_predict, int n_threads, bool cpu_only);
BenchmarkResult RunLlamaBaseline(const std::string& bin_path, const std::string& model_path, int n_predict, int n_threads, int seed);
BenchmarkResult RunLlamaBench(const std::string& bin_path, const std::string& model_path, int n_predict, int n_threads, bool cpu_only);

std::string RunCommand(const std::string& cmd) {
    spdlog::info("Running command: {}", cmd);
    std::array<char, 128> buffer;
    std::string result;
    std::unique_ptr<FILE, decltype(&pclose)> pipe(popen(cmd.c_str(), "r"), pclose);
    if (!pipe) {
        throw std::runtime_error("popen() failed!");
    }
    while (fgets(buffer.data(), buffer.size(), pipe.get()) != nullptr) {
        result += buffer.data();
    }
    // int ret = pclose(pipe.release()); // optional: check return status
    return result;
}

// In-process LFG benchmark
BenchmarkResult RunLFG(const std::string& model_path, const std::string& mmproj_path, const std::string& image_path, int n_predict, int n_threads, int seed, bool cpu_only) {
    BenchmarkResult res;
    res.engine = "LFG";
    res.n_generated = n_predict;
    res.output_text = "";

    auto start_load = std::chrono::high_resolution_clock::now();

    lfg_session_config config = lfg_session_default_config();
    config.n_threads = n_threads;
    config.n_batch = 2048; // Increased for vision tokens
    config.n_ctx = 4096;
    config.sampling.seed = seed;
    // Use only dist sampler for deterministic comparison testing
    config.sampling.temp = 0.0f;        // greedy
    config.sampling.top_k = 0;          // disabled
    config.sampling.top_p = 1.0f;       // disabled
    config.sampling.min_p = 0.0f;       // disabled
    config.sampling.penalty_repeat = 1.0f; // disabled

    // Load Backends
    if (cpu_only) {
        spdlog::info("Forcing CPU-only mode: Setting GGML_METAL_DISABLE=1");
        setenv("GGML_METAL_DISABLE", "1", 1);
    }

    ggml_backend_load_all();

    // Load Model
    lfg_model_params params = lfg_model_default_params();

    std::vector<ggml_backend_dev_t> devices;
    if (cpu_only) {
        ggml_backend_dev_t cpu_dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
        if (cpu_dev) {
            spdlog::info("Restricting model to CPU backend.");
            devices.push_back(cpu_dev);
            devices.push_back(nullptr);
            params.devices = devices.data();
        } else {
            spdlog::warn("Warning: CPU backend not found!");
        }
    }

    lfg_model* m = lfg_model_load_from_file(model_path.c_str(), params);
    if (!m) throw std::runtime_error("Failed to load model");

    std::unique_ptr<lfg_model, void(*)(lfg_model*)> model(m, lfg_model_free);

    // Vision / Multimodal setup
    std::vector<float> vision_embeddings;
    int n_vision_tokens = 0;

    if (!mmproj_path.empty() && !image_path.empty()) {
        spdlog::info("Loading vision projector: {}", mmproj_path);
        clip_context_params clip_params = {};
        clip_params.use_gpu = !cpu_only;

        // Initializa clip
        auto clip_res = clip_init(mmproj_path.c_str(), clip_params);
        clip_ctx* ctx_clip = clip_res.ctx_v; // Assuming vision model for now
        if (!ctx_clip) {
            throw std::runtime_error("Failed to load clip model");
        }

        spdlog::info("Loading image: {}", image_path);
        int w, h, c;
        unsigned char* data = stbi_load(image_path.c_str(), &w, &h, &c, 3);
        if (!data) {
            throw std::runtime_error("Failed to load image: " + image_path);
        }

        spdlog::info("Image loaded: {}x{} channels={}", w, h, c);

        // Create clip image structure
        auto* img_u8 = clip_image_u8_init();
        clip_build_img_from_pixels(data, w, h, img_u8);
        stbi_image_free(data);

        // Preprocess
        auto* img_res = clip_image_f32_batch_init();
        if (!clip_image_preprocess(ctx_clip, img_u8, img_res)) {
             throw std::runtime_error("Failed to preprocess image");
        }

        // Encode
        // Determine embedding size
        int n_embd_vision = clip_n_mmproj_embd(ctx_clip);
        // We know batch size 1
        clip_image_f32* img_f32 = clip_image_f32_get_img(img_res, 0);
        n_vision_tokens = clip_n_output_tokens(ctx_clip, img_f32);

        vision_embeddings.resize(n_vision_tokens * n_embd_vision);

        spdlog::info("Encoding image... (n_tokens={}, n_embd={})", n_vision_tokens, n_embd_vision);
        if (!clip_image_encode(ctx_clip, n_threads, img_f32, vision_embeddings.data())) {
             throw std::runtime_error("Failed to encode image");
        }

        // Cleanup clip resources
        clip_image_f32_batch_free(img_res);
        clip_image_u8_free(img_u8);
        clip_free(ctx_clip);
    }

    lfg_session *session = lfg_session_create(model.get(), &config);

    if (!vision_embeddings.empty()) {
        // TODO: lfg_session_ingest_embeddings is not yet available in the C API.
        // Vision embedding ingestion requires a raw lfg_decode with embd set in the batch.
        spdlog::warn("Vision embedding ingestion via session C API is not yet supported. Skipping.");
    }

    auto end_load = std::chrono::high_resolution_clock::now();
    res.load_time_ms = std::chrono::duration<double, std::milli>(end_load - start_load).count();

    // Warmup / Prompt
    // Using empty prompt or BOS?
    // Run generation loop

    auto start_eval = std::chrono::high_resolution_clock::now();

    // Ingest BOS
    auto* vocab = lfg_model_get_vocab(model.get());
    lfg_token bos = lfg_vocab_bos(vocab);
    lfg_session_ingest_tokens(session, &bos, 1, true);

    for (int i = 0; i < n_predict; ++i) {
        lfg_token id = lfg_session_sample(session);
        // Ingest the token back (do not update sampler again as Sample() did it)
        if (!lfg_session_ingest_tokens(session, &id, 1, false)) break;

        // Print token
        {
             char buf[256];
             int n = lfg_token_to_piece(vocab, id, buf, sizeof(buf), 0, false);
             if (n < 0) {
                 // Error or buffer too small, try one more time with larger buffer?
                 // Or just skip.
             } else {
                 std::string piece(buf, n);
                 // Using spdlog::info might add newlines/prefixes.
                 // For streaming eval, we might want raw output, but let's use spdlog::info for now.
                 spdlog::info("{}", piece);
                 res.output_text += piece;
             }
        }
    }

    auto end_eval = std::chrono::high_resolution_clock::now();
    res.eval_time_ms = std::chrono::duration<double, std::milli>(end_eval - start_eval).count();
    res.tps = (double)n_predict / (res.eval_time_ms / 1000.0);

    lfg_session_free(session);

    return res;
}

// Raw decode benchmark -- matches llama-bench: lfg_decode + random token, no sampler chain
BenchmarkResult RunLFGRawDecode(const std::string& model_path, int n_predict, int n_threads, bool cpu_only) {
    BenchmarkResult res;
    res.engine = "LFG-raw-decode";
    res.n_generated = n_predict;

    if (cpu_only) {
        setenv("GGML_METAL_DISABLE", "1", 1);
    }
    ggml_backend_load_all();

    lfg_model_params mparams = lfg_model_default_params();
    std::vector<ggml_backend_dev_t> devices;
    if (cpu_only) {
        ggml_backend_dev_t cpu_dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
        if (cpu_dev) {
            devices.push_back(cpu_dev);
            devices.push_back(nullptr);
            mparams.devices = devices.data();
        }
    }

    lfg_model* model = lfg_model_load_from_file(model_path.c_str(), mparams);
    if (!model) throw std::runtime_error("Failed to load model");

    lfg_context_params cparams = lfg_context_default_params();
    cparams.n_ctx = 2048;
    cparams.n_batch = 2048;
    cparams.n_threads = n_threads;
    cparams.n_threads_batch = n_threads;
    cparams.no_perf = false;

    lfg_context* ctx = lfg_init_from_model(model, cparams);
    if (!ctx) { lfg_model_free(model); throw std::runtime_error("Failed to create context"); }

    const auto* vocab = lfg_model_get_vocab(model);
    int32_t n_vocab = lfg_vocab_n_tokens(vocab);
    lfg_token token = lfg_vocab_bos(vocab);

    // Decode BOS first
    lfg_decode(ctx, lfg_batch_get_one(&token, 1));

    // Timed decode loop -- matches llama-bench test_gen exactly
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < n_predict; i++) {
        lfg_decode(ctx, lfg_batch_get_one(&token, 1));
        token = std::rand() % n_vocab;
    }
    auto t1 = std::chrono::high_resolution_clock::now();

    res.eval_time_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    res.tps = (double)n_predict / (res.eval_time_ms / 1000.0);

    lfg_free(ctx);
    lfg_model_free(model);
    return res;
}

// Baseline Runner
BenchmarkResult RunLlamaBaseline(const std::string& bin_path, const std::string& model_path, int n_predict, int n_threads, int seed) {
    BenchmarkResult res;
    res.engine = "llama.cpp";
    res.n_generated = n_predict;

    // Construct command
    // llama-baseline -m <model> -n <n_predict> -t <threads> -s <seed> --ignore-eos
    // We parse stderr for speeds. output looks like:
    // "eval time = ... ms / ... runs   ( ... ms per token, ... tokens per second)"

    std::stringstream cmd;
    cmd << bin_path << " -m " << model_path
        << " -n " << n_predict
        << " -t " << n_threads
        << " -s " << seed
        << " --ignore-eos 2>&1"; // redirect stderr to stdout to capture log

    res.command = cmd.str();
    std::string output = RunCommand(cmd.str());
    spdlog::info("--- Llama Baseline Output ---");
    spdlog::info("{}", output);
    spdlog::info("-----------------------------");

    // Naive parsing
    // Look for "eval time ="
    size_t pos = output.find("eval time =");
    if (pos != std::string::npos) {
        // eval time =    1234.56 ms /    10 runs   (  123.46 ms per token,     8.10 tokens per second)
        // Need to extract the ms (1234.56) and tps (8.10).
        // Skip "eval time ="
        // This is fragile but sufficient for specific eval harness.
        // Assuming strict format from llama.cpp

        // Actually, let's just create a dummy result for now if parsing fails, or assume the user will inspect logs.
        // But the requirement says "record results".
        // I'll try to parse generic floating points after "eval time ="

        // Regex would be ideal, but std::regex?
        // Let's just dump the raw output to file if needed, or parse simply.
        res.eval_time_ms = 0.0;
        res.tps = 0.0;

        // Try parsing manually
        // Finding "tokens per second)" and working backwards?
        // output: "... ( 123.46 ms per token, 8.10 tokens per second)"
        size_t tps_pos = output.rfind(" tokens per second");
        if (tps_pos != std::string::npos) {
             // Find last comma before that
             size_t comma_pos = output.rfind(",", tps_pos);
             if (comma_pos != std::string::npos) {
                 std::string tps_str = output.substr(comma_pos + 1, tps_pos - (comma_pos + 1));
                 res.tps = std::stod(tps_str);
             }
        }
    }

    // load time
    // "load time = ..."
    // ...
    res.load_time_ms = 0; // Skip for now unless strictly needed

    return res;
}

// Run llama-bench for comparison
BenchmarkResult RunLlamaBench(const std::string& bin_path, const std::string& model_path, int n_predict, int n_threads, bool cpu_only) {
    BenchmarkResult res;
    res.engine = "llama-bench";
    res.n_generated = n_predict;

    std::stringstream cmd;
    cmd << bin_path << " -m " << model_path
        << " -n " << n_predict
        << " -t " << n_threads
        << " -p 1";
    if (cpu_only) {
        cmd << " -ngl 0";
    }
    cmd << " 2>&1";

    res.command = cmd.str();
    std::string output = RunCommand(cmd.str());
    spdlog::info("--- llama-bench Output ---");
    spdlog::info("{}", output);
    spdlog::info("-----------------------------");

    // Parse output for TPS (token generation line: tg<N>)
    // Format: "| lfm2 350M Q4_K - Medium | ... | tg128 |        328.61 +/- 2.52 |"
    size_t tg_pos = output.find("tg" + std::to_string(n_predict));
    if (tg_pos != std::string::npos) {
        // Find the pipe after tg128, then extract the TPS value
        size_t pos = output.find("|", tg_pos);
        if (pos != std::string::npos) {
            pos++; // Skip the pipe
            // Skip whitespace
            while (pos < output.size() && std::isspace(output[pos])) pos++;
            // Extract the number (e.g., "328.61")
            std::string tps_str;
            while (pos < output.size() && (std::isdigit(output[pos]) || output[pos] == '.')) {
                tps_str += output[pos];
                pos++;
            }
            if (!tps_str.empty()) {
                res.tps = std::stod(tps_str);
                res.eval_time_ms = (double)n_predict / res.tps * 1000.0;
            }
        }
    }

    return res;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        spdlog::error("Usage: lfg-eval <model_path> [n_threads] [seed] [--cpu] [--image <image_path>] [--mmproj <mmproj_path>]");
        return 1;
    }

    std::string model_path = argv[1];
    int n_threads = 4;
    int seed = 1337;
    bool force_cpu = false;
    std::string image_path;
    std::string mmproj_path;

    // improved arg parsing
    for (int i = 2; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--cpu") {
            force_cpu = true;
        } else if (arg == "--image" && i + 1 < argc) {
            image_path = argv[++i];
        } else if (arg == "--mmproj" && i + 1 < argc) {
            mmproj_path = argv[++i];
        } else if (isdigit(arg[0])) {
             // legacy positional args
             // if we haven't seen them yet, assume order: threads, seed
             // This is a bit weak but maintains compat with original "broken" parsing
             // Actually, original code used fixed indices 2 and 3 for threads and seed.
        }
    }

    // Check if threads/seed provided as positional 2 and 3 (simple backward compat check)
    // Only if they look like numbers
    if (argc > 2 && isdigit(argv[2][0])) n_threads = std::stoi(argv[2]);
    if (argc > 3 && isdigit(argv[3][0])) seed = std::stoi(argv[3]);

    int n_predict = 128; // Fixed for bench

    fs::create_directories(".snapshot");

    json report;
    report["timestamp"] = std::time(nullptr);
    report["config"] = {
        {"model", model_path},
        {"mmproj", mmproj_path},
        {"image", image_path},
        {"threads", n_threads},
        {"seed", seed},
        {"n_predict", n_predict},
        {"cpu_only", force_cpu},
        {"args", std::vector<std::string>(argv, argv + argc)}
    };

    // Raw decode benchmark (matches llama-bench: decode + random token, no sampler)
    spdlog::info("Running LFG raw decode benchmark...");
    try {
        auto r0 = RunLFGRawDecode(model_path, n_predict, n_threads, force_cpu);
        report["results"]["lfg_raw_decode"] = {
            {"tps", r0.tps},
            {"eval_time_ms", r0.eval_time_ms}
        };
        spdlog::info("LFG raw decode TPS: {}", r0.tps);
    } catch (const std::exception& e) {
        spdlog::error("LFG raw decode Failed: {}", e.what());
    }

    spdlog::info("Running LFG Eval...");
    try {
        auto r1 = RunLFG(model_path, mmproj_path, image_path, n_predict, n_threads, seed, force_cpu);
        report["results"]["lfg"] = {
            {"tps", r1.tps},
            {"eval_time_ms", r1.eval_time_ms},
            {"output", r1.output_text}
        };
        spdlog::info("LFG TPS: {}", r1.tps);
    } catch (const std::exception& e) {
        spdlog::error("LFG Failed: {}", e.what());
    }

    // Run llama-bench for performance comparison
    spdlog::info("Running llama-bench comparison...");
    fs::path bench_path = fs::absolute(argv[0]).parent_path() / "llama-bench";
    // Also check in bin/ directory relative to CWD
    if (!fs::exists(bench_path)) {
        bench_path = "bin/llama-bench";
    }
    if (fs::exists(bench_path)) {
        try {
            auto r2 = RunLlamaBench(bench_path.string(), model_path, n_predict, n_threads, force_cpu);
            report["results"]["llama_bench"] = {
                {"tps", r2.tps},
                {"eval_time_ms", r2.eval_time_ms},
                {"command", r2.command}
            };
            spdlog::info("llama-bench TPS: {}", r2.tps);
        } catch (const std::exception& e) {
            spdlog::error("llama-bench Failed: {}", e.what());
        }
    } else {
        spdlog::warn("llama-bench not found at {}", bench_path.string());
    }

    // Save snapshot
    std::string filename = ".snapshot/result_" + std::to_string(std::time(nullptr)) + ".json";
    std::ofstream f(filename);
    f << report.dump(4);
    spdlog::info("Report saved to {}", filename);

    return 0;
}
