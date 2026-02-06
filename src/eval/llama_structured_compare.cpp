// Compare llama.cpp structured decoding (no healing/checkpoint).
// Emits a JSON report to stdout.

#include "../inference/json_schema_to_grammar.h"
#include <nlohmann/json.hpp>

#include "llama.h"

#include <spdlog/spdlog.h>
#include <iostream>
#include <string>
#include <vector>

using json = nlohmann::ordered_json;

namespace {

const std::string kToolSchema = R"({
  "type": "object",
  "properties": {
    "tool_name": { "const": "get_weather" },
    "parameters": {
      "type": "object",
      "properties": {
        "location": { "type": "string" },
        "unit": { "type": "string", "enum": ["celsius", "fahrenheit"] }
      },
      "required": ["location", "unit"]
    }
  },
  "required": ["tool_name", "parameters"]
})";

const std::string kSystemPrompt =
    "You are a helpful assistant. Respond with a tool call JSON only.";

const std::string kUserPrompt =
    "Call the get_weather tool for Paris in Celsius.";

const std::string kJsonPrefix =
    "{\"tool_name\": \"get_weather\", \"parameters\": {\"location\": \"Paris\", \"unit\": \"cels";

std::string BuildGrammarWithPreamble() {
    auto schema_json = json::parse(kToolSchema);
    std::string grammar = json_schema_to_grammar(schema_json, true);

    size_t root_pos = grammar.find("root ::= ");
    if (root_pos != std::string::npos) {
        grammar.replace(root_pos, 9, "json-root ::= ");
    }

    std::string preamble =
        "root ::= preamble json-root\n"
        "preamble ::= [^\\x7B]*\n";

    return preamble + grammar;
}

bool JsonIsComplete(const std::string& text) {
    int depth = 0;
    bool in_string = false;
    bool escape = false;
    bool saw_object = false;

    for (char c : text) {
        if (escape) {
            escape = false;
            continue;
        }
        if (c == '\\') {
            if (in_string) escape = true;
            continue;
        }
        if (c == '"') {
            in_string = !in_string;
            continue;
        }
        if (in_string) continue;
        if (c == '{') {
            depth++;
            saw_object = true;
        } else if (c == '}') {
            depth--;
            if (depth == 0 && saw_object) {
                return true;
            }
        }
    }
    return false;
}

std::string TokenToPiece(const llama_vocab* vocab, llama_token token) {
    char buf[256];
    int n = llama_token_to_piece(vocab, token, buf, sizeof(buf), 0, false);
    if (n <= 0) return "";
    return std::string(buf, n);
}

} // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <model_path> [n_predict] [seed] [n_threads]\n", argv[0]);
        return 1;
    }

    const std::string model_path = argv[1];
    const int n_predict = argc > 2 ? std::stoi(argv[2]) : 64;
    const uint32_t seed = argc > 3 ? static_cast<uint32_t>(std::stoul(argv[3])) : 42;
    const int n_threads = argc > 4 ? std::stoi(argv[4]) : 4;

    llama_backend_init();

    llama_model_params model_params = llama_model_default_params();
    llama_model* model = llama_model_load_from_file(model_path.c_str(), model_params);
    if (!model) {
        spdlog::error("Failed to load model: {}", model_path);
        return 1;
    }

    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx = 2048;
    ctx_params.n_threads = n_threads;
    ctx_params.n_threads_batch = n_threads;

    llama_context* ctx = llama_init_from_model(model, ctx_params);
    if (!ctx) {
        spdlog::error("Failed to create llama context");
        llama_model_free(model);
        return 1;
    }

    const llama_vocab* vocab = llama_model_get_vocab(model);

    const std::string grammar = BuildGrammarWithPreamble();
    llama_sampler* sampler = llama_sampler_chain_init(llama_sampler_chain_default_params());
    llama_sampler_chain_add(sampler, llama_sampler_init_grammar(vocab, grammar.c_str(), "root"));
    llama_sampler_chain_add(sampler, llama_sampler_init_greedy());

    const char * model_template = llama_model_chat_template(model, nullptr);
    const char * tmpl = model_template ? model_template : "chatml";

    std::vector<llama_chat_message> messages;
    messages.push_back({"system", kSystemPrompt.c_str()});
    messages.push_back({"user", kUserPrompt.c_str()});

    int32_t required = llama_chat_apply_template(
        tmpl, messages.data(), messages.size(), true, nullptr, 0);
    if (required < 0) {
        spdlog::error("Failed to apply chat template");
        llama_sampler_free(sampler);
        llama_free(ctx);
        llama_model_free(model);
        llama_backend_free();
        return 1;
    }

    std::string prompt(required + 1, '\0');
    int32_t written = llama_chat_apply_template(
        tmpl, messages.data(), messages.size(), true, prompt.data(), (int32_t)prompt.size());
    if (written < 0) {
        spdlog::error("Failed to render chat template");
        llama_sampler_free(sampler);
        llama_free(ctx);
        llama_model_free(model);
        llama_backend_free();
        return 1;
    }
    prompt.resize((size_t)written);
    prompt += kJsonPrefix;
    int n_tokens = llama_tokenize(vocab, prompt.c_str(),
                                  static_cast<int32_t>(prompt.size()), nullptr, 0, false, false);
    if (n_tokens < 0) n_tokens = -n_tokens;
    std::vector<llama_token> tokens(n_tokens);
    llama_tokenize(vocab, prompt.c_str(), static_cast<int32_t>(prompt.size()),
                   tokens.data(), n_tokens, false, false);

    bool runtime_error = false;
    std::string runtime_error_msg;
    std::string generated;

    try {
        for (int i = 0; i < n_tokens; ++i) {
            llama_batch batch = llama_batch_get_one(&tokens[i], 1);
            if (llama_decode(ctx, batch)) {
                spdlog::error("Failed to decode prompt token {}", i);
                runtime_error = true;
                runtime_error_msg = "decode prompt failed";
                break;
            }
            llama_sampler_accept(sampler, tokens[i]);
        }

        if (!runtime_error) {
            for (int i = 0; i < n_predict; ++i) {
                llama_token token = llama_sampler_sample(sampler, ctx, -1);
                if (llama_vocab_is_eog(vocab, token)) {
                    break;
                }

                llama_batch batch = llama_batch_get_one(&token, 1);
                if (llama_decode(ctx, batch)) {
                    spdlog::error("Failed to decode generated token {}", i);
                    runtime_error = true;
                    runtime_error_msg = "decode generated failed";
                    break;
                }
                llama_sampler_accept(sampler, token);

                generated += TokenToPiece(vocab, token);
                if (JsonIsComplete(kJsonPrefix + generated)) {
                    break;
                }
            }
        }
    } catch (const std::exception& e) {
        runtime_error = true;
        runtime_error_msg = e.what();
    }

    const std::string full_json_text = kJsonPrefix + generated;
    bool valid_json = true;
    std::string json_error;
    if (runtime_error) {
        valid_json = false;
        json_error = runtime_error_msg;
    } else {
        try {
            auto parsed = json::parse(full_json_text);
            (void)parsed;
        } catch (const std::exception& e) {
            valid_json = false;
            json_error = e.what();
        }
    }

    json report;
    report["engine"] = "llama.cpp";
    report["healed"] = false;
    report["checkpoint_supported"] = false;
    report["checkpoint_match"] = nullptr;
    report["valid_json"] = valid_json;
    report["json_error"] = json_error;
    report["prompt"] = prompt;
    report["chat_template"] = model_template ? json("model") : json("chatml");
    report["runtime_error"] = runtime_error ? json(runtime_error_msg) : json(nullptr);
    report["json_text"] = full_json_text;
    report["generated_suffix"] = generated;

    std::cout << report.dump(2) << std::endl;

    llama_sampler_free(sampler);
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();

    return 0;
}
