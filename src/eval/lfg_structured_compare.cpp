// Compare LFG structured decoding with healing + checkpointing.
// Emits a JSON report to stdout.

#include "../inference/inference_core.h"
#include "../loader/model_loader.h"
#include "../inference/json_schema_to_grammar.h"
#include <nlohmann/json.hpp>

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

    // Rename root to json-root so we can prepend a preamble rule.
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

std::string TokenToPiece(const lfm_vocab* vocab, lfm_token token) {
    char buf[256];
    int n = lfm_token_to_piece(vocab, token, buf, sizeof(buf), 0, false);
    if (n <= 0) return "";
    return std::string(buf, n);
}

} // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <model_path> [n_predict] [seed] [n_threads] [checkpoint_step]\n", argv[0]);
        return 1;
    }

    const std::string model_path = argv[1];
    const int n_predict = argc > 2 ? std::stoi(argv[2]) : 64;
    const uint32_t seed = argc > 3 ? static_cast<uint32_t>(std::stoul(argv[3])) : 42;
    const int n_threads = argc > 4 ? std::stoi(argv[4]) : 4;
    const int checkpoint_step = argc > 5 ? std::stoi(argv[5]) : 1;

    liquid::ModelLoader::ModelConfig load_config;
    load_config.model_path = model_path;
    load_config.n_gpu_layers = 0;

    auto* model = liquid::ModelLoader::LoadModel(load_config);
    if (!model) {
        spdlog::error("Failed to load model: {}", model_path);
        return 1;
    }

    liquid::InferenceCore::Config config;
    config.n_ctx = 2048;
    config.n_threads = n_threads;
    config.enable_healing = true;
    config.sampling.seed = seed;
    config.sampling.temp = 0.0f;
    config.sampling.top_k = 0;
    config.sampling.top_p = 1.0f;
    config.sampling.min_p = 0.0f;
    config.sampling.penalty_repeat = 1.0f;

    liquid::InferenceCore core(model, config);

    const std::string grammar = BuildGrammarWithPreamble();
    core.ConfigureStructuredDecoding(grammar, "root");

    const char * model_template = lfm_model_chat_template(model, nullptr);
    const char * tmpl = model_template ? model_template : "chatml";

    std::vector<lfm_chat_message> messages;
    messages.push_back({"system", kSystemPrompt.c_str()});
    messages.push_back({"user", kUserPrompt.c_str()});

    int32_t required = lfm_chat_apply_template(
        tmpl, messages.data(), messages.size(), true, nullptr, 0);
    if (required < 0) {
        spdlog::error("Failed to apply chat template");
        liquid::ModelLoader::FreeModel(model);
        return 1;
    }

    std::string prompt(required + 1, '\0');
    int32_t written = lfm_chat_apply_template(
        tmpl, messages.data(), messages.size(), true, prompt.data(), (int32_t)prompt.size());
    if (written < 0) {
        spdlog::error("Failed to render chat template");
        liquid::ModelLoader::FreeModel(model);
        return 1;
    }
    prompt.resize((size_t)written);
    prompt += kJsonPrefix;

    int n_tokens = lfm_tokenize(lfm_model_get_vocab(model), prompt.c_str(),
                                   static_cast<int32_t>(prompt.size()), nullptr, 0, false, false);
    if (n_tokens < 0) n_tokens = -n_tokens;
    std::vector<lfm_token> tokens(n_tokens);
    lfm_tokenize(lfm_model_get_vocab(model), prompt.c_str(),
                    static_cast<int32_t>(prompt.size()), tokens.data(), n_tokens, false, false);

    core.IngestTokens(tokens, true);
    bool healed = core.HealLastToken();

    std::string generated;
    std::string generated_at_checkpoint;
    liquid::InferenceCore::Checkpoint checkpoint;
    bool have_checkpoint = false;

    const lfm_vocab* vocab = lfm_model_get_vocab(model);
    for (int i = 0; i < n_predict; ++i) {
        core.Decode();
        lfm_token token = core.Sample();
        core.IngestTokens({token}, false);
        generated += TokenToPiece(vocab, token);

        if (!have_checkpoint && checkpoint_step > 0 && (i + 1) == checkpoint_step) {
            checkpoint = core.CreateCheckpoint();
            generated_at_checkpoint = generated;
            have_checkpoint = true;
        }

        if (JsonIsComplete(kJsonPrefix + generated)) {
            break;
        }
    }

    bool checkpoint_match = false;
    std::string resumed;
    if (have_checkpoint) {
        if (!core.RestoreCheckpoint(checkpoint)) {
            spdlog::error("Failed to restore checkpoint.");
        } else {
            for (int i = 0; i < n_predict; ++i) {
                core.Decode();
                lfm_token token = core.Sample();
                core.IngestTokens({token}, false);
                resumed += TokenToPiece(vocab, token);
                if (JsonIsComplete(kJsonPrefix + generated_at_checkpoint + resumed)) {
                    break;
                }
            }
            checkpoint_match = (generated == generated_at_checkpoint + resumed);
        }
    }

    const std::string full_json_text = kJsonPrefix + generated;
    bool valid_json = true;
    std::string json_error;
    try {
        auto parsed = json::parse(full_json_text);
        (void)parsed;
    } catch (const std::exception& e) {
        valid_json = false;
        json_error = e.what();
    }

    json report;
    report["engine"] = "lfg";
    report["healed"] = healed;
    report["checkpoint_supported"] = true;
    report["checkpoint_match"] = have_checkpoint ? json(checkpoint_match) : json(nullptr);
    report["valid_json"] = valid_json;
    report["json_error"] = json_error;
    report["prompt"] = prompt;
    report["chat_template"] = model_template ? json("model") : json("chatml");
    report["json_text"] = full_json_text;
    report["generated_suffix"] = generated;

    std::cout << report.dump(2) << std::endl;

    liquid::ModelLoader::FreeModel(model);
    return 0;
}
