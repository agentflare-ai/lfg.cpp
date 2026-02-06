#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"

#include "../inference/inference_core.h"
#include "../inference/lfm_inference.h"
#include "../inference/lfm_model.h"
#include "../loader/model_loader.h"

#include <nlohmann/json.hpp>
#include <spdlog/spdlog.h>

#include <fstream>
#include <regex>
#include <set>
#include <string>
#include <vector>

using namespace liquid;
using json = nlohmann::json;

namespace {

struct GenerationResult {
    std::string text;
    std::vector<lfm_token> tokens;
};

const std::string kChatToolSchema = R"({
  "type": "object",
  "properties": {
    "schema_version": { "const": "3.0" },
    "tool_calls": {
      "type": "array",
      "minItems": 5,
      "maxItems": 5,
      "items": {
        "oneOf": [
          {
            "type": "object",
            "properties": {
              "function": { "const": "get_weather" },
              "parameters": {
                "type": "object",
                "properties": {
                  "location": { "type": "string", "pattern": "^[A-Za-z ]+, [A-Z]{2}$" },
                  "unit": { "enum": ["celsius", "fahrenheit"] },
                  "days": { "type": "integer", "minimum": 1, "maximum": 7 }
                },
                "required": ["location", "unit", "days"],
                "additionalProperties": false
              }
            },
            "required": ["function", "parameters"],
            "additionalProperties": false
          },
          {
            "type": "object",
            "properties": {
              "function": { "const": "calculator" },
              "parameters": {
                "type": "object",
                "properties": {
                  "operation": { "enum": ["add", "subtract", "multiply", "divide"] },
                  "operands": {
                    "type": "array",
                    "items": { "type": "number" },
                    "minItems": 2,
                    "maxItems": 4
                  },
                  "rounding": { "enum": ["none", "floor", "ceil"] }
                },
                "required": ["operation", "operands", "rounding"],
                "additionalProperties": false
              }
            },
            "required": ["function", "parameters"],
            "additionalProperties": false
          },
          {
            "type": "object",
            "properties": {
              "function": { "const": "search_docs" },
              "parameters": {
                "type": "object",
                "properties": {
                  "query": { "type": "string", "minLength": 6 },
                  "top_k": { "type": "integer", "minimum": 1, "maximum": 5 },
                  "filters": {
                    "type": "object",
                    "properties": {
                      "tags": {
                        "type": "array",
                        "items": { "type": "string", "minLength": 2 },
                        "minItems": 1,
                        "maxItems": 3
                      },
                      "since": { "type": "string", "pattern": "^[2][0-9]{3}-[0-9]{2}-[0-9]{2}$" }
                    },
                    "required": ["tags", "since"],
                    "additionalProperties": false
                  }
                },
                "required": ["query", "top_k", "filters"],
                "additionalProperties": false
              }
            },
            "required": ["function", "parameters"],
            "additionalProperties": false
          },
          {
            "type": "object",
            "properties": {
              "function": { "const": "schedule_meeting" },
              "parameters": {
                "type": "object",
                "properties": {
                  "title": { "type": "string", "minLength": 4 },
                  "start_time": { "type": "string" },
                  "duration_minutes": { "type": "integer", "minimum": 15, "maximum": 120 },
                  "attendees": {
                    "type": "array",
                    "items": {
                      "type": "object",
                      "properties": {
                        "name": { "type": "string", "minLength": 2 },
                        "email": { "type": "string", "minLength": 5 }
                      },
                      "required": ["name", "email"],
                      "additionalProperties": false
                    },
                    "minItems": 1,
                    "maxItems": 4
                  }
                },
                "required": ["title", "start_time", "duration_minutes", "attendees"],
                "additionalProperties": false
              }
            },
            "required": ["function", "parameters"],
            "additionalProperties": false
          },
          {
            "type": "object",
            "properties": {
              "function": { "const": "summarize_alert" },
              "parameters": {
                "type": "object",
                "properties": {
                  "alert_id": { "type": "string", "pattern": "^[A-Z]{2}-[0-9]{4}$" },
                  "severity": { "enum": ["low", "medium", "high"] },
                  "include_timeline": { "type": "boolean" }
                },
                "required": ["alert_id", "severity", "include_timeline"],
                "additionalProperties": false
              }
            },
            "required": ["function", "parameters"],
            "additionalProperties": false
          }
        ]
      }
    }
  },
  "required": ["schema_version", "tool_calls"],
  "additionalProperties": false
})";

const std::string kSingleToolSchema = R"({
  "type": "object",
  "properties": {
    "function": { "const": "send_email" },
    "parameters": {
      "type": "object",
      "properties": {
        "to": { "type": "string", "minLength": 5 },
        "subject": { "type": "string", "minLength": 4 },
        "body": { "type": "string", "minLength": 8 }
      },
      "required": ["to", "subject", "body"],
      "additionalProperties": false
    }
  },
  "required": ["function", "parameters"],
  "additionalProperties": false
})";

std::string ApplyChatTemplate(const char * tmpl,
                              const std::vector<lfm_chat_message> & messages,
                              bool add_assistant) {
    int32_t required = lfm_chat_apply_template(
        tmpl, messages.data(), messages.size(), add_assistant, nullptr, 0);
    if (required < 0) {
        return {};
    }

    std::string buffer(required + 1, '\0');
    int32_t written = lfm_chat_apply_template(
        tmpl, messages.data(), messages.size(), add_assistant, buffer.data(), (int32_t)buffer.size());
    if (written < 0) {
        return {};
    }
    buffer.resize((size_t)written);
    return buffer;
}

GenerationResult GenerateStructured(InferenceCore & core,
                                    lfm_model * model,
                                    const std::string & schema,
                                    const std::string & prompt,
                                    int max_tokens) {
    core.Reset();
    core.ConfigureStructuredDecoding(schema);

    auto prompt_tokens = model->vocab.tokenize(prompt, true);
    core.IngestTokens(prompt_tokens, false);

    GenerationResult result;
    result.tokens.reserve(max_tokens);

    for (int i = 0; i < max_tokens; ++i) {
        lfm_token token = core.Sample();
        if (token == model->vocab.token_eos()) {
            break;
        }
        result.tokens.push_back(token);
        result.text += model->vocab.token_to_piece(token);
        core.IngestTokens({token}, false);

        if (i > 24) {
            const auto start = result.text.find('{');
            const auto end = result.text.rfind('}');
            if (start != std::string::npos && end != std::string::npos && end > start) {
                try {
                    auto parsed = json::parse(result.text.substr(start, end - start + 1));
                    (void)parsed;
                    break;
                } catch (const std::exception &) {
                }
            }
        }
    }

    return result;
}

json ParseJsonOrFail(const std::string & output) {
    const auto start = output.find('{');
    const auto end = output.rfind('}');
    if (start == std::string::npos || end == std::string::npos || end <= start) {
        throw std::runtime_error("No JSON object found in output");
    }
    const auto slice = output.substr(start, end - start + 1);
    return json::parse(slice);
}

void ValidateSingleTool(const json & obj) {
    REQUIRE(obj.contains("function"));
    REQUIRE(obj.contains("parameters"));
    CHECK(obj["function"] == "send_email");
    const auto & params = obj["parameters"];
    REQUIRE(params.contains("to"));
    REQUIRE(params.contains("subject"));
    REQUIRE(params.contains("body"));
    CHECK(params["to"].get<std::string>().find('@') != std::string::npos);
    CHECK(params["subject"].get<std::string>().size() >= 4);
    CHECK(params["body"].get<std::string>().size() >= 8);
}

} // namespace

TEST_CASE("LFM2.5-1.2B-Thinking Chat Template Tooling") {
    lfm_backend_init();

    const std::string model_path = "models/LFM2.5-1.2B-Thinking-GGUF/LFM2.5-1.2B-Thinking-Q4_K_M.gguf";
    std::ifstream f(model_path);
    if (!f.good()) {
        MESSAGE("Model not found at " << model_path << ". Skipping test.");
        return;
    }

    ModelLoader::ModelConfig load_config;
    load_config.model_path = model_path;
    load_config.n_gpu_layers = 0;

    lfm_model * model = ModelLoader::LoadModel(load_config);
    REQUIRE(model != nullptr);

    InferenceCore::Config config;
    config.n_ctx = 4096;
    config.sampling.temp = 0.0f;
    config.sampling.top_k = 40;
    config.sampling.top_p = 1.0f;
    config.sampling.seed = 12345;

    InferenceCore core(model, config);

    const char * model_template = lfm_model_chat_template(model, nullptr);
    const char * template_name = model_template ? model_template : "chatml";

    SUBCASE("Chat template multi-tool schema (model template)") {
        std::vector<lfm_chat_message> messages = {
            {"system", "You are a tool-using assistant. Return only JSON."},
            {"user", "Call get_weather, calculator, search_docs, schedule_meeting, and summarize_alert. Use realistic values."},
        };

        const std::string prompt = ApplyChatTemplate(template_name, messages, true);
        REQUIRE(!prompt.empty());

        auto result = GenerateStructured(core, model, kChatToolSchema, prompt, 768);
        MESSAGE("Generated JSON: " << result.text);

        json parsed = ParseJsonOrFail(result.text);
        REQUIRE(parsed.contains("schema_version"));
        CHECK(parsed["schema_version"] == "3.0");
        REQUIRE(parsed.contains("tool_calls"));
        REQUIRE(parsed["tool_calls"].is_array());
        CHECK(parsed["tool_calls"].size() == 5);
    }

    SUBCASE("Chat template single tool schema (multi-turn)") {
        std::vector<lfm_chat_message> messages = {
            {"system", "You are a tool-using assistant. Return only JSON."},
            {"user", "Let us draft an email."},
            {"assistant", "Sure. Provide the recipient and message."},
            {"user", "Send an email to alice@example.com with subject 'Status' and a short update."},
        };

        const std::string prompt = ApplyChatTemplate(template_name, messages, true);
        REQUIRE(!prompt.empty());

        auto result = GenerateStructured(core, model, kSingleToolSchema, prompt, 256);
        MESSAGE("Generated JSON: " << result.text);

        json parsed = ParseJsonOrFail(result.text);
        ValidateSingleTool(parsed);
    }

    SUBCASE("Chat template determinism (single-thread)") {
        InferenceCore::Config one_thread = config;
        one_thread.n_threads = 1;
        InferenceCore single_core(model, one_thread);

        std::vector<lfm_chat_message> messages = {
            {"system", "You are a tool-using assistant. Return only JSON."},
            {"user", "Call send_email with a short subject and body."},
        };

        const std::string prompt = ApplyChatTemplate("liquid2-sys", messages, true);
        REQUIRE(!prompt.empty());

        single_core.ConfigureStructuredDecoding(kSingleToolSchema);
        auto tokens = model->vocab.tokenize(prompt, true);
        single_core.IngestTokens(tokens, false);

        auto prefix = single_core.Sample();
        single_core.IngestTokens({prefix}, false);
        auto checkpoint = single_core.CreateCheckpoint();

        GenerationResult a = {};
        for (int i = 0; i < 128; ++i) {
            lfm_token t = single_core.Sample();
            if (t == model->vocab.token_eos()) break;
            a.tokens.push_back(t);
            a.text += model->vocab.token_to_piece(t);
            single_core.IngestTokens({t}, false);
        }

        REQUIRE(single_core.RestoreCheckpoint(checkpoint));

        GenerationResult b = {};
        for (int i = 0; i < 128; ++i) {
            lfm_token t = single_core.Sample();
            if (t == model->vocab.token_eos()) break;
            b.tokens.push_back(t);
            b.text += model->vocab.token_to_piece(t);
            single_core.IngestTokens({t}, false);
        }

        CHECK(a.tokens == b.tokens);
        CHECK(a.text == b.text);
    }

    lfm_model_free(model);
}
