#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"

#include "../inference/inference_core.h"
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

const std::string kToolSchema = R"({
  "type": "object",
  "properties": {
    "schema_version": { "const": "2.0" },
    "tool_calls": {
      "type": "array",
      "minItems": 4,
      "maxItems": 4,
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
          }
        ]
      }
    }
  },
  "required": ["schema_version", "tool_calls"],
  "additionalProperties": false
})";

const std::string kToolPrompt =
    "Return ONLY a JSON object matching the schema. "
    "Include exactly one tool call each for get_weather, calculator, search_docs, and schedule_meeting. "
    "Use realistic values with ISO-like dates for start_time.";

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

GenerationResult GenerateContinuation(InferenceCore & core,
                                      lfm_model * model,
                                      int max_tokens) {
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

void ValidateToolCall(const json & tool_call) {
    REQUIRE(tool_call.contains("function"));
    REQUIRE(tool_call.contains("parameters"));
    REQUIRE(tool_call["function"].is_string());
    REQUIRE(tool_call["parameters"].is_object());

    const std::string fn = tool_call["function"].get<std::string>();
    const auto & params = tool_call["parameters"];

    if (fn == "get_weather") {
        REQUIRE(params.contains("location"));
        REQUIRE(params.contains("unit"));
        REQUIRE(params.contains("days"));
        CHECK(std::regex_match(params["location"].get<std::string>(),
                               std::regex(R"(^[A-Za-z ]+, [A-Z]{2}$)")));
        CHECK(params["days"].get<int>() >= 1);
        CHECK(params["days"].get<int>() <= 7);
    } else if (fn == "calculator") {
        REQUIRE(params.contains("operation"));
        REQUIRE(params.contains("operands"));
        REQUIRE(params["operands"].is_array());
        CHECK(params["operands"].size() >= 2);
        CHECK(params["operands"].size() <= 4);
    } else if (fn == "search_docs") {
        REQUIRE(params.contains("query"));
        REQUIRE(params.contains("top_k"));
        REQUIRE(params.contains("filters"));
        CHECK(params["query"].get<std::string>().size() >= 6);
        CHECK(params["top_k"].get<int>() >= 1);
        CHECK(params["top_k"].get<int>() <= 5);
        const auto & filters = params["filters"];
        REQUIRE(filters.contains("tags"));
        REQUIRE(filters.contains("since"));
        REQUIRE(filters["tags"].is_array());
        CHECK(filters["tags"].size() >= 1);
        CHECK(filters["tags"].size() <= 3);
        CHECK(std::regex_match(filters["since"].get<std::string>(),
                               std::regex(R"(^[2][0-9]{3}-[0-9]{2}-[0-9]{2}$)")));
    } else if (fn == "schedule_meeting") {
        REQUIRE(params.contains("title"));
        REQUIRE(params.contains("start_time"));
        REQUIRE(params.contains("duration_minutes"));
        REQUIRE(params.contains("attendees"));
        CHECK(params["duration_minutes"].get<int>() >= 15);
        CHECK(params["duration_minutes"].get<int>() <= 120);
        const auto & attendees = params["attendees"];
        REQUIRE(attendees.is_array());
        CHECK(attendees.size() >= 1);
        CHECK(attendees.size() <= 4);
        for (const auto & attendee : attendees) {
            REQUIRE(attendee.contains("name"));
            REQUIRE(attendee.contains("email"));
            CHECK(attendee["name"].get<std::string>().size() >= 2);
            CHECK(attendee["email"].get<std::string>().find('@') != std::string::npos);
        }
    } else {
        FAIL("Unexpected tool function: " << fn);
    }
}

bool ConfigureReasoningFromMarkers(InferenceCore & core, lfm_model * model) {
    auto start_tokens = model->vocab.tokenize("<think>", false);
    auto end_tokens = model->vocab.tokenize("</think>", false);
    if (start_tokens.empty() || end_tokens.empty()) {
        return false;
    }
    core.ConfigureReasoning(start_tokens, end_tokens);
    return true;
}

} // namespace

TEST_CASE("LFM2.5-1.2B-Thinking Exhaustive Tooling + Healing + Checkpointing") {
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
    ConfigureReasoningFromMarkers(core, model);

    SUBCASE("Multi-tool structured decoding (strict schema)") {
        auto result = GenerateStructured(core, model, kToolSchema, kToolPrompt, 768);
        MESSAGE("Generated JSON: " << result.text);

        json parsed = ParseJsonOrFail(result.text);
        REQUIRE(parsed.contains("schema_version"));
        CHECK(parsed["schema_version"] == "2.0");

        REQUIRE(parsed.contains("tool_calls"));
        REQUIRE(parsed["tool_calls"].is_array());
        CHECK(parsed["tool_calls"].size() == 4);

        std::multiset<std::string> functions;
        for (const auto & call : parsed["tool_calls"]) {
            ValidateToolCall(call);
            functions.insert(call["function"].get<std::string>());
        }

        CHECK(functions.count("get_weather") == 1);
        CHECK(functions.count("calculator") == 1);
        CHECK(functions.count("search_docs") == 1);
        CHECK(functions.count("schedule_meeting") == 1);
    }

    SUBCASE("Checkpoint determinism under tool schema") {
        core.Reset();
        core.ConfigureStructuredDecoding(kToolSchema);

        auto prompt_tokens = model->vocab.tokenize(kToolPrompt, true);
        core.IngestTokens(prompt_tokens, false);

        auto prefix = GenerateContinuation(core, model, 80);
        CHECK(!prefix.tokens.empty());

        auto checkpoint = core.CreateCheckpoint();
        auto suffix_a = GenerateContinuation(core, model, 160);
        REQUIRE(core.RestoreCheckpoint(checkpoint));
        auto suffix_b = GenerateContinuation(core, model, 160);

        CHECK(suffix_a.tokens == suffix_b.tokens);
        CHECK(suffix_a.text == suffix_b.text);
    }

    SUBCASE("Token healing + checkpoint on partial tool call") {
        InferenceCore::Config heal_config = config;
        heal_config.enable_healing = true;
        InferenceCore healing_core(model, heal_config);
        ConfigureReasoningFromMarkers(healing_core, model);

        healing_core.ConfigureStructuredDecoding(kToolSchema);

        const std::string prefix = "<think>Deciding to check the weather...\n</think>{\"schema_version\":\"2.0\",\"tool_calls\":[";
        auto prefix_tokens = model->vocab.tokenize(prefix, false);
        if (!prefix_tokens.empty() && prefix_tokens[0] == model->vocab.token_bos()) {
            prefix_tokens.erase(prefix_tokens.begin());
        }
        REQUIRE(healing_core.IngestTokens(prefix_tokens, true));

        const std::string partial = "{\"function\":\"get_wea";
        auto partial_tokens = model->vocab.tokenize(partial, false);
        if (!partial_tokens.empty() && partial_tokens[0] == model->vocab.token_bos()) {
            partial_tokens.erase(partial_tokens.begin());
        }
        REQUIRE(healing_core.IngestTokens(partial_tokens, true));

        auto checkpoint = healing_core.CreateCheckpoint();
        bool healed_a = healing_core.HealLastToken();
        MESSAGE("Token healing triggered: " << (healed_a ? "true" : "false"));

        auto result_a = GenerateContinuation(healing_core, model, 240);
        MESSAGE("Generated after healing: " << result_a.text);
        CHECK(result_a.text.find("\"parameters\"") != std::string::npos);

        REQUIRE(healing_core.RestoreCheckpoint(checkpoint));
        bool healed_b = healing_core.HealLastToken();
        MESSAGE("Token healing after restore: " << (healed_b ? "true" : "false"));

        auto result_b = GenerateContinuation(healing_core, model, 240);
        CHECK(result_a.tokens == result_b.tokens);
        CHECK(result_a.text == result_b.text);
    }

    lfm_model_free(model);
}
