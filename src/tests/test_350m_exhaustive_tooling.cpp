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
#include <stdexcept>
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
    "schema_version": { "const": "1.0" },
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
                  "query": { "type": "string", "minLength": 4 },
                  "top_k": { "type": "integer", "minimum": 1, "maximum": 5 },
                  "filters": {
                    "type": "object",
                    "properties": {
                      "tags": {
                        "type": "array",
                        "items": { "type": "string" },
                        "minItems": 1,
                        "maxItems": 3
                      },
                      "since": { "type": "string", "pattern": "^[2][0][0-9]{2}-[0-9]{2}-[0-9]{2}$" }
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
              "function": { "const": "generate_report" },
              "parameters": {
                "type": "object",
                "properties": {
                  "run_id": { "type": "string", "pattern": "^[A-Z]{2}-[0-9]{4}$" },
                  "cases": {
                    "type": "object",
                    "properties": {
                      "success_case": {
                        "type": "object",
                        "properties": {
                          "name": { "type": "string" },
                          "status": { "const": "success" },
                          "detail": { "type": "string" }
                        },
                        "required": ["name", "status", "detail"],
                        "additionalProperties": false
                      },
                      "failure_case": {
                        "type": "object",
                        "properties": {
                          "name": { "type": "string" },
                          "status": { "const": "failure" },
                          "detail": { "type": "string" }
                        },
                        "required": ["name", "status", "detail"],
                        "additionalProperties": false
                      }
                    },
                    "required": ["success_case", "failure_case"],
                    "additionalProperties": false
                  },
                  "counts": {
                    "type": "object",
                    "properties": {
                      "pass": { "type": "integer", "minimum": 1, "maximum": 9 },
                      "fail": { "type": "integer", "minimum": 1, "maximum": 9 }
                    },
                    "required": ["pass", "fail"],
                    "additionalProperties": false
                  }
                },
                "required": ["run_id", "cases", "counts"],
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
    "Include tool calls for get_weather, calculator, search_docs, and generate_report. "
    "Use a run_id like AB-1234 and include one success case and one failure case in the report.";

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

        if (i > 16) {
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

void ValidateToolReport(const json & tool_call) {
    REQUIRE(tool_call.contains("function"));
    REQUIRE(tool_call.contains("parameters"));
    REQUIRE(tool_call["function"].is_string());
    REQUIRE(tool_call["parameters"].is_object());

    if (tool_call["function"] == "generate_report") {
        const auto & params = tool_call["parameters"];
        REQUIRE(params.contains("run_id"));
        REQUIRE(params.contains("cases"));
        REQUIRE(params.contains("counts"));

        const auto run_id = params["run_id"].get<std::string>();
        CHECK(std::regex_match(run_id, std::regex(R"(^[A-Z]{2}-[0-9]{4}$)")));

        const auto & cases = params["cases"];
        REQUIRE(cases.contains("success_case"));
        REQUIRE(cases.contains("failure_case"));

        const auto & success_case = cases["success_case"];
        const auto & failure_case = cases["failure_case"];

        CHECK(success_case["status"] == "success");
        CHECK(failure_case["status"] == "failure");

        const auto & counts = params["counts"];
        CHECK(counts["pass"].get<int>() >= 1);
        CHECK(counts["fail"].get<int>() >= 1);
    }
}

} // namespace

TEST_CASE("LFM2-350M Exhaustive Tool Calling + Structured Decoding") {
    lfm_backend_init();

    const std::string model_path = "models/lfm2-350M.gguf";
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
    config.n_ctx = 2048;
    config.sampling.temp = 0.0f;
    config.sampling.top_k = 40;
    config.sampling.top_p = 1.0f;

    InferenceCore core(model, config);

    SUBCASE("Multi-tool structured decoding with report") {
        auto result = GenerateStructured(core, model, kToolSchema, kToolPrompt, 512);
        MESSAGE("Generated JSON: " << result.text);

        json parsed = ParseJsonOrFail(result.text);
        REQUIRE(parsed.contains("schema_version"));
        CHECK(parsed["schema_version"] == "1.0");

        REQUIRE(parsed.contains("tool_calls"));
        REQUIRE(parsed["tool_calls"].is_array());
        CHECK(parsed["tool_calls"].size() >= 4);

        std::set<std::string> functions;
        for (const auto & call : parsed["tool_calls"]) {
            REQUIRE(call.contains("function"));
            functions.insert(call["function"].get<std::string>());
            ValidateToolReport(call);
        }

        CHECK(functions.count("get_weather") == 1);
        CHECK(functions.count("calculator") == 1);
        CHECK(functions.count("search_docs") == 1);
        CHECK(functions.count("generate_report") == 1);
    }

    SUBCASE("Checkpointing determinism under complex schema") {
        core.Reset();
        core.ConfigureStructuredDecoding("");

        auto prompt_tokens = model->vocab.tokenize(kToolPrompt, true);
        core.IngestTokens(prompt_tokens, false);

        auto prefix = GenerateContinuation(core, model, 80);
        auto checkpoint = core.CreateCheckpoint();

        auto suffix_a = GenerateContinuation(core, model, 120);
        bool restored = core.RestoreCheckpoint(checkpoint);
        CHECK(restored);

        auto suffix_b = GenerateContinuation(core, model, 120);

        CHECK(suffix_a.tokens == suffix_b.tokens);
        CHECK(suffix_a.text == suffix_b.text);
    }

    SUBCASE("Token healing with JSON schema + report") {
        InferenceCore::Config heal_config = config;
        heal_config.enable_healing = true;
        InferenceCore healing_core(model, heal_config);

        // Use a valid JSON prefix so healing can replay grammar state safely.
        const std::string prompt_prefix = "{\"schema_version\": \"1.0\", \"tool_calls\": [";
        const std::string partial_token = "{";
        healing_core.ConfigureStructuredDecoding(kToolSchema);

        auto prefix_tokens = model->vocab.tokenize(prompt_prefix, false);
        if (!prefix_tokens.empty() && prefix_tokens[0] == model->vocab.token_bos()) {
            prefix_tokens.erase(prefix_tokens.begin());
        }
        healing_core.IngestTokens(prefix_tokens, false);

        auto partial_tokens = model->vocab.tokenize(partial_token, false);
        if (!partial_tokens.empty() && partial_tokens[0] == model->vocab.token_bos()) {
            partial_tokens.erase(partial_tokens.begin());
        }
        healing_core.IngestTokens(partial_tokens, false);

        bool healed = healing_core.HealLastToken();
        MESSAGE("Token healing triggered: " << (healed ? "true" : "false"));

        auto result = GenerateContinuation(healing_core, model, 512);
        MESSAGE("Generated JSON after healing: " << result.text);

        CHECK(result.text.find("\"function\"") != std::string::npos);
        CHECK(result.text.find("\"search_docs\"") != std::string::npos);
        CHECK(result.text.find("\"since\"") != std::string::npos);
    }

    lfm_model_free(model);
}
