#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"

#include "../inference/lfg_api.h"
#include "../inference/lfg_model.h"

#include <nlohmann/json.hpp>
#include <spdlog/spdlog.h>

#include <fstream>
#include <regex>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

using json = nlohmann::json;

namespace {

struct GenerationResult {
    std::string text;
    std::vector<lfg_token> tokens;
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

GenerationResult GenerateStructured(lfg_session * session,
                                    lfg_model * model,
                                    const std::string & schema,
                                    const std::string & prompt,
                                    int max_tokens) {
    lfg_session_reset(session);
    lfg_session_configure_structured(session, schema.c_str(), "root");

    auto prompt_tokens = model->vocab.tokenize(prompt, true);
    lfg_session_ingest_tokens(session, prompt_tokens.data(), prompt_tokens.size(), false);

    GenerationResult result;
    result.tokens.reserve(max_tokens);

    for (int i = 0; i < max_tokens; ++i) {
        lfg_token token = lfg_session_sample(session);
        if (token == model->vocab.token_eos()) {
            break;
        }
        result.tokens.push_back(token);
        result.text += model->vocab.token_to_piece(token);
        lfg_session_ingest_tokens(session, &token, 1, false);

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

GenerationResult GenerateContinuation(lfg_session * session,
                                      lfg_model * model,
                                      int max_tokens) {
    GenerationResult result;
    result.tokens.reserve(max_tokens);

    for (int i = 0; i < max_tokens; ++i) {
        lfg_token token = lfg_session_sample(session);
        if (token == model->vocab.token_eos()) {
            break;
        }
        result.tokens.push_back(token);
        result.text += model->vocab.token_to_piece(token);
        lfg_session_ingest_tokens(session, &token, 1, false);
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
    lfg_backend_init();

    const std::string model_path = "models/lfm2-350M.gguf";
    std::ifstream f(model_path);
    if (!f.good()) {
        MESSAGE("Model not found at " << model_path << ". Skipping test.");
        return;
    }

    lfg_model_load_config load_config = lfg_model_load_default_config();
    load_config.model_path = model_path.c_str();
    load_config.n_gpu_layers = 0;

    lfg_model * model = lfg_load_model(&load_config);
    REQUIRE(model != nullptr);

    lfg_session_config config = lfg_session_default_config();
    config.n_ctx = 2048;
    config.sampling.temp = 0.0f;
    config.sampling.top_k = 40;
    config.sampling.top_p = 1.0f;

    lfg_session * session = lfg_session_create(model, &config);

    SUBCASE("Multi-tool structured decoding with report") {
        auto result = GenerateStructured(session, model, kToolSchema, kToolPrompt, 512);
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
        lfg_session_reset(session);
        lfg_session_configure_structured(session, "", "root");

        auto prompt_tokens = model->vocab.tokenize(kToolPrompt, true);
        lfg_session_ingest_tokens(session, prompt_tokens.data(), prompt_tokens.size(), false);

        auto prefix = GenerateContinuation(session, model, 80);
        lfg_checkpoint * checkpoint = lfg_session_create_checkpoint(session);

        auto suffix_a = GenerateContinuation(session, model, 120);
        bool restored = lfg_session_restore_checkpoint(session, checkpoint);
        CHECK(restored);

        auto suffix_b = GenerateContinuation(session, model, 120);

        CHECK(suffix_a.tokens == suffix_b.tokens);
        CHECK(suffix_a.text == suffix_b.text);

        lfg_checkpoint_free(checkpoint);
    }

    SUBCASE("Token healing with JSON schema + report") {
        lfg_session_config heal_config = config;
        heal_config.enable_healing = true;
        lfg_session * healing_session = lfg_session_create(model, &heal_config);

        // Use a valid JSON prefix so healing can replay grammar state safely.
        const std::string prompt_prefix = "{\"schema_version\": \"1.0\", \"tool_calls\": [";
        const std::string partial_token = "{";
        lfg_session_configure_structured(healing_session, kToolSchema.c_str(), "root");

        auto prefix_tokens = model->vocab.tokenize(prompt_prefix, false);
        if (!prefix_tokens.empty() && prefix_tokens[0] == model->vocab.token_bos()) {
            prefix_tokens.erase(prefix_tokens.begin());
        }
        lfg_session_ingest_tokens(healing_session, prefix_tokens.data(), prefix_tokens.size(), false);

        auto partial_tokens = model->vocab.tokenize(partial_token, false);
        if (!partial_tokens.empty() && partial_tokens[0] == model->vocab.token_bos()) {
            partial_tokens.erase(partial_tokens.begin());
        }
        lfg_session_ingest_tokens(healing_session, partial_tokens.data(), partial_tokens.size(), false);

        bool healed = lfg_session_heal_last_token(healing_session);
        MESSAGE("Token healing triggered: " << (healed ? "true" : "false"));

        auto result = GenerateContinuation(healing_session, model, 512);
        MESSAGE("Generated JSON after healing: " << result.text);

        CHECK(result.text.find("\"function\"") != std::string::npos);
        CHECK(result.text.find("\"search_docs\"") != std::string::npos);
        CHECK(result.text.find("\"since\"") != std::string::npos);

        lfg_session_free(healing_session);
    }

    lfg_session_free(session);
    lfg_model_free(model);
}
