#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"

#include "../inference/lfg_api.h"

#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// Test helper: parse and auto-free
// ---------------------------------------------------------------------------

struct tc_result {
    std::string name;
    std::string arguments;
};

static std::vector<tc_result> parse_pythonic(const std::string & input) {
    lfg_tool_call buf[16];
    int32_t n = lfg_parse_pythonic_tool_calls(input.c_str(), (int32_t)input.size(), buf, 16);
    std::vector<tc_result> out;
    for (int32_t i = 0; i < n; i++) {
        tc_result r;
        r.name = buf[i].name ? buf[i].name : "";
        r.arguments = buf[i].arguments ? buf[i].arguments : "";
        free((void *)buf[i].name);
        free((void *)buf[i].arguments);
        out.push_back(std::move(r));
    }
    return out;
}

// ---------------------------------------------------------------------------
// JSON-to-Pythonic converter (same logic as lfg_api.cpp — duplicated for testing)
// ---------------------------------------------------------------------------

static char * json_args_to_pythonic(const char * json_args) {
    if (!json_args || json_args[0] != '{') return nullptr;

    std::string result;
    std::string input(json_args);

    size_t pos = 1;
    bool first = true;

    while (pos < input.size()) {
        while (pos < input.size() && (input[pos] == ' ' || input[pos] == '\t' || input[pos] == '\n')) pos++;
        if (pos >= input.size() || input[pos] == '}') break;

        if (input[pos] != '"') break;
        size_t key_start = pos + 1;
        size_t key_end = input.find('"', key_start);
        if (key_end == std::string::npos) break;
        std::string key = input.substr(key_start, key_end - key_start);
        pos = key_end + 1;

        while (pos < input.size() && (input[pos] == ' ' || input[pos] == ':')) pos++;

        if (!first) result += ", ";
        first = false;
        result += key + "=";

        if (pos >= input.size()) break;

        if (input[pos] == '"') {
            size_t val_start = pos + 1;
            size_t val_end = val_start;
            while (val_end < input.size()) {
                if (input[val_end] == '\\' && val_end + 1 < input.size()) {
                    val_end += 2;
                    continue;
                }
                if (input[val_end] == '"') break;
                val_end++;
            }
            result += "\"" + input.substr(val_start, val_end - val_start) + "\"";
            pos = val_end + 1;
        } else if (input.compare(pos, 4, "true") == 0) {
            result += "True";
            pos += 4;
        } else if (input.compare(pos, 5, "false") == 0) {
            result += "False";
            pos += 5;
        } else if (input.compare(pos, 4, "null") == 0) {
            result += "None";
            pos += 4;
        } else if (input[pos] == '{' || input[pos] == '[') {
            char open = input[pos], close = (open == '{') ? '}' : ']';
            int depth = 1;
            size_t nest_start = pos;
            pos++;
            while (pos < input.size() && depth > 0) {
                if (input[pos] == '"') {
                    pos++;
                    while (pos < input.size() && input[pos] != '"') {
                        if (input[pos] == '\\') pos++;
                        pos++;
                    }
                }
                else if (input[pos] == open) depth++;
                else if (input[pos] == close) depth--;
                pos++;
            }
            result += input.substr(nest_start, pos - nest_start);
        } else {
            size_t val_start = pos;
            while (pos < input.size() && input[pos] != ',' && input[pos] != '}'
                   && input[pos] != ' ' && input[pos] != '\n') pos++;
            result += input.substr(val_start, pos - val_start);
        }

        while (pos < input.size() && (input[pos] == ' ' || input[pos] == ',')) pos++;
    }

    return strdup(result.c_str());
}

// =========================================================================
// Parser correctness tests
// =========================================================================

TEST_CASE("Parser: empty args") {
    auto calls = parse_pythonic("[func()]");
    REQUIRE(calls.size() == 1);
    CHECK(calls[0].name == "func");
    CHECK(calls[0].arguments == "{}");
}

TEST_CASE("Parser: single string arg (double quotes)") {
    auto calls = parse_pythonic("[func(key=\"val\")]");
    REQUIRE(calls.size() == 1);
    CHECK(calls[0].name == "func");
    CHECK(calls[0].arguments.find("\"key\"") != std::string::npos);
    CHECK(calls[0].arguments.find("\"val\"") != std::string::npos);
}

TEST_CASE("Parser: single string arg (single quotes)") {
    auto calls = parse_pythonic("[func(key='val')]");
    REQUIRE(calls.size() == 1);
    CHECK(calls[0].name == "func");
    CHECK(calls[0].arguments.find("\"key\"") != std::string::npos);
    CHECK(calls[0].arguments.find("\"val\"") != std::string::npos);
}

TEST_CASE("Parser: mixed types") {
    auto calls = parse_pythonic("[func(a=\"hello\", b=123, c=1.5, d=True, e=False, f=None)]");
    REQUIRE(calls.size() == 1);
    CHECK(calls[0].name == "func");
    auto & args = calls[0].arguments;
    CHECK(args.find("\"a\"") != std::string::npos);
    CHECK(args.find("\"hello\"") != std::string::npos);
    CHECK(args.find("123") != std::string::npos);
    CHECK(args.find("1.5") != std::string::npos);
    CHECK(args.find("true") != std::string::npos);
    CHECK(args.find("false") != std::string::npos);
    CHECK(args.find("null") != std::string::npos);
    // Should NOT contain Python-style values
    CHECK(args.find("True") == std::string::npos);
    CHECK(args.find("False") == std::string::npos);
    CHECK(args.find("None") == std::string::npos);
}

TEST_CASE("Parser: nested dict") {
    auto calls = parse_pythonic("[func(opts={\"a\": 1, \"b\": 2})]");
    REQUIRE(calls.size() == 1);
    CHECK(calls[0].name == "func");
    CHECK(calls[0].arguments.find("\"opts\"") != std::string::npos);
    CHECK(calls[0].arguments.find("{\"a\": 1, \"b\": 2}") != std::string::npos);
}

TEST_CASE("Parser: nested list") {
    auto calls = parse_pythonic("[func(ids=[1, 2, 3])]");
    REQUIRE(calls.size() == 1);
    CHECK(calls[0].name == "func");
    CHECK(calls[0].arguments.find("\"ids\"") != std::string::npos);
    CHECK(calls[0].arguments.find("[1, 2, 3]") != std::string::npos);
}

TEST_CASE("Parser: multiple tool calls") {
    auto calls = parse_pythonic("[func1(a=1)]\n[func2(b=2)]");
    REQUIRE(calls.size() == 2);
    CHECK(calls[0].name == "func1");
    CHECK(calls[1].name == "func2");
}

TEST_CASE("Parser: unicode strings") {
    auto calls = parse_pythonic("[func(text=\"日本語\")]");
    REQUIRE(calls.size() == 1);
    CHECK(calls[0].arguments.find("日本語") != std::string::npos);
}

TEST_CASE("Parser: negative numbers") {
    auto calls = parse_pythonic("[func(x=-1, y=-3.14)]");
    REQUIRE(calls.size() == 1);
    CHECK(calls[0].arguments.find("-1") != std::string::npos);
    CHECK(calls[0].arguments.find("-3.14") != std::string::npos);
}

TEST_CASE("Parser: whitespace tolerance") {
    auto calls = parse_pythonic("  [  func  (  a  =  \"hello\"  ,  b  =  42  )  ]  ");
    REQUIRE(calls.size() == 1);
    CHECK(calls[0].name == "func");
}

TEST_CASE("Parser: underscore in names") {
    auto calls = parse_pythonic("[get_weather(location_name=\"Tokyo\")]");
    REQUIRE(calls.size() == 1);
    CHECK(calls[0].name == "get_weather");
    CHECK(calls[0].arguments.find("\"location_name\"") != std::string::npos);
}

// =========================================================================
// Parser robustness / fuzz tests (must never crash, leak, or infinite loop)
// =========================================================================

TEST_CASE("Robustness: empty input") {
    auto calls = parse_pythonic("");
    CHECK(calls.empty());
}

TEST_CASE("Robustness: NULL input") {
    lfg_tool_call buf[4];
    CHECK(lfg_parse_pythonic_tool_calls(nullptr, 0, buf, 4) == 0);
    CHECK(lfg_parse_pythonic_tool_calls(nullptr, 10, buf, 4) == 0);
    CHECK(lfg_parse_pythonic_tool_calls("test", 4, nullptr, 4) == 0);
    CHECK(lfg_parse_pythonic_tool_calls("test", 4, buf, 0) == 0);
}

TEST_CASE("Robustness: no brackets") {
    auto calls = parse_pythonic("func(a=1)");
    CHECK(calls.empty());

    calls = parse_pythonic("hello world");
    CHECK(calls.empty());
}

TEST_CASE("Robustness: unclosed bracket") {
    auto calls = parse_pythonic("[func(a=1");
    CHECK(calls.empty());
}

TEST_CASE("Robustness: unclosed paren") {
    auto calls = parse_pythonic("[func(a=1]");
    CHECK(calls.empty());
}

TEST_CASE("Robustness: unclosed string") {
    auto calls = parse_pythonic("[func(a=\"hello)]");
    CHECK(calls.empty());

    calls = parse_pythonic("[func(a='hello)]");
    CHECK(calls.empty());
}

TEST_CASE("Robustness: missing value") {
    auto calls = parse_pythonic("[func(a=)]");
    CHECK(calls.empty());
}

TEST_CASE("Robustness: missing key") {
    auto calls = parse_pythonic("[func(=1)]");
    CHECK(calls.empty());

    calls = parse_pythonic("[func(\"val\")]");
    CHECK(calls.empty());
}

TEST_CASE("Robustness: extra commas") {
    auto calls = parse_pythonic("[func(a=1,,b=2)]");
    // Just ensure no crash
    (void)calls;
}

TEST_CASE("Robustness: deeply nested") {
    auto calls = parse_pythonic("[func(x=[[[[[[1]]]]]])]");
    // Just ensure no crash or infinite loop
    (void)calls;
}

TEST_CASE("Robustness: long string arg") {
    std::string long_val(10000, 'x');
    std::string input = "[func(s=\"" + long_val + "\")]";
    auto calls = parse_pythonic(input);
    if (!calls.empty()) {
        CHECK(calls[0].name == "func");
    }
}

TEST_CASE("Robustness: binary garbage") {
    std::string garbage;
    for (int i = 0; i < 256; i++) {
        garbage += static_cast<char>(i);
    }
    auto calls = parse_pythonic(garbage);
    CHECK(calls.empty());
}

TEST_CASE("Robustness: partial special tokens") {
    auto calls = parse_pythonic("<|tool_call_sta");
    CHECK(calls.empty());

    calls = parse_pythonic("<|tool_call_end");
    CHECK(calls.empty());
}

TEST_CASE("Robustness: only whitespace in brackets") {
    auto calls = parse_pythonic("[  ]");
    CHECK(calls.empty());
}

TEST_CASE("Robustness: only name, no parens") {
    auto calls = parse_pythonic("[func]");
    CHECK(calls.empty());

    calls = parse_pythonic("[func garbage stuff]");
    CHECK(calls.empty());
}

TEST_CASE("Robustness: empty name") {
    auto calls = parse_pythonic("[(a=1)]");
    CHECK(calls.empty());
}

TEST_CASE("Robustness: very long function name") {
    std::string long_name(10000, 'a');
    std::string input = "[" + long_name + "(x=1)]";
    auto calls = parse_pythonic(input);
    // Should handle gracefully
    (void)calls;
}

TEST_CASE("Robustness: integer overflow") {
    auto calls = parse_pythonic("[func(n=99999999999999999999999)]");
    // Should parse the number as raw text, not crash
    if (!calls.empty()) {
        CHECK(calls[0].name == "func");
    }
}

TEST_CASE("Robustness: trailing comma") {
    auto calls = parse_pythonic("[func(a=1,)]");
    // Should handle gracefully (parse or fail without crash)
    (void)calls;
}

// =========================================================================
// JSON-to-Pythonic round-trip tests
// =========================================================================

TEST_CASE("JSON-to-Pythonic: simple string") {
    char *result = json_args_to_pythonic("{\"key\":\"val\"}");
    REQUIRE(result != nullptr);
    CHECK(std::string(result) == "key=\"val\"");
    free(result);
}

TEST_CASE("JSON-to-Pythonic: mixed types") {
    char *result = json_args_to_pythonic("{\"a\":1,\"b\":true,\"c\":null}");
    REQUIRE(result != nullptr);
    std::string s(result);
    CHECK(s.find("a=1") != std::string::npos);
    CHECK(s.find("b=True") != std::string::npos);
    CHECK(s.find("c=None") != std::string::npos);
    free(result);
}

TEST_CASE("JSON-to-Pythonic: empty object") {
    char *result = json_args_to_pythonic("{}");
    REQUIRE(result != nullptr);
    CHECK(std::string(result).empty());
    free(result);
}

TEST_CASE("JSON-to-Pythonic: false value") {
    char *result = json_args_to_pythonic("{\"enabled\":false}");
    REQUIRE(result != nullptr);
    CHECK(std::string(result) == "enabled=False");
    free(result);
}

TEST_CASE("JSON-to-Pythonic: nested objects pass through") {
    char *result = json_args_to_pythonic("{\"opts\":{\"a\":1}}");
    REQUIRE(result != nullptr);
    std::string s(result);
    CHECK(s.find("opts=") != std::string::npos);
    CHECK(s.find("{\"a\":1}") != std::string::npos);
    free(result);
}

TEST_CASE("JSON-to-Pythonic: invalid JSON returns NULL") {
    CHECK(json_args_to_pythonic(nullptr) == nullptr);
    CHECK(json_args_to_pythonic("") == nullptr);
    CHECK(json_args_to_pythonic("not json") == nullptr);
    CHECK(json_args_to_pythonic("[1,2,3]") == nullptr);
}

TEST_CASE("JSON-to-Pythonic: number value") {
    char *result = json_args_to_pythonic("{\"x\":42,\"y\":3.14}");
    REQUIRE(result != nullptr);
    std::string s(result);
    CHECK(s.find("x=42") != std::string::npos);
    CHECK(s.find("y=3.14") != std::string::npos);
    free(result);
}

// =========================================================================
// Full round-trip: Pythonic → parse → JSON → Pythonic
// =========================================================================

TEST_CASE("Round-trip: parse then convert back") {
    auto calls = parse_pythonic("[get_weather(location=\"Tokyo\", units=\"celsius\")]");
    REQUIRE(calls.size() == 1);
    CHECK(calls[0].name == "get_weather");

    char *pythonic = json_args_to_pythonic(calls[0].arguments.c_str());
    REQUIRE(pythonic != nullptr);
    std::string s(pythonic);
    CHECK(s.find("location=\"Tokyo\"") != std::string::npos);
    CHECK(s.find("units=\"celsius\"") != std::string::npos);
    free(pythonic);
}

// =========================================================================
// Public API NULL safety
// =========================================================================

TEST_CASE("API: get_tool_calls with NULL session") {
    int32_t n = -1;
    const lfg_tool_call *calls = lfg_session_get_tool_calls(nullptr, &n);
    CHECK(calls == nullptr);
    CHECK(n == 0);
}

TEST_CASE("API: get_last_output with NULL session") {
    int32_t len = -1;
    const char *out = lfg_session_get_last_output(nullptr, &len);
    CHECK(out == nullptr);
    CHECK(len == 0);
}

TEST_CASE("API: set_tool_call_format with NULL session") {
    // Should not crash
    lfg_session_set_tool_call_format(nullptr, LFG_TOOL_CALL_FORMAT_PYTHONIC);
    lfg_session_set_tool_call_format(nullptr, LFG_TOOL_CALL_FORMAT_JSON);
}
