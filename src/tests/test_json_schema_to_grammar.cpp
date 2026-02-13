#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"

#include "../inference/lfg_api.h"

#include <string>

TEST_CASE("Valid JSON schema produces grammar") {
    const char *schema = R"({"type":"object","properties":{"name":{"type":"string"}},"required":["name"]})";

    // Query required size first (buf=NULL, size=0)
    int32_t needed = lfg_json_schema_to_grammar(schema, false, nullptr, 0);
    REQUIRE(needed > 0);

    // Allocate and convert
    std::string buf(needed + 1, '\0');
    int32_t written = lfg_json_schema_to_grammar(schema, false, &buf[0], buf.size());
    REQUIRE(written > 0);
    CHECK(written == needed);

    // Grammar should contain "root" rule
    CHECK(std::string(buf.c_str()).find("root") != std::string::npos);
}

TEST_CASE("Invalid JSON returns -1 and sets error") {
    lfg_clear_last_error();
    char buf[512];
    int32_t ret = lfg_json_schema_to_grammar("not valid json {{{", false, buf, sizeof(buf));
    CHECK(ret == -1);

    char err[256];
    auto code = lfg_get_last_error(err, sizeof(err));
    CHECK(code != LFG_ERROR_NONE);
}

TEST_CASE("NULL schema returns -1 and sets error") {
    lfg_clear_last_error();
    int32_t ret = lfg_json_schema_to_grammar(nullptr, false, nullptr, 0);
    CHECK(ret == -1);

    char err[256];
    auto code = lfg_get_last_error(err, sizeof(err));
    CHECK(code == LFG_ERROR_INVALID_ARGUMENT);
}

TEST_CASE("Buffer too small returns required size without overflow") {
    const char *schema = R"({"type":"object","properties":{"x":{"type":"integer"}},"required":["x"]})";

    // Get required size
    int32_t needed = lfg_json_schema_to_grammar(schema, false, nullptr, 0);
    REQUIRE(needed > 0);

    // Use a tiny buffer — snprintf truncates, returns what would have been written
    char tiny[4] = {};
    int32_t ret = lfg_json_schema_to_grammar(schema, false, tiny, sizeof(tiny));
    // snprintf returns the size that would have been needed (excluding null)
    CHECK(ret == needed);
    // The buffer should be null-terminated and not overflowed
    CHECK(tiny[sizeof(tiny) - 1] == '\0');
}
