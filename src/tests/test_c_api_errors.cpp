#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"

#include "../inference/lfg_inference.h"

#include <string>

TEST_CASE("C API error handling") {
    char buf[256];

    lfg_clear_last_error();
    auto code = lfg_get_last_error(buf, sizeof(buf));
    CHECK(code == LFG_ERROR_NONE);
    CHECK(std::string(buf).empty());

    lfg_model_params params = lfg_model_default_params();
    auto * model = lfg_model_load_from_file(nullptr, params);
    CHECK(model == nullptr);
    code = lfg_get_last_error(buf, sizeof(buf));
    CHECK(code == LFG_ERROR_INVALID_ARGUMENT);
    CHECK(std::string(buf).find("path_model") != std::string::npos);

    lfg_clear_last_error();
    int32_t res = lfg_tokenize(nullptr, "hi", 2, nullptr, 0, false, false);
    CHECK(res < 0);
    code = lfg_get_last_error(buf, sizeof(buf));
    CHECK(code == LFG_ERROR_INVALID_ARGUMENT);
}

TEST_CASE("C API version helpers") {
    uint32_t major = 0;
    uint32_t minor = 0;
    uint32_t patch = 0;

    lfg_api_version(&major, &minor, &patch);
    CHECK(major == LFG_API_VERSION_MAJOR);
    CHECK(minor == LFG_API_VERSION_MINOR);
    CHECK(patch == LFG_API_VERSION_PATCH);

    const std::string version_str = lfg_api_version_string();
    CHECK(version_str.find('.') != std::string::npos);
    CHECK(lfg_abi_version() == LFG_ABI_VERSION);
}
