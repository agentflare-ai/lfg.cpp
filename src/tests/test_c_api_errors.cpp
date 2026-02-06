#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"

#include "../inference/lfm_inference.h"

#include <string>

TEST_CASE("C API error handling") {
    char buf[256];

    lfm_clear_last_error();
    auto code = lfm_get_last_error(buf, sizeof(buf));
    CHECK(code == LFM_ERROR_NONE);
    CHECK(std::string(buf).empty());

    lfm_model_params params = lfm_model_default_params();
    auto * model = lfm_model_load_from_file(nullptr, params);
    CHECK(model == nullptr);
    code = lfm_get_last_error(buf, sizeof(buf));
    CHECK(code == LFM_ERROR_INVALID_ARGUMENT);
    CHECK(std::string(buf).find("path_model") != std::string::npos);

    lfm_clear_last_error();
    int32_t res = lfm_tokenize(nullptr, "hi", 2, nullptr, 0, false, false);
    CHECK(res < 0);
    code = lfm_get_last_error(buf, sizeof(buf));
    CHECK(code == LFM_ERROR_INVALID_ARGUMENT);
}

TEST_CASE("C API version helpers") {
    uint32_t major = 0;
    uint32_t minor = 0;
    uint32_t patch = 0;

    lfm_api_version(&major, &minor, &patch);
    CHECK(major == LFM_API_VERSION_MAJOR);
    CHECK(minor == LFM_API_VERSION_MINOR);
    CHECK(patch == LFM_API_VERSION_PATCH);

    const std::string version_str = lfm_api_version_string();
    CHECK(version_str.find('.') != std::string::npos);
    CHECK(lfm_abi_version() == LFM_ABI_VERSION);
}
