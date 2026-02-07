#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"

#include "../inference/lfg_api.h"

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

TEST_CASE("Session API NULL safety") {
    // lfg_session_create with NULL model
    lfg_session *s = lfg_session_create(nullptr, nullptr);
    CHECK(s == nullptr);

    // lfg_session_sample with NULL session
    lfg_token t = lfg_session_sample(nullptr);
    CHECK(t == 0);

    // lfg_session_reset with NULL session — no crash
    lfg_session_reset(nullptr);

    // lfg_session_configure_structured with NULL session
    bool ok = lfg_session_configure_structured(nullptr, "root ::= \"hi\"", nullptr);
    CHECK(ok == false);

    // lfg_session_configure_reasoning with NULL session — no crash
    lfg_token start = 1, end = 2;
    lfg_session_configure_reasoning(nullptr, &start, 1, &end, 1);

    // lfg_session_configure_stop_sequences with NULL session
    const lfg_token seq[] = {1, 2};
    const lfg_token *seqs[] = {seq};
    size_t lens[] = {2};
    ok = lfg_session_configure_stop_sequences(nullptr, seqs, lens, 1);
    CHECK(ok == false);

    // lfg_checkpoint_free with NULL — no crash
    lfg_checkpoint_free(nullptr);

    // lfg_session_get_logits with NULL session
    int32_t n = lfg_session_get_logits(nullptr, nullptr, 0);
    CHECK(n <= 0);

    // lfg_session_get_vocab_size with NULL session
    n = lfg_session_get_vocab_size(nullptr);
    CHECK(n <= 0);

    // lfg_session_ingest_tokens with NULL session
    lfg_token tok = 1;
    ok = lfg_session_ingest_tokens(nullptr, &tok, 1, false);
    CHECK(ok == false);

    // lfg_session_heal_last_token with NULL session
    ok = lfg_session_heal_last_token(nullptr);
    CHECK(ok == false);

    // lfg_session_create_checkpoint with NULL session
    lfg_checkpoint *ck = lfg_session_create_checkpoint(nullptr);
    CHECK(ck == nullptr);

    // lfg_session_restore_checkpoint with NULL session
    ok = lfg_session_restore_checkpoint(nullptr, nullptr);
    CHECK(ok == false);
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
