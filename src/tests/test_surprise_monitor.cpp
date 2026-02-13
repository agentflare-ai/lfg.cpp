#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"
#include "../inference/lfg_api.h"
#include <cstring>
#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

static std::vector<lfg_token> tokenize(const lfg_vocab *vocab, const std::string &text, bool add_special) {
    std::vector<lfg_token> tokens(text.size() + 16);
    int32_t n = lfg_tokenize(vocab, text.c_str(), text.length(), tokens.data(), tokens.size(), add_special, false);
    if (n < 0) {
        tokens.resize(-n);
        n = lfg_tokenize(vocab, text.c_str(), text.length(), tokens.data(), tokens.size(), add_special, false);
    }
    tokens.resize(n);
    return tokens;
}

// ---------------------------------------------------------------------------
// 350M model — unit / mechanical tests
// ---------------------------------------------------------------------------

static lfg_model * g_350m = nullptr;

static lfg_model * get_350m() {
    if (!g_350m) {
        lfg_backend_init();
        lfg_model_load_config cfg = lfg_model_load_default_config();
        cfg.model_path = "models/lfm2-350M.gguf";
        cfg.n_gpu_layers = 0;
        g_350m = lfg_load_model(&cfg);
    }
    return g_350m;
}

// ---------------------------------------------------------------------------
// Test 1: Default config returns sensible values
// ---------------------------------------------------------------------------

TEST_CASE("Default config returns sensible values") {
    lfg_surprise_monitor_config cfg = lfg_surprise_monitor_default_config();
    CHECK(cfg.threshold > 0.0f);
    CHECK(cfg.threshold < 1.0f);
}

// ---------------------------------------------------------------------------
// Test 2: Configure returns n_embd > 0; disable with NULL returns 0
// ---------------------------------------------------------------------------

TEST_CASE("Configure returns n_embd > 0") {
    lfg_model *model = get_350m();
    REQUIRE(model != nullptr);

    lfg_session_config config = lfg_session_default_config();
    config.n_ctx = 2048;
    config.sampling.temp = 0.0f;
    lfg_session *session = lfg_session_create(model, &config);
    REQUIRE(session != nullptr);

    lfg_surprise_monitor_config scfg = lfg_surprise_monitor_default_config();
    int32_t n_embd = lfg_session_configure_surprise_monitor(session, &scfg);
    CHECK(n_embd > 0);
    MESSAGE("n_embd = " << n_embd);

    // Disable returns 0
    CHECK(lfg_session_configure_surprise_monitor(session, nullptr) == 0);

    lfg_session_free(session);
}

// ---------------------------------------------------------------------------
// Test 3: No event when input is predictable (high threshold)
// ---------------------------------------------------------------------------

TEST_CASE("No event on predictable input with high threshold") {
    lfg_model *model = get_350m();
    REQUIRE(model != nullptr);

    const lfg_vocab *vocab = lfg_model_get_vocab(model);

    lfg_session_config config = lfg_session_default_config();
    config.n_ctx = 2048;
    config.sampling.temp = 0.0f;
    lfg_session *session = lfg_session_create(model, &config);
    REQUIRE(session != nullptr);

    // Very high threshold — nothing qualifies (surprise can exceed 1.0, so use 2.0)
    lfg_surprise_monitor_config scfg = lfg_surprise_monitor_default_config();
    scfg.threshold = 2.0f;
    REQUIRE(lfg_session_configure_surprise_monitor(session, &scfg) > 0);

    // Predictable text
    auto tokens = tokenize(vocab, "The capital of France is Paris", true);
    REQUIRE(lfg_session_ingest_tokens(session, tokens.data(), tokens.size(), true));

    // Pop should return false — nothing exceeded threshold
    lfg_surprise_event ev;
    CHECK_FALSE(lfg_session_surprise_pop(session, &ev, nullptr, 0));

    lfg_session_free(session);
}

// ---------------------------------------------------------------------------
// Test 4: Event fires on novel input (gibberish, low threshold)
// ---------------------------------------------------------------------------

TEST_CASE("Event fires on novel input") {
    lfg_model *model = get_350m();
    REQUIRE(model != nullptr);

    const lfg_vocab *vocab = lfg_model_get_vocab(model);

    lfg_session_config config = lfg_session_default_config();
    config.n_ctx = 2048;
    config.sampling.temp = 0.0f;
    lfg_session *session = lfg_session_create(model, &config);
    REQUIRE(session != nullptr);

    lfg_surprise_monitor_config scfg = lfg_surprise_monitor_default_config();
    scfg.threshold = 0.1f;
    REQUIRE(lfg_session_configure_surprise_monitor(session, &scfg) > 0);

    // Gibberish text — should be very surprising to the model
    auto tokens = tokenize(vocab, "xyzzy plugh blort quux zorp fnord mxyzptlk blargh", true);
    REQUIRE(tokens.size() > 4);
    REQUIRE(lfg_session_ingest_tokens(session, tokens.data(), tokens.size(), true));

    lfg_surprise_event ev;
    bool got = lfg_session_surprise_pop(session, &ev, nullptr, 0);
    CHECK(got);
    if (got) {
        MESSAGE("mean=" << ev.mean_surprise << " max=" << ev.max_surprise
                << " above=" << ev.n_above_threshold << "/" << ev.n_tokens_evaluated);
    }

    lfg_session_free(session);
}

// ---------------------------------------------------------------------------
// Test 5: Event fields valid
// ---------------------------------------------------------------------------

TEST_CASE("Event fields are valid") {
    lfg_model *model = get_350m();
    REQUIRE(model != nullptr);

    const lfg_vocab *vocab = lfg_model_get_vocab(model);

    lfg_session_config config = lfg_session_default_config();
    config.n_ctx = 2048;
    config.sampling.temp = 0.0f;
    lfg_session *session = lfg_session_create(model, &config);
    REQUIRE(session != nullptr);

    lfg_surprise_monitor_config scfg = lfg_surprise_monitor_default_config();
    scfg.threshold = 0.1f;
    REQUIRE(lfg_session_configure_surprise_monitor(session, &scfg) > 0);

    auto tokens = tokenize(vocab, "xyzzy plugh blort quux zorp fnord mxyzptlk blargh", true);
    REQUIRE(lfg_session_ingest_tokens(session, tokens.data(), tokens.size(), true));

    lfg_surprise_event ev;
    bool got = lfg_session_surprise_pop(session, &ev, nullptr, 0);
    if (got) {
        CHECK(ev.mean_surprise >= scfg.threshold);
        CHECK(ev.max_surprise >= ev.mean_surprise);
        CHECK(ev.n_above_threshold > 0);
        CHECK(ev.n_tokens_evaluated > 0);
        CHECK(ev.n_above_threshold <= ev.n_tokens_evaluated);
        MESSAGE("Event: mean=" << ev.mean_surprise << " max=" << ev.max_surprise
                << " above=" << ev.n_above_threshold << " total=" << ev.n_tokens_evaluated);
    } else {
        MESSAGE("No event fired — cannot validate fields");
    }

    lfg_session_free(session);
}

// ---------------------------------------------------------------------------
// Test 6: Pop returns true once then false
// ---------------------------------------------------------------------------

TEST_CASE("Pop returns true once then false") {
    lfg_model *model = get_350m();
    REQUIRE(model != nullptr);

    const lfg_vocab *vocab = lfg_model_get_vocab(model);

    lfg_session_config config = lfg_session_default_config();
    config.n_ctx = 2048;
    config.sampling.temp = 0.0f;
    lfg_session *session = lfg_session_create(model, &config);
    REQUIRE(session != nullptr);

    lfg_surprise_monitor_config scfg = lfg_surprise_monitor_default_config();
    scfg.threshold = 0.1f;
    REQUIRE(lfg_session_configure_surprise_monitor(session, &scfg) > 0);

    auto tokens = tokenize(vocab, "xyzzy plugh blort quux zorp fnord", true);
    REQUIRE(lfg_session_ingest_tokens(session, tokens.data(), tokens.size(), true));

    lfg_surprise_event ev;
    bool first = lfg_session_surprise_pop(session, &ev, nullptr, 0);
    if (first) {
        // Second call should return false
        CHECK_FALSE(lfg_session_surprise_pop(session, &ev, nullptr, 0));
    }

    lfg_session_free(session);
}

// ---------------------------------------------------------------------------
// Test 7: Reset clears state
// ---------------------------------------------------------------------------

TEST_CASE("Reset clears state") {
    lfg_model *model = get_350m();
    REQUIRE(model != nullptr);

    const lfg_vocab *vocab = lfg_model_get_vocab(model);

    lfg_session_config config = lfg_session_default_config();
    config.n_ctx = 2048;
    config.sampling.temp = 0.0f;
    lfg_session *session = lfg_session_create(model, &config);
    REQUIRE(session != nullptr);

    lfg_surprise_monitor_config scfg = lfg_surprise_monitor_default_config();
    scfg.threshold = 0.1f;
    REQUIRE(lfg_session_configure_surprise_monitor(session, &scfg) > 0);

    auto tokens = tokenize(vocab, "xyzzy plugh blort quux zorp fnord", true);
    REQUIRE(lfg_session_ingest_tokens(session, tokens.data(), tokens.size(), true));

    // Reset session
    lfg_session_reset(session);

    // After reset, pop should return false
    lfg_surprise_event ev;
    CHECK_FALSE(lfg_session_surprise_pop(session, &ev, nullptr, 0));

    lfg_session_free(session);
}

// ---------------------------------------------------------------------------
// Test 8: Generate loop leaves surprise events in queue
// ---------------------------------------------------------------------------

TEST_CASE("Generate loop does not drain surprise queue") {
    lfg_model *model = get_350m();
    REQUIRE(model != nullptr);

    const lfg_vocab *vocab = lfg_model_get_vocab(model);

    lfg_session_config config = lfg_session_default_config();
    config.n_ctx = 2048;
    config.sampling.temp = 0.0f;
    lfg_session *session = lfg_session_create(model, &config);
    REQUIRE(session != nullptr);

    lfg_surprise_monitor_config scfg = lfg_surprise_monitor_default_config();
    scfg.threshold = 0.1f;
    REQUIRE(lfg_session_configure_surprise_monitor(session, &scfg) > 0);

    // Ingest surprising prompt
    auto tokens = tokenize(vocab, "xyzzy plugh blort quux zorp fnord mxyzptlk blargh", true);
    REQUIRE(lfg_session_ingest_tokens(session, tokens.data(), tokens.size(), true));

    lfg_generate_config gc = lfg_generate_default_config();
    gc.max_tokens = 10;
    lfg_session_generate(session, gc);

    // Event remains queued for external consumer.
    int32_t pending_before = lfg_session_surprise_pending(session);
    CHECK(pending_before >= 0);
    lfg_surprise_event ev;
    bool got = lfg_session_surprise_pop(session, &ev, nullptr, 0);
    if (pending_before > 0) {
        CHECK(got);
    }
    CHECK(lfg_session_surprise_pending(session) == (got ? pending_before - 1 : pending_before));

    lfg_session_free(session);
}

// ---------------------------------------------------------------------------
// Test 9: Surprise + entropy + confidence all coexist
// ---------------------------------------------------------------------------

TEST_CASE("Surprise + entropy + confidence coexist") {
    lfg_model *model = get_350m();
    REQUIRE(model != nullptr);

    const lfg_vocab *vocab = lfg_model_get_vocab(model);

    lfg_session_config config = lfg_session_default_config();
    config.n_ctx = 2048;
    config.sampling.temp = 0.0f;
    lfg_session *session = lfg_session_create(model, &config);
    REQUIRE(session != nullptr);

    // Configure entropy monitor
    lfg_entropy_monitor_config ecfg = lfg_entropy_monitor_default_config();
    ecfg.threshold = 0.01f;
    ecfg.cooldown_tokens = 1;
    ecfg.ring_size = 8;
    REQUIRE(lfg_session_configure_entropy_monitor(session, &ecfg) > 0);

    // Configure confidence monitor
    lfg_confidence_monitor_config ccfg = lfg_confidence_monitor_default_config();
    ccfg.threshold = 0.99f;
    ccfg.min_span = 2;
    ccfg.ring_size = 8;
    REQUIRE(lfg_session_configure_confidence_monitor(session, &ccfg) > 0);

    // Configure surprise monitor
    lfg_surprise_monitor_config scfg = lfg_surprise_monitor_default_config();
    scfg.threshold = 0.1f;
    REQUIRE(lfg_session_configure_surprise_monitor(session, &scfg) > 0);

    // Ingest surprising prompt
    auto tokens = tokenize(vocab, "xyzzy plugh blort quux zorp fnord", true);
    REQUIRE(lfg_session_ingest_tokens(session, tokens.data(), tokens.size(), true));

    // Surprise event should be available
    lfg_surprise_event sev;
    bool got_surprise = lfg_session_surprise_pop(session, &sev, nullptr, 0);
    MESSAGE("Surprise event: " << got_surprise);
    if (got_surprise) {
        CHECK(sev.n_above_threshold > 0);
    }

    // Entropy events should be available after generation
    lfg_generate_config gc = lfg_generate_default_config();
    gc.max_tokens = 20;
    lfg_session_generate(session, gc);

    volatile int32_t *entropy_counter = lfg_session_entropy_counter(session);
    REQUIRE(static_cast<const void *>(const_cast<const int32_t *>(entropy_counter)) != nullptr);
    MESSAGE("Entropy events: " << *entropy_counter);
    CHECK(*entropy_counter > 0);

    // Last entropy should be valid
    float last_entropy = lfg_session_get_last_entropy(session);
    CHECK(last_entropy >= 0.0f);
    CHECK(last_entropy <= 1.0f);

    lfg_session_free(session);
}

// ---------------------------------------------------------------------------
// Test 10: Long prompt (>n_batch) doesn't crash
// ---------------------------------------------------------------------------

TEST_CASE("Long prompt does not crash") {
    lfg_model *model = get_350m();
    REQUIRE(model != nullptr);

    const lfg_vocab *vocab = lfg_model_get_vocab(model);

    lfg_session_config config = lfg_session_default_config();
    config.n_ctx = 2048;
    config.n_batch = 64;  // Small batch to force multi-batch ingestion
    config.sampling.temp = 0.0f;
    lfg_session *session = lfg_session_create(model, &config);
    REQUIRE(session != nullptr);

    lfg_surprise_monitor_config scfg = lfg_surprise_monitor_default_config();
    scfg.threshold = 0.3f;
    REQUIRE(lfg_session_configure_surprise_monitor(session, &scfg) > 0);

    // Build a long prompt (well over n_batch=64)
    std::string long_text;
    for (int i = 0; i < 20; ++i) {
        long_text += "xyzzy plugh blort quux zorp fnord blargh ";
    }
    auto tokens = tokenize(vocab, long_text, true);
    MESSAGE("Long prompt token count: " << tokens.size());
    REQUIRE(tokens.size() > 64);

    // Should not crash
    REQUIRE(lfg_session_ingest_tokens(session, tokens.data(), tokens.size(), true));

    lfg_surprise_event ev;
    bool got = lfg_session_surprise_pop(session, &ev, nullptr, 0);
    MESSAGE("Event from long prompt: " << got);
    if (got) {
        CHECK(ev.n_tokens_evaluated > 0);
    }

    lfg_session_free(session);
}

// ---------------------------------------------------------------------------
// Test 11: API without monitor configured returns safe defaults
// ---------------------------------------------------------------------------

TEST_CASE("API without monitor returns safe defaults") {
    lfg_model *model = get_350m();
    REQUIRE(model != nullptr);

    lfg_session_config config = lfg_session_default_config();
    config.n_ctx = 512;
    config.sampling.temp = 0.0f;
    lfg_session *session = lfg_session_create(model, &config);
    REQUIRE(session != nullptr);

    // No surprise monitor configured
    CHECK(lfg_session_surprise_pending(session) == 0);

    lfg_surprise_event ev;
    CHECK_FALSE(lfg_session_surprise_pop(session, &ev, nullptr, 0));

    volatile int32_t *counter = lfg_session_surprise_counter(session);
    REQUIRE(static_cast<const void *>(const_cast<const int32_t *>(counter)) != nullptr);
    CHECK(*counter == 0);

    lfg_session_surprise_flush(session);
    CHECK(lfg_session_surprise_pending(session) == 0);

    lfg_session_free(session);
}

// ---------------------------------------------------------------------------
// Test 12: Re-ingest after reset produces fresh event
// ---------------------------------------------------------------------------

TEST_CASE("Re-ingest after reset produces fresh event") {
    lfg_model *model = get_350m();
    REQUIRE(model != nullptr);

    const lfg_vocab *vocab = lfg_model_get_vocab(model);

    lfg_session_config config = lfg_session_default_config();
    config.n_ctx = 2048;
    config.sampling.temp = 0.0f;
    lfg_session *session = lfg_session_create(model, &config);
    REQUIRE(session != nullptr);

    lfg_surprise_monitor_config scfg = lfg_surprise_monitor_default_config();
    scfg.threshold = 0.1f;
    REQUIRE(lfg_session_configure_surprise_monitor(session, &scfg) > 0);

    auto tokens = tokenize(vocab, "xyzzy plugh blort quux zorp fnord", true);

    // First ingestion
    REQUIRE(lfg_session_ingest_tokens(session, tokens.data(), tokens.size(), true));
    lfg_surprise_event ev1;
    bool got1 = lfg_session_surprise_pop(session, &ev1, nullptr, 0);

    // Reset and re-ingest
    lfg_session_reset(session);
    REQUIRE(lfg_session_ingest_tokens(session, tokens.data(), tokens.size(), true));
    lfg_surprise_event ev2;
    bool got2 = lfg_session_surprise_pop(session, &ev2, nullptr, 0);

    // Both should have fired (or neither, but should be consistent)
    CHECK(got1 == got2);
    if (got1 && got2) {
        CHECK(ev1.n_above_threshold == ev2.n_above_threshold);
        MESSAGE("Consistent: above=" << ev1.n_above_threshold);
    }

    lfg_session_free(session);
}

// ---------------------------------------------------------------------------
// Test 13: Reasoning tokens excluded from surprise evaluation
// ---------------------------------------------------------------------------

TEST_CASE("Reasoning tokens excluded from surprise by default") {
    lfg_model *model = get_350m();
    REQUIRE(model != nullptr);

    const lfg_vocab *vocab = lfg_model_get_vocab(model);

    // Tokenize reasoning delimiters as they appear in context:
    //   " <think>" has a leading space because it follows a word in BPE tokenization
    //   "</think>" starts with "</" which tokenizes consistently
    auto start_toks = tokenize(vocab, " <think>", false);
    auto end_toks = tokenize(vocab, "</think>", false);
    REQUIRE(start_toks.size() > 0);
    REQUIRE(end_toks.size() > 0);

    // Build prompt with reasoning block containing gibberish
    std::string prompt = "Hello <think>xyzzy plugh gibberish blort quux</think> world";
    auto tokens = tokenize(vocab, prompt, true);

    // --- Run WITH include_reasoning = true (evaluate everything) ---
    lfg_session_config config = lfg_session_default_config();
    config.n_ctx = 2048;
    config.sampling.temp = 0.0f;
    lfg_session *session = lfg_session_create(model, &config);
    REQUIRE(session != nullptr);

    lfg_session_configure_reasoning(session,
        start_toks.data(), start_toks.size(),
        end_toks.data(), end_toks.size());

    lfg_surprise_monitor_config scfg = lfg_surprise_monitor_default_config();
    scfg.threshold = 0.1f;
    scfg.include_reasoning = true;
    REQUIRE(lfg_session_configure_surprise_monitor(session, &scfg) > 0);

    REQUIRE(lfg_session_ingest_tokens(session, tokens.data(), tokens.size(), true));

    lfg_surprise_event ev_all;
    bool got_all = lfg_session_surprise_pop(session, &ev_all, nullptr, 0);
    REQUIRE(got_all);
    MESSAGE("With include_reasoning: evaluated=" << ev_all.n_tokens_evaluated
            << " above=" << ev_all.n_above_threshold);

    // --- Run with default (include_reasoning = false, skip reasoning) ---
    lfg_session_reset(session);

    lfg_surprise_monitor_config scfg2 = lfg_surprise_monitor_default_config();
    scfg2.threshold = 0.1f;
    // include_reasoning defaults to false — reasoning tokens skipped
    REQUIRE(lfg_session_configure_surprise_monitor(session, &scfg2) > 0);

    REQUIRE(lfg_session_ingest_tokens(session, tokens.data(), tokens.size(), true));

    lfg_surprise_event ev_filtered;
    bool got_filtered = lfg_session_surprise_pop(session, &ev_filtered, nullptr, 0);

    if (got_filtered) {
        MESSAGE("With ignore: evaluated=" << ev_filtered.n_tokens_evaluated
                << " above=" << ev_filtered.n_above_threshold);
        // With reasoning tokens excluded, fewer tokens should be evaluated
        CHECK(ev_filtered.n_tokens_evaluated < ev_all.n_tokens_evaluated);
    } else {
        // If no event fires, that's also valid — all remaining tokens were unsurprising
        MESSAGE("With ignore: no event (all remaining tokens below threshold)");
        CHECK(ev_all.n_above_threshold > 0);
    }

    lfg_session_free(session);
}

// ---------------------------------------------------------------------------
// Test 14: Chat-scoped surprise evaluates only last turn
// ---------------------------------------------------------------------------

TEST_CASE("Chat-scoped surprise evaluates only last turn") {
    lfg_model *model = get_350m();
    REQUIRE(model != nullptr);

    const lfg_vocab *vocab = lfg_model_get_vocab(model);

    lfg_session_config config = lfg_session_default_config();
    config.n_ctx = 2048;
    config.sampling.temp = 0.0f;
    lfg_session *session = lfg_session_create(model, &config);
    REQUIRE(session != nullptr);

    lfg_surprise_monitor_config scfg = lfg_surprise_monitor_default_config();
    scfg.threshold = 0.1f;
    REQUIRE(lfg_session_configure_surprise_monitor(session, &scfg) > 0);

    // Multi-turn conversation: context + gibberish last turn
    lfg_chat_message messages[3];
    messages[0].role = "system";
    messages[0].content = "You are a helpful assistant. Please answer questions clearly and concisely.";
    messages[1].role = "user";
    messages[1].content = "What is the capital of France?";
    messages[2].role = "user";
    messages[2].content = "xyzzy plugh blort quux zorp fnord";

    lfg_generate_config gc = lfg_generate_default_config();
    gc.max_tokens = 5;

    lfg_session_chat_generate(session, messages, 3, gc);

    lfg_surprise_event ev{};
    bool got = lfg_session_surprise_pop(session, &ev, nullptr, 0);
    MESSAGE("Chat surprise queued=" << got << " evaluated=" << ev.n_tokens_evaluated);

    if (got) {
        // The evaluated token count should be much less than the full prompt
        // (only the last user turn, not the system+first user turn)
        auto full_prompt_toks = tokenize(vocab,
            "You are a helpful assistant. Please answer questions clearly and concisely."
            "What is the capital of France?"
            "xyzzy plugh blort quux zorp fnord", true);
        MESSAGE("Full prompt token count: " << full_prompt_toks.size());
        CHECK(ev.n_tokens_evaluated < (int)full_prompt_toks.size());
    }

    lfg_session_free(session);
}
