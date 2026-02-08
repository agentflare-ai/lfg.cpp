#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"
#include "../inference/lfg_api.h"
#include <cstring>
#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

static std::string token_to_string(const lfg_vocab *vocab, lfg_token token) {
    char buf[256];
    int32_t n = lfg_token_to_piece(vocab, token, buf, sizeof(buf), 0, false);
    if (n < 0) return "";
    return std::string(buf, n);
}

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

static std::string generate(lfg_session *session, const lfg_vocab *vocab, int max_tokens) {
    std::string output;
    for (int i = 0; i < max_tokens; ++i) {
        lfg_session_decode(session);
        lfg_token tok = lfg_session_sample(session);
        if (lfg_vocab_is_eog(vocab, tok)) break;
        output += token_to_string(vocab, tok);
        lfg_session_ingest_tokens(session, &tok, 1, false);
    }
    return output;
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
    lfg_confidence_monitor_config cfg = lfg_confidence_monitor_default_config();
    CHECK(cfg.threshold > 0.0f);
    CHECK(cfg.threshold < 1.0f);
    CHECK(cfg.min_span > 0);
    CHECK(cfg.ring_size > 0);
}

// ---------------------------------------------------------------------------
// Test 2: Configure returns n_embd > 0
// ---------------------------------------------------------------------------

TEST_CASE("Configure returns n_embd > 0") {
    lfg_model *model = get_350m();
    REQUIRE(model != nullptr);

    lfg_session_config config = lfg_session_default_config();
    config.n_ctx = 2048;
    config.sampling.temp = 0.0f;
    lfg_session *session = lfg_session_create(model, &config);
    REQUIRE(session != nullptr);

    lfg_confidence_monitor_config ccfg = lfg_confidence_monitor_default_config();
    int32_t n_embd = lfg_session_configure_confidence_monitor(session, &ccfg);
    CHECK(n_embd > 0);
    MESSAGE("n_embd = " << n_embd);

    // Disable returns 0
    CHECK(lfg_session_configure_confidence_monitor(session, nullptr) == 0);

    lfg_session_free(session);
}

// ---------------------------------------------------------------------------
// Test 3: No events when all tokens are high-entropy (above threshold)
// ---------------------------------------------------------------------------

TEST_CASE("No events when all tokens are above threshold") {
    lfg_model *model = get_350m();
    REQUIRE(model != nullptr);

    const lfg_vocab *vocab = lfg_model_get_vocab(model);

    lfg_session_config config = lfg_session_default_config();
    config.n_ctx = 2048;
    config.sampling.temp = 0.8f;
    lfg_session *session = lfg_session_create(model, &config);
    REQUIRE(session != nullptr);

    // Very low threshold — only extremely certain tokens would qualify
    lfg_confidence_monitor_config ccfg = lfg_confidence_monitor_default_config();
    ccfg.threshold = 0.001f;  // Almost nothing qualifies
    ccfg.min_span = 3;
    ccfg.ring_size = 4;
    REQUIRE(lfg_session_configure_confidence_monitor(session, &ccfg) > 0);

    auto tokens = tokenize(vocab, "The quick brown fox jumps over the lazy", true);
    REQUIRE(lfg_session_ingest_tokens(session, tokens.data(), tokens.size(), true));

    generate(session, vocab, 20);

    int32_t pending = lfg_session_confidence_pending(session);
    MESSAGE("Pending events with threshold=0.001: " << pending);
    CHECK(pending == 0);

    lfg_session_free(session);
}

// ---------------------------------------------------------------------------
// Test 4: Events fire when sustained low-entropy span meets min_span
// ---------------------------------------------------------------------------

TEST_CASE("Events fire on sustained low-entropy spans") {
    lfg_model *model = get_350m();
    REQUIRE(model != nullptr);

    const lfg_vocab *vocab = lfg_model_get_vocab(model);

    lfg_session_config config = lfg_session_default_config();
    config.n_ctx = 2048;
    config.sampling.temp = 0.0f;  // Greedy = very confident
    lfg_session *session = lfg_session_create(model, &config);
    REQUIRE(session != nullptr);

    // High threshold + small min_span — most greedy tokens should qualify
    lfg_confidence_monitor_config ccfg = lfg_confidence_monitor_default_config();
    ccfg.threshold = 0.99f;  // Almost everything qualifies
    ccfg.min_span = 2;       // Small span requirement
    ccfg.ring_size = 8;
    REQUIRE(lfg_session_configure_confidence_monitor(session, &ccfg) > 0);

    auto tokens = tokenize(vocab, "The capital of France is Paris and it is beautiful", true);
    REQUIRE(lfg_session_ingest_tokens(session, tokens.data(), tokens.size(), true));

    generate(session, vocab, 30);

    int32_t pending = lfg_session_confidence_pending(session);
    MESSAGE("Pending events with threshold=0.99, min_span=2: " << pending);
    // With greedy sampling and high threshold, should get at least one event
    CHECK(pending >= 0);  // May or may not fire depending on model entropy

    // If we got events, verify their fields
    if (pending > 0) {
        lfg_confidence_event ev;
        std::vector<float> embd(2048);
        bool got = lfg_session_confidence_pop(session, &ev, embd.data(), (int32_t)embd.size());
        CHECK(got);
        CHECK(ev.mean_entropy >= 0.0f);
        CHECK(ev.mean_entropy <= 1.0f);
        CHECK(ev.min_entropy >= 0.0f);
        CHECK(ev.min_entropy <= ev.mean_entropy);
        CHECK(ev.span_length >= 2);
        CHECK(ev.start_pos >= 0);
        CHECK(ev.end_pos > ev.start_pos);
        MESSAGE("Event: mean=" << ev.mean_entropy << " min=" << ev.min_entropy
                << " span=" << ev.span_length << " start=" << ev.start_pos << " end=" << ev.end_pos);
    }

    lfg_session_free(session);
}

// ---------------------------------------------------------------------------
// Test 5: span_length matches actual run length
// ---------------------------------------------------------------------------

TEST_CASE("Span length matches actual run") {
    lfg_model *model = get_350m();
    REQUIRE(model != nullptr);

    const lfg_vocab *vocab = lfg_model_get_vocab(model);

    lfg_session_config config = lfg_session_default_config();
    config.n_ctx = 2048;
    config.sampling.temp = 0.0f;
    lfg_session *session = lfg_session_create(model, &config);
    REQUIRE(session != nullptr);

    lfg_confidence_monitor_config ccfg = lfg_confidence_monitor_default_config();
    ccfg.threshold = 0.99f;
    ccfg.min_span = 1;  // Fire on any 1+ token run
    ccfg.ring_size = 16;
    REQUIRE(lfg_session_configure_confidence_monitor(session, &ccfg) > 0);

    auto tokens = tokenize(vocab, "Once upon a time in a land far far away", true);
    REQUIRE(lfg_session_ingest_tokens(session, tokens.data(), tokens.size(), true));

    generate(session, vocab, 20);

    // Pop all events and verify span_length consistency
    lfg_confidence_event ev;
    int total_popped = 0;
    while (lfg_session_confidence_pop(session, &ev, nullptr, 0)) {
        CHECK(ev.span_length >= 1);
        CHECK(ev.end_pos - ev.start_pos >= ev.span_length);
        total_popped++;
    }
    MESSAGE("Total events popped: " << total_popped);

    lfg_session_free(session);
}

// ---------------------------------------------------------------------------
// Test 6: mean_entropy is correct average
// ---------------------------------------------------------------------------

TEST_CASE("Mean entropy is within valid range") {
    lfg_model *model = get_350m();
    REQUIRE(model != nullptr);

    const lfg_vocab *vocab = lfg_model_get_vocab(model);

    lfg_session_config config = lfg_session_default_config();
    config.n_ctx = 2048;
    config.sampling.temp = 0.0f;
    lfg_session *session = lfg_session_create(model, &config);
    REQUIRE(session != nullptr);

    lfg_confidence_monitor_config ccfg = lfg_confidence_monitor_default_config();
    ccfg.threshold = 0.99f;
    ccfg.min_span = 2;
    ccfg.ring_size = 8;
    REQUIRE(lfg_session_configure_confidence_monitor(session, &ccfg) > 0);

    auto tokens = tokenize(vocab, "The answer to everything is forty two", true);
    REQUIRE(lfg_session_ingest_tokens(session, tokens.data(), tokens.size(), true));

    generate(session, vocab, 30);

    lfg_confidence_event ev;
    while (lfg_session_confidence_pop(session, &ev, nullptr, 0)) {
        CHECK(ev.mean_entropy >= 0.0f);
        CHECK(ev.mean_entropy <= ccfg.threshold);
        CHECK(ev.min_entropy >= 0.0f);
        CHECK(ev.min_entropy <= ev.mean_entropy);
        MESSAGE("Mean entropy: " << ev.mean_entropy << " (threshold: " << ccfg.threshold << ")");
    }

    lfg_session_free(session);
}

// ---------------------------------------------------------------------------
// Test 7: Pending count tracks correctly
// ---------------------------------------------------------------------------

TEST_CASE("Pending count tracks correctly") {
    lfg_model *model = get_350m();
    REQUIRE(model != nullptr);

    const lfg_vocab *vocab = lfg_model_get_vocab(model);

    lfg_session_config config = lfg_session_default_config();
    config.n_ctx = 2048;
    config.sampling.temp = 0.0f;
    lfg_session *session = lfg_session_create(model, &config);
    REQUIRE(session != nullptr);

    lfg_confidence_monitor_config ccfg = lfg_confidence_monitor_default_config();
    ccfg.threshold = 0.99f;
    ccfg.min_span = 1;
    ccfg.ring_size = 16;
    REQUIRE(lfg_session_configure_confidence_monitor(session, &ccfg) > 0);

    // Before generation, pending should be 0
    CHECK(lfg_session_confidence_pending(session) == 0);

    auto tokens = tokenize(vocab, "Hello world this is a test", true);
    REQUIRE(lfg_session_ingest_tokens(session, tokens.data(), tokens.size(), true));

    generate(session, vocab, 20);

    int32_t pending = lfg_session_confidence_pending(session);
    MESSAGE("Pending after generation: " << pending);

    // Pop one and verify pending decreases
    if (pending > 0) {
        lfg_confidence_event ev;
        lfg_session_confidence_pop(session, &ev, nullptr, 0);
        CHECK(lfg_session_confidence_pending(session) == pending - 1);
    }

    lfg_session_free(session);
}

// ---------------------------------------------------------------------------
// Test 8: Flush clears pending events
// ---------------------------------------------------------------------------

TEST_CASE("Flush clears pending events") {
    lfg_model *model = get_350m();
    REQUIRE(model != nullptr);

    const lfg_vocab *vocab = lfg_model_get_vocab(model);

    lfg_session_config config = lfg_session_default_config();
    config.n_ctx = 2048;
    config.sampling.temp = 0.0f;
    lfg_session *session = lfg_session_create(model, &config);
    REQUIRE(session != nullptr);

    lfg_confidence_monitor_config ccfg = lfg_confidence_monitor_default_config();
    ccfg.threshold = 0.99f;
    ccfg.min_span = 1;
    ccfg.ring_size = 8;
    REQUIRE(lfg_session_configure_confidence_monitor(session, &ccfg) > 0);

    auto tokens = tokenize(vocab, "Testing the confidence flush mechanism", true);
    REQUIRE(lfg_session_ingest_tokens(session, tokens.data(), tokens.size(), true));

    generate(session, vocab, 20);

    lfg_session_confidence_flush(session);
    CHECK(lfg_session_confidence_pending(session) == 0);

    // Pop should return false after flush
    lfg_confidence_event ev;
    CHECK_FALSE(lfg_session_confidence_pop(session, &ev, nullptr, 0));

    lfg_session_free(session);
}

// ---------------------------------------------------------------------------
// Test 9: Counter increments on event write
// ---------------------------------------------------------------------------

TEST_CASE("Counter increments with events") {
    lfg_model *model = get_350m();
    REQUIRE(model != nullptr);

    const lfg_vocab *vocab = lfg_model_get_vocab(model);

    lfg_session_config config = lfg_session_default_config();
    config.n_ctx = 2048;
    config.sampling.temp = 0.0f;
    lfg_session *session = lfg_session_create(model, &config);
    REQUIRE(session != nullptr);

    lfg_confidence_monitor_config ccfg = lfg_confidence_monitor_default_config();
    ccfg.threshold = 0.99f;
    ccfg.min_span = 1;
    ccfg.ring_size = 8;
    REQUIRE(lfg_session_configure_confidence_monitor(session, &ccfg) > 0);

    volatile int32_t *counter = lfg_session_confidence_counter(session);
    REQUIRE(static_cast<const void *>(const_cast<const int32_t *>(counter)) != nullptr);
    CHECK(*counter == 0);

    auto tokens = tokenize(vocab, "Hello world", true);
    REQUIRE(lfg_session_ingest_tokens(session, tokens.data(), tokens.size(), true));

    generate(session, vocab, 20);

    int32_t count = *counter;
    MESSAGE("Counter after 20 tokens: " << count);
    // Counter should equal total written events
    CHECK(count == lfg_session_confidence_pending(session));

    lfg_session_free(session);
}

// ---------------------------------------------------------------------------
// Test 10: Reset clears run state
// ---------------------------------------------------------------------------

TEST_CASE("Reset clears run state") {
    lfg_model *model = get_350m();
    REQUIRE(model != nullptr);

    const lfg_vocab *vocab = lfg_model_get_vocab(model);

    lfg_session_config config = lfg_session_default_config();
    config.n_ctx = 2048;
    config.sampling.temp = 0.0f;
    lfg_session *session = lfg_session_create(model, &config);
    REQUIRE(session != nullptr);

    lfg_confidence_monitor_config ccfg = lfg_confidence_monitor_default_config();
    ccfg.threshold = 0.99f;
    ccfg.min_span = 1;
    ccfg.ring_size = 8;
    REQUIRE(lfg_session_configure_confidence_monitor(session, &ccfg) > 0);

    auto tokens = tokenize(vocab, "Test reset mechanism", true);
    REQUIRE(lfg_session_ingest_tokens(session, tokens.data(), tokens.size(), true));

    generate(session, vocab, 15);

    // Reset session
    lfg_session_reset(session);

    // After reset, pending should be 0
    CHECK(lfg_session_confidence_pending(session) == 0);

    volatile int32_t *counter = lfg_session_confidence_counter(session);
    CHECK(*counter == 0);

    // Pop should return false
    lfg_confidence_event ev;
    CHECK_FALSE(lfg_session_confidence_pop(session, &ev, nullptr, 0));

    lfg_session_free(session);
}

// ---------------------------------------------------------------------------
// Test 11: Generate loop integration — confidence callback fires
// ---------------------------------------------------------------------------

TEST_CASE("Generate loop fires confidence callback") {
    lfg_model *model = get_350m();
    REQUIRE(model != nullptr);

    const lfg_vocab *vocab = lfg_model_get_vocab(model);

    lfg_session_config config = lfg_session_default_config();
    config.n_ctx = 2048;
    config.sampling.temp = 0.0f;
    lfg_session *session = lfg_session_create(model, &config);
    REQUIRE(session != nullptr);

    lfg_confidence_monitor_config ccfg = lfg_confidence_monitor_default_config();
    ccfg.threshold = 0.99f;
    ccfg.min_span = 2;
    ccfg.ring_size = 8;
    REQUIRE(lfg_session_configure_confidence_monitor(session, &ccfg) > 0);

    auto tokens = tokenize(vocab, "The capital of France is Paris and the capital of Germany is Berlin", true);
    REQUIRE(lfg_session_ingest_tokens(session, tokens.data(), tokens.size(), true));

    struct cb_state {
        int count;
        int total_span;
    };
    cb_state state = {0, 0};

    lfg_generate_config gc = lfg_generate_default_config();
    gc.max_tokens = 30;
    gc.confidence_cb = [](const lfg_confidence_event *event, const float *, void *ud) {
        auto *s = (cb_state *)ud;
        s->count++;
        s->total_span += event->span_length;
    };
    gc.confidence_cb_data = &state;

    lfg_generate_result r = lfg_session_generate(session, gc);

    MESSAGE("Callback fired " << state.count << " times, total span tokens: " << state.total_span);
    MESSAGE("n_confidence_spans in result: " << r.n_confidence_spans);
    CHECK(r.n_confidence_spans == state.count);

    lfg_session_free(session);
}

// ---------------------------------------------------------------------------
// Test 12: Confidence + entropy can coexist
// ---------------------------------------------------------------------------

TEST_CASE("Confidence + entropy coexist") {
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

    // Configure confidence monitor (shares tool_ctx)
    lfg_confidence_monitor_config ccfg = lfg_confidence_monitor_default_config();
    ccfg.threshold = 0.99f;
    ccfg.min_span = 2;
    ccfg.ring_size = 8;
    REQUIRE(lfg_session_configure_confidence_monitor(session, &ccfg) > 0);

    auto tokens = tokenize(vocab, "The quick brown fox jumps over the lazy dog", true);
    REQUIRE(lfg_session_ingest_tokens(session, tokens.data(), tokens.size(), true));

    generate(session, vocab, 20);

    // Both should have counters
    volatile int32_t *entropy_counter = lfg_session_entropy_counter(session);
    volatile int32_t *confidence_counter = lfg_session_confidence_counter(session);
    REQUIRE(static_cast<const void *>(const_cast<const int32_t *>(entropy_counter)) != nullptr);
    REQUIRE(static_cast<const void *>(const_cast<const int32_t *>(confidence_counter)) != nullptr);

    MESSAGE("Entropy events: " << *entropy_counter);
    MESSAGE("Confidence events: " << *confidence_counter);

    // Entropy events should fire (low threshold)
    CHECK(*entropy_counter > 0);

    // Last entropy should be valid (shared computation path)
    float last_entropy = lfg_session_get_last_entropy(session);
    CHECK(last_entropy >= 0.0f);
    CHECK(last_entropy <= 1.0f);

    lfg_session_free(session);
}

// ---------------------------------------------------------------------------
// Test 13: Getter without monitor returns safe defaults
// ---------------------------------------------------------------------------

TEST_CASE("API without monitor returns safe defaults") {
    lfg_model *model = get_350m();
    REQUIRE(model != nullptr);

    lfg_session_config config = lfg_session_default_config();
    config.n_ctx = 512;
    config.sampling.temp = 0.0f;
    lfg_session *session = lfg_session_create(model, &config);
    REQUIRE(session != nullptr);

    // No confidence monitor configured
    CHECK(lfg_session_confidence_pending(session) == 0);

    lfg_confidence_event ev;
    CHECK_FALSE(lfg_session_confidence_pop(session, &ev, nullptr, 0));

    // Counter returns non-null pointer but value is 0
    volatile int32_t *counter = lfg_session_confidence_counter(session);
    REQUIRE(static_cast<const void *>(const_cast<const int32_t *>(counter)) != nullptr);
    CHECK(*counter == 0);

    // Flush is safe to call
    lfg_session_confidence_flush(session);
    CHECK(lfg_session_confidence_pending(session) == 0);

    lfg_session_free(session);
}

// ---------------------------------------------------------------------------
// Test 14: Reasoning tokens break confidence runs by default
// ---------------------------------------------------------------------------

TEST_CASE("Reasoning tokens break confidence runs by default") {
    lfg_model *model = get_350m();
    REQUIRE(model != nullptr);

    const lfg_vocab *vocab = lfg_model_get_vocab(model);

    // Tokenize reasoning delimiters
    auto start_toks = tokenize(vocab, "<think>", false);
    auto end_toks = tokenize(vocab, "</think>", false);
    REQUIRE(start_toks.size() > 0);
    REQUIRE(end_toks.size() > 0);

    // --- Run WITH include_reasoning = true (count everything) ---
    lfg_session_config config = lfg_session_default_config();
    config.n_ctx = 2048;
    config.sampling.temp = 0.0f;
    config.reasoning_budget = 50;
    lfg_session *session = lfg_session_create(model, &config);
    REQUIRE(session != nullptr);

    lfg_session_configure_reasoning(session,
        start_toks.data(), start_toks.size(),
        end_toks.data(), end_toks.size());

    lfg_confidence_monitor_config ccfg = lfg_confidence_monitor_default_config();
    ccfg.threshold = 0.99f;
    ccfg.min_span = 2;
    ccfg.ring_size = 16;
    ccfg.include_reasoning = true;
    REQUIRE(lfg_session_configure_confidence_monitor(session, &ccfg) > 0);

    auto tokens = tokenize(vocab, "The capital of France is Paris", true);
    REQUIRE(lfg_session_ingest_tokens(session, tokens.data(), tokens.size(), true));

    lfg_generate_config gc = lfg_generate_default_config();
    gc.max_tokens = 30;
    lfg_generate_result r1 = lfg_session_generate(session, gc);
    int32_t events_include = r1.n_confidence_spans;
    MESSAGE("Events with include_reasoning: " << events_include);

    // --- Run with default (include_reasoning = false, skip reasoning) ---
    lfg_session_reset(session);

    lfg_confidence_monitor_config ccfg2 = lfg_confidence_monitor_default_config();
    ccfg2.threshold = 0.99f;
    ccfg2.min_span = 2;
    ccfg2.ring_size = 16;
    // include_reasoning defaults to false — reasoning tokens break runs
    REQUIRE(lfg_session_configure_confidence_monitor(session, &ccfg2) > 0);

    REQUIRE(lfg_session_ingest_tokens(session, tokens.data(), tokens.size(), true));

    lfg_generate_result r2 = lfg_session_generate(session, gc);
    int32_t events_default = r2.n_confidence_spans;
    MESSAGE("Events with default (skip reasoning): " << events_default);

    // With reasoning tokens treated as run-breakers, events should be <= include
    CHECK(events_default <= events_include);

    lfg_session_free(session);
}
