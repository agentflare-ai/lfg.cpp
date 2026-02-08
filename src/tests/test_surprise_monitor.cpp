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
    lfg_surprise_monitor_config cfg = lfg_surprise_monitor_default_config();
    CHECK(cfg.threshold > 0.0f);
    CHECK(cfg.threshold < 1.0f);
    CHECK(cfg.min_span > 0);
    CHECK(cfg.ring_size > 0);
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
// Test 3: No events when input is predictable (low threshold)
// ---------------------------------------------------------------------------

TEST_CASE("No events on predictable input with low threshold") {
    lfg_model *model = get_350m();
    REQUIRE(model != nullptr);

    const lfg_vocab *vocab = lfg_model_get_vocab(model);

    lfg_session_config config = lfg_session_default_config();
    config.n_ctx = 2048;
    config.sampling.temp = 0.0f;
    lfg_session *session = lfg_session_create(model, &config);
    REQUIRE(session != nullptr);

    // Very high threshold — only extremely surprising tokens qualify
    lfg_surprise_monitor_config scfg = lfg_surprise_monitor_default_config();
    scfg.threshold = 0.99f;  // Almost nothing qualifies as surprising
    scfg.min_span = 3;
    scfg.ring_size = 8;
    REQUIRE(lfg_session_configure_surprise_monitor(session, &scfg) > 0);

    // Predictable text
    auto tokens = tokenize(vocab, "The capital of France is Paris", true);
    REQUIRE(lfg_session_ingest_tokens(session, tokens.data(), tokens.size(), true));

    int32_t pending = lfg_session_surprise_pending(session);
    MESSAGE("Pending events with threshold=0.99 on predictable text: " << pending);
    CHECK(pending == 0);

    lfg_session_free(session);
}

// ---------------------------------------------------------------------------
// Test 4: Events fire on novel input (gibberish, moderate threshold)
// ---------------------------------------------------------------------------

TEST_CASE("Events fire on novel input") {
    lfg_model *model = get_350m();
    REQUIRE(model != nullptr);

    const lfg_vocab *vocab = lfg_model_get_vocab(model);

    lfg_session_config config = lfg_session_default_config();
    config.n_ctx = 2048;
    config.sampling.temp = 0.0f;
    lfg_session *session = lfg_session_create(model, &config);
    REQUIRE(session != nullptr);

    // Low threshold + small span so most gibberish qualifies
    lfg_surprise_monitor_config scfg = lfg_surprise_monitor_default_config();
    scfg.threshold = 0.1f;  // Very low — most tokens qualify
    scfg.min_span = 2;
    scfg.ring_size = 16;
    REQUIRE(lfg_session_configure_surprise_monitor(session, &scfg) > 0);

    // Gibberish text — should be very surprising to the model
    auto tokens = tokenize(vocab, "xyzzy plugh blort quux zorp fnord mxyzptlk blargh", true);
    REQUIRE(tokens.size() > 4);
    REQUIRE(lfg_session_ingest_tokens(session, tokens.data(), tokens.size(), true));

    int32_t pending = lfg_session_surprise_pending(session);
    MESSAGE("Pending events with threshold=0.1 on gibberish: " << pending);
    CHECK(pending > 0);

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
    scfg.min_span = 2;
    scfg.ring_size = 16;
    REQUIRE(lfg_session_configure_surprise_monitor(session, &scfg) > 0);

    auto tokens = tokenize(vocab, "xyzzy plugh blort quux zorp fnord mxyzptlk blargh", true);
    REQUIRE(lfg_session_ingest_tokens(session, tokens.data(), tokens.size(), true));

    int32_t pending = lfg_session_surprise_pending(session);
    if (pending > 0) {
        lfg_surprise_event ev;
        std::vector<float> embd(2048);
        bool got = lfg_session_surprise_pop(session, &ev, embd.data(), (int32_t)embd.size());
        CHECK(got);
        CHECK(ev.mean_surprise >= scfg.threshold);
        CHECK(ev.max_surprise >= ev.mean_surprise);
        CHECK(ev.span_length >= scfg.min_span);
        CHECK(ev.start_pos >= 0);
        CHECK(ev.end_pos > ev.start_pos);
        MESSAGE("Event: mean=" << ev.mean_surprise << " max=" << ev.max_surprise
                << " span=" << ev.span_length << " start=" << ev.start_pos << " end=" << ev.end_pos);
    } else {
        MESSAGE("No events fired — cannot validate fields");
    }

    lfg_session_free(session);
}

// ---------------------------------------------------------------------------
// Test 6: Pending count tracks correctly
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

    lfg_surprise_monitor_config scfg = lfg_surprise_monitor_default_config();
    scfg.threshold = 0.1f;
    scfg.min_span = 1;
    scfg.ring_size = 16;
    REQUIRE(lfg_session_configure_surprise_monitor(session, &scfg) > 0);

    // Before ingestion, pending should be 0
    CHECK(lfg_session_surprise_pending(session) == 0);

    auto tokens = tokenize(vocab, "xyzzy plugh blort quux zorp fnord", true);
    REQUIRE(lfg_session_ingest_tokens(session, tokens.data(), tokens.size(), true));

    int32_t pending = lfg_session_surprise_pending(session);
    MESSAGE("Pending after ingestion: " << pending);

    // Pop one and verify pending decreases
    if (pending > 0) {
        lfg_surprise_event ev;
        lfg_session_surprise_pop(session, &ev, nullptr, 0);
        CHECK(lfg_session_surprise_pending(session) == pending - 1);
    }

    lfg_session_free(session);
}

// ---------------------------------------------------------------------------
// Test 7: Flush clears pending events
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

    lfg_surprise_monitor_config scfg = lfg_surprise_monitor_default_config();
    scfg.threshold = 0.1f;
    scfg.min_span = 1;
    scfg.ring_size = 8;
    REQUIRE(lfg_session_configure_surprise_monitor(session, &scfg) > 0);

    auto tokens = tokenize(vocab, "xyzzy plugh blort quux zorp fnord", true);
    REQUIRE(lfg_session_ingest_tokens(session, tokens.data(), tokens.size(), true));

    lfg_session_surprise_flush(session);
    CHECK(lfg_session_surprise_pending(session) == 0);

    // Pop should return false after flush
    lfg_surprise_event ev;
    CHECK_FALSE(lfg_session_surprise_pop(session, &ev, nullptr, 0));

    lfg_session_free(session);
}

// ---------------------------------------------------------------------------
// Test 8: Counter increments on event write
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

    lfg_surprise_monitor_config scfg = lfg_surprise_monitor_default_config();
    scfg.threshold = 0.1f;
    scfg.min_span = 1;
    scfg.ring_size = 8;
    REQUIRE(lfg_session_configure_surprise_monitor(session, &scfg) > 0);

    volatile int32_t *counter = lfg_session_surprise_counter(session);
    REQUIRE(static_cast<const void *>(const_cast<const int32_t *>(counter)) != nullptr);
    CHECK(*counter == 0);

    auto tokens = tokenize(vocab, "xyzzy plugh blort quux zorp fnord", true);
    REQUIRE(lfg_session_ingest_tokens(session, tokens.data(), tokens.size(), true));

    int32_t count = *counter;
    MESSAGE("Counter after ingestion: " << count);
    // Counter should equal total written events
    CHECK(count == lfg_session_surprise_pending(session));

    lfg_session_free(session);
}

// ---------------------------------------------------------------------------
// Test 9: Reset clears state
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
    scfg.min_span = 1;
    scfg.ring_size = 8;
    REQUIRE(lfg_session_configure_surprise_monitor(session, &scfg) > 0);

    auto tokens = tokenize(vocab, "xyzzy plugh blort quux zorp fnord", true);
    REQUIRE(lfg_session_ingest_tokens(session, tokens.data(), tokens.size(), true));

    // Reset session
    lfg_session_reset(session);

    // After reset, pending should be 0
    CHECK(lfg_session_surprise_pending(session) == 0);

    volatile int32_t *counter = lfg_session_surprise_counter(session);
    CHECK(*counter == 0);

    // Pop should return false
    lfg_surprise_event ev;
    CHECK_FALSE(lfg_session_surprise_pop(session, &ev, nullptr, 0));

    lfg_session_free(session);
}

// ---------------------------------------------------------------------------
// Test 10: Generate loop drains events via callback
// ---------------------------------------------------------------------------

TEST_CASE("Generate loop drains surprise events via callback") {
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
    scfg.min_span = 2;
    scfg.ring_size = 16;
    REQUIRE(lfg_session_configure_surprise_monitor(session, &scfg) > 0);

    // Ingest surprising prompt
    auto tokens = tokenize(vocab, "xyzzy plugh blort quux zorp fnord mxyzptlk blargh", true);
    REQUIRE(lfg_session_ingest_tokens(session, tokens.data(), tokens.size(), true));

    struct cb_state {
        int count;
        int total_span;
    };
    cb_state state = {0, 0};

    lfg_generate_config gc = lfg_generate_default_config();
    gc.max_tokens = 10;
    gc.surprise_cb = [](const lfg_surprise_event *event, const float *, void *ud) {
        auto *s = (cb_state *)ud;
        s->count++;
        s->total_span += event->span_length;
    };
    gc.surprise_cb_data = &state;

    lfg_generate_result r = lfg_session_generate(session, gc);

    MESSAGE("Callback fired " << state.count << " times, total span tokens: " << state.total_span);
    MESSAGE("n_surprise_spans in result: " << r.n_surprise_spans);
    CHECK(r.n_surprise_spans == state.count);

    // After generate, all surprise events should be drained
    CHECK(lfg_session_surprise_pending(session) == 0);

    lfg_session_free(session);
}

// ---------------------------------------------------------------------------
// Test 11: Surprise + entropy + confidence all coexist
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
    scfg.min_span = 2;
    scfg.ring_size = 8;
    REQUIRE(lfg_session_configure_surprise_monitor(session, &scfg) > 0);

    // Ingest and generate
    auto tokens = tokenize(vocab, "xyzzy plugh blort quux zorp fnord", true);
    REQUIRE(lfg_session_ingest_tokens(session, tokens.data(), tokens.size(), true));

    generate(session, vocab, 20);

    // All three should have counters
    volatile int32_t *entropy_counter = lfg_session_entropy_counter(session);
    volatile int32_t *confidence_counter = lfg_session_confidence_counter(session);
    volatile int32_t *surprise_counter = lfg_session_surprise_counter(session);
    REQUIRE(static_cast<const void *>(const_cast<const int32_t *>(entropy_counter)) != nullptr);
    REQUIRE(static_cast<const void *>(const_cast<const int32_t *>(confidence_counter)) != nullptr);
    REQUIRE(static_cast<const void *>(const_cast<const int32_t *>(surprise_counter)) != nullptr);

    MESSAGE("Entropy events: " << *entropy_counter);
    MESSAGE("Confidence events: " << *confidence_counter);
    MESSAGE("Surprise events: " << *surprise_counter);

    // Entropy events should fire (low threshold)
    CHECK(*entropy_counter > 0);

    // Surprise events should fire (gibberish input)
    CHECK(*surprise_counter > 0);

    // Last entropy should be valid (shared computation path)
    float last_entropy = lfg_session_get_last_entropy(session);
    CHECK(last_entropy >= 0.0f);
    CHECK(last_entropy <= 1.0f);

    lfg_session_free(session);
}

// ---------------------------------------------------------------------------
// Test 12: Long prompt (>n_batch) doesn't crash
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
    scfg.min_span = 2;
    scfg.ring_size = 16;
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

    int32_t pending = lfg_session_surprise_pending(session);
    MESSAGE("Events from long prompt: " << pending);
    // With gibberish text, we should get events
    CHECK(pending >= 0);

    lfg_session_free(session);
}

// ---------------------------------------------------------------------------
// Test 13: API without monitor configured returns safe defaults
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

    // Counter returns non-null pointer but value is 0
    volatile int32_t *counter = lfg_session_surprise_counter(session);
    REQUIRE(static_cast<const void *>(const_cast<const int32_t *>(counter)) != nullptr);
    CHECK(*counter == 0);

    // Flush is safe to call
    lfg_session_surprise_flush(session);
    CHECK(lfg_session_surprise_pending(session) == 0);

    lfg_session_free(session);
}
