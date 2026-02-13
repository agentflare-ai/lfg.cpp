#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"
#include "../inference/lfg_api.h"
#include <cstring>
#include <fstream>
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
// Test 1: Events fire on uncertain tokens
// ---------------------------------------------------------------------------

TEST_CASE("Events fire on uncertain tokens") {
    lfg_model *model = get_350m();
    REQUIRE(model != nullptr);

    const lfg_vocab *vocab = lfg_model_get_vocab(model);

    lfg_session_config config = lfg_session_default_config();
    config.n_ctx = 2048;
    config.sampling.temp = 0.8f;  // some randomness to ensure entropy
    lfg_session *session = lfg_session_create(model, &config);
    REQUIRE(session != nullptr);

    // Very low threshold — should fire on most tokens
    lfg_entropy_monitor_config ecfg = lfg_entropy_monitor_default_config();
    ecfg.threshold = 0.01f;  // fire on almost anything
    ecfg.cooldown_tokens = 1;
    ecfg.ring_size = 8;
    REQUIRE(lfg_session_configure_entropy_monitor(session, &ecfg));

    auto tokens = tokenize(vocab, "The quick brown fox jumps over the lazy", true);
    REQUIRE(lfg_session_ingest_tokens(session, tokens.data(), tokens.size(), true));

    // Generate a few tokens
    std::string output = generate(session, vocab, 10);
    MESSAGE("Output: " << output);

    // Should have at least one event
    int32_t pending = lfg_session_entropy_pending(session);
    MESSAGE("Pending events: " << pending);
    CHECK(pending > 0);

    // Pop and verify event fields
    lfg_entropy_event ev;
    std::vector<float> embd(2048);
    bool got = lfg_session_entropy_pop(session, &ev, embd.data(), (int32_t)embd.size());
    CHECK(got);
    CHECK(ev.entropy > 0.0f);
    CHECK(ev.normalized >= 0.0f);
    CHECK(ev.normalized <= 1.0f);
    CHECK(ev.checkpoint_id >= 0);
    CHECK(ev.n_embd > 0);

    lfg_session_free(session);
}

// ---------------------------------------------------------------------------
// Test 2: Cooldown prevents rapid events
// ---------------------------------------------------------------------------

TEST_CASE("Cooldown prevents rapid events") {
    lfg_model *model = get_350m();
    REQUIRE(model != nullptr);

    const lfg_vocab *vocab = lfg_model_get_vocab(model);

    lfg_session_config config = lfg_session_default_config();
    config.n_ctx = 2048;
    config.sampling.temp = 0.8f;
    lfg_session *session = lfg_session_create(model, &config);
    REQUIRE(session != nullptr);

    // threshold=0.01 fires easily, but cooldown=100 prevents rapid events
    lfg_entropy_monitor_config ecfg = lfg_entropy_monitor_default_config();
    ecfg.threshold = 0.01f;
    ecfg.cooldown_tokens = 100;
    ecfg.ring_size = 8;
    REQUIRE(lfg_session_configure_entropy_monitor(session, &ecfg));

    auto tokens = tokenize(vocab, "Once upon a time in a land far away there lived a", true);
    REQUIRE(lfg_session_ingest_tokens(session, tokens.data(), tokens.size(), true));

    // Generate 50 tokens — less than cooldown
    generate(session, vocab, 50);

    // Should have exactly 1 event (initial), not more due to cooldown
    int32_t pending = lfg_session_entropy_pending(session);
    MESSAGE("Pending events with cooldown=100: " << pending);
    CHECK(pending == 1);

    lfg_session_free(session);
}

// ---------------------------------------------------------------------------
// Test 3: Rewind + inject changes output
// ---------------------------------------------------------------------------

TEST_CASE("Rewind changes output") {
    lfg_model *model = get_350m();
    REQUIRE(model != nullptr);

    const lfg_vocab *vocab = lfg_model_get_vocab(model);

    lfg_session_config config = lfg_session_default_config();
    config.n_ctx = 2048;
    config.sampling.temp = 0.0f;  // deterministic
    lfg_session *session = lfg_session_create(model, &config);
    REQUIRE(session != nullptr);

    lfg_entropy_monitor_config ecfg = lfg_entropy_monitor_default_config();
    ecfg.threshold = 0.01f;
    ecfg.cooldown_tokens = 1;
    ecfg.ring_size = 4;
    REQUIRE(lfg_session_configure_entropy_monitor(session, &ecfg));

    auto tokens = tokenize(vocab, "The capital of France is", true);
    REQUIRE(lfg_session_ingest_tokens(session, tokens.data(), tokens.size(), true));

    // Generate a few tokens to trigger events
    std::string output1 = generate(session, vocab, 5);
    MESSAGE("Before rewind: " << output1);

    // Pop an event and rewind
    lfg_entropy_event ev;
    bool got = lfg_session_entropy_pop(session, &ev, nullptr, 0);
    if (got) {
        MESSAGE("Rewinding to checkpoint " << ev.checkpoint_id << " (n_past=" << ev.n_past << ")");
        bool rewound = lfg_session_rewind(session, ev.checkpoint_id);
        CHECK(rewound);

        if (rewound) {
            // Inject different context
            auto inject = tokenize(vocab, " Paris, which", false);
            REQUIRE(lfg_session_ingest_tokens(session, inject.data(), inject.size(), false));

            // Generate again
            std::string output2 = generate(session, vocab, 5);
            MESSAGE("After rewind+inject: " << output2);

            // Output should differ since we injected different tokens
            CHECK(output2 != output1);
        }
    } else {
        MESSAGE("No entropy events fired — skipping rewind test");
    }

    lfg_session_free(session);
}

// ---------------------------------------------------------------------------
// Test 4: Counter increments
// ---------------------------------------------------------------------------

TEST_CASE("Counter increments with events") {
    lfg_model *model = get_350m();
    REQUIRE(model != nullptr);

    const lfg_vocab *vocab = lfg_model_get_vocab(model);

    lfg_session_config config = lfg_session_default_config();
    config.n_ctx = 2048;
    config.sampling.temp = 0.8f;
    lfg_session *session = lfg_session_create(model, &config);
    REQUIRE(session != nullptr);

    lfg_entropy_monitor_config ecfg = lfg_entropy_monitor_default_config();
    ecfg.threshold = 0.01f;
    ecfg.cooldown_tokens = 1;
    ecfg.ring_size = 8;
    REQUIRE(lfg_session_configure_entropy_monitor(session, &ecfg));

    volatile int32_t *counter = lfg_session_entropy_counter(session);
    REQUIRE(static_cast<const void *>(const_cast<const int32_t *>(counter)) != nullptr);
    CHECK(*counter == 0);

    auto tokens = tokenize(vocab, "Hello world", true);
    REQUIRE(lfg_session_ingest_tokens(session, tokens.data(), tokens.size(), true));

    generate(session, vocab, 10);

    // Counter should have incremented
    int32_t count = *counter;
    MESSAGE("Counter after 10 tokens: " << count);
    CHECK(count > 0);
    CHECK(count == lfg_session_entropy_pending(session));  // pending = write - read, read=0

    lfg_session_free(session);
}

// ---------------------------------------------------------------------------
// Test 5: Getter without monitor returns -1
// ---------------------------------------------------------------------------

TEST_CASE("Getter without monitor returns -1") {
    lfg_model *model = get_350m();
    REQUIRE(model != nullptr);

    lfg_session_config config = lfg_session_default_config();
    config.n_ctx = 512;
    config.sampling.temp = 0.0f;
    lfg_session *session = lfg_session_create(model, &config);
    REQUIRE(session != nullptr);

    // No entropy monitor configured
    float e = lfg_session_get_last_entropy(session);
    CHECK(e == doctest::Approx(-1.0f));

    // Also verify counter returns non-null pointer (but value is 0)
    volatile int32_t *counter = lfg_session_entropy_counter(session);
    REQUIRE(static_cast<const void *>(const_cast<const int32_t *>(counter)) != nullptr);
    CHECK(*counter == 0);

    // Entropy pending returns 0 with no slots
    CHECK(lfg_session_entropy_pending(session) == 0);

    // Pop returns false
    lfg_entropy_event ev;
    CHECK_FALSE(lfg_session_entropy_pop(session, &ev, nullptr, 0));

    // Rewind returns false
    CHECK_FALSE(lfg_session_rewind(session, 0));

    lfg_session_free(session);
}

// ---------------------------------------------------------------------------
// Test 6: Coexists with tool ranking
// ---------------------------------------------------------------------------

TEST_CASE("Coexists with tool ranking") {
    lfg_model *model = get_350m();
    REQUIRE(model != nullptr);

    const lfg_vocab *vocab = lfg_model_get_vocab(model);

    lfg_session_config config = lfg_session_default_config();
    config.n_ctx = 2048;
    config.sampling.temp = 0.0f;
    lfg_session *session = lfg_session_create(model, &config);
    REQUIRE(session != nullptr);

    // Register tools first
    lfg_tool_desc tools[] = {
        {"get_weather", "Get current weather for a location", nullptr},
        {"calculator", "Perform arithmetic operations", nullptr},
    };
    REQUIRE(lfg_session_register_tools(session, tools, 2, 2) == 2);

    // Then configure entropy monitor (shares tool_ctx)
    lfg_entropy_monitor_config ecfg = lfg_entropy_monitor_default_config();
    ecfg.threshold = 0.01f;
    ecfg.cooldown_tokens = 1;
    ecfg.ring_size = 4;
    REQUIRE(lfg_session_configure_entropy_monitor(session, &ecfg));

    auto tokens = tokenize(vocab, "What is the weather in Paris?", true);
    REQUIRE(lfg_session_ingest_tokens(session, tokens.data(), tokens.size(), true));

    // Generate should work with both features active
    std::string output = generate(session, vocab, 10);
    MESSAGE("Output with tools+entropy: " << output);
    CHECK(!output.empty());

    // Entropy events should have fired
    CHECK(lfg_session_entropy_pending(session) > 0);

    // Clear tools — entropy should still be active
    lfg_session_clear_tools(session);
    float e = lfg_session_get_last_entropy(session);
    CHECK(e >= 0.0f);  // last entropy should be valid from previous generation

    lfg_session_free(session);
}

// ---------------------------------------------------------------------------
// Test 7: Ring wraps gracefully
// ---------------------------------------------------------------------------

TEST_CASE("Ring wraps gracefully") {
    lfg_model *model = get_350m();
    REQUIRE(model != nullptr);

    const lfg_vocab *vocab = lfg_model_get_vocab(model);

    lfg_session_config config = lfg_session_default_config();
    config.n_ctx = 2048;
    config.sampling.temp = 0.8f;
    lfg_session *session = lfg_session_create(model, &config);
    REQUIRE(session != nullptr);

    // Small ring (2 slots), low threshold, low cooldown → lots of events
    lfg_entropy_monitor_config ecfg = lfg_entropy_monitor_default_config();
    ecfg.threshold = 0.01f;
    ecfg.cooldown_tokens = 1;
    ecfg.ring_size = 2;
    REQUIRE(lfg_session_configure_entropy_monitor(session, &ecfg));

    auto tokens = tokenize(vocab, "Tell me a long story about dragons and", true);
    REQUIRE(lfg_session_ingest_tokens(session, tokens.data(), tokens.size(), true));

    // Generate enough tokens to overflow the ring
    generate(session, vocab, 20);

    volatile int32_t *counter = lfg_session_entropy_counter(session);
    int32_t total_written = *counter;
    MESSAGE("Total events written: " << total_written);
    CHECK(total_written > 2);  // more events than ring capacity

    // Pop all available events — they should have valid fields
    lfg_entropy_event ev;
    int popped = 0;
    while (lfg_session_entropy_pop(session, &ev, nullptr, 0)) {
        CHECK(ev.entropy >= 0.0f);
        CHECK(ev.normalized >= 0.0f);
        CHECK(ev.normalized <= 1.0f);
        popped++;
    }
    MESSAGE("Popped events: " << popped);

    // After popping all, pending should be 0
    CHECK(lfg_session_entropy_pending(session) == 0);

    // Rewind to an old checkpoint should fail (ring wrapped, snap expired)
    CHECK_FALSE(lfg_session_rewind(session, 0));

    lfg_session_free(session);
}

// ---------------------------------------------------------------------------
// 1.2B Thinking model — full retrieval integration tests
// ---------------------------------------------------------------------------

static const char * MODEL_1_2B_PATH = "models/LFM2.5-1.2B-Thinking-GGUF/LFM2.5-1.2B-Thinking-Q4_K_M.gguf";

static lfg_model * g_1_2b = nullptr;

static lfg_model * get_1_2b() {
    if (!g_1_2b) {
        lfg_backend_init();
        std::ifstream f(MODEL_1_2B_PATH);
        if (!f.good()) return nullptr;
        lfg_model_load_config cfg = lfg_model_load_default_config();
        cfg.model_path = MODEL_1_2B_PATH;
        cfg.n_gpu_layers = 0;
        g_1_2b = lfg_load_model(&cfg);
    }
    return g_1_2b;
}

// Knowledge base entry: text + pre-computed embedding.
struct kb_entry {
    std::string text;
    std::vector<float> embedding;
};

static float cosine_similarity(const float *a, const float *b, int n) {
    float dot = 0.0f;
    for (int i = 0; i < n; ++i) dot += a[i] * b[i];
    return dot;  // both L2-normalized, so dot == cosine
}

// Build a knowledge base using the public lfg_session_embed API.
static std::vector<kb_entry> build_kb(lfg_session *session, const char **texts, int n_texts, int32_t n_embd) {
    std::vector<kb_entry> kb;
    for (int i = 0; i < n_texts; ++i) {
        kb_entry entry;
        entry.text = texts[i];
        entry.embedding.resize(n_embd);
        int32_t got = lfg_session_embed(session, texts[i], (int32_t)std::strlen(texts[i]),
                                         entry.embedding.data(), n_embd);
        if (got != n_embd) entry.embedding.assign(n_embd, 0.0f);
        kb.push_back(std::move(entry));
    }
    return kb;
}

// Generate with entropy-triggered retrieval from a knowledge base.
// max_retrievals caps how many rewind+inject cycles we do (prevents runaway).
static std::string generate_with_retrieval(
    lfg_session *session, const lfg_vocab *vocab,
    const std::vector<kb_entry> &kb, int32_t n_embd,
    int max_tokens, int max_retrievals, int *out_retrievals)
{
    std::string output;
    int retrievals = 0;
    std::vector<float> embd_buf(n_embd);

    for (int i = 0; i < max_tokens; ++i) {
        lfg_session_decode(session);
        lfg_token tok = lfg_session_sample(session);
        if (lfg_vocab_is_eog(vocab, tok)) break;

        // Check for entropy events (only retrieve if under budget)
        lfg_entropy_event ev;
        if (retrievals < max_retrievals &&
            lfg_session_entropy_pop(session, &ev, embd_buf.data(), n_embd)) {
            // Find best matching KB entry by cosine similarity
            int best_idx = -1;
            float best_score = -1.0f;
            for (int k = 0; k < (int)kb.size(); ++k) {
                float score = cosine_similarity(embd_buf.data(), kb[k].embedding.data(), n_embd);
                if (score > best_score) { best_score = score; best_idx = k; }
            }

            if (best_idx >= 0 && best_score > 0.0f) {
                if (lfg_session_rewind(session, ev.checkpoint_id)) {
                    // Inject retrieved knowledge as context
                    std::string inject = " " + kb[best_idx].text + " ";
                    auto inject_toks = tokenize(vocab, inject, false);
                    lfg_session_ingest_tokens(session, inject_toks.data(), inject_toks.size(), false);
                    retrievals++;
                    lfg_session_entropy_flush(session);
                    continue;  // re-decode from new position
                }
            }
        } else {
            lfg_session_entropy_flush(session);
        }

        output += token_to_string(vocab, tok);
        lfg_session_ingest_tokens(session, &tok, 1, false);
    }

    if (out_retrievals) *out_retrievals = retrievals;
    return output;
}

TEST_CASE("Integration: Entropy-triggered retrieval changes output") {
    lfg_model *model = get_1_2b();
    if (!model) { MESSAGE("Skipping: 1.2B model not found"); return; }

    const lfg_vocab *vocab = lfg_model_get_vocab(model);
    int32_t n_embd = lfg_model_n_embd(model);

    const char *facts[] = {
        "The Eiffel Tower is 330 meters tall and was completed in 1889 for the World's Fair in Paris, France.",
        "The speed of light in vacuum is exactly 299,792,458 meters per second.",
        "Mount Everest has an elevation of 8,849 meters above sea level, located in the Himalayas.",
        "The Pacific Ocean covers approximately 165.25 million square kilometers, making it the largest ocean.",
        "DNA was first identified by Friedrich Miescher in 1869, and its double helix structure was discovered by Watson and Crick in 1953.",
    };
    constexpr int N_FACTS = 5;

    std::string prompt = "Tell me about the Eiffel Tower and how tall it is.\n";

    // --- Run 1: Generation WITHOUT retrieval (baseline) ---
    std::string output_baseline;
    {
        lfg_session_config config = lfg_session_default_config();
        config.n_ctx = 2048;
        config.sampling.temp = 0.0f;
        lfg_session *session = lfg_session_create(model, &config);
        REQUIRE(session != nullptr);

        auto toks = tokenize(vocab, prompt, true);
        lfg_session_ingest_tokens(session, toks.data(), toks.size(), true);
        output_baseline = generate(session, vocab, 80);
        MESSAGE("Baseline: " << output_baseline);

        lfg_session_free(session);
    }

    // --- Run 2: Generation WITH entropy-triggered retrieval ---
    std::string output_retrieval;
    int n_retrievals = 0;
    {
        lfg_session_config config = lfg_session_default_config();
        config.n_ctx = 2048;
        config.sampling.temp = 0.0f;
        lfg_session *session = lfg_session_create(model, &config);
        REQUIRE(session != nullptr);

        // Build KB using the public embed API
        auto kb = build_kb(session, facts, N_FACTS, n_embd);
        MESSAGE("Knowledge base: " << kb.size() << " entries, n_embd=" << n_embd);

        // Verify embeddings are valid
        for (auto &entry : kb) {
            float norm = 0;
            for (float v : entry.embedding) norm += v * v;
            CHECK(norm > 0.5f);
        }

        // Configure entropy monitor: low threshold to ensure events fire
        lfg_entropy_monitor_config ecfg = lfg_entropy_monitor_default_config();
        ecfg.threshold = 0.05f;
        ecfg.cooldown_tokens = 16;
        ecfg.ring_size = 4;
        REQUIRE(lfg_session_configure_entropy_monitor(session, &ecfg));

        auto toks = tokenize(vocab, prompt, true);
        lfg_session_ingest_tokens(session, toks.data(), toks.size(), true);

        // Allow at most 2 retrievals to avoid output collapse
        output_retrieval = generate_with_retrieval(session, vocab, kb, n_embd, 80, 2, &n_retrievals);
        MESSAGE("With retrieval (" << n_retrievals << " retrievals): " << output_retrieval);

        lfg_session_free(session);
    }

    MESSAGE("Retrievals performed: " << n_retrievals);

    if (n_retrievals > 0) {
        // Retrieval injects different context, so output must differ
        CHECK_MESSAGE(output_retrieval != output_baseline,
                      "Expected different output after retrieval injection");
        CHECK(!output_retrieval.empty());
    } else {
        MESSAGE("No retrievals triggered — entropy stayed below threshold with greedy sampling");
    }
}

TEST_CASE("Integration: Embed API produces valid embeddings") {
    lfg_model *model = get_1_2b();
    if (!model) { MESSAGE("Skipping: 1.2B model not found"); return; }

    int32_t n_embd = lfg_model_n_embd(model);

    const char *texts[] = {
        "The Eiffel Tower in Paris stands 330 meters tall and was built in 1889.",
        "Mount Everest is the tallest mountain at 8,849 meters in the Himalayas.",
        "DNA's double helix structure was discovered by Watson and Crick in 1953.",
    };
    constexpr int N = 3;

    lfg_session_config config = lfg_session_default_config();
    config.n_ctx = 2048;
    config.sampling.temp = 0.0f;
    lfg_session *session = lfg_session_create(model, &config);
    REQUIRE(session != nullptr);

    auto kb = build_kb(session, texts, N, n_embd);

    // 1. All embeddings have unit norm (L2-normalized)
    for (int i = 0; i < N; ++i) {
        float norm_sq = 0;
        for (float v : kb[i].embedding) norm_sq += v * v;
        CHECK(norm_sq == doctest::Approx(1.0f).epsilon(0.01));
    }

    // 2. Self-similarity is 1.0 (sanity)
    for (int i = 0; i < N; ++i) {
        float self = cosine_similarity(kb[i].embedding.data(), kb[i].embedding.data(), n_embd);
        CHECK(self == doctest::Approx(1.0f).epsilon(0.001));
    }

    // 3. Cross-similarity < 1.0 (embeddings are distinct, not collapsed)
    for (int i = 0; i < N; ++i) {
        for (int j = i + 1; j < N; ++j) {
            float sim = cosine_similarity(kb[i].embedding.data(), kb[j].embedding.data(), n_embd);
            MESSAGE("sim(" << i << "," << j << ") = " << sim);
            CHECK(sim < 0.99f);
        }
    }

    // 4. Same text produces identical embedding (deterministic)
    std::vector<float> embd2(n_embd);
    int32_t got = lfg_session_embed(session, texts[0], (int32_t)std::strlen(texts[0]),
                                     embd2.data(), n_embd);
    REQUIRE(got == n_embd);
    float self_check = cosine_similarity(kb[0].embedding.data(), embd2.data(), n_embd);
    CHECK(self_check == doctest::Approx(1.0f).epsilon(0.001));

    // 5. Error cases
    CHECK(lfg_session_embed(session, nullptr, 10, embd2.data(), n_embd) == 0);
    CHECK(lfg_session_embed(session, texts[0], 0, embd2.data(), n_embd) == 0);
    CHECK(lfg_session_embed(session, texts[0], (int32_t)std::strlen(texts[0]), embd2.data(), 1) == 0);

    lfg_session_free(session);
}

// ---------------------------------------------------------------------------
// Integration: Retrieve + Store — entropy retrieves, confidence stores
// ---------------------------------------------------------------------------

TEST_CASE("Integration: Confidence store collects spans during retrieval generation") {
    lfg_model *model = get_1_2b();
    if (!model) { MESSAGE("Skipping: 1.2B model not found"); return; }

    const lfg_vocab *vocab = lfg_model_get_vocab(model);
    int32_t n_embd = lfg_model_n_embd(model);

    // Seed KB for retrieval
    const char *facts[] = {
        "The Eiffel Tower is 330 meters tall and was completed in 1889 for the World's Fair in Paris, France.",
        "The speed of light in vacuum is exactly 299,792,458 meters per second.",
        "Mount Everest has an elevation of 8,849 meters above sea level, located in the Himalayas.",
    };
    constexpr int N_FACTS = 3;

    // Unrelated topic — for verifying stored embeddings have semantic signal
    const char *unrelated_text = "The recipe for chocolate cake requires flour, sugar, cocoa powder, and eggs.";

    lfg_session_config config = lfg_session_default_config();
    config.n_ctx = 2048;
    config.sampling.temp = 0.0f;
    lfg_session *session = lfg_session_create(model, &config);
    REQUIRE(session != nullptr);

    // Build seed KB
    auto kb = build_kb(session, facts, N_FACTS, n_embd);

    // Compute unrelated embedding for later comparison
    std::vector<float> unrelated_embd(n_embd);
    int32_t got_unrelated = lfg_session_embed(session, unrelated_text, (int32_t)std::strlen(unrelated_text),
                                                unrelated_embd.data(), n_embd);
    REQUIRE(got_unrelated == n_embd);

    // Compute prompt topic embedding
    const char *prompt_topic = "Eiffel Tower Paris France height";
    std::vector<float> topic_embd(n_embd);
    int32_t got_topic = lfg_session_embed(session, prompt_topic, (int32_t)std::strlen(prompt_topic),
                                            topic_embd.data(), n_embd);
    REQUIRE(got_topic == n_embd);

    // Configure entropy monitor (retrieve)
    lfg_entropy_monitor_config ecfg = lfg_entropy_monitor_default_config();
    ecfg.threshold = 0.05f;
    ecfg.cooldown_tokens = 16;
    ecfg.ring_size = 4;
    REQUIRE(lfg_session_configure_entropy_monitor(session, &ecfg) > 0);

    // Configure confidence monitor (store)
    lfg_confidence_monitor_config ccfg = lfg_confidence_monitor_default_config();
    ccfg.threshold = 0.5f;   // Moderate — capture reasonably confident spans
    ccfg.min_span = 3;
    ccfg.ring_size = 8;
    REQUIRE(lfg_session_configure_confidence_monitor(session, &ccfg) > 0);

    // Accumulate stored spans via confidence callback
    struct store_entry {
        lfg_confidence_event event;
        std::vector<float>   embedding;
    };

    struct loop_state {
        std::vector<store_entry> stored;
        int retrievals;
        int max_retrievals;
    };

    loop_state state{};
    state.retrievals = 0;
    state.max_retrievals = 2;

    // Ingest prompt
    std::string prompt = "Tell me about the Eiffel Tower and how tall it is.\n";
    auto toks = tokenize(vocab, prompt, true);
    lfg_session_ingest_tokens(session, toks.data(), toks.size(), true);

    int generated = 0;
    std::vector<float> embd_buf(n_embd);
    std::vector<float> conf_embd_buf(n_embd);

    for (int i = 0; i < 80; ++i) {
        lfg_session_decode(session);
        lfg_token tok = lfg_session_sample(session);
        generated++;
        if (lfg_vocab_is_eog(vocab, tok)) {
            break;
        }

        // Entropy-triggered retrieval (external loop owned by caller).
        lfg_entropy_event ev;
        if (state.retrievals < state.max_retrievals &&
            lfg_session_entropy_pop(session, &ev, embd_buf.data(), n_embd)) {
            int best_idx = -1;
            float best_score = -1.0f;
            for (int k = 0; k < (int)kb.size(); ++k) {
                float score = cosine_similarity(embd_buf.data(), kb[k].embedding.data(), n_embd);
                if (score > best_score) { best_score = score; best_idx = k; }
            }

            if (best_idx >= 0 && best_score > 0.0f && lfg_session_rewind(session, ev.checkpoint_id)) {
                auto inj = tokenize(vocab, kb[best_idx].text, false);
                lfg_session_ingest_tokens(session, inj.data(), inj.size(), false);
                state.retrievals++;
                lfg_session_entropy_flush(session);
                continue;
            }
            lfg_session_entropy_flush(session);
        } else {
            lfg_session_entropy_flush(session);
        }

        // Drain confidence events and store for verification.
        lfg_confidence_event cev;
        while (lfg_session_confidence_pop(session, &cev, conf_embd_buf.data(), n_embd)) {
            store_entry entry;
            entry.event = cev;
            if (cev.n_embd > 0) {
                entry.embedding.assign(conf_embd_buf.begin(), conf_embd_buf.begin() + cev.n_embd);
            }
            state.stored.push_back(std::move(entry));
        }

        lfg_session_ingest_tokens(session, &tok, 1, false);
    }

    lfg_confidence_event cev;
    while (lfg_session_confidence_pop(session, &cev, conf_embd_buf.data(), n_embd)) {
        store_entry entry;
        entry.event = cev;
        if (cev.n_embd > 0) {
            entry.embedding.assign(conf_embd_buf.begin(), conf_embd_buf.begin() + cev.n_embd);
        }
        state.stored.push_back(std::move(entry));
    }

    MESSAGE("Generated " << generated << " tokens");
    MESSAGE("Retrievals: " << state.retrievals);
    MESSAGE("Confidence spans stored: " << state.stored.size());

    // --- Verify stored spans ---
    if (!state.stored.empty()) {
        MESSAGE("--- Stored span details ---");
        for (int i = 0; i < (int)state.stored.size(); ++i) {
            auto &s = state.stored[i];
            MESSAGE("  Span " << i << ": len=" << s.event.span_length
                    << " mean_H=" << s.event.mean_entropy
                    << " min_H=" << s.event.min_entropy
                    << " pos=[" << s.event.start_pos << "," << s.event.end_pos << "]"
                    << " has_embd=" << !s.embedding.empty());

            // Event field sanity
            CHECK(s.event.span_length >= ccfg.min_span);
            CHECK(s.event.mean_entropy >= 0.0f);
            CHECK(s.event.mean_entropy <= ccfg.threshold);
            CHECK(s.event.min_entropy <= s.event.mean_entropy);
            CHECK(s.event.end_pos > s.event.start_pos);
        }

        // Find the first stored span that has an embedding
        int embd_idx = -1;
        for (int i = 0; i < (int)state.stored.size(); ++i) {
            if (!state.stored[i].embedding.empty()) { embd_idx = i; break; }
        }

        if (embd_idx >= 0) {
            auto &entry = state.stored[embd_idx];

            // Embedding should be L2-normalized
            float norm_sq = 0;
            for (float v : entry.embedding) norm_sq += v * v;
            CHECK(norm_sq == doctest::Approx(1.0f).epsilon(0.05));

            // Stored embedding should be more similar to the prompt topic
            // than to an unrelated topic (chocolate cake)
            float sim_topic = cosine_similarity(entry.embedding.data(), topic_embd.data(), n_embd);
            float sim_unrelated = cosine_similarity(entry.embedding.data(), unrelated_embd.data(), n_embd);

            MESSAGE("  Stored span " << embd_idx << " sim to topic: " << sim_topic);
            MESSAGE("  Stored span " << embd_idx << " sim to unrelated: " << sim_unrelated);
            CHECK_MESSAGE(sim_topic > sim_unrelated,
                          "Stored confident span should be more related to Eiffel Tower than chocolate cake");
        } else {
            MESSAGE("No stored spans had embeddings — skipping semantic check");
        }
    } else {
        MESSAGE("No confidence spans stored — model entropy may not have dropped below threshold");
    }

    lfg_session_free(session);
}
