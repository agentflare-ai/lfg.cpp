#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"

#include "../inference/lfg_inference.h"

#include <cmath>
#include <vector>

TEST_CASE("Greedy sampler with all-NINF logits does not crash") {
    // Greedy just picks argmax — all -INF means any index is fine, shouldn't crash
    const size_t n = 128;
    std::vector<lfg_token_data> data(n);
    for (size_t i = 0; i < n; ++i) {
        data[i] = { (lfg_token)i, -INFINITY, 0.0f };
    }
    lfg_token_data_array cur_p = { data.data(), n, -1, false };

    lfg_sampler *greedy = lfg_sampler_init_greedy();
    REQUIRE(greedy != nullptr);
    lfg_sampler_apply(greedy, &cur_p);

    CHECK(cur_p.selected >= 0);
    CHECK(cur_p.selected < (int64_t)n);

    lfg_sampler_free(greedy);
}

TEST_CASE("Dist sampler with all-NINF logits does not crash") {
    // The dist sampler calls softmax internally — this is the key safety test.
    // Without the safety guard, this would divide by zero (undefined behavior).
    const size_t n = 64;
    std::vector<lfg_token_data> data(n);
    for (size_t i = 0; i < n; ++i) {
        data[i] = { (lfg_token)i, -INFINITY, 0.0f };
    }
    lfg_token_data_array cur_p = { data.data(), n, -1, false };

    lfg_sampler *dist = lfg_sampler_init_dist(42);
    REQUIRE(dist != nullptr);
    lfg_sampler_apply(dist, &cur_p);

    CHECK(cur_p.selected >= 0);
    CHECK(cur_p.selected < (int64_t)n);
    CHECK(!std::isnan(data[cur_p.selected].p));
    CHECK(data[cur_p.selected].p > 0.0f);

    lfg_sampler_free(dist);
}

TEST_CASE("Temp_ext + dist sampler with all-NINF logits does not crash") {
    // Tests the temp_ext inline softmax safety guard (bug fix A1).
    // Without the guard, the re-normalization after dynamic temperature divides by zero.
    const size_t n = 64;
    std::vector<lfg_token_data> data(n);
    for (size_t i = 0; i < n; ++i) {
        data[i] = { (lfg_token)i, -INFINITY, 0.0f };
    }
    lfg_token_data_array cur_p = { data.data(), n, -1, false };

    // Chain: temp_ext → dist
    lfg_sampler *chain = lfg_sampler_chain_init(lfg_sampler_chain_default_params());
    REQUIRE(chain != nullptr);
    lfg_sampler_chain_add(chain, lfg_sampler_init_temp_ext(0.8f, 0.2f, 1.0f));
    lfg_sampler_chain_add(chain, lfg_sampler_init_dist(42));

    lfg_sampler_apply(chain, &cur_p);

    CHECK(cur_p.selected >= 0);
    CHECK(cur_p.selected < (int64_t)n);
    CHECK(!std::isnan(data[cur_p.selected].p));
    CHECK(data[cur_p.selected].p > 0.0f);

    lfg_sampler_free(chain);
}

TEST_CASE("Dist sampler with normal logits works correctly") {
    // Sanity check that the safety guard doesn't break normal softmax
    const size_t n = 4;
    std::vector<lfg_token_data> data = {
        { 0, 2.0f, 0.0f },
        { 1, 1.0f, 0.0f },
        { 2, 0.5f, 0.0f },
        { 3, -1.0f, 0.0f },
    };
    lfg_token_data_array cur_p = { data.data(), n, -1, false };

    lfg_sampler *dist = lfg_sampler_init_dist(42);
    REQUIRE(dist != nullptr);
    lfg_sampler_apply(dist, &cur_p);

    CHECK(cur_p.selected >= 0);
    CHECK(cur_p.selected < (int64_t)n);

    // Probabilities should be ordered (higher logit = higher probability)
    CHECK(data[0].p > data[1].p);
    CHECK(data[1].p > data[2].p);
    CHECK(data[2].p > data[3].p);

    // Sum of probabilities should be ~1.0
    float sum = 0.0f;
    for (size_t i = 0; i < n; ++i) sum += data[i].p;
    CHECK(sum == doctest::Approx(1.0f).epsilon(0.001f));

    lfg_sampler_free(dist);
}
