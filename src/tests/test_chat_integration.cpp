#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"

#include "../inference/lfg_api.h"

#include <cstring>
#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// Natural multi-turn chat integration test against the 1.2B thinking model.
// Builds a real conversation over 12+ turns, feeding each model response
// back as the assistant message for the next turn.
// ---------------------------------------------------------------------------

static const char *MODEL_PATH =
    "models/LFM2.5-1.2B-Thinking-GGUF/LFM2.5-1.2B-Thinking-Q4_K_M.gguf";

struct test_env {
    lfg_model   *model   = nullptr;
    lfg_session *session = nullptr;
    const lfg_vocab *vocab = nullptr;
};

static bool setup(test_env *env, int n_ctx = 4096) {
    lfg_model_load_config lcfg = lfg_model_load_default_config();
    lcfg.model_path = MODEL_PATH;
    env->model = lfg_load_model(&lcfg);
    if (!env->model) return false;

    env->vocab = lfg_model_get_vocab(env->model);

    lfg_session_config cfg = lfg_session_default_config();
    cfg.n_ctx = n_ctx;
    cfg.sampling.temp = 0.0f;
    cfg.sampling.seed = 42;
    env->session = lfg_session_create(env->model, &cfg);
    if (!env->session) {
        lfg_model_free(env->model);
        return false;
    }
    return true;
}

static void teardown(test_env *env) {
    if (env->session) lfg_session_free(env->session);
    if (env->model) lfg_model_free(env->model);
}

struct collect_state { std::string text; };

static lfg_generate_action collect_cb(lfg_token, const char *piece, int32_t len, void *ud) {
    auto *st = static_cast<collect_state *>(ud);
    if (piece && len > 0) st->text.append(piece, len);
    return LFG_GENERATE_CONTINUE;
}

static bool contains(const std::string &s, const char *sub) {
    return s.find(sub) != std::string::npos;
}

// Strip <think>...</think> blocks from model output
static std::string strip_thinking(const std::string &s) {
    auto think_end = s.find("</think>");
    std::string cleaned = s;
    if (think_end != std::string::npos) {
        cleaned = s.substr(think_end + 8);
    }
    // Also strip any leading <|im_end|> that leaks through
    auto im_end = cleaned.find("<|im_end|>");
    if (im_end != std::string::npos && im_end < 5) {
        cleaned = cleaned.substr(im_end + 10);
    }
    auto start = cleaned.find_first_not_of(" \t\n\r");
    if (start == std::string::npos) return "";
    auto end = cleaned.find_last_not_of(" \t\n\r");
    return cleaned.substr(start, end - start + 1);
}

// Run one turn of conversation: reset session, feed full history, generate.
static std::string run_turn(test_env *env,
                            const std::vector<lfg_chat_message> &msgs,
                            int max_tokens = 256) {
    lfg_session_reset(env->session);

    collect_state st;
    lfg_generate_config gc = lfg_generate_default_config();
    gc.max_tokens = max_tokens;
    gc.token_cb = collect_cb;
    gc.token_cb_data = &st;

    lfg_generate_result r = lfg_session_chat_generate(
        env->session, msgs.data(), msgs.size(), gc);

    (void)r;
    return strip_thinking(st.text);
}

// ===========================================================================

TEST_CASE("Multi-turn: 14-turn geography quiz conversation") {
    test_env env;
    if (!setup(&env, 4096)) {
        MESSAGE("Skipping: model not available");
        return;
    }

    struct turn {
        const char *user_msg;
        const char *check;  // substring the response must contain (or nullptr)
    };

    // A natural conversation: geography quiz with follow-ups
    turn script[] = {
        // Turn 1: establish context
        {"What is the capital of France?", "Paris"},
        // Turn 2: follow-up on same topic
        {"What river runs through it?", "Seine"},
        // Turn 3: pivot to a new country
        {"What about Japan — what is its capital?", "Tokyo"},
        // Turn 4: dig deeper
        {"What is the population of Tokyo approximately?", "million"},
        // Turn 5: test recall of earlier context
        {"Going back to France, what is a famous landmark in its capital?",
         nullptr},  // Eiffel Tower likely, but accept anything
        // Turn 6: new topic entirely
        {"What is the largest desert in the world?", nullptr},
        // Turn 7: follow-up
        {"Is it hot or cold?", nullptr},
        // Turn 8: arithmetic in context
        {"If France has about 67 million people and Japan has about 125 million, "
         "how many combined?", "192"},
        // Turn 9: opinion/reasoning
        {"Which city would you recommend visiting first, Paris or Tokyo?",
         nullptr},
        // Turn 10: meta question about the conversation
        {"What was the first question I asked you in this conversation?",
         nullptr},
        // Turn 11: simple factual
        {"How many continents are there?", nullptr},
        // Turn 12: follow-up on continents
        {"Name them.", nullptr},
        // Turn 13: back to geography
        {"What is the smallest country in the world by area?", nullptr},
        // Turn 14: wrap up
        {"Thanks for the chat! Summarize the main topics we discussed.",
         nullptr},
    };

    int n_turns = sizeof(script) / sizeof(script[0]);
    REQUIRE(n_turns >= 12);

    // Conversation history — start with a system message
    std::vector<lfg_chat_message> history;
    std::vector<std::string> stored_responses;  // keep strings alive

    history.push_back({"system",
        "You are a helpful geography assistant. Give concise, factual answers."});

    int passed_content_checks = 0;

    for (int i = 0; i < n_turns; i++) {
        // Add user message
        history.push_back({"user", script[i].user_msg});

        // Generate
        std::string response = run_turn(&env, history, 256);

        // Store response and add to history (must keep string alive)
        stored_responses.push_back(response);
        history.push_back({"assistant", stored_responses.back().c_str()});

        // Print for manual verification
        MESSAGE("--- Turn ", i + 1, " ---");
        MESSAGE("User: ", script[i].user_msg);
        MESSAGE("Assistant: ", response);

        // Basic sanity: response is non-empty
        CHECK(!response.empty());

        // Content check if specified
        if (script[i].check) {
            bool found = contains(response, script[i].check);
            if (found) passed_content_checks++;
            CHECK_MESSAGE(found,
                "Turn ", i + 1, ": expected '", script[i].check,
                "' in response");
        }
    }

    MESSAGE("=== Content checks passed: ", passed_content_checks, " ===");
    MESSAGE("=== Total turns completed: ", n_turns, " ===");

    teardown(&env);
}

TEST_CASE("Multi-turn: 12-turn coding help conversation") {
    test_env env;
    if (!setup(&env, 4096)) {
        MESSAGE("Skipping: model not available");
        return;
    }

    struct turn {
        const char *user_msg;
        const char *check;
    };

    turn script[] = {
        // Turn 1
        {"What is a linked list?", nullptr},
        // Turn 2
        {"How is it different from an array?", nullptr},
        // Turn 3
        {"What is the time complexity of inserting at the head of a linked list?",
         "O(1)"},
        // Turn 4
        {"What about inserting at the end?", nullptr},
        // Turn 5
        {"Can you show me a simple linked list node struct in C?", "struct"},
        // Turn 6
        {"How would I insert a node at the head?", nullptr},
        // Turn 7
        {"What is a doubly linked list?", nullptr},
        // Turn 8
        {"What are the advantages of a doubly linked list over a singly linked list?",
         nullptr},
        // Turn 9
        {"What is the space complexity of a linked list with n nodes?", "O(n)"},
        // Turn 10
        {"When should I use a linked list instead of an array?", nullptr},
        // Turn 11
        {"What is a circular linked list?", nullptr},
        // Turn 12
        {"Summarize the types of linked lists we discussed.", nullptr},
    };

    int n_turns = sizeof(script) / sizeof(script[0]);
    REQUIRE(n_turns >= 12);

    std::vector<lfg_chat_message> history;
    std::vector<std::string> stored_responses;

    history.push_back({"system",
        "You are a computer science tutor. Give clear, concise explanations."});

    int passed_content_checks = 0;

    for (int i = 0; i < n_turns; i++) {
        history.push_back({"user", script[i].user_msg});

        std::string response = run_turn(&env, history, 256);

        stored_responses.push_back(response);
        history.push_back({"assistant", stored_responses.back().c_str()});

        MESSAGE("--- Turn ", i + 1, " ---");
        MESSAGE("User: ", script[i].user_msg);
        MESSAGE("Assistant: ", response);

        CHECK(!response.empty());

        if (script[i].check) {
            bool found = contains(response, script[i].check);
            if (found) passed_content_checks++;
            CHECK_MESSAGE(found,
                "Turn ", i + 1, ": expected '", script[i].check,
                "' in response");
        }
    }

    MESSAGE("=== Content checks passed: ", passed_content_checks, " ===");
    MESSAGE("=== Total turns completed: ", n_turns, " ===");

    teardown(&env);
}
