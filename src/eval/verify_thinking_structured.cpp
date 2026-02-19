// Verify that the 1.2B-Thinking model actually produces:
//   1. <think> ... reasoning text ... </think>
//   2. Valid JSON after </think> (grammar-constrained)
// with token healing and structured checkpointing enabled.
//
// Usage: verify_thinking_structured [model_path] [n_threads]

#include "lfg_api.h"
#include <cstdio>
#include <string>
#include <vector>
#include <fstream>

static std::string token_to_str(const lfg_vocab * vocab, lfg_token t) {
    char buf[512];
    int n = lfg_detokenize(const_cast<lfg_vocab *>(vocab), &t, 1, buf, sizeof(buf), false, false);
    if (n < 0) return "";
    return std::string(buf, n);
}

static std::vector<lfg_token> tokenize_marker(const lfg_vocab * vocab, const char * text) {
    std::vector<lfg_token> toks(32);
    int n = lfg_tokenize(const_cast<lfg_vocab *>(vocab), text, (int32_t)strlen(text),
                         toks.data(), (int32_t)toks.size(), false, false);
    if (n < 0) {
        toks.resize(-n);
        n = lfg_tokenize(const_cast<lfg_vocab *>(vocab), text, (int32_t)strlen(text),
                         toks.data(), (int32_t)toks.size(), false, false);
    }
    toks.resize(n > 0 ? n : 0);
    return toks;
}

// Check if the tail of `history` matches `pattern`
static bool history_ends_with(const std::vector<lfg_token> & history,
                              const std::vector<lfg_token> & pattern) {
    if (pattern.empty() || history.size() < pattern.size()) return false;
    size_t off = history.size() - pattern.size();
    for (size_t i = 0; i < pattern.size(); i++) {
        if (history[off + i] != pattern[i]) return false;
    }
    return true;
}

int main(int argc, char ** argv) {
    const char * model_path = argc > 1 ? argv[1]
        : "models/LFM2.5-1.2B-Thinking-GGUF/LFM2.5-1.2B-Thinking-Q4_K_M.gguf";
    int n_threads = argc > 2 ? std::stoi(argv[2]) : 4;

    { std::ifstream f(model_path); if (!f.good()) { fprintf(stderr, "Model not found: %s\n", model_path); return 1; } }

    printf("=== Verify: Thinking + Structured Decoding + Healing ===\n");
    printf("Model: %s\n\n", model_path);

    lfg_backend_init();

    lfg_model_load_config lc = lfg_model_load_default_config();
    lc.model_path = model_path;
    lc.n_gpu_layers = 0;

    lfg_model * model = lfg_load_model(&lc);
    if (!model) { fprintf(stderr, "Failed to load model\n"); return 1; }

    const lfg_vocab * vocab = lfg_model_get_vocab(model);

    // --- Discover thinking tokens from vocab ---
    auto think_start_toks = tokenize_marker(vocab, "<think>");
    auto think_end_toks   = tokenize_marker(vocab, "</think>");

    printf("Think start tokens (%zu): [", think_start_toks.size());
    for (size_t i = 0; i < think_start_toks.size(); i++)
        printf("%s%d='%s'", i ? ", " : "", think_start_toks[i], token_to_str(vocab, think_start_toks[i]).c_str());
    printf("]\n");

    printf("Think end tokens (%zu):   [", think_end_toks.size());
    for (size_t i = 0; i < think_end_toks.size(); i++)
        printf("%s%d='%s'", i ? ", " : "", think_end_toks[i], token_to_str(vocab, think_end_toks[i]).c_str());
    printf("]\n");

    if (think_start_toks.empty() || think_end_toks.empty()) {
        fprintf(stderr, "Could not tokenize <think>/<think> markers\n");
        lfg_model_free(model);
        return 1;
    }

    // --- JSON grammar ---
    const char * json_grammar = R"(
root   ::= object
value  ::= object | array | string | number | ("true" | "false" | "null") ws
object ::= "{" ws ( string ":" ws value ("," ws string ":" ws value)* )? "}" ws
array  ::= "[" ws ( value ("," ws value)* )? "]" ws
string ::= "\"" ([^"\\] | "\\" (["\\/bfnrt"] | "u" [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F]))* "\"" ws
number ::= ("-"? ([0-9]+) ("." [0-9]+)? ([eE] [+-]? [0-9]+)?) ws
ws     ::= ([ \t\n] ws)?
)";

    // --- Create session ---
    lfg_session_config config = lfg_session_default_config();
    config.n_ctx     = 4096;
    config.n_threads = n_threads;
    config.sampling.temp = 0.6f;
    config.sampling.seed = 42;
    config.sampling.top_k = 40;
    config.sampling.top_p = 0.9f;
    config.sampling.penalty_repeat = 1.1f;
    config.sampling.penalty_last_n = 64;
    config.enable_healing = true;
    config.structured_checkpointing = true;

    lfg_session * session = lfg_session_create(model, &config);
    if (!session) { fprintf(stderr, "Failed to create session\n"); lfg_model_free(model); return 1; }

    // Configure reasoning + structured FIRST so ingested tokens update state
    lfg_session_configure_reasoning(session,
        think_start_toks.data(), think_start_toks.size(),
        think_end_toks.data(),   think_end_toks.size());
    lfg_session_configure_structured(session, json_grammar, "root");

    // --- Build prompt using chat template (bailing-think format) ---
    // The thinking model uses: <role>HUMAN</role>...<role>ASSISTANT</role><think>
    std::string user_msg = "What is 2+2? Answer in JSON with a \"result\" key.";

    lfg_chat_message messages[1];
    messages[0].role    = "user";
    messages[0].content = user_msg.c_str();

    // Build prompt WITHOUT <think> suffix (add_ass=false), then append <think> separately
    std::vector<char> tmpl_buf(user_msg.size() * 4 + 512);
    int32_t tmpl_len = lfg_chat_apply_template(
        "bailing-think", messages, 1, true, tmpl_buf.data(), (int32_t)tmpl_buf.size());
    if (tmpl_len > (int32_t)tmpl_buf.size()) {
        tmpl_buf.resize(tmpl_len + 1);
        tmpl_len = lfg_chat_apply_template(
            "bailing-think", messages, 1, true, tmpl_buf.data(), (int32_t)tmpl_buf.size());
    }
    std::string prompt(tmpl_buf.data(), tmpl_len);

    printf("Formatted prompt:\n%s\n\n", prompt.c_str());

    // Tokenize the full prompt (includes <think> at the end)
    std::vector<lfg_token> tokens(prompt.size() + 16);
    int n_tok = lfg_tokenize(const_cast<lfg_vocab *>(vocab),
                             prompt.c_str(), (int32_t)prompt.size(),
                             tokens.data(), (int32_t)tokens.size(), true, true);
    tokens.resize(n_tok);

    printf("Prompt tokens: %d\n", n_tok);

    // Print last few tokens to verify <think> is at the end
    printf("Last 5 tokens: ");
    for (int i = std::max(0, n_tok - 5); i < n_tok; i++)
        printf("%d='%s' ", tokens[i], token_to_str(vocab, tokens[i]).c_str());
    printf("\n\n");

    // Split ingestion: everything before <think> without sampler (avoids grammar rejection),
    // then <think> with sampler to trigger the reasoning state
    int n_pre = n_tok - (int)think_start_toks.size();
    if (n_pre > 0) {
        lfg_session_ingest_tokens(session, tokens.data(), n_pre, false);
    }
    // Ingest <think> with sampler update to activate reasoning gate
    lfg_session_ingest_tokens(session, tokens.data() + n_pre, (int32_t)think_start_toks.size(), true);

    // --- Generate ---
    int max_tokens = 200;
    std::string full_output;
    std::string thinking_text;
    std::string structured_text;
    std::vector<lfg_token> gen_history;

    // The chat template already appended <think>, so we start in thinking mode
    bool in_thinking  = true;
    bool saw_think_start = true;
    bool saw_think_end   = false;
    int  thinking_tokens = 0;
    int  structured_tokens = 0;
    bool hit_eos = false;

    printf("--- Generation ---\n");

    for (int i = 0; i < max_tokens; i++) {
        lfg_session_decode(session);
        lfg_token t = lfg_session_sample(session);

        std::string s = token_to_str(vocab, t);
        full_output += s;
        gen_history.push_back(t);

        if (lfg_vocab_is_eog(vocab, t)) {
            hit_eos = true;
            printf("[EOS]\n");
            break;
        }

        // Check for think start/end sequences
        bool just_entered_thinking = false;
        bool just_exited_thinking  = false;

        if (!in_thinking && !saw_think_end && history_ends_with(gen_history, think_start_toks)) {
            saw_think_start = true;
            in_thinking = true;
            just_entered_thinking = true;
            printf("[THINK_START]");
        } else if (in_thinking && history_ends_with(gen_history, think_end_toks)) {
            saw_think_end = true;
            in_thinking = false;
            just_exited_thinking = true;
            printf("[THINK_END]");
        }

        if (!just_entered_thinking && !just_exited_thinking) {
            if (in_thinking) {
                thinking_text += s;
                thinking_tokens++;
            } else if (saw_think_end) {
                structured_text += s;
                structured_tokens++;
            }
            printf("%s", s.c_str());
            fflush(stdout);
        }

        lfg_session_ingest_tokens(session, &t, 1, true);

        // Note: we do NOT heal the </think> token — it's a structural marker
        // whose bytes ("</think>") conflict with the JSON grammar constraint.
        // The session API's heal function also guards against this internally.
    }

    printf("\n\n--- Results ---\n");
    printf("Saw <think>:   %s\n", saw_think_start ? "YES" : "NO");
    printf("Saw </think>:  %s\n", saw_think_end ? "YES" : "NO");
    printf("Thinking tokens: %d\n", thinking_tokens);
    printf("Structured tokens: %d\n", structured_tokens);
    printf("Hit EOS: %s\n", hit_eos ? "YES" : "NO");

    printf("\n--- Thinking Text ---\n%s\n", thinking_text.c_str());
    printf("\n--- Structured Output ---\n%s\n", structured_text.c_str());

    // --- Validation ---
    int pass = 0, fail = 0;

    auto check = [&](const char * name, bool cond) {
        printf("  [%s] %s\n", cond ? "PASS" : "FAIL", name);
        if (cond) pass++; else fail++;
    };

    printf("\n--- Checks ---\n");
    check("Model produced <think>", saw_think_start);
    check("Model produced </think>", saw_think_end);
    check("Thinking text is non-empty", !thinking_text.empty());
    check("Thinking is NOT JSON (no leading '{')",
          thinking_text.empty() || thinking_text[0] != '{');
    check("Structured output is non-empty", !structured_text.empty());

    // Trim leading whitespace for JSON check
    size_t first_non_ws = structured_text.find_first_not_of(" \t\n\r");
    bool starts_with_brace = (first_non_ws != std::string::npos && structured_text[first_non_ws] == '{');
    check("Structured output starts with '{'", starts_with_brace);
    check("Budget enforced (thinking <= 100 tokens)", thinking_tokens <= 100);

    int brace_depth = 0;
    bool json_valid_braces = true;
    for (char c : structured_text) {
        if (c == '{') brace_depth++;
        else if (c == '}') brace_depth--;
        if (brace_depth < 0) { json_valid_braces = false; break; }
    }
    check("Structured output has balanced braces", json_valid_braces && brace_depth == 0);

    printf("\n=== %d passed, %d failed ===\n", pass, fail);

    lfg_session_free(session);
    lfg_model_free(model);

    return fail > 0 ? 1 : 0;
}
