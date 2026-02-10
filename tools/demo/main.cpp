// lfg-demo: Interactive ImGui demo/tools application for the lfg.cpp inference engine.
// Single-file application with model loading, settings configuration, and streaming chat.

#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"

#ifdef __APPLE__
#define GL_SILENCE_DEPRECATION
#endif
#include <GLFW/glfw3.h>

#include "lfg_api.h"
#include "file_dialog.h"

#include <cstdio>
#include <cstring>
#include <cmath>
#include <mutex>
#include <thread>
#include <string>
#include <vector>
#include <atomic>
#include <algorithm>

// ---------------------------------------------------------------------------
// Application state (shared between main and inference thread)
// ---------------------------------------------------------------------------

struct ChatMessage {
    std::string role;   // "system", "user", "assistant"
    std::string text;

    // Split assistant text into thinking and output parts for rendering.
    // Handles streaming (partial tags) gracefully.
    void split_thinking(std::string &thinking_out, std::string &output_out) const {
        static const char *THINK_OPEN  = "<think>";
        static const char *THINK_CLOSE = "</think>";
        static const size_t OPEN_LEN  = 7;
        static const size_t CLOSE_LEN = 8;

        thinking_out.clear();
        output_out.clear();

        size_t open_pos = text.find(THINK_OPEN);
        if (open_pos == std::string::npos) {
            // No thinking block — everything is output
            output_out = text;
            return;
        }

        // Text before <think> (if any) is output
        output_out = text.substr(0, open_pos);

        size_t think_start = open_pos + OPEN_LEN;
        size_t close_pos = text.find(THINK_CLOSE, think_start);
        if (close_pos == std::string::npos) {
            // Still thinking (no close tag yet)
            thinking_out = text.substr(think_start);
        } else {
            thinking_out = text.substr(think_start, close_pos - think_start);
            // Text after </think> is output
            output_out += text.substr(close_pos + CLOSE_LEN);
        }
    }
};

struct SurpriseLogEntry {
    std::string prompt_text;         // The user message that triggered the event
    float       mean_surprise;
    float       max_surprise;
    int32_t     n_above_threshold;
    int32_t     n_tokens_evaluated;
    int         turn_index;          // Which chat turn (1-based)
};

struct RetrievalLogEntry {
    std::string context_text;        // Generated text up to the high-entropy point
    float       entropy;             // Raw Shannon entropy
    float       normalized;          // Normalized entropy [0,1]
    float       top_logprob;         // Log probability of the sampled token
    int32_t     n_past;              // Token position when event fired
    int         turn_index;          // Which chat turn (1-based)
};

struct ConfidenceLogEntry {
    std::string context_text;        // Generated text at span end
    float       mean_entropy;        // Average normalized entropy over the span
    float       min_entropy;         // Minimum normalized entropy in the span
    int32_t     span_length;         // Number of consecutive low-entropy tokens
    int32_t     start_pos;           // n_past at span start
    int32_t     end_pos;             // n_past at span end
    int         turn_index;          // Which chat turn (1-based)
};

struct ToolRankingEntry {
    std::string query;          // User message used as ranking query
    std::string ranked_output;  // Full formatted output from rank_tools
    int turn_index;
};

// ---------------------------------------------------------------------------
// Example tools for tool ranking visualization
// ---------------------------------------------------------------------------

static const lfg_tool_desc EXAMPLE_TOOLS[] = {
    {"get_weather",
     "Get the current weather forecast for a given location",
     R"({"type":"object","properties":{"location":{"type":"string","description":"City or coordinates"},"units":{"type":"string","enum":["celsius","fahrenheit"]}},"required":["location"]})"},
    {"search_web",
     "Search the internet for information on a topic",
     R"({"type":"object","properties":{"query":{"type":"string","description":"Search query"},"num_results":{"type":"integer","description":"Number of results to return"}},"required":["query"]})"},
    {"send_email",
     "Send an email message to a recipient",
     R"({"type":"object","properties":{"to":{"type":"string","description":"Recipient email address"},"subject":{"type":"string"},"body":{"type":"string"}},"required":["to","subject","body"]})"},
    {"calculator",
     "Evaluate an arithmetic or mathematical expression",
     R"({"type":"object","properties":{"expression":{"type":"string","description":"Math expression to evaluate"}},"required":["expression"]})"},
    {"set_reminder",
     "Set a timed reminder with a message",
     R"json({"type":"object","properties":{"message":{"type":"string"},"time":{"type":"string","description":"When to remind (ISO 8601 or natural language)"}},"required":["message","time"]})json"},
    {"create_calendar_event",
     "Create a calendar event with a title, time, and optional attendees",
     R"({"type":"object","properties":{"title":{"type":"string"},"start_time":{"type":"string"},"end_time":{"type":"string"},"attendees":{"type":"array","items":{"type":"string"}}},"required":["title","start_time"]})"},
    {"translate_text",
     "Translate text from one language to another",
     R"({"type":"object","properties":{"text":{"type":"string"},"source_language":{"type":"string"},"target_language":{"type":"string"}},"required":["text","target_language"]})"},
    {"get_stock_price",
     "Look up the current stock price for a ticker symbol",
     R"({"type":"object","properties":{"symbol":{"type":"string","description":"Stock ticker symbol"}},"required":["symbol"]})"},
    {"play_music",
     "Play a song or playlist by name or artist",
     R"({"type":"object","properties":{"query":{"type":"string","description":"Song, artist, or playlist name"},"shuffle":{"type":"boolean"}},"required":["query"]})"},
    {"set_timer",
     "Set a countdown timer for a specified duration",
     R"({"type":"object","properties":{"duration_seconds":{"type":"integer","description":"Timer duration in seconds"},"label":{"type":"string"}},"required":["duration_seconds"]})"},
    {"take_screenshot",
     "Capture a screenshot of the current screen",
     R"({"type":"object","properties":{"region":{"type":"string","description":"Full screen or window name","default":"full"}}})"},
    {"read_file",
     "Read the contents of a file from the filesystem",
     R"({"type":"object","properties":{"path":{"type":"string","description":"File path to read"},"encoding":{"type":"string","default":"utf-8"}},"required":["path"]})"},
};
static constexpr int N_EXAMPLE_TOOLS = sizeof(EXAMPLE_TOOLS) / sizeof(EXAMPLE_TOOLS[0]);

struct AppState {
    std::mutex mtx;

    // Model
    lfg_model *model = nullptr;
    lfg_session *session = nullptr;
    bool model_loaded = false;
    lfg_model_stats model_stats = {};

    // Generation
    std::string pending_output;     // tokens accumulated by inference thread
    bool generating = false;
    bool stop_requested = false;
    lfg_generate_result last_result = {};
    double gen_start_time = 0.0;
    double gen_elapsed = 0.0;

    // Config
    lfg_session_config session_cfg = {};
    lfg_sampling_config sampling_cfg = {};
    int32_t gen_max_tokens = 0;  // Per-generation limit. 0 = context-limited.

    // Monitor configs
    lfg_entropy_monitor_config entropy_cfg = {};
    lfg_confidence_monitor_config confidence_cfg = {};
    lfg_surprise_monitor_config surprise_cfg = {};
    bool entropy_enabled = false;
    bool confidence_enabled = false;
    bool surprise_enabled = false;

    // Status bar
    float last_entropy = -1.0f;
    int32_t last_surprise_count = 0;
    float last_surprise_mean = 0.0f;

    // Chat
    std::vector<ChatMessage> messages;
    std::string system_prompt;
    bool needs_session_recreate = false;

    // Surprise log
    std::vector<SurpriseLogEntry> surprise_log;
    std::string surprise_prompt_snapshot; // set before generation starts

    // Retrieval log (entropy events — where model needed more info)
    std::vector<RetrievalLogEntry> retrieval_log;

    // Confidence log (sustained low-entropy spans — store candidates)
    std::vector<ConfidenceLogEntry> confidence_log;

    // Context log (exact formatted prompts + raw output per turn)
    struct ContextLogEntry {
        std::string input;   // Formatted prompt sent to tokenizer
        std::string output;  // Raw unparsed model output
    };
    std::vector<ContextLogEntry> context_log;

    // Generated token IDs (for raw output with special tokens)
    std::vector<lfg_token> generated_tokens;

    // Tool ranking
    bool tools_enabled = true;
    int32_t tool_top_k = 3;
    std::vector<ToolRankingEntry> tool_ranking_log;

    // Log
    std::string log_buffer;
};

static AppState g_state;

// ---------------------------------------------------------------------------
// Token callback for streaming generation
// ---------------------------------------------------------------------------

static lfg_generate_action token_callback(
    lfg_token token, const char *piece, int32_t piece_len, void *user_data)
{
    auto *state = static_cast<AppState *>(user_data);
    std::lock_guard<std::mutex> lock(state->mtx);
    if (state->stop_requested) {
        return LFG_GENERATE_STOP;
    }
    state->pending_output.append(piece, static_cast<size_t>(piece_len));
    state->generated_tokens.push_back(token);
    return LFG_GENERATE_CONTINUE;
}

// ---------------------------------------------------------------------------
// Surprise callback for logging events
// ---------------------------------------------------------------------------

static void surprise_callback(
    const lfg_surprise_event *event, const float * /*embedding*/, void *user_data)
{
    auto *state = static_cast<AppState *>(user_data);
    std::lock_guard<std::mutex> lock(state->mtx);

    SurpriseLogEntry entry;
    entry.prompt_text = state->surprise_prompt_snapshot;
    entry.mean_surprise = event->mean_surprise;
    entry.max_surprise = event->max_surprise;
    entry.n_above_threshold = event->n_above_threshold;
    entry.n_tokens_evaluated = event->n_tokens_evaluated;

    // Count user turns to get turn index
    int turn = 0;
    for (auto &m : state->messages) {
        if (m.role == "user") turn++;
    }
    entry.turn_index = turn;

    state->surprise_log.push_back(std::move(entry));
}

// ---------------------------------------------------------------------------
// Entropy callback for logging retrieval events
// ---------------------------------------------------------------------------

static const char *entropy_callback(
    const lfg_entropy_event *event, const float * /*embedding*/, void *user_data)
{
    auto *state = static_cast<AppState *>(user_data);
    std::lock_guard<std::mutex> lock(state->mtx);

    RetrievalLogEntry entry;
    // Full text = already-drained portion + pending (we hold the lock)
    if (!state->messages.empty() && state->messages.back().role == "assistant") {
        entry.context_text = state->messages.back().text + state->pending_output;
    } else {
        entry.context_text = state->pending_output;
    }
    entry.entropy = event->entropy;
    entry.normalized = event->normalized;
    entry.top_logprob = event->top_logprob;
    entry.n_past = event->n_past;

    int turn = 0;
    for (auto &m : state->messages) {
        if (m.role == "user") turn++;
    }
    entry.turn_index = turn;

    state->retrieval_log.push_back(std::move(entry));

    return nullptr; // Don't inject — just log
}

// ---------------------------------------------------------------------------
// Confidence callback for logging low-entropy spans
// ---------------------------------------------------------------------------

static void confidence_callback(
    const lfg_confidence_event *event, const float * /*embedding*/, void *user_data)
{
    auto *state = static_cast<AppState *>(user_data);
    std::lock_guard<std::mutex> lock(state->mtx);

    ConfidenceLogEntry entry;
    // Full text = already-drained portion + pending (we hold the lock)
    if (!state->messages.empty() && state->messages.back().role == "assistant") {
        entry.context_text = state->messages.back().text + state->pending_output;
    } else {
        entry.context_text = state->pending_output;
    }
    entry.mean_entropy = event->mean_entropy;
    entry.min_entropy = event->min_entropy;
    entry.span_length = event->span_length;
    entry.start_pos = event->start_pos;
    entry.end_pos = event->end_pos;

    int turn = 0;
    for (auto &m : state->messages) {
        if (m.role == "user") turn++;
    }
    entry.turn_index = turn;

    state->confidence_log.push_back(std::move(entry));
}

// ---------------------------------------------------------------------------
// Simple tool executor for demo (handles calculator, stubs for others)
// ---------------------------------------------------------------------------

// Tiny expression evaluator: handles +, -, *, / with parentheses, integer/float operands.
static double eval_expr(const char *&s);

static double eval_atom(const char *&s) {
    while (*s == ' ') s++;
    bool neg = false;
    if (*s == '-') { neg = true; s++; }
    double val;
    if (*s == '(') {
        s++; // skip '('
        val = eval_expr(s);
        if (*s == ')') s++;
    } else {
        char *end;
        val = std::strtod(s, &end);
        s = end;
    }
    return neg ? -val : val;
}

static double eval_term(const char *&s) {
    double val = eval_atom(s);
    while (*s == ' ') s++;
    while (*s == '*' || *s == '/') {
        char op = *s++;
        double rhs = eval_atom(s);
        if (op == '*') val *= rhs; else if (rhs != 0) val /= rhs;
        while (*s == ' ') s++;
    }
    return val;
}

static double eval_expr(const char *&s) {
    double val = eval_term(s);
    while (*s == ' ') s++;
    while (*s == '+' || *s == '-') {
        char op = *s++;
        double rhs = eval_term(s);
        if (op == '+') val += rhs; else val -= rhs;
        while (*s == ' ') s++;
    }
    return val;
}

static std::string execute_tool(const char *name, const char *arguments) {
    if (!name) return "{\"error\": \"no tool name\"}";

    if (std::strcmp(name, "calculator") == 0) {
        // Parse "expression" from JSON args: {"expression": "2 + 2"}
        std::string args(arguments ? arguments : "{}");
        // Simple extraction: find "expression" value
        size_t key_pos = args.find("\"expression\"");
        if (key_pos == std::string::npos) return "{\"error\": \"missing expression\"}";
        size_t colon = args.find(':', key_pos);
        if (colon == std::string::npos) return "{\"error\": \"malformed args\"}";
        // Find the string value
        size_t q1 = args.find('"', colon + 1);
        size_t q2 = (q1 != std::string::npos) ? args.find('"', q1 + 1) : std::string::npos;
        if (q1 == std::string::npos || q2 == std::string::npos)
            return "{\"error\": \"missing expression value\"}";
        std::string expr = args.substr(q1 + 1, q2 - q1 - 1);
        const char *p = expr.c_str();
        double result = eval_expr(p);
        // Format result — use integer format if it's a whole number
        char buf[64];
        if (result == (double)(long long)result && std::fabs(result) < 1e15) {
            std::snprintf(buf, sizeof(buf), "%lld", (long long)result);
        } else {
            std::snprintf(buf, sizeof(buf), "%.6g", result);
        }
        return std::string("{\"result\": ") + buf + "}";
    }

    if (std::strcmp(name, "get_weather") == 0) {
        return R"({"temperature": 22, "condition": "sunny", "humidity": 45})";
    }
    if (std::strcmp(name, "search_web") == 0) {
        return R"({"results": [{"title": "Example result", "snippet": "This is a demo search result."}]})";
    }

    // Generic stub for unimplemented tools
    return std::string("{\"status\": \"ok\", \"note\": \"stub response for ") + name + "\"}";
}

// ---------------------------------------------------------------------------
// Inference thread function
// ---------------------------------------------------------------------------

static void inference_thread_func(AppState *state) {
    // Build message array from chat history
    std::vector<ChatMessage> msgs_copy;
    lfg_generate_config gen_cfg;
    {
        std::lock_guard<std::mutex> lock(state->mtx);
        msgs_copy = state->messages;
        gen_cfg = lfg_generate_default_config();
        gen_cfg.max_tokens = state->gen_max_tokens;
        gen_cfg.token_cb = token_callback;
        gen_cfg.token_cb_data = state;
        if (state->surprise_enabled) {
            gen_cfg.surprise_cb = surprise_callback;
            gen_cfg.surprise_cb_data = state;
        }
        if (state->entropy_enabled) {
            gen_cfg.entropy_cb = entropy_callback;
            gen_cfg.entropy_cb_data = state;
        }
        if (state->confidence_enabled) {
            gen_cfg.confidence_cb = confidence_callback;
            gen_cfg.confidence_cb_data = state;
        }
    }

    // Convert to lfg_chat_message array.
    // Exclude the trailing empty assistant message — it's a UI placeholder for
    // streaming text.  lfg_session_chat_generate() adds the assistant prompt
    // prefix via add_ass=true; including the empty message would produce a
    // double assistant turn and the model would emit EOS immediately.
    std::vector<lfg_chat_message> c_msgs;
    c_msgs.reserve(msgs_copy.size());
    for (size_t i = 0; i < msgs_copy.size(); ++i) {
        auto &m = msgs_copy[i];
        if (i == msgs_copy.size() - 1 && m.role == "assistant" && m.text.empty()) {
            continue;
        }
        lfg_chat_message cm{};
        cm.role = m.role.c_str();
        cm.content = m.text.c_str();
        c_msgs.push_back(cm);
    }

    lfg_generate_result result = lfg_session_chat_generate(
        state->session,
        c_msgs.data(), c_msgs.size(),
        gen_cfg);

    // Tool call loop: execute tools and re-generate until model produces a
    // non-tool-call response (max 5 rounds to prevent infinite loops).
    // Intermediate messages (raw assistant output with tool markers, tool results)
    // are kept ONLY in the c_msgs API array — state->messages stays clean for the UI.
    std::string tool_call_info;
    // Accumulate intermediate messages separately from UI-visible chat history.
    // These hold the raw assistant output (with <|tool_call_start|> markers) and
    // tool result messages that the model needs to see but the user shouldn't.
    struct IntermediateMsg { std::string role; std::string text; };
    std::vector<IntermediateMsg> intermediate_msgs;

    for (int tool_round = 0;
         tool_round < 5 && result.stop_reason == LFG_STOP_TOOL_CALL;
         tool_round++)
    {
        int32_t n_calls = 0;
        const lfg_tool_call *calls = lfg_session_get_tool_calls(state->session, &n_calls);

        if (tool_round > 0) tool_call_info += "--- Round " + std::to_string(tool_round + 1) + " ---\n";

        // Get the raw assistant output (with <|tool_call_start|>...<|tool_call_end|>
        // markers) that the model needs in history per LFM2.5 protocol.
        // This goes into intermediate_msgs, NOT state->messages.
        {
            int32_t raw_len = 0;
            const char *raw_ptr = lfg_session_get_last_output(state->session, &raw_len);
            std::string raw_assistant;
            if (raw_ptr && raw_len > 0) {
                raw_assistant.assign(raw_ptr, static_cast<size_t>(raw_len));
            }
            intermediate_msgs.push_back({"assistant", std::move(raw_assistant)});
        }

        // Execute each tool call, log it, and add results to intermediate_msgs
        for (int32_t tc = 0; tc < n_calls; ++tc) {
            std::string name = calls[tc].name ? calls[tc].name : "?";
            std::string args = calls[tc].arguments ? calls[tc].arguments : "{}";
            std::string id   = calls[tc].id ? calls[tc].id : "?";

            std::string tool_result = execute_tool(calls[tc].name, calls[tc].arguments);

            tool_call_info += name + "(" + args + ")\n";
            tool_call_info += "  id: " + id + "\n";
            tool_call_info += "  result: " + tool_result + "\n\n";

            intermediate_msgs.push_back({"tool", std::move(tool_result)});
        }

        // Clear pending output for the continuation — the existing empty assistant
        // message in state->messages will receive the next generation's tokens
        {
            std::lock_guard<std::mutex> lock(state->mtx);
            state->pending_output.clear();
        }

        // Build c_msgs: original messages (minus trailing empty assistant) +
        // intermediate messages (raw assistant + tool results)
        std::vector<ChatMessage> base_msgs_copy;
        {
            std::lock_guard<std::mutex> lock(state->mtx);
            base_msgs_copy = state->messages;
        }
        c_msgs.clear();
        for (size_t i = 0; i < base_msgs_copy.size(); ++i) {
            auto &m = base_msgs_copy[i];
            // Skip trailing empty assistant placeholder
            if (i == base_msgs_copy.size() - 1 && m.role == "assistant" && m.text.empty()) {
                continue;
            }
            lfg_chat_message cm{};
            cm.role = m.role.c_str();
            cm.content = m.text.c_str();
            c_msgs.push_back(cm);
        }
        // Append intermediate messages (raw assistant outputs + tool results)
        for (auto &im : intermediate_msgs) {
            lfg_chat_message cm{};
            cm.role = im.role.c_str();
            cm.content = im.text.c_str();
            c_msgs.push_back(cm);
        }

        // Reset session so chat_generate re-ingests the full conversation
        lfg_session_reset(state->session);

        // Re-generate with the tool results in context
        result = lfg_session_chat_generate(
            state->session,
            c_msgs.data(), c_msgs.size(),
            gen_cfg);
    }

    // Capture the formatted prompt for the Context tab
    int32_t prompt_len = 0;
    const char *prompt_text = lfg_session_get_last_prompt(state->session, &prompt_len);

    // Capture tool ranking if tools are enabled
    std::string rank_output;
    std::string rank_query;
    bool do_rank = false;
    {
        std::lock_guard<std::mutex> lock(state->mtx);
        do_rank = state->tools_enabled;
        // Find the last user message as ranking query
        for (int i = (int)state->messages.size() - 1; i >= 0; i--) {
            if (state->messages[i].role == "user") {
                rank_query = state->messages[i].text;
                break;
            }
        }
    }
    if (do_rank && !rank_query.empty()) {
        // First call to get required size
        int32_t needed = lfg_session_rank_tools(state->session,
            rank_query.c_str(), (int32_t)rank_query.size(), nullptr, 0);
        if (needed > 0) {
            rank_output.resize(needed);
            lfg_session_rank_tools(state->session,
                rank_query.c_str(), (int32_t)rank_query.size(),
                &rank_output[0], needed + 1);
        }
    }

    // Get raw output from session (already accumulated with special tokens)
    std::string raw_output;
    {
        int32_t out_len = 0;
        const char *out_ptr = lfg_session_get_last_output(state->session, &out_len);
        if (out_ptr && out_len > 0) {
            raw_output.assign(out_ptr, static_cast<size_t>(out_len));
        }
    }

    {
        std::lock_guard<std::mutex> lock(state->mtx);
        // Capture context input + raw output (with special tokens visible)
        {
            AppState::ContextLogEntry ctx_entry;
            if (prompt_text && prompt_len > 0) {
                ctx_entry.input.assign(prompt_text, static_cast<size_t>(prompt_len));
            }
            ctx_entry.output = raw_output;
            if (!ctx_entry.input.empty() || !ctx_entry.output.empty()) {
                state->context_log.push_back(std::move(ctx_entry));
            }
        }
        // Store tool ranking entry (includes structured calls if present)
        if (!rank_output.empty() || !tool_call_info.empty()) {
            int turn = 0;
            for (auto &m : state->messages) {
                if (m.role == "user") turn++;
            }
            std::string combined = rank_output;
            if (!tool_call_info.empty()) {
                if (!combined.empty()) combined += "\n\n--- Parsed Tool Calls ---\n";
                combined += tool_call_info;
            }
            if (!rank_query.empty() || !combined.empty()) {
                state->tool_ranking_log.push_back({rank_query, combined, turn});
            }
        }
        state->last_result = result;
        state->gen_elapsed = glfwGetTime() - state->gen_start_time;
        state->last_entropy = lfg_session_get_last_entropy(state->session);
        state->generating = false;
    }
}

// ---------------------------------------------------------------------------
// Helper: recreate session with current config
// ---------------------------------------------------------------------------

static void recreate_session(AppState *state) {
    if (state->session) {
        lfg_session_free(state->session);
        state->session = nullptr;
    }
    if (!state->model) return;

    state->session = lfg_session_create(state->model, &state->session_cfg);
    if (!state->session) {
        state->log_buffer += "ERROR: Failed to create session\n";
        return;
    }

    // Configure monitors
    if (state->entropy_enabled) {
        lfg_session_configure_entropy_monitor(state->session, &state->entropy_cfg);
    }
    if (state->confidence_enabled) {
        lfg_session_configure_confidence_monitor(state->session, &state->confidence_cfg);
    }
    if (state->surprise_enabled) {
        lfg_session_configure_surprise_monitor(state->session, &state->surprise_cfg);
    }

    // Register tools
    if (state->tools_enabled) {
        int32_t n = lfg_session_register_tools(state->session, EXAMPLE_TOOLS, N_EXAMPLE_TOOLS, state->tool_top_k);
        if (n > 0) {
            state->log_buffer += "Registered " + std::to_string(n) + " tools (top_k=" +
                                 std::to_string(state->tool_top_k) + ")\n";
        } else {
            state->log_buffer += "WARNING: Failed to register tools\n";
        }
    }

    state->needs_session_recreate = false;
    state->log_buffer += "Session created (n_ctx=" + std::to_string(state->session_cfg.n_ctx) +
                         ", threads=" + std::to_string(state->session_cfg.n_threads) + ")\n";
}

// ---------------------------------------------------------------------------
// Helper: selectable read-only text block (supports select + copy)
// ---------------------------------------------------------------------------

// Word-wrap text to fit within available width, inserting newlines at word boundaries.
static std::string wrap_text(const std::string &text, float wrap_width) {
    if (text.empty() || wrap_width <= 0.0f) return text;

    std::string result;
    result.reserve(text.size() + text.size() / 40);

    float space_w = ImGui::CalcTextSize(" ").x;
    float line_w = 0.0f;

    size_t i = 0;
    while (i < text.size()) {
        // Handle explicit newlines
        if (text[i] == '\n') {
            result += '\n';
            line_w = 0.0f;
            ++i;
            continue;
        }

        // Find next word (or whitespace run)
        size_t word_start = i;
        if (text[i] == ' ' || text[i] == '\t') {
            while (i < text.size() && (text[i] == ' ' || text[i] == '\t')) ++i;
            float ws_w = space_w * (float)(i - word_start);
            if (line_w + ws_w <= wrap_width) {
                result.append(i - word_start, ' ');
                line_w += ws_w;
            }
            // If whitespace would overflow, just skip it (line break absorbs it)
            continue;
        }

        // Measure word
        size_t word_end = i;
        while (word_end < text.size() && text[word_end] != ' ' && text[word_end] != '\t' && text[word_end] != '\n')
            ++word_end;

        ImVec2 word_size = ImGui::CalcTextSize(text.c_str() + i, text.c_str() + word_end);

        if (line_w > 0.0f && line_w + word_size.x > wrap_width) {
            result += '\n';
            line_w = 0.0f;
        }

        result.append(text, i, word_end - i);
        line_w += word_size.x;
        i = word_end;
    }

    return result;
}

static void selectable_text(const char *label, const std::string &text, const ImVec4 *color = nullptr) {
    if (text.empty()) return;
    if (color) ImGui::PushStyleColor(ImGuiCol_Text, *color);

    float avail_width = ImGui::GetContentRegionAvail().x;
    std::string wrapped = wrap_text(text, avail_width - ImGui::GetStyle().FramePadding.x * 2);

    // Count lines for height
    float line_height = ImGui::GetTextLineHeightWithSpacing();
    int n_lines = 1;
    for (char c : wrapped) {
        if (c == '\n') n_lines++;
    }
    float height = line_height * (n_lines + 1);

    ImGui::InputTextMultiline(label, const_cast<char *>(wrapped.c_str()), wrapped.size() + 1,
                              ImVec2(-1.0f, height),
                              ImGuiInputTextFlags_ReadOnly);
    if (color) ImGui::PopStyleColor();
}

// ---------------------------------------------------------------------------
// UI Panels
// ---------------------------------------------------------------------------

static char model_path_buf[1024] = "models/LFM2.5-1.2B-Thinking-GGUF/LFM2.5-1.2B-Thinking-Q4_K_M.gguf";

static void draw_model_panel(AppState *state) {
    ImGui::Text("Model");
    ImGui::Separator();

    ImGui::InputText("Path", model_path_buf, sizeof(model_path_buf));
    ImGui::SameLine();
    if (ImGui::Button("Browse...")) {
        const char *path = open_file_dialog("Select Model", "models");
        if (path) {
            snprintf(model_path_buf, sizeof(model_path_buf), "%s", path);
            free((void *)path);
        }
    }

    bool can_load = !state->model_loaded && !state->generating;
    bool can_unload = state->model_loaded && !state->generating;

    if (!can_load) ImGui::BeginDisabled();
    if (ImGui::Button("Load Model")) {
        state->log_buffer += "Loading model: " + std::string(model_path_buf) + "...\n";

        lfg_model_load_config load_cfg = lfg_model_load_default_config();
        load_cfg.model_path = model_path_buf;

        state->model = lfg_load_model(&load_cfg);
        if (state->model) {
            state->model_loaded = true;
            state->model_stats = lfg_model_get_stats(state->model);
            state->log_buffer += "Model loaded. Params: " +
                std::to_string(state->model_stats.n_params) +
                ", vocab: " + std::to_string(state->model_stats.n_vocab) +
                ", ctx_train: " + std::to_string(state->model_stats.n_ctx_train) + "\n";

            // Create session with current config
            recreate_session(state);
        } else {
            state->log_buffer += "ERROR: Failed to load model\n";
        }
    }
    if (!can_load) ImGui::EndDisabled();

    ImGui::SameLine();

    if (!can_unload) ImGui::BeginDisabled();
    if (ImGui::Button("Unload")) {
        if (state->session) {
            lfg_session_free(state->session);
            state->session = nullptr;
        }
        if (state->model) {
            lfg_model_free(state->model);
            state->model = nullptr;
        }
        state->model_loaded = false;
        state->model_stats = {};
        state->messages.clear();
        state->log_buffer += "Model unloaded\n";
    }
    if (!can_unload) ImGui::EndDisabled();

    if (state->model_loaded) {
        ImGui::Text("Params: %llu", (unsigned long long)state->model_stats.n_params);
        ImGui::Text("Vocab:  %d", state->model_stats.n_vocab);
        ImGui::Text("Ctx:    %d", state->model_stats.n_ctx_train);
    }
}

static void draw_settings_panel(AppState *state) {
    ImGui::Text("Settings");
    ImGui::Separator();

    bool changed = false;

    if (ImGui::CollapsingHeader("Session", ImGuiTreeNodeFlags_DefaultOpen)) {
        changed |= ImGui::SliderInt("Threads", &state->session_cfg.n_threads, 1, 16);
        changed |= ImGui::SliderInt("Context", &state->session_cfg.n_ctx, 128, 8192);
        changed |= ImGui::SliderInt("Batch", &state->session_cfg.n_batch, 32, 2048);
        changed |= ImGui::InputInt("Reasoning Budget", &state->session_cfg.reasoning_budget);
    }

    if (ImGui::CollapsingHeader("Generation", ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::InputInt("Max Tokens", &state->gen_max_tokens);
        if (state->gen_max_tokens < 0) state->gen_max_tokens = 0;
        ImGui::SameLine();
        ImGui::TextDisabled("(0 = unlimited)");
    }

    if (ImGui::CollapsingHeader("Sampling", ImGuiTreeNodeFlags_DefaultOpen)) {
        int seed_int = static_cast<int>(state->sampling_cfg.seed);
        if (ImGui::InputInt("Seed", &seed_int)) {
            state->sampling_cfg.seed = static_cast<uint32_t>(seed_int);
            changed = true;
        }
        changed |= ImGui::SliderFloat("Temperature", &state->sampling_cfg.temp, 0.0f, 2.0f);
        changed |= ImGui::SliderInt("Top K", &state->sampling_cfg.top_k, 0, 200);
        changed |= ImGui::SliderFloat("Top P", &state->sampling_cfg.top_p, 0.0f, 1.0f);
        changed |= ImGui::SliderFloat("Min P", &state->sampling_cfg.min_p, 0.0f, 0.5f);
        changed |= ImGui::SliderFloat("Repeat Penalty", &state->sampling_cfg.penalty_repeat, 1.0f, 2.0f);
    }

    if (ImGui::CollapsingHeader("Monitors")) {
        bool monitors_changed = false;

        if (ImGui::Checkbox("Entropy", &state->entropy_enabled)) monitors_changed = true;
        if (state->entropy_enabled) {
            ImGui::Indent();
            monitors_changed |= ImGui::SliderFloat("Ent. Threshold", &state->entropy_cfg.threshold, 0.01f, 1.0f);
            monitors_changed |= ImGui::SliderInt("Ent. Cooldown", &state->entropy_cfg.cooldown_tokens, 0, 50);
            ImGui::Unindent();
        }

        if (ImGui::Checkbox("Confidence", &state->confidence_enabled)) monitors_changed = true;
        if (state->confidence_enabled) {
            ImGui::Indent();
            monitors_changed |= ImGui::SliderFloat("Conf. Threshold", &state->confidence_cfg.threshold, 0.01f, 1.0f);
            monitors_changed |= ImGui::SliderInt("Conf. Min Span", &state->confidence_cfg.min_span, 1, 50);
            ImGui::Unindent();
        }

        if (ImGui::Checkbox("Surprise", &state->surprise_enabled)) monitors_changed = true;
        if (state->surprise_enabled) {
            ImGui::Indent();
            monitors_changed |= ImGui::SliderFloat("Surp. Threshold", &state->surprise_cfg.threshold, 0.01f, 1.0f);
            ImGui::Unindent();
        }

        // Configure monitors live on the existing session — no recreation needed
        if (monitors_changed && state->session && !state->generating) {
            if (state->entropy_enabled) {
                lfg_session_configure_entropy_monitor(state->session, &state->entropy_cfg);
            }
            if (state->confidence_enabled) {
                lfg_session_configure_confidence_monitor(state->session, &state->confidence_cfg);
            }
            if (state->surprise_enabled) {
                lfg_session_configure_surprise_monitor(state->session, &state->surprise_cfg);
            }
        }
    }

    if (ImGui::CollapsingHeader("Tools")) {
        if (ImGui::Checkbox("Enable Tools", &state->tools_enabled)) {
            changed = true;
        }
        if (state->tools_enabled) {
            ImGui::Indent();
            if (ImGui::SliderInt("Top K", &state->tool_top_k, 1, 12)) {
                changed = true;
            }
            ImGui::TextDisabled("%d tools registered", N_EXAMPLE_TOOLS);
            ImGui::Unindent();
        }
    }

    if (changed) {
        state->session_cfg.sampling = state->sampling_cfg;
        state->needs_session_recreate = true;
    }

    if (state->needs_session_recreate && state->model_loaded && !state->generating) {
        if (ImGui::Button("Apply Settings")) {
            state->messages.clear();
            recreate_session(state);
        }
        ImGui::SameLine();
        ImGui::TextColored(ImVec4(1.0f, 0.8f, 0.0f, 1.0f), "Settings changed");
    }
}

static char chat_input_buf[4096] = "";
static char system_prompt_buf[2048] = "You are a helpful assistant.";

static void draw_chat_panel(AppState *state) {
    ImGui::Text("Chat");
    ImGui::Separator();

    // System prompt
    if (ImGui::CollapsingHeader("System Prompt")) {
        ImGui::InputTextMultiline("##sys_prompt", system_prompt_buf, sizeof(system_prompt_buf),
                                  ImVec2(-1.0f, ImGui::GetTextLineHeight() * 4));
        state->system_prompt = system_prompt_buf;
    }

    // Chat history
    float footer_height = ImGui::GetFrameHeightWithSpacing() * 2 + 8;
    ImGui::BeginChild("ChatHistory", ImVec2(0, -footer_height), ImGuiChildFlags_Borders);

    for (size_t i = 0; i < state->messages.size(); i++) {
        auto &msg = state->messages[i];
        if (msg.role == "system") continue;

        ImVec4 color;
        const char *label;
        if (msg.role == "user") {
            color = ImVec4(0.4f, 0.7f, 1.0f, 1.0f);
            label = "You";
        } else {
            color = ImVec4(0.5f, 1.0f, 0.5f, 1.0f);
            label = "Assistant";
        }

        // Note: pending_output is drained in the main loop (before ImGui frame),
        // so msg.text is always up to date here — no secondary drain needed.

        if (msg.role == "assistant") {
            std::string thinking, output;
            msg.split_thinking(thinking, output);

            ImGui::TextColored(color, "%s:", label);

            // Show output text (the actual response)
            if (!output.empty()) {
                char out_id[32];
                snprintf(out_id, sizeof(out_id), "##aout%d", (int)i);
                selectable_text(out_id, output);
            }

            // Show thinking as dim collapsed section below
            if (!thinking.empty()) {
                ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.5f, 0.5f, 0.5f, 0.7f));
                bool still_thinking = state->generating && i == state->messages.size() - 1 &&
                                      msg.text.find("</think>") == std::string::npos;
                char header[32];
                snprintf(header, sizeof(header), "%s##think%d",
                         still_thinking ? "Thinking..." : "Thinking", (int)i);
                if (ImGui::TreeNode(header)) {
                    char think_id[32];
                    snprintf(think_id, sizeof(think_id), "##athink%d", (int)i);
                    ImVec4 dim(0.5f, 0.5f, 0.5f, 0.7f);
                    selectable_text(think_id, thinking, &dim);
                    ImGui::TreePop();
                }
                ImGui::PopStyleColor();
            }
        } else {
            ImGui::TextColored(color, "%s:", label);
            char user_id[32];
            snprintf(user_id, sizeof(user_id), "##umsg%d", (int)i);
            selectable_text(user_id, msg.text, &color);
        }

        ImGui::Spacing();
    }

    // Auto-scroll
    if (ImGui::GetScrollY() >= ImGui::GetScrollMaxY() - 20.0f) {
        ImGui::SetScrollHereY(1.0f);
    }

    ImGui::EndChild();

    // Input area
    bool can_send = state->model_loaded && state->session && !state->generating;

    ImGui::PushItemWidth(-80);
    bool enter_pressed = ImGui::InputText("##chat_input", chat_input_buf, sizeof(chat_input_buf),
                                          ImGuiInputTextFlags_EnterReturnsTrue);
    ImGui::PopItemWidth();
    ImGui::SameLine();

    if (state->generating) {
        if (ImGui::Button("Stop", ImVec2(70, 0))) {
            std::lock_guard<std::mutex> lock(state->mtx);
            state->stop_requested = true;
        }
    } else {
        if (!can_send) ImGui::BeginDisabled();
        bool send = ImGui::Button("Send", ImVec2(70, 0)) || (enter_pressed && can_send);
        if (!can_send) ImGui::EndDisabled();

        if (send && strlen(chat_input_buf) > 0) {
            // Reset session for new conversation turn
            lfg_session_reset(state->session);

            // Add system prompt if messages empty
            if (state->messages.empty() && strlen(system_prompt_buf) > 0) {
                state->messages.push_back({"system", system_prompt_buf});
            }

            // Add user message
            state->messages.push_back({"user", chat_input_buf});

            // Snapshot prompt for surprise log before clearing input
            state->surprise_prompt_snapshot = chat_input_buf;
            chat_input_buf[0] = '\0';

            // Add empty assistant message to fill
            state->messages.push_back({"assistant", ""});

            // Start generation
            {
                std::lock_guard<std::mutex> lock(state->mtx);
                state->generating = true;
                state->stop_requested = false;
                state->pending_output.clear();
                state->generated_tokens.clear();
                state->gen_start_time = glfwGetTime();
                state->last_result = {};
            }

            std::thread(inference_thread_func, state).detach();
        }
    }

    // Clear chat button
    ImGui::SameLine();
    if (state->generating) ImGui::BeginDisabled();
    if (ImGui::Button("Clear")) {
        state->messages.clear();
        if (state->session) lfg_session_reset(state->session);
    }
    if (state->generating) ImGui::EndDisabled();
}

static void draw_status_bar(AppState *state) {
    ImGui::Separator();

    if (state->generating) {
        double elapsed = glfwGetTime() - state->gen_start_time;
        ImGui::Text("Generating... (%.1fs)", elapsed);
    } else if (state->last_result.n_tokens > 0) {
        float tps = state->gen_elapsed > 0.0 ?
            static_cast<float>(state->last_result.n_tokens) / static_cast<float>(state->gen_elapsed) : 0.0f;

        const char *stop_str = "unknown";
        switch (state->last_result.stop_reason) {
            case LFG_STOP_EOS:        stop_str = "EOS"; break;
            case LFG_STOP_MAX_TOKENS: stop_str = "max_tokens"; break;
            case LFG_STOP_CALLBACK:   stop_str = "stopped"; break;
            case LFG_STOP_TOOL_CALL:  stop_str = "tool_call"; break;
        }

        ImGui::Text("Tokens: %d | %.1f tok/s | Stop: %s",
                     state->last_result.n_tokens, tps, stop_str);
    } else {
        ImGui::Text("Ready");
    }

    ImGui::SameLine(ImGui::GetWindowWidth() - 300);

    if (state->last_entropy >= 0.0f) {
        ImGui::Text("Entropy: %.3f", state->last_entropy);
        ImGui::SameLine();
    }
    if (state->last_result.n_surprise_events > 0) {
        ImGui::Text("Surprise: %.3f", state->last_surprise_mean);
    }
}

static void draw_surprise_panel(AppState *state) {
    if (state->surprise_log.empty()) {
        ImGui::TextDisabled("No surprise events yet. Enable the Surprise monitor and send messages.");
        return;
    }

    // Summary
    ImGui::Text("Events: %d", (int)state->surprise_log.size());
    ImGui::SameLine();
    if (ImGui::Button("Clear##surprise")) {
        state->surprise_log.clear();
        return;
    }
    ImGui::Separator();

    // Scrollable list
    ImGui::BeginChild("SurpriseList", ImVec2(0, 0), ImGuiChildFlags_Borders);

    for (int idx = (int)state->surprise_log.size() - 1; idx >= 0; idx--) {
        auto &e = state->surprise_log[idx];
        ImGui::PushID(idx);

        // Header with turn number and key metric
        bool open = ImGui::CollapsingHeader(
            ("Turn " + std::to_string(e.turn_index) +
             "  mean=" + std::to_string(e.mean_surprise).substr(0, 5) +
             "  max=" + std::to_string(e.max_surprise).substr(0, 5)).c_str(),
            ImGuiTreeNodeFlags_DefaultOpen);

        if (open) {
            ImGui::Indent();

            // Metrics table
            if (ImGui::BeginTable("##metrics", 2, ImGuiTableFlags_SizingStretchProp)) {
                ImGui::TableNextRow();
                ImGui::TableNextColumn(); ImGui::TextDisabled("Mean Surprise");
                ImGui::TableNextColumn(); ImGui::Text("%.4f", e.mean_surprise);

                ImGui::TableNextRow();
                ImGui::TableNextColumn(); ImGui::TextDisabled("Max Surprise");
                ImGui::TableNextColumn(); ImGui::Text("%.4f", e.max_surprise);

                ImGui::TableNextRow();
                ImGui::TableNextColumn(); ImGui::TextDisabled("Tokens Above");
                ImGui::TableNextColumn(); ImGui::Text("%d / %d", e.n_above_threshold, e.n_tokens_evaluated);

                float ratio = e.n_tokens_evaluated > 0
                    ? (float)e.n_above_threshold / (float)e.n_tokens_evaluated : 0.0f;
                ImGui::TableNextRow();
                ImGui::TableNextColumn(); ImGui::TextDisabled("Ratio");
                ImGui::TableNextColumn(); ImGui::Text("%.1f%%", ratio * 100.0f);

                ImGui::EndTable();
            }

            // Prompt text
            ImGui::TextDisabled("Prompt:");
            ImVec4 prompt_color(0.7f, 0.8f, 1.0f, 1.0f);
            selectable_text("##prompt", e.prompt_text, &prompt_color);

            ImGui::Unindent();
        }

        ImGui::PopID();
        ImGui::Spacing();
    }

    ImGui::EndChild();
}

static void draw_retrieval_panel(AppState *state) {
    if (state->retrieval_log.empty()) {
        ImGui::TextDisabled("No retrieval events yet. Enable the Entropy monitor and generate text.");
        return;
    }

    // Summary
    ImGui::Text("Events: %d", (int)state->retrieval_log.size());
    ImGui::SameLine();
    if (ImGui::Button("Clear##retrieval")) {
        state->retrieval_log.clear();
        return;
    }
    ImGui::Separator();

    // Scrollable list (newest first)
    ImGui::BeginChild("RetrievalList", ImVec2(0, 0), ImGuiChildFlags_Borders);

    for (int idx = (int)state->retrieval_log.size() - 1; idx >= 0; idx--) {
        auto &e = state->retrieval_log[idx];
        ImGui::PushID(idx + 10000); // offset to avoid ID collision with surprise panel

        char header[128];
        snprintf(header, sizeof(header), "Turn %d  pos=%d  norm=%.4f",
                 e.turn_index, e.n_past, e.normalized);

        bool open = ImGui::CollapsingHeader(header, ImGuiTreeNodeFlags_DefaultOpen);

        if (open) {
            ImGui::Indent();

            // Metrics table
            if (ImGui::BeginTable("##rmetrics", 2, ImGuiTableFlags_SizingStretchProp)) {
                ImGui::TableNextRow();
                ImGui::TableNextColumn(); ImGui::TextDisabled("Normalized Entropy");
                ImGui::TableNextColumn(); ImGui::Text("%.4f", e.normalized);

                ImGui::TableNextRow();
                ImGui::TableNextColumn(); ImGui::TextDisabled("Raw Entropy");
                ImGui::TableNextColumn(); ImGui::Text("%.2f", e.entropy);

                ImGui::TableNextRow();
                ImGui::TableNextColumn(); ImGui::TextDisabled("Top Logprob");
                ImGui::TableNextColumn(); ImGui::Text("%.4f", e.top_logprob);

                ImGui::TableNextRow();
                ImGui::TableNextColumn(); ImGui::TextDisabled("Token Position");
                ImGui::TableNextColumn(); ImGui::Text("%d", e.n_past);

                ImGui::EndTable();
            }

            // Context: text generated up to this point
            if (!e.context_text.empty()) {
                ImGui::TextDisabled("Generated text at trigger:");
                ImVec4 ctx_color(1.0f, 0.85f, 0.5f, 1.0f);
                selectable_text("##ctx", e.context_text, &ctx_color);
            } else {
                ImGui::TextDisabled("(no generated text yet — triggered early)");
            }

            ImGui::Unindent();
        }

        ImGui::PopID();
        ImGui::Spacing();
    }

    ImGui::EndChild();
}

static void draw_confidence_panel(AppState *state) {
    if (state->confidence_log.empty()) {
        ImGui::TextDisabled("No confidence events yet. Enable the Confidence monitor and generate text.");
        return;
    }

    // Summary
    ImGui::Text("Events: %d", (int)state->confidence_log.size());
    ImGui::SameLine();
    if (ImGui::Button("Clear##confidence")) {
        state->confidence_log.clear();
        return;
    }
    ImGui::Separator();

    // Scrollable list (newest first)
    ImGui::BeginChild("ConfidenceList", ImVec2(0, 0), ImGuiChildFlags_Borders);

    for (int idx = (int)state->confidence_log.size() - 1; idx >= 0; idx--) {
        auto &e = state->confidence_log[idx];
        ImGui::PushID(idx + 20000);

        char header[128];
        snprintf(header, sizeof(header), "Turn %d  span=%d  mean=%.4f  [%d-%d]",
                 e.turn_index, e.span_length, e.mean_entropy, e.start_pos, e.end_pos);

        bool open = ImGui::CollapsingHeader(header, ImGuiTreeNodeFlags_DefaultOpen);

        if (open) {
            ImGui::Indent();

            if (ImGui::BeginTable("##cmetrics", 2, ImGuiTableFlags_SizingStretchProp)) {
                ImGui::TableNextRow();
                ImGui::TableNextColumn(); ImGui::TextDisabled("Mean Entropy");
                ImGui::TableNextColumn(); ImGui::Text("%.4f", e.mean_entropy);

                ImGui::TableNextRow();
                ImGui::TableNextColumn(); ImGui::TextDisabled("Min Entropy");
                ImGui::TableNextColumn(); ImGui::Text("%.4f", e.min_entropy);

                ImGui::TableNextRow();
                ImGui::TableNextColumn(); ImGui::TextDisabled("Span Length");
                ImGui::TableNextColumn(); ImGui::Text("%d tokens", e.span_length);

                ImGui::TableNextRow();
                ImGui::TableNextColumn(); ImGui::TextDisabled("Position");
                ImGui::TableNextColumn(); ImGui::Text("%d - %d", e.start_pos, e.end_pos);

                ImGui::EndTable();
            }

            // Context: text generated at span end
            if (!e.context_text.empty()) {
                ImGui::TextDisabled("Generated text at span end:");
                ImVec4 ctx_color(0.5f, 1.0f, 0.7f, 1.0f);
                selectable_text("##ctx", e.context_text, &ctx_color);
            } else {
                ImGui::TextDisabled("(no generated text yet)");
            }

            ImGui::Unindent();
        }

        ImGui::PopID();
        ImGui::Spacing();
    }

    ImGui::EndChild();
}

static void draw_tools_panel(AppState *state) {
    if (state->tool_ranking_log.empty()) {
        ImGui::TextDisabled("No tool ranking results yet. Enable tools in Settings and send a message.");
        return;
    }

    ImGui::Text("Rankings: %d", (int)state->tool_ranking_log.size());
    ImGui::SameLine();
    if (ImGui::Button("Clear##tools")) {
        state->tool_ranking_log.clear();
        return;
    }
    ImGui::Separator();

    ImGui::BeginChild("ToolsList", ImVec2(0, 0), ImGuiChildFlags_Borders);

    for (int idx = (int)state->tool_ranking_log.size() - 1; idx >= 0; idx--) {
        auto &e = state->tool_ranking_log[idx];
        ImGui::PushID(idx + 40000);

        char header[128];
        snprintf(header, sizeof(header), "Turn %d  \"%.*s\"",
                 e.turn_index,
                 (int)std::min(e.query.size(), (size_t)50),
                 e.query.c_str());

        bool open = ImGui::CollapsingHeader(header, ImGuiTreeNodeFlags_DefaultOpen);

        if (open) {
            ImGui::Indent();

            ImGui::TextDisabled("Query:");
            ImVec4 query_color(0.4f, 0.7f, 1.0f, 1.0f);
            char qid[32];
            snprintf(qid, sizeof(qid), "##tquery%d", idx);
            selectable_text(qid, e.query, &query_color);

            ImGui::Spacing();
            ImGui::TextDisabled("Ranked Tools:");
            ImVec4 rank_color(0.9f, 0.85f, 0.5f, 1.0f);
            char rid[32];
            snprintf(rid, sizeof(rid), "##tranked%d", idx);
            selectable_text(rid, e.ranked_output, &rank_color);

            ImGui::Unindent();
        }

        ImGui::PopID();
        ImGui::Spacing();
    }

    ImGui::EndChild();
}

static void draw_context_panel(AppState *state) {
    if (state->context_log.empty()) {
        ImGui::TextDisabled("No context entries yet. Send a message to see the formatted prompt and raw output.");
        return;
    }

    ImGui::Text("Entries: %d", (int)state->context_log.size());
    ImGui::SameLine();
    if (ImGui::Button("Clear##context")) {
        state->context_log.clear();
        return;
    }
    ImGui::Separator();

    // Scrollable list (newest first)
    ImGui::BeginChild("ContextList", ImVec2(0, 0), ImGuiChildFlags_Borders);

    for (int idx = (int)state->context_log.size() - 1; idx >= 0; idx--) {
        auto &entry = state->context_log[idx];
        ImGui::PushID(idx + 30000);

        char header[64];
        snprintf(header, sizeof(header), "Turn %d", idx + 1);

        if (ImGui::CollapsingHeader(header)) {
            ImGui::Indent();

            // Context Input (formatted prompt)
            if (!entry.input.empty()) {
                ImGui::TextDisabled("Context Input (%d chars):", (int)entry.input.size());
                ImVec4 input_color(0.8f, 0.8f, 0.6f, 1.0f);
                char in_id[32];
                snprintf(in_id, sizeof(in_id), "##ctxin%d", idx);
                selectable_text(in_id, entry.input, &input_color);
            } else {
                ImGui::TextDisabled("(no context input captured)");
            }

            ImGui::Spacing();

            // Raw Output (unparsed model output)
            if (!entry.output.empty()) {
                ImGui::TextDisabled("Raw Output (%d chars):", (int)entry.output.size());
                ImVec4 output_color(0.5f, 1.0f, 0.5f, 1.0f);
                char out_id[32];
                snprintf(out_id, sizeof(out_id), "##ctxout%d", idx);
                selectable_text(out_id, entry.output, &output_color);
            } else {
                ImGui::TextDisabled("(no output captured)");
            }

            ImGui::Unindent();
        }

        ImGui::PopID();
        ImGui::Spacing();
    }

    ImGui::EndChild();
}

static void draw_log_panel(AppState *state) {
    if (ImGui::CollapsingHeader("Log")) {
        ImGui::BeginChild("LogScroll", ImVec2(0, 120), ImGuiChildFlags_Borders);
        ImGui::TextUnformatted(state->log_buffer.c_str());
        if (ImGui::GetScrollY() >= ImGui::GetScrollMaxY() - 10.0f) {
            ImGui::SetScrollHereY(1.0f);
        }
        ImGui::EndChild();
        if (ImGui::Button("Clear Log")) {
            state->log_buffer.clear();
        }
    }
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

static void glfw_error_callback(int error, const char *description) {
    fprintf(stderr, "GLFW Error %d: %s\n", error, description);
}

int main(int /*argc*/, char ** /*argv*/) {
    glfwSetErrorCallback(glfw_error_callback);
    if (!glfwInit()) {
        fprintf(stderr, "Failed to initialize GLFW\n");
        return 1;
    }

    // OpenGL 3.2 + GLSL 150 (macOS)
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 2);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GLFW_TRUE);
    const char *glsl_version = "#version 150";

    GLFWwindow *window = glfwCreateWindow(1280, 800, "lfg-demo", nullptr, nullptr);
    if (!window) {
        fprintf(stderr, "Failed to create GLFW window\n");
        glfwTerminate();
        return 1;
    }
    glfwMakeContextCurrent(window);
    glfwSwapInterval(1); // vsync

    // Setup ImGui
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO &io = ImGui::GetIO();
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;

    ImGui::StyleColorsDark();

    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init(glsl_version);

    // Initialize defaults
    g_state.session_cfg = lfg_session_default_config();
    g_state.session_cfg.n_ctx = 4096;
    g_state.sampling_cfg = lfg_sampling_default_config();
    g_state.session_cfg.sampling = g_state.sampling_cfg;
    g_state.entropy_cfg = lfg_entropy_monitor_default_config();
    g_state.confidence_cfg = lfg_confidence_monitor_default_config();
    g_state.surprise_cfg = lfg_surprise_monitor_default_config();

    g_state.log_buffer = "lfg-demo started\n";

    // Main loop
    while (!glfwWindowShouldClose(window)) {
        glfwPollEvents();

        // Drain pending output — must run even after generation finishes,
        // otherwise fast models (350M) complete between frames and output is lost.
        {
            std::lock_guard<std::mutex> lock(g_state.mtx);
            if (!g_state.pending_output.empty() && !g_state.messages.empty()) {
                auto &last = g_state.messages.back();
                if (last.role == "assistant") {
                    last.text += g_state.pending_output;
                    g_state.pending_output.clear();
                }
            }
        }

        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        // Full-window dockspace
        ImGuiViewport *viewport = ImGui::GetMainViewport();
        ImGui::SetNextWindowPos(viewport->WorkPos);
        ImGui::SetNextWindowSize(viewport->WorkSize);
        ImGui::Begin("##MainWindow", nullptr,
                     ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize |
                     ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoCollapse |
                     ImGuiWindowFlags_NoBringToFrontOnFocus);

        // Layout: left sidebar + main area
        float sidebar_width = 320.0f;

        // Left sidebar
        ImGui::BeginChild("Sidebar", ImVec2(sidebar_width, 0));
        draw_model_panel(&g_state);
        ImGui::Spacing();
        draw_settings_panel(&g_state);
        ImGui::Spacing();
        draw_log_panel(&g_state);
        ImGui::EndChild();

        ImGui::SameLine();

        // Main area with tabs
        ImGui::BeginChild("MainArea", ImVec2(0, 0));
        if (ImGui::BeginTabBar("MainTabs", ImGuiTabBarFlags_FittingPolicyScroll)) {
            if (ImGui::BeginTabItem("Chat")) {
                draw_chat_panel(&g_state);
                draw_status_bar(&g_state);
                ImGui::EndTabItem();
            }
            // Show surprise count badge in tab label
            char surprise_label[64];
            if (g_state.surprise_log.empty()) {
                snprintf(surprise_label, sizeof(surprise_label), "Surprise");
            } else {
                snprintf(surprise_label, sizeof(surprise_label), "Surprise (%d)",
                         (int)g_state.surprise_log.size());
            }
            if (ImGui::BeginTabItem(surprise_label)) {
                draw_surprise_panel(&g_state);
                ImGui::EndTabItem();
            }
            // Show retrieval count badge in tab label
            char retrieval_label[64];
            if (g_state.retrieval_log.empty()) {
                snprintf(retrieval_label, sizeof(retrieval_label), "Retrieval");
            } else {
                snprintf(retrieval_label, sizeof(retrieval_label), "Retrieval (%d)",
                         (int)g_state.retrieval_log.size());
            }
            if (ImGui::BeginTabItem(retrieval_label)) {
                draw_retrieval_panel(&g_state);
                ImGui::EndTabItem();
            }
            // Show confidence count badge in tab label
            char confidence_label[64];
            if (g_state.confidence_log.empty()) {
                snprintf(confidence_label, sizeof(confidence_label), "Confidence");
            } else {
                snprintf(confidence_label, sizeof(confidence_label), "Confidence (%d)",
                         (int)g_state.confidence_log.size());
            }
            if (ImGui::BeginTabItem(confidence_label)) {
                draw_confidence_panel(&g_state);
                ImGui::EndTabItem();
            }
            // Show context count badge in tab label
            char context_label[64];
            if (g_state.context_log.empty()) {
                snprintf(context_label, sizeof(context_label), "Context");
            } else {
                snprintf(context_label, sizeof(context_label), "Context (%d)",
                         (int)g_state.context_log.size());
            }
            if (ImGui::BeginTabItem(context_label)) {
                draw_context_panel(&g_state);
                ImGui::EndTabItem();
            }
            // Show tools count badge in tab label
            char tools_label[64];
            if (g_state.tool_ranking_log.empty()) {
                snprintf(tools_label, sizeof(tools_label), "Tools");
            } else {
                snprintf(tools_label, sizeof(tools_label), "Tools (%d)",
                         (int)g_state.tool_ranking_log.size());
            }
            if (ImGui::BeginTabItem(tools_label)) {
                draw_tools_panel(&g_state);
                ImGui::EndTabItem();
            }
            ImGui::EndTabBar();
        }
        ImGui::EndChild();

        ImGui::End();

        // Render
        ImGui::Render();
        int display_w, display_h;
        glfwGetFramebufferSize(window, &display_w, &display_h);
        glViewport(0, 0, display_w, display_h);
        glClearColor(0.1f, 0.1f, 0.1f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

        glfwSwapBuffers(window);
    }

    // Cleanup
    if (g_state.session) lfg_session_free(g_state.session);
    if (g_state.model) lfg_model_free(g_state.model);

    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();

    glfwDestroyWindow(window);
    glfwTerminate();

    return 0;
}
