// eval_tool_similarity: Dump cosine similarity matrix between tool descriptions
// and queries to diagnose whether the model's embedding space carries signal.

#include "lfg_api.h"
#include <cstdio>
#include <cstring>
#include <cmath>
#include <cstdlib>
#include <vector>
#include <string>

static float dot(const float *a, const float *b, int n) {
    float sum = 0.0f;
    for (int i = 0; i < n; i++) sum += a[i] * b[i];
    return sum;
}

int main() {
    lfg_backend_init();

    lfg_model_load_config cfg = lfg_model_load_default_config();
    cfg.model_path = "models/lfm2-350M.gguf";
    cfg.n_gpu_layers = 0;

    auto *model = lfg_load_model(&cfg);
    if (!model) {
        printf("FAIL: model not found (tried %s)\n", cfg.model_path);
        // Try 1.2B fallback
        cfg.model_path = "models/LFM2.5-1.2B-Thinking-GGUF/LFM2.5-1.2B-Thinking-Q4_K_M.gguf";
        model = lfg_load_model(&cfg);
        if (!model) { printf("FAIL: no model found\n"); return 1; }
    }
    printf("Model: %s\n\n", cfg.model_path);

    lfg_session_config sc = lfg_session_default_config();
    sc.n_ctx = 2048;
    auto *session = lfg_session_create(model, &sc);
    if (!session) { printf("FAIL: session create\n"); return 1; }

    int32_t n_embd = lfg_model_n_embd(model);
    printf("n_embd = %d\n\n", n_embd);

    // Tool descriptions (same as demo EXAMPLE_TOOLS)
    const char *tool_texts[] = {
        "get_weather: Get the current weather forecast for a given location",
        "search_web: Search the internet for information on a topic",
        "send_email: Send an email message to a recipient",
        "calculator: Evaluate an arithmetic or mathematical expression",
        "set_reminder: Set a timed reminder with a message",
        "create_calendar_event: Create a calendar event with a title, time, and optional attendees",
        "translate_text: Translate text from one language to another",
        "get_stock_price: Look up the current stock price for a ticker symbol",
        "play_music: Play a song or playlist by name or artist",
        "set_timer: Set a countdown timer for a specified duration",
        "take_screenshot: Capture a screenshot of the current screen",
        "read_file: Read the contents of a file from the filesystem",
        "list_tools: List all the tools and capabilities you can do",
    };
    const int N_TOOLS = sizeof(tool_texts) / sizeof(tool_texts[0]);

    // Queries — mix of tool-relevant and non-tool
    const char *queries[] = {
        "What's the weather like in Paris?",
        "Search for news about AI",
        "Send an email to bob about the meeting",
        "What is 42 * 17?",
        "1 + 1",
        "what is 1 + 1",
        "calculate 1 + 1",
        "What tools do you have?",
        "What can you do?",
        "List your capabilities",
        "Hello, how are you?",
        "Tell me a joke",
        "Remind me to buy groceries in 30 minutes",
        "What's the stock price of AAPL?",
        "Play some jazz music",
        "Translate hello to French",
    };
    const int N_QUERIES = sizeof(queries) / sizeof(queries[0]);

    // Compute tool embeddings
    std::vector<std::vector<float>> tool_embds(N_TOOLS);
    for (int i = 0; i < N_TOOLS; i++) {
        tool_embds[i].resize(n_embd);
        int32_t r = lfg_session_embed(session, tool_texts[i], (int32_t)strlen(tool_texts[i]),
                                       tool_embds[i].data(), n_embd);
        if (r <= 0) {
            printf("FAIL: embed tool %d (%s)\n", i, tool_texts[i]);
            return 1;
        }
    }
    printf("Computed %d tool embeddings\n", N_TOOLS);

    // Compute query embeddings
    std::vector<std::vector<float>> query_embds(N_QUERIES);
    for (int i = 0; i < N_QUERIES; i++) {
        query_embds[i].resize(n_embd);
        int32_t r = lfg_session_embed(session, queries[i], (int32_t)strlen(queries[i]),
                                       query_embds[i].data(), n_embd);
        if (r <= 0) {
            printf("FAIL: embed query %d (%s)\n", i, queries[i]);
            return 1;
        }
    }
    printf("Computed %d query embeddings\n\n", N_QUERIES);

    // Print similarity matrix
    printf("%-45s", "");
    for (int t = 0; t < N_TOOLS; t++) {
        // Print short tool name
        const char *colon = strchr(tool_texts[t], ':');
        int name_len = colon ? (int)(colon - tool_texts[t]) : 12;
        printf(" %.*s", name_len > 12 ? 12 : name_len, tool_texts[t]);
        // Pad to 13 chars
        for (int p = (name_len > 12 ? 12 : name_len); p < 13; p++) printf(" ");
    }
    printf("\n");

    for (int q = 0; q < N_QUERIES; q++) {
        printf("%-45.45s", queries[q]);
        float best_score = -1.0f;
        int best_idx = -1;
        float sum = 0.0f;
        for (int t = 0; t < N_TOOLS; t++) {
            float sim = dot(query_embds[q].data(), tool_embds[t].data(), n_embd);
            sum += sim;
            if (sim > best_score) { best_score = sim; best_idx = t; }
            // Color: bold if highest for this query
            printf(" %+.4f      ", sim);
        }
        float mean = sum / N_TOOLS;
        float gap = best_score - mean;

        // Print best match + gap
        const char *colon = strchr(tool_texts[best_idx], ':');
        int name_len = colon ? (int)(colon - tool_texts[best_idx]) : 12;
        printf("  BEST: %.*s (%.4f, gap=%.4f)\n", name_len, tool_texts[best_idx], best_score, gap);
    }

    // Also print tool-to-tool similarity to check if tools are distinguishable
    printf("\n\nTool-to-tool similarity:\n");
    printf("%-30s", "");
    for (int t = 0; t < N_TOOLS; t++) {
        const char *colon = strchr(tool_texts[t], ':');
        int name_len = colon ? (int)(colon - tool_texts[t]) : 8;
        printf(" %.*s", name_len > 8 ? 8 : name_len, tool_texts[t]);
        for (int p = (name_len > 8 ? 8 : name_len); p < 9; p++) printf(" ");
    }
    printf("\n");

    for (int i = 0; i < N_TOOLS; i++) {
        const char *colon = strchr(tool_texts[i], ':');
        int name_len = colon ? (int)(colon - tool_texts[i]) : 25;
        printf("%-30.*s", name_len > 30 ? 30 : name_len, tool_texts[i]);
        for (int j = 0; j < N_TOOLS; j++) {
            float sim = dot(tool_embds[i].data(), tool_embds[j].data(), n_embd);
            printf(" %+.4f  ", sim);
        }
        printf("\n");
    }

    lfg_session_free(session);
    lfg_model_free(model);
    return 0;
}
