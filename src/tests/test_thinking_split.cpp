#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"

#include <string>

static void split_thinking_text(const std::string &text, std::string &thinking_out, std::string &output_out) {
    static const char *THINK_OPEN  = "<think>";
    static const char *THINK_CLOSE = "</think>";
    static const size_t OPEN_LEN  = 7;
    static const size_t CLOSE_LEN = 8;

    thinking_out.clear();
    output_out.clear();

    size_t i = 0;
    while (i < text.size()) {
        size_t open_pos = text.find(THINK_OPEN, i);
        if (open_pos == std::string::npos) {
            output_out.append(text.substr(i));
            break;
        }
        output_out.append(text.substr(i, open_pos - i));
        size_t content_start = open_pos + OPEN_LEN;
        size_t close_pos = text.find(THINK_CLOSE, content_start);
        if (close_pos == std::string::npos) {
            thinking_out.append(text.substr(content_start));
            break;
        }
        thinking_out.append(text.substr(content_start, close_pos - content_start));
        i = close_pos + CLOSE_LEN;
    }
}

TEST_CASE("Split thinking handles multiple blocks without reclassifying output") {
    std::string thinking;
    std::string output;

    split_thinking_text("A<think>t1</think>B", thinking, output);
    CHECK(thinking == "t1");
    CHECK(output == "AB");

    split_thinking_text("A<think>t1</think>B<think>t2</think>C", thinking, output);
    CHECK(thinking.find("t1") != std::string::npos);
    CHECK(thinking.find("t2") != std::string::npos);
    CHECK(output == "ABC");

    split_thinking_text("A<think>t1", thinking, output);
    CHECK(thinking == "t1");
    CHECK(output == "A");

    split_thinking_text("No think", thinking, output);
    CHECK(thinking.empty());
    CHECK(output == "No think");
}
