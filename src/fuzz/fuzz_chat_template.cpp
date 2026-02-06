#include "lfg_inference.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t * data, size_t size) {
    if (!data || size == 0) {
        return 0;
    }

    const uint8_t * p = data;
    size_t remaining = size;

    const size_t tmpl_len = std::min(static_cast<size_t>(p[0]), remaining - 1);
    p += 1;
    remaining -= 1;

    std::string tmpl(reinterpret_cast<const char *>(p), tmpl_len);
    p += tmpl_len;
    remaining -= tmpl_len;

    size_t n_msg = 0;
    if (remaining > 0) {
        n_msg = std::min<size_t>(p[0] % 4, 4);
        p += 1;
        remaining -= 1;
    }

    std::vector<std::string> roles;
    std::vector<std::string> contents;
    std::vector<lfg_chat_message> msgs;
    roles.reserve(n_msg);
    contents.reserve(n_msg);
    msgs.reserve(n_msg);

    for (size_t i = 0; i < n_msg && remaining > 0; ++i) {
        size_t role_len = std::min(static_cast<size_t>(p[0] % 16), remaining - 1);
        p += 1;
        remaining -= 1;

        size_t content_len = 0;
        if (remaining > 0) {
            content_len = std::min(static_cast<size_t>(p[0] % 64), remaining - 1);
            p += 1;
            remaining -= 1;
        }

        role_len = std::min(role_len, remaining);
        std::string role(reinterpret_cast<const char *>(p), role_len);
        p += role_len;
        remaining -= role_len;

        content_len = std::min(content_len, remaining);
        std::string content(reinterpret_cast<const char *>(p), content_len);
        p += content_len;
        remaining -= content_len;

        roles.push_back(std::move(role));
        contents.push_back(std::move(content));
        msgs.push_back({roles.back().c_str(), contents.back().c_str()});
    }

    const bool add_ass = (size % 2) == 1;
    const int32_t out_len = static_cast<int32_t>(std::min<size_t>(remaining > 0 ? p[0] : 0, 255));

    std::vector<char> out_buf;
    char * out_ptr = nullptr;
    if (out_len > 0) {
        out_buf.resize(static_cast<size_t>(out_len));
        out_ptr = out_buf.data();
    }

    (void) lfg_chat_apply_template(
        tmpl.c_str(),
        msgs.empty() ? nullptr : msgs.data(),
        msgs.size(),
        add_ass,
        out_ptr,
        out_len);

    return 0;
}
