#include "lfg_inference.h"
#include "lfg_impl.h"

#include <cstdarg>
#include <cstdio>

namespace {
constexpr size_t kLiquidErrorMsgMax = 512;
thread_local lfg_error g_last_error_code = LFG_ERROR_NONE;
thread_local char g_last_error_msg[kLiquidErrorMsgMax] = {0};
}

void lfg_set_last_error(enum lfg_error code, const char * format, ...) {
    g_last_error_code = code;
    if (!format) {
        g_last_error_msg[0] = '\0';
        return;
    }

    va_list args;
    va_start(args, format);
    vsnprintf(g_last_error_msg, kLiquidErrorMsgMax, format, args);
    va_end(args);
}

enum lfg_error lfg_get_last_error(char * buf, size_t buf_size) {
    if (buf && buf_size > 0) {
        std::snprintf(buf, buf_size, "%s", g_last_error_msg);
        buf[buf_size - 1] = '\0';
    }
    return g_last_error_code;
}

void lfg_clear_last_error(void) {
    g_last_error_code = LFG_ERROR_NONE;
    g_last_error_msg[0] = '\0';
}

const char * lfg_error_string(enum lfg_error code) {
    switch (code) {
        case LFG_ERROR_NONE:
            return "none";
        case LFG_ERROR_INVALID_ARGUMENT:
            return "invalid_argument";
        case LFG_ERROR_IO:
            return "io_error";
        case LFG_ERROR_OUT_OF_MEMORY:
            return "out_of_memory";
        case LFG_ERROR_UNSUPPORTED:
            return "unsupported";
        case LFG_ERROR_CANCELLED:
            return "cancelled";
        case LFG_ERROR_INTERNAL:
            return "internal_error";
    }
    return "unknown";
}

