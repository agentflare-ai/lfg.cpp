#include "lfm_inference.h"
#include "lfm_impl.h"

#include <cstdarg>
#include <cstdio>

namespace {
constexpr size_t kLiquidErrorMsgMax = 512;
thread_local lfm_error g_last_error_code = LFM_ERROR_NONE;
thread_local char g_last_error_msg[kLiquidErrorMsgMax] = {0};
}

void lfm_set_last_error(enum lfm_error code, const char * format, ...) {
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

enum lfm_error lfm_get_last_error(char * buf, size_t buf_size) {
    if (buf && buf_size > 0) {
        std::snprintf(buf, buf_size, "%s", g_last_error_msg);
        buf[buf_size - 1] = '\0';
    }
    return g_last_error_code;
}

void lfm_clear_last_error(void) {
    g_last_error_code = LFM_ERROR_NONE;
    g_last_error_msg[0] = '\0';
}

const char * lfm_error_string(enum lfm_error code) {
    switch (code) {
        case LFM_ERROR_NONE:
            return "none";
        case LFM_ERROR_INVALID_ARGUMENT:
            return "invalid_argument";
        case LFM_ERROR_IO:
            return "io_error";
        case LFM_ERROR_OUT_OF_MEMORY:
            return "out_of_memory";
        case LFM_ERROR_UNSUPPORTED:
            return "unsupported";
        case LFM_ERROR_CANCELLED:
            return "cancelled";
        case LFM_ERROR_INTERNAL:
            return "internal_error";
    }
    return "unknown";
}

