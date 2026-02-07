#pragma once

#ifdef __cplusplus
#include "lfg_inference.h"
#include "ggml.h" // for ggml_log_level

#include <string>
#include <vector>
#include <spdlog/spdlog.h>

#ifdef __GNUC__
#    if defined(__MINGW32__) && !defined(__clang__)
#        define LFG_ATTRIBUTE_FORMAT(...) __attribute__((format(gnu_printf, __VA_ARGS__)))
#    else
#        define LFG_ATTRIBUTE_FORMAT(...) __attribute__((format(printf, __VA_ARGS__)))
#    endif
#else
#    define LFG_ATTRIBUTE_FORMAT(...)
#endif

//
// logging
//

#ifndef SPDLOG_ACTIVE_LEVEL
#  ifdef NDEBUG
#    define SPDLOG_ACTIVE_LEVEL SPDLOG_LEVEL_ERROR
#  else
#    define SPDLOG_ACTIVE_LEVEL SPDLOG_LEVEL_DEBUG
#  endif
#endif

#include <spdlog/spdlog.h>

#define LFG_LOG(...)       SPDLOG_DEBUG(__VA_ARGS__)
#define LFG_LOG_INFO(...)  SPDLOG_INFO(__VA_ARGS__)
#define LFG_LOG_WARN(...)  SPDLOG_WARN(__VA_ARGS__)
#define LFG_LOG_ERROR(...) SPDLOG_ERROR(__VA_ARGS__)
#define LFG_LOG_DEBUG(...) SPDLOG_DEBUG(__VA_ARGS__)
#define LFG_LOG_CONT(...)  SPDLOG_DEBUG(__VA_ARGS__) // continued logs are usually detailed noise

// Legacy lfg_log_internal for C/Printf compatibility if needed, but spdlog is preferred.
LFG_ATTRIBUTE_FORMAT(2, 3)
void lfg_log_internal        (ggml_log_level level, const char * format, ...);
void lfg_log_callback_default(ggml_log_level level, const char * text, void * user_data);

//
// helpers
//

template <typename T>
struct lfg_no_init {
    T value;
    lfg_no_init() = default;
};

// error handling (thread-local, implemented in lfg_error.cpp)
LFG_ATTRIBUTE_FORMAT(2, 3)
void lfg_set_last_error(enum lfg_error code, const char * format, ...);

inline bool lfg_check_ptr(const void * ptr, enum lfg_error code, const char * fn, const char * name) {
    if (ptr) {
        return true;
    }
    lfg_set_last_error(code, "%s: %s is NULL", fn, name);
    return false;
}

struct lfg_time_meas {
    lfg_time_meas(int64_t & t_acc, bool disable = false);
    ~lfg_time_meas();

    const int64_t t_start_us;

    int64_t & t_acc;
};

void lfg_replace_all(std::string & s, const std::string & search, const std::string & replace);

LFG_ATTRIBUTE_FORMAT(1, 2)
std::string lfg_format(const char * fmt, ...);

std::string lfg_format_tensor_shape(const std::vector<int64_t> & ne);
std::string lfg_format_tensor_shape(const struct ggml_tensor * t);

std::string lfg_gguf_kv_to_str(const struct gguf_context * ctx_gguf, int i);

#define LFG_TENSOR_NAME_FATTN "__fattn__"
#endif // __cplusplus
