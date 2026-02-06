#pragma once

#ifdef __cplusplus
#include "lfm_inference.h"
#include "ggml.h" // for ggml_log_level

#include <string>
#include <vector>
#include <spdlog/spdlog.h>

#ifdef __GNUC__
#    if defined(__MINGW32__) && !defined(__clang__)
#        define LFM_ATTRIBUTE_FORMAT(...) __attribute__((format(gnu_printf, __VA_ARGS__)))
#    else
#        define LFM_ATTRIBUTE_FORMAT(...) __attribute__((format(printf, __VA_ARGS__)))
#    endif
#else
#    define LFM_ATTRIBUTE_FORMAT(...)
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

#define LFM_LOG(...)       SPDLOG_DEBUG(__VA_ARGS__)
#define LFM_LOG_INFO(...)  SPDLOG_INFO(__VA_ARGS__)
#define LFM_LOG_WARN(...)  SPDLOG_WARN(__VA_ARGS__)
#define LFM_LOG_ERROR(...) SPDLOG_ERROR(__VA_ARGS__)
#define LFM_LOG_DEBUG(...) SPDLOG_DEBUG(__VA_ARGS__)
#define LFM_LOG_CONT(...)  SPDLOG_DEBUG(__VA_ARGS__) // continued logs are usually detailed noise

// Legacy lfm_log_internal for C/Printf compatibility if needed, but spdlog is preferred.
LFM_ATTRIBUTE_FORMAT(2, 3)
void lfm_log_internal        (ggml_log_level level, const char * format, ...);
void lfm_log_callback_default(ggml_log_level level, const char * text, void * user_data);

//
// helpers
//

template <typename T>
struct no_init {
    T value;
    no_init() = default;
};

// error handling (thread-local, implemented in lfm_error.cpp)
LFM_ATTRIBUTE_FORMAT(2, 3)
void lfm_set_last_error(enum lfm_error code, const char * format, ...);

inline bool lfm_check_ptr(const void * ptr, enum lfm_error code, const char * fn, const char * name) {
    if (ptr) {
        return true;
    }
    lfm_set_last_error(code, "%s: %s is NULL", fn, name);
    return false;
}

struct time_meas {
    time_meas(int64_t & t_acc, bool disable = false);
    ~time_meas();

    const int64_t t_start_us;

    int64_t & t_acc;
};

void replace_all(std::string & s, const std::string & search, const std::string & replace);

// TODO: rename to lfm_format ?
LFM_ATTRIBUTE_FORMAT(1, 2)
std::string format(const char * fmt, ...);

std::string lfm_format_tensor_shape(const std::vector<int64_t> & ne);
std::string lfm_format_tensor_shape(const struct ggml_tensor * t);

std::string gguf_kv_to_str(const struct gguf_context * ctx_gguf, int i);

#define LFM_TENSOR_NAME_FATTN "__fattn__"
#endif // __cplusplus
