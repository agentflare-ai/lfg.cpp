// Stub spdlog header for WASI builds (no threading support)
#pragma once

#include <cstdio>

#define SPDLOG_TRACE(...)    ((void)0)
#define SPDLOG_DEBUG(...)    ((void)0)
#define SPDLOG_INFO(...)     ((void)0)
#define SPDLOG_WARN(...)     ((void)0)
#define SPDLOG_ERROR(...)    ((void)0)
#define SPDLOG_CRITICAL(...) ((void)0)

#define SPDLOG_LOGGER_TRACE(logger, ...)    ((void)0)
#define SPDLOG_LOGGER_DEBUG(logger, ...)    ((void)0)
#define SPDLOG_LOGGER_INFO(logger, ...)     ((void)0)
#define SPDLOG_LOGGER_WARN(logger, ...)     ((void)0)
#define SPDLOG_LOGGER_ERROR(logger, ...)    ((void)0)
#define SPDLOG_LOGGER_CRITICAL(logger, ...) ((void)0)

namespace spdlog {
    template<typename... Args> inline void trace(Args&&...) {}
    template<typename... Args> inline void debug(Args&&...) {}
    template<typename... Args> inline void info(Args&&...) {}
    template<typename... Args> inline void warn(Args&&...) {}
    template<typename... Args> inline void error(Args&&...) {}
    template<typename... Args> inline void critical(Args&&...) {}
}
