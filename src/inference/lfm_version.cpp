#include "lfm_inference.h"

#define LFM_STR_HELPER(x) #x
#define LFM_STR(x) LFM_STR_HELPER(x)

static const char kLiquidApiVersionString[] =
    LFM_STR(LFM_API_VERSION_MAJOR) "."
    LFM_STR(LFM_API_VERSION_MINOR) "."
    LFM_STR(LFM_API_VERSION_PATCH);

#undef LFM_STR
#undef LFM_STR_HELPER

void lfm_api_version(uint32_t * major, uint32_t * minor, uint32_t * patch) {
    if (major) {
        *major = LFM_API_VERSION_MAJOR;
    }
    if (minor) {
        *minor = LFM_API_VERSION_MINOR;
    }
    if (patch) {
        *patch = LFM_API_VERSION_PATCH;
    }
}

const char * lfm_api_version_string(void) {
    return kLiquidApiVersionString;
}

uint32_t lfm_abi_version(void) {
    return LFM_ABI_VERSION;
}
