#include "lfg_inference.h"

#define LFG_STR_HELPER(x) #x
#define LFG_STR(x) LFG_STR_HELPER(x)

static const char kLiquidApiVersionString[] =
    LFG_STR(LFG_API_VERSION_MAJOR) "."
    LFG_STR(LFG_API_VERSION_MINOR) "."
    LFG_STR(LFG_API_VERSION_PATCH);

#undef LFG_STR
#undef LFG_STR_HELPER

void lfg_api_version(uint32_t * major, uint32_t * minor, uint32_t * patch) {
    if (major) {
        *major = LFG_API_VERSION_MAJOR;
    }
    if (minor) {
        *minor = LFG_API_VERSION_MINOR;
    }
    if (patch) {
        *patch = LFG_API_VERSION_PATCH;
    }
}

const char * lfg_api_version_string(void) {
    return kLiquidApiVersionString;
}

uint32_t lfg_abi_version(void) {
    return LFG_ABI_VERSION;
}
