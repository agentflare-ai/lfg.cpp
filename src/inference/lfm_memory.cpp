#include "lfm_memory.h"

lfm_memory_status lfm_memory_status_combine(lfm_memory_status s0, lfm_memory_status s1) {
    bool has_update = false;

    switch (s0) {
        case LFM_MEMORY_STATUS_SUCCESS:
            {
                has_update = true;
                break;
            }
        case LFM_MEMORY_STATUS_NO_UPDATE:
            {
                break;
            }
        case LFM_MEMORY_STATUS_FAILED_PREPARE:
        case LFM_MEMORY_STATUS_FAILED_COMPUTE:
            {
                return s0;
            }
    }

    switch (s1) {
        case LFM_MEMORY_STATUS_SUCCESS:
            {
                has_update = true;
                break;
            }
        case LFM_MEMORY_STATUS_NO_UPDATE:
            {
                break;
            }
        case LFM_MEMORY_STATUS_FAILED_PREPARE:
        case LFM_MEMORY_STATUS_FAILED_COMPUTE:
            {
                return s1;
            }
    }

    // if either status has an update, then the combined status has an update
    return has_update ? LFM_MEMORY_STATUS_SUCCESS : LFM_MEMORY_STATUS_NO_UPDATE;
}

bool lfm_memory_status_is_fail(lfm_memory_status status) {
    switch (status) {
        case LFM_MEMORY_STATUS_SUCCESS:
        case LFM_MEMORY_STATUS_NO_UPDATE:
            {
                return false;
            }
        case LFM_MEMORY_STATUS_FAILED_PREPARE:
        case LFM_MEMORY_STATUS_FAILED_COMPUTE:
            {
                return true;
            }
    }

    return false;
}
