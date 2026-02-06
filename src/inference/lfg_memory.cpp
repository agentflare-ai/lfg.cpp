#include "lfg_memory.h"

lfg_memory_status lfg_memory_status_combine(lfg_memory_status s0, lfg_memory_status s1) {
    bool has_update = false;

    switch (s0) {
        case LFG_MEMORY_STATUS_SUCCESS:
            {
                has_update = true;
                break;
            }
        case LFG_MEMORY_STATUS_NO_UPDATE:
            {
                break;
            }
        case LFG_MEMORY_STATUS_FAILED_PREPARE:
        case LFG_MEMORY_STATUS_FAILED_COMPUTE:
            {
                return s0;
            }
    }

    switch (s1) {
        case LFG_MEMORY_STATUS_SUCCESS:
            {
                has_update = true;
                break;
            }
        case LFG_MEMORY_STATUS_NO_UPDATE:
            {
                break;
            }
        case LFG_MEMORY_STATUS_FAILED_PREPARE:
        case LFG_MEMORY_STATUS_FAILED_COMPUTE:
            {
                return s1;
            }
    }

    // if either status has an update, then the combined status has an update
    return has_update ? LFG_MEMORY_STATUS_SUCCESS : LFG_MEMORY_STATUS_NO_UPDATE;
}

bool lfg_memory_status_is_fail(lfg_memory_status status) {
    switch (status) {
        case LFG_MEMORY_STATUS_SUCCESS:
        case LFG_MEMORY_STATUS_NO_UPDATE:
            {
                return false;
            }
        case LFG_MEMORY_STATUS_FAILED_PREPARE:
        case LFG_MEMORY_STATUS_FAILED_COMPUTE:
            {
                return true;
            }
    }

    return false;
}
