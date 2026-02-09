#pragma once

#ifdef __cplusplus
extern "C" {
#endif

// Returns malloc'd path (caller frees), or NULL if cancelled.
const char *open_file_dialog(const char *title, const char *default_dir);

#ifdef __cplusplus
}
#endif
