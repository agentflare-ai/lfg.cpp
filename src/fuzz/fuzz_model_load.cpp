#include "lfg_inference.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <mutex>
#include <string>

#if !defined(_WIN32)
#include <unistd.h>
#endif

namespace {
constexpr size_t kMaxInputSize = 1 << 20;

struct TempFile {
    std::string path;
    FILE * file = nullptr;

    TempFile() {
#if defined(_WIN32)
        char buf[L_tmpnam];
        if (tmpnam_s(buf, sizeof(buf)) == 0) {
            file = fopen(buf, "wb");
            if (file) {
                path = buf;
            }
        }
#else
        char tmpl[] = "/tmp/lfg_fuzz_XXXXXX";
        int fd = mkstemp(tmpl);
        if (fd >= 0) {
            file = fdopen(fd, "wb");
            path = tmpl;
        }
#endif
    }

    ~TempFile() {
        if (file) {
            fclose(file);
            file = nullptr;
        }
        if (!path.empty()) {
#if defined(_WIN32)
            _unlink(path.c_str());
#else
            unlink(path.c_str());
#endif
        }
    }

    bool valid() const {
        return file != nullptr && !path.empty();
    }
};
}

extern "C" int LLVMFuzzerTestOneInput(const uint8_t * data, size_t size) {
    static std::once_flag init_once;
    std::call_once(init_once, []() {
        lfg_backend_init();
    });

    if (!data || size == 0) {
        return 0;
    }

    size = std::min(size, kMaxInputSize);

    TempFile temp;
    if (!temp.valid()) {
        return 0;
    }

    const size_t written = fwrite(data, 1, size, temp.file);
    if (written != size) {
        return 0;
    }

    lfg_model_params params = lfg_model_default_params();
    params.vocab_only = true;

    try {
        lfg_model * model = lfg_model_load_from_file(temp.path.c_str(), params);
        if (model) {
            lfg_model_free(model);
        }
    } catch (...) {
        // swallow exceptions to keep fuzzing; C API should not throw
    }

    return 0;
}
