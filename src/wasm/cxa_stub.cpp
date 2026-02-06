// Minimal C++ exception ABI stubs for WASI (no exception support)
// Exceptions will abort instead of unwinding.
#include <cstdio>
#include <cstdlib>
#include <cstddef>

namespace std {
    class type_info;
}

extern "C" {

void* __cxa_allocate_exception(size_t size) {
    // Allocate space for the exception object
    void* ptr = malloc(size);
    if (!ptr) {
        fprintf(stderr, "WASI: failed to allocate exception\n");
        abort();
    }
    return ptr;
}

void __cxa_free_exception(void* ptr) {
    free(ptr);
}

[[noreturn]] void __cxa_throw(void* thrown_exception, std::type_info* tinfo, void (*dest)(void*)) {
    (void)thrown_exception;
    (void)tinfo;
    (void)dest;
    fprintf(stderr, "WASI: C++ exception thrown (exceptions not supported), aborting\n");
    abort();
}

void* __cxa_begin_catch(void* exception_object) {
    return exception_object;
}

void __cxa_end_catch() {}

[[noreturn]] void __cxa_rethrow() {
    fprintf(stderr, "WASI: C++ exception rethrown, aborting\n");
    abort();
}

// Guard for static initializers (one-time init)
int __cxa_guard_acquire(long long* guard_object) {
    if (*((char*)guard_object) != 0) return 0;
    return 1;
}

void __cxa_guard_release(long long* guard_object) {
    *((char*)guard_object) = 1;
}

void __cxa_guard_abort(long long* guard_object) {
    (void)guard_object;
}

// Personality function for exception handling
int __gxx_personality_v0(...) {
    abort();
}

// Unwind resume
[[noreturn]] void _Unwind_Resume(void* exception_object) {
    (void)exception_object;
    abort();
}

} // extern "C"
