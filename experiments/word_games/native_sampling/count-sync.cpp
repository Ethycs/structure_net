// Optional identical instrumentation for both benchmark arms; no CUDA changes.
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <dlfcn.h>
static std::atomic<unsigned long long> count{0};
extern "C" void ggml_backend_sched_synchronize(void * sched) {
    using function = void (*)(void *);
    static auto real = [] {
        void * symbol = dlsym(RTLD_NEXT, "ggml_backend_sched_synchronize");
        if (!symbol) {
            // The launcher may load its implementation with a local ELF scope.
            void * handle = dlopen("libggml-base.so.0", RTLD_NOW | RTLD_NOLOAD);
            if (handle) symbol = dlsym(handle, "ggml_backend_sched_synchronize");
        }
        return reinterpret_cast<function>(symbol);
    }();
    if (!real) std::abort();
    count.fetch_add(1, std::memory_order_relaxed);
    real(sched);
}
__attribute__((destructor)) static void report() {
    std::fprintf(stderr, "WORD_GAMES_SCHED_SYNCHRONIZE_CALLS=%llu\n", count.load());
}
