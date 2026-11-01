#include "ggml.h"
#include "ggml-backend.h"
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>

template<class T> static T wg_symbol(const char * name) {
    void * symbol = dlsym(RTLD_NEXT, name);
    if (!symbol) {
        void * handle = dlopen("libggml-base.so.0", RTLD_NOW | RTLD_NOLOAD);
        if (handle) symbol = dlsym(handle, name);
    }
    if (!symbol) { std::fprintf(stderr, "missing symbol: %s\n", name); std::abort(); }
    return reinterpret_cast<T>(symbol);
}

static std::atomic<unsigned long long> wg_sync{0}, wg_raw_bytes{0}, wg_raw_calls{0};
static std::atomic<unsigned long long> wg_preview_bytes{0}, wg_other_bytes{0}, wg_cpu_bytes{0};

extern "C" void ggml_backend_sched_synchronize(ggml_backend_sched_t sched) {
    static auto real = wg_symbol<decltype(&ggml_backend_sched_synchronize)>("ggml_backend_sched_synchronize");
    wg_sync.fetch_add(1, std::memory_order_relaxed);
    real(sched);
}

extern "C" void ggml_backend_tensor_get_async(ggml_backend_t backend, const ggml_tensor * tensor,
                                               void * data, size_t offset, size_t size) {
    static auto real = wg_symbol<decltype(&ggml_backend_tensor_get_async)>("ggml_backend_tensor_get_async");
    static auto name_of = wg_symbol<decltype(&ggml_get_name)>("ggml_get_name");
    static auto device_of = wg_symbol<decltype(&ggml_backend_get_device)>("ggml_backend_get_device");
    static auto type_of = wg_symbol<decltype(&ggml_backend_dev_type)>("ggml_backend_dev_type");
    const char * name = name_of(tensor);
    if (type_of(device_of(backend)) != GGML_BACKEND_DEVICE_TYPE_GPU) {
        wg_cpu_bytes.fetch_add(size, std::memory_order_relaxed);
    } else if (std::strcmp(name, "result_output") == 0) {
        wg_raw_bytes.fetch_add(size, std::memory_order_relaxed);
        wg_raw_calls.fetch_add(1, std::memory_order_relaxed);
    } else if (std::strcmp(name, "top_k_rows") == 0 || std::strcmp(name, "top_k") == 0 ||
               std::strcmp(name, "greedy_preview_sum") == 0) {
        wg_preview_bytes.fetch_add(size, std::memory_order_relaxed);
    } else {
        wg_other_bytes.fetch_add(size, std::memory_order_relaxed);
    }
    real(backend,tensor,data,offset,size);
}

static void wg_reset() {
    wg_sync=0; wg_raw_bytes=0; wg_raw_calls=0; wg_preview_bytes=0; wg_other_bytes=0; wg_cpu_bytes=0;
}

__attribute__((destructor)) static void wg_report() {
    std::fprintf(stderr, "WORD_GAMES_SCHED_SYNCHRONIZE_CALLS=%llu\n", wg_sync.load());
    std::fprintf(stderr, "WORD_GAMES_OUTPUT_DOWNLOADS={\"raw_calls\":%llu,\"raw_bytes\":%llu,"
        "\"preview_bytes\":%llu,\"other_gpu_bytes\":%llu,\"cpu_bytes\":%llu}\n",
        wg_raw_calls.load(),wg_raw_bytes.load(),wg_preview_bytes.load(),wg_other_bytes.load(),wg_cpu_bytes.load());
}
