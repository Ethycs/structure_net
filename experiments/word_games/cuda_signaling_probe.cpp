// Exploratory systems check, not model throughput or quality evidence.
// Run from the repo root through Pixi; keep binaries in .pixi and JSON in data.
// Fixed design: 8 KiB payload, 20 warmups, 5 x 100 handoffs, all directed pairs
// and 0->1->2->0. Active services remain resident. Each iteration waits only
// at the final destination. Verify every final payload byte after each trial.
// "host_sync" is a deliberately synchronized control, NOT the current runtime.
// The current GGML GPU copy path is closest to "peer_copy_event".
// Stream memops retain CUDA-visible event dependencies, as the API requires;
// this establishes compatibility, not an event-free GPU scheduling protocol.
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

static void rt(cudaError_t e, const char * what) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(e));
        std::exit(2);
    }
}
static void dr(CUresult e, const char * what) {
    if (e != CUDA_SUCCESS) {
        const char * s = nullptr; cuGetErrorString(e, &s);
        std::fprintf(stderr, "%s: %s\n", what, s ? s : "unknown");
        std::exit(3);
    }
}
#define RT(x) rt((x), #x)
#define DR(x) dr((x), #x)
constexpr size_t bytes = 8192;
constexpr int warmups = 20, trials = 5, iterations = 100;
struct Device {
    void * buf = nullptr;
    cudaStream_t stream = nullptr;
    cudaEvent_t done = nullptr;
    CUdeviceptr flags = 0;
};

int main() {
    int count = 0; RT(cudaGetDeviceCount(&count));
    if (count < 2) return 4;
    std::vector<Device> dev(count);
    void * flags = nullptr;
    RT(cudaSetDevice(0));
    RT(cudaHostAlloc(&flags, 4096, cudaHostAllocPortable | cudaHostAllocMapped));
    std::memset(flags, 0, 4096);
    std::printf("{\"schema_version\":1,\"classification\":\"systems_lifecycle_only_not_quality_evidence\","
                "\"payload_bytes\":%zu,\"warmups\":%d,\"trials\":%d,\"iterations\":%d,\"devices\":[",
                bytes, warmups, trials, iterations);
    for (int i = 0; i < count; ++i) {
        RT(cudaSetDevice(i));
        cudaDeviceProp p; RT(cudaGetDeviceProperties(&p, i));
        size_t free = 0, total = 0; RT(cudaMemGetInfo(&free, &total));
        if (free < 256ULL * 1024 * 1024) {
            std::fprintf(stderr, "insufficient spare memory on device %d\n", i); return 5;
        }
        RT(cudaMalloc(&dev[i].buf, bytes));
        RT(cudaStreamCreateWithFlags(&dev[i].stream, cudaStreamNonBlocking));
        RT(cudaEventCreateWithFlags(&dev[i].done, cudaEventDisableTiming));
        void * ptr = nullptr; RT(cudaHostGetDevicePointer(&ptr, flags, 0));
        dev[i].flags = reinterpret_cast<CUdeviceptr>(ptr);
        int mem64 = 0, ipc = 0, flush = 0;
        DR(cuDeviceGetAttribute(&mem64, CU_DEVICE_ATTRIBUTE_CAN_USE_64_BIT_STREAM_MEM_OPS, i));
        DR(cuDeviceGetAttribute(&ipc, CU_DEVICE_ATTRIBUTE_IPC_EVENT_SUPPORTED, i));
        DR(cuDeviceGetAttribute(&flush, CU_DEVICE_ATTRIBUTE_CAN_FLUSH_REMOTE_WRITES, i));
        std::printf("%s{\"ordinal\":%d,\"name\":\"%s\",\"can_map_host_memory\":%d,"
                    "\"host_native_atomics\":%d,\"stream_memops_64\":%d,\"ipc_event\":%d,"
                    "\"flush_remote_writes\":%d,\"peer_access\":[",
                    i ? "," : "", i, p.name, p.canMapHostMemory,
                    p.hostNativeAtomicSupported, mem64, ipc, flush);
        for (int j = 0; j < count; ++j) {
            int peer = 0; if (j != i) RT(cudaDeviceCanAccessPeer(&peer, i, j));
            std::printf("%s%d", j ? "," : "", peer);
        }
        std::printf("]}");
    }
    std::vector<std::vector<int>> routes;
    for (int i = 0; i < count; ++i) for (int j = 0; j < count; ++j)
        if (i != j) routes.push_back({i, j});
    if (count >= 3) routes.push_back({0, 1, 2, 0});
    const char * modes[] = {"host_sync_staged", "cross_gpu_event_staged", "peer_copy_event", "event_guarded_memops_staged"};
    void * stage[3] = {}; void * result = nullptr;
    for (auto & p : stage) RT(cudaHostAlloc(&p, bytes, cudaHostAllocPortable));
    RT(cudaHostAlloc(&result, bytes, cudaHostAllocPortable));
    std::vector<unsigned char> expected(bytes);
    unsigned int sequence = 0;
    bool first = true;
    std::printf("],\"checks\":[");
    for (size_t ri = 0; ri < routes.size(); ++ri) {
        const auto & route = routes[ri];
        // Rotate method order to reduce systematic ordering bias.
        for (int mi = 0; mi < 4; ++mi) {
            const int mode = (mi + ri) % 4;
            std::vector<double> times;
            for (int trial = -1; trial < trials; ++trial) {
                for (size_t k = 0; k < bytes; ++k)
                    expected[k] = static_cast<unsigned char>(k * 17 + trial + mode * 31 + ri);
                for (int i = 0; i < count; ++i) {
                    RT(cudaSetDevice(i));
                    RT(cudaMemsetAsync(dev[i].buf, 0, bytes, dev[i].stream));
                    RT(cudaStreamSynchronize(dev[i].stream));
                }
                RT(cudaSetDevice(route.front()));
                RT(cudaMemcpy(dev[route.front()].buf, expected.data(), bytes, cudaMemcpyHostToDevice));
                const int repeats = trial < 0 ? warmups : iterations;
                const auto t0 = std::chrono::steady_clock::now();
                for (int it = 0; it < repeats; ++it) {
                    ++sequence;
                    for (size_t hop = 0; hop + 1 < route.size(); ++hop) {
                        const int a = route[hop], b = route[hop + 1];
                        RT(cudaSetDevice(a));
                        if (mode == 2) {
                            RT(cudaMemcpyPeerAsync(dev[b].buf, b, dev[a].buf, a, bytes, dev[a].stream));
                        } else {
                            RT(cudaMemcpyAsync(stage[hop], dev[a].buf, bytes, cudaMemcpyDeviceToHost, dev[a].stream));
                        }
                        if (mode == 0) {
                            RT(cudaStreamSynchronize(dev[a].stream));
                        } else {
                            if (mode == 3) DR(cuStreamWriteValue32(dev[a].stream, dev[a].flags + hop * 64, sequence, 0));
                            RT(cudaEventRecord(dev[a].done, dev[a].stream));
                        }
                        RT(cudaSetDevice(b));
                        if (mode != 0) RT(cudaStreamWaitEvent(dev[b].stream, dev[a].done, 0));
                        if (mode == 3) DR(cuStreamWaitValue32(dev[b].stream, dev[b].flags + hop * 64, sequence, CU_STREAM_WAIT_VALUE_EQ));
                        if (mode != 2) RT(cudaMemcpyAsync(dev[b].buf, stage[hop], bytes, cudaMemcpyHostToDevice, dev[b].stream));
                    }
                    RT(cudaStreamSynchronize(dev[route.back()].stream));
                }
                const double elapsed_us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count();
                for (int visited : route) {
                    RT(cudaSetDevice(visited));
                    RT(cudaMemcpy(result, dev[visited].buf, bytes, cudaMemcpyDeviceToHost));
                    if (std::memcmp(result, expected.data(), bytes) != 0) {
                        std::fprintf(stderr, "payload mismatch route=%zu mode=%d trial=%d device=%d\n", ri, mode, trial, visited); return 6;
                    }
                }
                if (trial >= 0) times.push_back(elapsed_us / repeats);
            }
            auto ordered = times; std::sort(ordered.begin(), ordered.end());
            std::printf("%s{\"route\":[", first ? "" : ","); first = false;
            for (size_t j = 0; j < route.size(); ++j) std::printf("%s%d", j ? "," : "", route[j]);
            std::printf("],\"mode\":\"%s\",\"payload_equal\":true,\"trial_us_per_route\":[", modes[mode]);
            for (size_t j = 0; j < times.size(); ++j) std::printf("%s%.6f", j ? "," : "", times[j]);
            std::printf("],\"median_us_per_route\":%.6f}", ordered[ordered.size()/2]);
            std::fflush(stdout);
        }
    }
    std::printf("]}\n");
    for (int i = 0; i < count; ++i) {
        RT(cudaSetDevice(i)); RT(cudaEventDestroy(dev[i].done));
        RT(cudaStreamDestroy(dev[i].stream)); RT(cudaFree(dev[i].buf));
    }
    for (auto p : stage) RT(cudaFreeHost(p));
    RT(cudaFreeHost(result)); RT(cudaFreeHost(flags));
    return 0;
}
