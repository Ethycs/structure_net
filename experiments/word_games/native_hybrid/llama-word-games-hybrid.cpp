// WORD_GAMES exploratory within-layer expert partition for llama.cpp b11960.
// Mechanism inspired by bdrazn/blobpager E7b (compact pools + CPU correction).
// This implementation reads final router weights, caches miss graphs, uses a
// persistent CPU worker/threadpool, and joins after GPU hit computation.
// Single model/context, single-token decode only; fail closed on any mismatch.
#include "llama-word-games-hybrid.h"
#include "llama-model.h"
#include "llama-impl.h"
#include "ggml-cpu.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

namespace {
using clock_type = std::chrono::steady_clock;
double us() { return std::chrono::duration<double, std::micro>(clock_type::now().time_since_epoch()).count(); }
struct cpu_graph {
    ggml_context * ctx = nullptr;
    ggml_backend_buffer_t buf = nullptr;
    ggml_cgraph * graph = nullptr;
    ggml_tensor * x = nullptr, * ids = nullptr, * weights = nullptr, * out = nullptr;
};
struct layer_state {
    ggml_tensor * cpu[3] = {}; // up, gate, down
    ggml_tensor * pool[3] = {};
    ggml_tensor * slots = nullptr, * correction = nullptr;
    ggml_tensor * graph_input = nullptr, * graph_ids = nullptr;
    std::vector<int32_t> map;
    std::unique_ptr<cpu_graph> graphs[9];
    std::vector<float> x, weights, result;
    std::vector<int32_t> ids;
    int32_t all_ids[8] = {};
    float all_weights[8] = {};
    bool validated = false;
    int misses = 0;
};
struct state {
    bool active = false, pending = false, done = false, stop = false;
    int pins = 0, n_layers = 26, embd = 0, n_expert = 0, job = -1;
    ggml_backend_t cpu = nullptr;
    ggml_backend_dev_t main_dev = nullptr, pool_dev = nullptr;
    ggml_context * ctx = nullptr;
    ggml_backend_buffer_t pool_buf = nullptr, corr_buf = nullptr;
    layer_state layers[26];
    std::mutex mutex;
    std::condition_variable cv;
    std::thread worker;
    ggml_backend_sched_eval_callback user = nullptr;
    void * user_data = nullptr;
    uint64_t calls = 0, hits = 0, misses = 0;
    double cpu_us = 0, join_us = 0, launch_us = 0, window_us = 0, launched_at = 0;
    int validated_layers = 0;
    double validation_max_relative_l2 = 0;
};
// Process lifetime state: exit hook joins before backend unloading. The server
// only permits one model/context in this exploratory executable.
state & s() { static auto * value = new state; return *value; }

void save_stats() {
    auto & g = s();
    const char * path = getenv("WORD_GAMES_HYBRID_STATS");
    if (!path) return;
    std::ofstream f(path);
    GGML_ASSERT(f.good());
    f << "{\"active\":true,\"pins\":" << g.pins << ",\"layers\":" << g.n_layers
      << ",\"calls\":" << g.calls << ",\"gpu_hits\":" << g.hits
      << ",\"cpu_misses\":" << g.misses << ",\"cpu_us\":" << g.cpu_us
      << ",\"join_wait_us\":" << g.join_us << ",\"launch_us\":" << g.launch_us
      << ",\"gpu_window_us\":" << g.window_us
      << ",\"validated_layers\":" << g.validated_layers
      << ",\"validation_max_relative_l2\":" << g.validation_max_relative_l2 << "}\n";
}
void finish() {
    auto & g = s();
    if (!g.active) return;
    { std::unique_lock<std::mutex> lock(g.mutex); g.stop = true; g.cv.notify_all(); }
    if (g.worker.joinable()) g.worker.join();
    save_stats();
}

cpu_graph & get_graph(layer_state & l, int n) {
    auto & g = s();
    GGML_ASSERT(n > 0 && n <= 8);
    if (l.graphs[n]) return *l.graphs[n];
    auto c = std::make_unique<cpu_graph>();
    c->ctx = ggml_init({1024*1024, nullptr, true});
    GGML_ASSERT(c->ctx);
    c->x = ggml_new_tensor_3d(c->ctx, GGML_TYPE_F32, g.embd, 1, 1);
    c->ids = ggml_new_tensor_2d(c->ctx, GGML_TYPE_I32, n, 1);
    c->weights = ggml_new_tensor_3d(c->ctx, GGML_TYPE_F32, 1, n, 1);
    auto up = ggml_mul_mat_id(c->ctx, l.cpu[0], c->x, c->ids);
    auto gate = ggml_mul_mat_id(c->ctx, l.cpu[1], c->x, c->ids);
    auto hid = ggml_swiglu_split(c->ctx, gate, up);
    auto down = ggml_mul_mat_id(c->ctx, l.cpu[2], hid, c->ids);
    auto weighted = ggml_mul(c->ctx, down, c->weights);
    auto out = ggml_view_2d(c->ctx, weighted, g.embd, 1, weighted->nb[2], 0);
    for (int i = 1; i < n; ++i) {
        auto v = ggml_view_2d(c->ctx, weighted, g.embd, 1, weighted->nb[2], i*weighted->nb[1]);
        out = ggml_add(c->ctx, out, v);
    }
    c->out = out;
    c->graph = ggml_new_graph(c->ctx);
    ggml_build_forward_expand(c->graph, out);
    c->buf = ggml_backend_alloc_ctx_tensors(c->ctx, g.cpu);
    GGML_ASSERT(c->buf);
    l.graphs[n] = std::move(c);
    return *l.graphs[n];
}
void cpu_compute(int il) {
    auto & g = s(); auto & l = g.layers[il];
    double start = us();
    auto & c = get_graph(l, l.misses);
    ggml_backend_tensor_set(c.x, l.x.data(), 0, g.embd*sizeof(float));
    ggml_backend_tensor_set(c.ids, l.ids.data(), 0, l.misses*sizeof(int32_t));
    ggml_backend_tensor_set(c.weights, l.weights.data(), 0, l.misses*sizeof(float));
    GGML_ASSERT(ggml_backend_graph_compute(g.cpu, c.graph) == GGML_STATUS_SUCCESS);
    ggml_backend_tensor_get(c.out, l.result.data(), 0, g.embd*sizeof(float));
    for (float v : l.result) GGML_ASSERT(std::isfinite(v));
    g.cpu_us += us()-start;
}
ggml_backend_t backend(ggml_backend_sched_t sched, ggml_backend_dev_t dev) {
    for (int i = 0; i < ggml_backend_sched_get_n_backends(sched); ++i) {
        auto b = ggml_backend_sched_get_backend(sched, i);
        if (ggml_backend_get_device(b) == dev) return b;
    }
    GGML_ABORT("hybrid backend missing");
}
} // namespace

void word_games_hybrid_setup(const llama_model & model, int threads) {
    const char * pins = getenv("WORD_GAMES_HYBRID_PINS");
    if (!pins) return;
    auto & g = s();
    GGML_ASSERT(!g.active && model.arch == LLM_ARCH_QWEN35MOE);
    g.pins = std::stoi(pins);
    GGML_ASSERT(g.pins >= 0 && g.pins <= 80);
    g.embd = model.hparams.n_embd; g.n_expert = model.hparams.n_expert;
    GGML_ASSERT(g.n_expert == 256 && model.hparams.n_expert_used(0) == 8);
    GGML_ASSERT(model.devices.size() == 2);
    g.main_dev = model.devices[0].dev; g.pool_dev = model.devices[1].dev;
    GGML_ASSERT(std::string(ggml_backend_dev_name(g.main_dev)) == "CUDA0");
    GGML_ASSERT(std::string(ggml_backend_dev_name(g.pool_dev)) == "CUDA1");
    g.ctx = ggml_init({1024*1024, nullptr, true});
    auto corr_ctx = ggml_init({128*1024, nullptr, true});
    GGML_ASSERT(g.ctx && corr_ctx);
    size_t copied = 0;
    for (int il = 0; il < g.n_layers; ++il) {
        auto & l = g.layers[il]; auto & ml = model.layers[il];
        l.cpu[0] = ml.ffn_up_exps; l.cpu[1] = ml.ffn_gate_exps; l.cpu[2] = ml.ffn_down_exps;
        l.map.assign(g.n_expert, g.pins);
        // Deterministic evenly spaced IDs; no task-answer-derived calibration.
        for (int p = 0; p < g.pins; ++p) l.map[p*g.n_expert/g.pins] = p;
        for (int k = 0; k < 3; ++k) {
            auto w = l.cpu[k];
            GGML_ASSERT(w && w->data && ggml_backend_buffer_is_host(w->buffer));
            GGML_ASSERT(w->ne[2] == g.n_expert && ggml_is_contiguous(w));
            l.pool[k] = ggml_new_tensor_3d(g.ctx, w->type, w->ne[0], w->ne[1], g.pins+1);
            ggml_format_name(l.pool[k], "hybrid_pool_%d_%d", il, k);
            GGML_ASSERT(l.pool[k]->nb[2] == w->nb[2]);
        }
        l.slots = ggml_new_tensor_1d(g.ctx, GGML_TYPE_I32, g.n_expert);
        l.correction = ggml_new_tensor_2d(corr_ctx, GGML_TYPE_F32, g.embd, 1);
        ggml_format_name(l.correction, "hybrid_correction-%d", il);
        l.x.resize(g.embd); l.result.resize(g.embd); l.weights.resize(8); l.ids.resize(8);
    }
    g.pool_buf = ggml_backend_alloc_ctx_tensors_from_buft(g.ctx, ggml_backend_dev_buffer_type(g.pool_dev));
    g.corr_buf = ggml_backend_alloc_ctx_tensors_from_buft(corr_ctx, ggml_backend_dev_buffer_type(g.main_dev));
    GGML_ASSERT(g.pool_buf && g.corr_buf);
    ggml_backend_buffer_set_usage(g.pool_buf, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    ggml_backend_buffer_clear(g.pool_buf, 0); ggml_backend_buffer_clear(g.corr_buf, 0);
    for (auto & l : g.layers) {
        for (int e = 0; e < g.n_expert; ++e) if (l.map[e] < g.pins) {
            for (int k = 0; k < 3; ++k) {
                const auto w = l.cpu[k]; const size_t n = w->nb[2];
                const auto src = static_cast<const char *>(w->data) + e*n;
                ggml_backend_tensor_set(l.pool[k], src, l.map[e]*n, n);
                std::vector<char> back(n);
                ggml_backend_tensor_get(l.pool[k], back.data(), l.map[e]*n, n);
                GGML_ASSERT(memcmp(src, back.data(), n) == 0);
                copied += n;
            }
        }
        ggml_backend_tensor_set(l.slots, l.map.data(), 0, g.n_expert*sizeof(int32_t));
    }
    g.cpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    GGML_ASSERT(g.cpu);
    auto reg = ggml_backend_dev_backend_reg(ggml_backend_get_device(g.cpu));
    auto set_threads = (ggml_backend_set_n_threads_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_set_n_threads");
    auto new_pool = (decltype(ggml_threadpool_new) *) ggml_backend_reg_get_proc_address(reg, "ggml_threadpool_new");
    auto set_pool = (decltype(ggml_backend_cpu_set_threadpool) *) ggml_backend_reg_get_proc_address(reg, "ggml_backend_cpu_set_threadpool");
    GGML_ASSERT(set_threads && new_pool && set_pool);
    auto params = ggml_threadpool_params_default(threads);
    params.poll = 0; // worker sleeps while GPU-only layers run
    auto pool = new_pool(&params); GGML_ASSERT(pool);
    set_threads(g.cpu, threads); set_pool(g.cpu, pool);
    g.active = true;
    g.worker = std::thread([] {
        auto & g = s();
        while (true) {
            std::unique_lock<std::mutex> lock(g.mutex);
            g.cv.wait(lock, [&] { return g.stop || g.pending; });
            if (g.stop) return;
            const int job = g.job; lock.unlock();
            cpu_compute(job);
            lock.lock(); g.pending = false; g.done = true; g.cv.notify_all();
        }
    });
    atexit(finish);
    LLAMA_LOG_INFO("WORD_GAMES_HYBRID: pins=%d layers=26 copied=%zu pool_MiB=%.2f byte_verification=passed threads=%d async_worker=1\n",
        g.pins, copied, ggml_backend_buffer_get_size(g.pool_buf)/1048576.0, threads);
}
bool word_games_hybrid_enabled() { return s().active; }
bool word_games_hybrid_active(int il, int64_t tokens) { return s().active && il >= 0 && il < 26 && tokens == 1; }
void word_games_hybrid_set_user(ggml_backend_sched_eval_callback cb, void * ud) { s().user = cb; s().user_data = ud; }

ggml_tensor * word_games_hybrid_build(ggml_context * ctx, ggml_cgraph * gf,
    ggml_backend_sched_t sched, ggml_tensor * input, ggml_tensor * ids, ggml_tensor * weights, int il) {
    auto & g = s(); auto & l = g.layers[il];
    l.graph_input = input; l.graph_ids = ids;
    auto main = backend(sched, g.main_dev), gpu = backend(sched, g.pool_dev);
    auto place = [&](ggml_tensor * t) { ggml_backend_sched_set_tensor_backend(sched, t, gpu); return t; };
    weights = ggml_cont(ctx, weights); // real node: callback reads final weights
    ggml_format_name(weights, "hybrid_launch-%d", il);
    ggml_backend_sched_set_tensor_backend(sched, weights, main);
    ggml_build_forward_expand(gf, weights);
    auto map = ggml_reshape_2d(ctx, l.slots, 1, g.n_expert);
    auto remap = place(ggml_get_rows(ctx, map, ids));
    remap = ggml_reshape_2d(ctx, remap, 8, 1);
    auto x = ggml_reshape_3d(ctx, input, g.embd, 1, 1);
    auto up = place(ggml_mul_mat_id(ctx, l.pool[0], x, remap));
    auto gate = place(ggml_mul_mat_id(ctx, l.pool[1], x, remap));
    auto hid = place(ggml_swiglu_split(ctx, gate, up));
    auto down = place(ggml_mul_mat_id(ctx, l.pool[2], hid, remap));
    auto weighted = place(ggml_mul(ctx, down, weights));
    auto sum = ggml_view_2d(ctx, weighted, g.embd, 1, weighted->nb[2], 0);
    for (int i = 1; i < 8; ++i) {
        auto v = ggml_view_2d(ctx, weighted, g.embd, 1, weighted->nb[2], i*weighted->nb[1]);
        sum = place(ggml_add(ctx, sum, v));
    }
    ggml_format_name(sum, "hybrid_join-%d", il);
    ggml_build_forward_expand(gf, sum);
    auto out = ggml_add(ctx, sum, l.correction);
    // Correction is written by the join callback on CUDA0 before this split.
    ggml_backend_sched_set_tensor_backend(sched, out, main);
    ggml_format_name(out, "ffn_moe_out-%d", il);
    ggml_build_forward_expand(gf, out);
    return out;
}

bool word_games_hybrid_callback(ggml_tensor * t, bool ask, void *) {
    auto & g = s();
    const bool launch = strncmp(t->name, "hybrid_launch-", 14) == 0;
    const bool join = strncmp(t->name, "hybrid_join-", 12) == 0;
    if (ask) return launch || join || (g.user && g.user(t, true, g.user_data));
    if (launch) {
        const double start = us();
        const int il = atoi(t->name+14); GGML_ASSERT(il >= 0 && il < 26);
        auto & l = g.layers[il];
        GGML_ASSERT(!g.pending && g.job == -1);
        int32_t ids[8]; float weights[8];
        ggml_backend_tensor_get(l.graph_ids, ids, 0, sizeof(ids));
        ggml_backend_tensor_get(t, weights, 0, sizeof(weights));
        memcpy(l.all_ids, ids, sizeof(ids)); memcpy(l.all_weights, weights, sizeof(weights));
        ggml_backend_tensor_get(l.graph_input, l.x.data(), 0, g.embd*sizeof(float));
        l.misses = 0;
        for (int k = 0; k < 8; ++k) {
            GGML_ASSERT(ids[k] >= 0 && ids[k] < g.n_expert && std::isfinite(weights[k]));
            for (int j = 0; j < k; ++j) GGML_ASSERT(ids[j] != ids[k]);
            if (l.map[ids[k]] == g.pins) {
                l.ids[l.misses] = ids[k]; l.weights[l.misses++] = weights[k];
            }
        }
        g.calls++; g.misses += l.misses; g.hits += 8-l.misses;
        std::fill(l.result.begin(), l.result.end(), 0.0f);
        { std::unique_lock<std::mutex> lock(g.mutex);
          g.job = il; g.done = l.misses == 0; g.pending = !g.done;
          if (g.pending) g.cv.notify_all(); }
        g.launch_us += us()-start; g.launched_at = us();
    }
    if (join) {
        const double start = us(); const int il = atoi(t->name+12);
        GGML_ASSERT(il == g.job);
        g.window_us += start-g.launched_at;
        { std::unique_lock<std::mutex> lock(g.mutex); g.cv.wait(lock, [&] { return g.done; }); g.job = -1; }
        auto & l = g.layers[il];
        if (!l.validated) {
            // On real activations, compare the combined result with all eight
            // experts evaluated by the unchanged CPU backend. Outside timing
            // once all layers have been exercised by the JSON warm-up.
            std::vector<float> hits(g.embd), reference(g.embd);
            ggml_backend_tensor_get(t, hits.data(), 0, g.embd*sizeof(float));
            auto & c = get_graph(l, 8);
            ggml_backend_tensor_set(c.x, l.x.data(), 0, g.embd*sizeof(float));
            ggml_backend_tensor_set(c.ids, l.all_ids, 0, sizeof(l.all_ids));
            ggml_backend_tensor_set(c.weights, l.all_weights, 0, sizeof(l.all_weights));
            GGML_ASSERT(ggml_backend_graph_compute(g.cpu, c.graph) == GGML_STATUS_SUCCESS);
            ggml_backend_tensor_get(c.out, reference.data(), 0, g.embd*sizeof(float));
            double diff2 = 0, ref2 = 0, max_abs = 0;
            for (int i = 0; i < g.embd; ++i) {
                double d = double(hits[i])+l.result[i]-reference[i];
                GGML_ASSERT(std::isfinite(d));
                diff2 += d*d; ref2 += double(reference[i])*reference[i];
                max_abs = std::max(max_abs, std::abs(d));
            }
            const double relative = std::sqrt(diff2/std::max(ref2, 1e-30));
            LLAMA_LOG_INFO("WORD_GAMES_HYBRID_VALIDATE: layer=%d cpu=%d gpu=%d relative_l2=%.9g max_abs=%.9g\n",
                il, l.misses, 8-l.misses, relative, max_abs);
            GGML_ASSERT(relative < (g.pins == 0 ? 1e-6 : 0.03));
            g.validation_max_relative_l2 = std::max(g.validation_max_relative_l2, relative);
            l.validated = true; g.validated_layers++;
        }
        ggml_backend_tensor_set(l.correction, l.result.data(), 0, g.embd*sizeof(float));
        g.join_us += us()-start;
        GGML_ASSERT(g.hits+g.misses == g.calls*8);
        if (g.calls % 2600 == 0) save_stats();
    }
    return !g.user || g.user(t, false, g.user_data);
}
