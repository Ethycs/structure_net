// Independent physical GGUF models, with one private context per replica.
#include "llama.h"
#include "ggml-backend.h"
#include "nlohmann/json.hpp"
#include <cuda_runtime_api.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <future>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

using json = nlohmann::json;
using Clock = std::chrono::steady_clock;

static size_t gpu_used() {
    size_t free, total;
    if (cudaMemGetInfo(&free, &total) != cudaSuccess) throw std::runtime_error("CUDA memory query failed");
    return total-free;
}

struct Replica {
    llama_model * model = nullptr;
    llama_context * ctx = nullptr;
    llama_batch_ext * batch = nullptr;
    const llama_vocab * vocab = nullptr;
    ~Replica() {
        if (batch) llama_batch_ext_free(batch);
        if (ctx) llama_free(ctx);
        if (model) llama_model_free(model);
    }
    void clear() { llama_memory_clear(llama_get_memory(ctx), false); }
    void forward(const std::vector<int> & tokens, int position=0, bool all=false) {
        llama_batch_ext_clear(batch);
        for (size_t i=0; i<tokens.size(); ++i) {
            int idx = llama_batch_ext_add_token(batch, 0, tokens[i]);
            llama_pos pos = position + i;
            if (idx < 0 || !llama_batch_ext_set_pos(batch, idx, &pos)) throw std::runtime_error("batch setup failed");
            llama_batch_ext_set_output_logits(batch, idx, all || i+1==tokens.size());
        }
        if (llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch) != 0) throw std::runtime_error("decode failed");
        llama_synchronize(ctx);
    }
    int greedy() {
        auto logits = llama_get_logits_ith(ctx, -1);
        if (!logits) throw std::runtime_error("missing logits");
        return std::max_element(logits, logits + llama_vocab_n_tokens(vocab)) - logits;
    }
    std::vector<int> generate(const std::vector<int> & prompt, int steps, bool stop) {
        clear(); forward(prompt);
        std::vector<int> result;
        for (int i=0; i<steps; ++i) {
            int token = greedy();
            result.push_back(token);
            if (stop && (token==128001 || token==128008 || token==128009)) break;
            if (i+1 < steps) forward({token}, prompt.size()+i);
        }
        return result;
    }
    double nll(const std::vector<int> & prompt, const std::vector<int> & reference) {
        clear();
        std::vector<int> tokens = prompt;
        tokens.insert(tokens.end(), reference.begin(), reference.end()-1);
        forward(tokens, 0, true);
        double total = 0;
        for (size_t i=0; i<reference.size(); ++i) {
            auto logits = llama_get_logits_ith(ctx, prompt.size()-1+i);
            double max = *std::max_element(logits, logits+llama_vocab_n_tokens(vocab));
            double sum = 0;
            for (int j=0; j<llama_vocab_n_tokens(vocab); ++j) sum += std::exp(logits[j]-max);
            total += std::log(sum)+max-logits[reference[i]];
        }
        return total/reference.size();
    }
};

int main(int argc, char ** argv) {
    if (argc != 6) {
        std::cerr << "usage: assay model.gguf fixtures.json replicas output.json validation|test|pilot\n";
        return 2;
    }
    json result;
    try {
        std::ifstream input(argv[2]); json fixtures; input >> fixtures;
        int count = std::stoi(argv[3]);
        std::string stage = argv[5];
        int repeats = stage=="pilot" ? 1 : 5;
        llama_backend_init();
        ggml_backend_load_all();
        auto dev = ggml_backend_dev_by_name("CUDA0");
        if (!dev) throw std::runtime_error("CUDA0 backend required; CPU fallback forbidden");
        ggml_backend_dev_t devices[] = {dev, nullptr};
        llama_model_tensor_buft_override overrides[] = {{".*", ggml_backend_dev_buffer_type(dev)}, {nullptr, nullptr}};
        size_t initial = gpu_used(), peak = initial;
        auto start = Clock::now();
        std::vector<std::unique_ptr<Replica>> models;
        std::vector<uint64_t> pointers;
        uint64_t parameters = 0;
        for (int i=0; i<count; ++i) {
            auto r = std::make_unique<Replica>();
            auto mp = llama_model_default_params();
            mp.n_gpu_layers = 99; mp.devices = devices;
            mp.load_mode = LLAMA_LOAD_MODE_NONE; mp.lazy_mode = LLAMA_LAZY_MODE_OFF;
            mp.split_mode = LLAMA_SPLIT_MODE_NONE; mp.tensor_buft_overrides = overrides;
            r->model = llama_model_load_from_file(argv[1], mp);
            if (!r->model) throw std::runtime_error("physical model allocation failed");
            r->vocab = llama_model_get_vocab(r->model);
            pointers.push_back(reinterpret_cast<uint64_t>(r->model));
            parameters += llama_model_size(r->model);
            auto cp = llama_context_default_params();
            cp.n_ctx=1024; cp.n_batch=1024; cp.n_ubatch=512; cp.n_seq_max=1;
            cp.n_threads=2; cp.n_threads_batch=2;
            cp.type_k=GGML_TYPE_F16; cp.type_v=GGML_TYPE_F16;
            cp.flash_attn_type=LLAMA_FLASH_ATTN_TYPE_ENABLED;
            r->ctx=llama_init_from_model(r->model, cp);
            if (!r->ctx) throw std::runtime_error("private context allocation failed");
            r->batch=llama_batch_ext_init(r->ctx);
            if (!r->batch) throw std::runtime_error("batch allocation failed");
            models.push_back(std::move(r));
            peak=std::max(peak,gpu_used());
        }
        result = {{"status","completed"},{"replicas",count},{"model_object_addresses",pointers},
                  {"model_weight_bytes",parameters},{"gpu_initial_used_bytes",initial},
                  {"gpu_resident_increment_bytes",gpu_used()-initial},
                  {"load_seconds",std::chrono::duration<double>(Clock::now()-start).count()},
                  {"context_per_model",1024},{"physical_models",true},{"shared_weight_model_slots",false},
                  {"kv_payload_bytes",static_cast<uint64_t>(count)*28*2*8*1024*128*2}};
        json timings=json::array();
        auto execute = [&](bool concurrent, int length, int steps) {
            auto operation = [&](int i) {
                auto ids=fixtures["benchmark_ids"][i].get<std::vector<int>>(); ids.resize(length);
                if (steps) models[i]->generate(ids,steps,false);
                else {models[i]->clear(); models[i]->forward(ids);}
            };
            if (concurrent) {
                std::vector<std::future<void>> jobs;
                for (int i=0;i<count;++i) jobs.push_back(std::async(std::launch::async,operation,i));
                for (auto & job:jobs) job.get();
            } else for (int i=0;i<count;++i) operation(i);
        };
        for (auto item: std::vector<std::pair<int,int>>{{128,0},{512,0},{128,32}}) {
            json samples={{"separate",json::array()},{"concurrent",json::array()}};
            execute(false,item.first,item.second); execute(true,item.first,item.second);
            for (int sample=0;sample<repeats;++sample) {
                for (int order=0;order<2;++order) {
                    bool concurrent = (sample+order)%2;
                    cudaDeviceSynchronize(); auto start=Clock::now();
                    execute(concurrent,item.first,item.second);
                    cudaDeviceSynchronize();
                    samples[concurrent?"concurrent":"separate"].push_back(1000*std::chrono::duration<double>(Clock::now()-start).count());
                    peak=std::max(peak,gpu_used());
                }
            }
            timings.push_back({{"length",item.first},{"decode_steps",item.second},{"samples_wall_ms",samples}});
        }
        result["timings"]=timings;
        result["quality"]=json::array();
        if (count==1 && stage!="pilot") {
            for (const auto & row:fixtures["examples"]) {
                if (row["split"]!=stage) continue;
                auto prompt=row["prompt_ids"].get<std::vector<int>>();
                auto reference=row["reference_ids"].get<std::vector<int>>();
                double nll=models[0]->nll(prompt,reference);
                auto generated=models[0]->generate(prompt,96,true);
                result["quality"].push_back({{"id",row["id"]},{"generated_ids",generated},{"reference_nll",nll}});
                peak=std::max(peak,gpu_used());
            }
        }
        result["observed_peak_gpu_increment_bytes"]=peak-initial;
        result["peak_sampling_boundary"]="after model/context allocation and each synchronized inference; transient peaks may be higher";
        result["workspace_and_runtime_increment_bytes"]=static_cast<int64_t>(peak-initial)-parameters-result["kv_payload_bytes"].get<int64_t>();
    } catch (const std::exception & error) {
        result["status"]="failed"; result["error"]=error.what();
    }
    std::ofstream output(argv[4]); output << result.dump(2) << '\n';
    return result["status"]=="completed" ? 0 : 1;
}
