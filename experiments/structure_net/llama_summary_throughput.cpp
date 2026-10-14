// Fresh-input Llama generation with target-verified prompt lookup.
#include "llama.h"
#include "ggml-backend.h"
#include "nlohmann/json.hpp"
#include <cuda_runtime_api.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

using json = nlohmann::json;
using Clock = std::chrono::steady_clock;
static void quiet_log(ggml_log_level level,const char * text,void *) {
    if(level>=GGML_LOG_LEVEL_WARN) std::cerr<<text;
}
static double elapsed(Clock::time_point start) {
    return std::chrono::duration<double>(Clock::now()-start).count();
}
static size_t gpu_used() {
    size_t free, total;
    if (cudaMemGetInfo(&free,&total)!=cudaSuccess) throw std::runtime_error("CUDA query failed");
    return total-free;
}

static std::vector<int> lookup(const std::vector<int> & prompt,
                               const std::vector<int> & history, int limit) {
    if (limit<=0) return {};
    for (int n=4;n>=2;--n) {
        if (history.size()<size_t(n)) continue;
        // Most recent matching source span wins, deterministically.
        for (int i=int(prompt.size())-n-1;i>=0;--i) {
            if (std::equal(history.end()-n,history.end(),prompt.begin()+i)) {
                int end=std::min(int(prompt.size()),i+n+limit);
                return {prompt.begin()+i+n,prompt.begin()+end};
            }
        }
    }
    return {};
}

struct Model {
    llama_model * model=nullptr;
    llama_context * ctx=nullptr;
    llama_batch_ext * batch=nullptr;
    const llama_vocab * vocab=nullptr;
    int chunk=512, capacity=8192;
    ~Model() {
        if(batch) llama_batch_ext_free(batch);
        if(ctx) llama_free(ctx);
        if(model) llama_model_free(model);
    }
    void load(const std::string & path, int ubatch, int context) {
        chunk=ubatch; capacity=context;
        auto dev=ggml_backend_dev_by_name("CUDA0");
        if(!dev) throw std::runtime_error("CUDA0 required; no CPU fallback");
        ggml_backend_dev_t devices[]={dev,nullptr};
        llama_model_tensor_buft_override overrides[]={{".*",ggml_backend_dev_buffer_type(dev)},{nullptr,nullptr}};
        auto mp=llama_model_default_params();
        mp.n_gpu_layers=99; mp.devices=devices;
        mp.load_mode=LLAMA_LOAD_MODE_NONE; mp.lazy_mode=LLAMA_LAZY_MODE_OFF;
        mp.split_mode=LLAMA_SPLIT_MODE_NONE; mp.tensor_buft_overrides=overrides;
        model=llama_model_load_from_file(path.c_str(),mp);
        if(!model) throw std::runtime_error("model load failed");
        vocab=llama_model_get_vocab(model);
        auto cp=llama_context_default_params();
        cp.n_ctx=context; cp.n_batch=ubatch; cp.n_ubatch=ubatch; cp.n_seq_max=1;
        cp.n_threads=2; cp.n_threads_batch=2;
        cp.type_k=GGML_TYPE_F16; cp.type_v=GGML_TYPE_F16;
        cp.flash_attn_type=LLAMA_FLASH_ATTN_TYPE_ENABLED;
        ctx=llama_init_from_model(model,cp);
        if(!ctx) throw std::runtime_error("context allocation failed");
        batch=llama_batch_ext_init(ctx);
        if(!batch) throw std::runtime_error("batch allocation failed");
    }
    void forward(const std::vector<int> & ids,int position,bool all=false) {
        if(ids.empty() || ids.size()>size_t(chunk) || position+ids.size()>size_t(capacity))
            throw std::runtime_error("invalid forward range");
        llama_batch_ext_clear(batch);
        for(size_t i=0;i<ids.size();++i) {
            int idx=llama_batch_ext_add_token(batch,0,ids[i]);
            llama_pos pos=position+i;
            if(idx<0 || !llama_batch_ext_set_pos(batch,idx,&pos)) throw std::runtime_error("batch setup failed");
            llama_batch_ext_set_output_logits(batch,idx,all || i+1==ids.size());
        }
        if(llama_process(ctx,LLAMA_PROCESS_TYPE_DECODE,batch)!=0) throw std::runtime_error("decode failed");
        llama_synchronize(ctx);
    }
    int greedy(int row=-1) {
        auto p=llama_get_logits_ith(ctx,row);
        if(!p) throw std::runtime_error("missing logits");
        int n=llama_vocab_n_tokens(vocab);
        for(int i=0;i<n;++i) if(!std::isfinite(p[i])) throw std::runtime_error("nonfinite logits");
        return std::max_element(p,p+n)-p;
    }
    std::string decode(const std::vector<int> & ids) {
        std::string result;
        for(int id:ids) {
            if(llama_vocab_is_eog(vocab,id)) continue;
            char buffer[256]; int n=llama_token_to_piece(vocab,id,buffer,sizeof(buffer),0,false);
            if(n<0) {
                std::vector<char> bigger(-n);
                n=llama_token_to_piece(vocab,id,bigger.data(),bigger.size(),0,false);
                if(n<0) throw std::runtime_error("token decode failed");
                result.append(bigger.data(),n);
            } else result.append(buffer,n);
        }
        return result;
    }
    json generate(const std::vector<int> & prompt,int cap,int draft,bool stop=true) {
        if(prompt.empty() || cap<1 || prompt.size()+cap>size_t(capacity)) throw std::runtime_error("invalid generation budget");
        cudaDeviceSynchronize(); auto start=Clock::now();
        llama_memory_clear(llama_get_memory(ctx),true);
        for(size_t i=0;i<prompt.size();i+=chunk) {
            size_t end=std::min(prompt.size(),i+chunk);
            forward({prompt.begin()+i,prompt.begin()+end},i);
        }
        std::vector<int> generated={greedy()};
        double ttft=elapsed(start);
        int proposed=0, accepted=0, calls=0, rollbacks=0;
        while(int(generated.size())<cap && !(stop && llama_vocab_is_eog(vocab,generated.back()))) {
            std::vector<int> history=prompt;
            history.insert(history.end(),generated.begin(),generated.end());
            int old_past=int(history.size())-1;
            auto candidates=lookup(prompt,history,std::min(draft,cap-int(generated.size())-1));
            std::vector<int> batch_ids={generated.back()};
            batch_ids.insert(batch_ids.end(),candidates.begin(),candidates.end());
            forward(batch_ids,old_past,true); ++calls; proposed+=candidates.size();
            for(size_t i=0;i<batch_ids.size();++i) {
                int next=greedy(i);
                generated.push_back(next);
                bool matched=i<candidates.size() && next==candidates[i];
                if(matched) ++accepted;
                if(int(generated.size())==cap || (stop && llama_vocab_is_eog(vocab,next)) || !matched) break;
            }
            int keep=int(prompt.size()+generated.size())-1;
            if(keep<old_past+int(batch_ids.size())) {
                if(!llama_memory_seq_rm(llama_get_memory(ctx),0,keep,-1)) throw std::runtime_error("KV rollback failed");
                ++rollbacks;
            }
        }
        cudaDeviceSynchronize(); double seconds=elapsed(start);
        int count=0; for(int token:generated) if(!llama_vocab_is_eog(vocab,token)) ++count;
        return {{"generated_ids",generated},{"text",decode(generated)},
                {"output_tokens",count},{"prompt_tokens",prompt.size()},{"cap",cap},{"draft",draft},
                {"wall_seconds",seconds},{"ttft_seconds",ttft},{"decode_seconds",seconds-ttft},
                {"decode_steps",generated.size()-1},{"verification_calls",calls},
                {"proposed_tokens",proposed},{"accepted_tokens",accepted},{"rollbacks",rollbacks},
                {"stopped_eos",llama_vocab_is_eog(vocab,generated.back())}};
    }
};

int main(int argc,char **argv) {
    if(argc==2 && std::string(argv[1])=="self-test") {
        if(lookup({1,2,3,4,5},{8,1,2},2)!=std::vector<int>({3,4})) return 1;
        if(lookup({1,2,3,1,2,9},{1,2},3)!=std::vector<int>({9})) return 1;
        if(!lookup({1,2,3},{1,2},0).empty()) return 1;
        if(!lookup({1,2,3},{8,9},4).empty()) return 1;
        if(lookup({1,2,3,4,5},{1,2,3,4},8)!=std::vector<int>({5})) return 1;
        std::cout<<"5 prompt lookup contracts passed\n";return 0;
    }
    // Preparation uses the same pinned quantization implementation, on CPU.
    if(argc==5 && std::string(argv[1])=="quantize") {
        auto p=llama_model_quantize_default_params(); p.nthread=4;
        std::string kind=argv[4];
        if(kind=="Q4_K_M") p.ftype=LLAMA_FTYPE_MOSTLY_Q4_K_M;
        else if(kind=="Q5_K_M") p.ftype=LLAMA_FTYPE_MOSTLY_Q5_K_M;
        else if(kind=="Q8_0") p.ftype=LLAMA_FTYPE_MOSTLY_Q8_0;
        else return 2;
        return llama_model_quantize(argv[2],argv[3],&p);
    }
    if(argc!=5) {std::cerr<<"usage: runner model.gguf fixtures.json output.json pilot|full\n";return 2;}
    json result={{"schema","nal.llama-summary-runtime.v1"},{"status","running"},{"rows",json::array()}};
    auto save=[&](){std::ofstream out(argv[3]);out<<result.dump(2)<<'\n';};
    try {
        std::ifstream input(argv[2]);json config;input>>config;
        bool pilot=std::string(argv[4])=="pilot";
        llama_log_set(quiet_log,nullptr);ggml_log_set(quiet_log,nullptr);
        llama_backend_init();ggml_backend_load_all();
        size_t initial=gpu_used();auto load_start=Clock::now();
        Model m;m.load(argv[1],config.value("ubatch",512),config.value("context",8192));
        result["load_seconds"]=elapsed(load_start);
        result["model_weight_bytes"]=llama_model_size(m.model);
        result["resident_increment_bytes"]=gpu_used()-initial;
        result["profile"]=pilot?"systems_lifecycle_only_not_quality_evidence":"exploratory";
        result["model"]=argv[1];result["config"]=config;
        int repeats=pilot?1:config.value("repeats",5);
        auto drafts=config.value("drafts",std::vector<int>{0,4,8,16});
        // Warm every distinct strategy; no previous request KV is retained.
        auto warm=config["examples"][0]["prompt_ids"].get<std::vector<int>>();
        warm.resize(std::min(size_t(64),warm.size()));
        for(int d:drafts)m.generate(warm,24,d,false);
        for(const auto & example:config["examples"]) {
            auto prompt=example["prompt_ids"].get<std::vector<int>>();
            int cap=example.value("cap",768);
            for(int repeat=0;repeat<repeats;++repeat) {
                auto order=drafts;
                if(repeat%2) std::reverse(order.begin(),order.end());
                for(int d:order) {
                    auto row=m.generate(prompt,cap,d,example.value("stop_eos",true));
                    row["id"]=example["id"];row["repeat"]=repeat;
                    result["rows"].push_back(row);save();
                    std::cout<<json{{"id",example["id"]},{"repeat",repeat},{"draft",d},
                        {"tokens",row["output_tokens"]},{"seconds",row["wall_seconds"]}}.dump()<<std::endl;
                }
            }
        }
        result["sampled_gpu_increment_bytes"]=gpu_used()-initial;
        result["status"]="completed";
    } catch(const std::exception & e) {result["status"]="failed";result["error"]=e.what();}
    save();return result["status"]=="completed"?0:1;
}
