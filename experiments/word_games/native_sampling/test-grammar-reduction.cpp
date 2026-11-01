#define main upstream_backend_sampler_main
#include "test-backend-sampler.cpp"
#undef main
#include "count-output.cpp"
#include <limits>
static unsigned reserve_calls = 0;
extern "C" bool ggml_backend_sched_reserve(ggml_backend_sched_t sched, ggml_cgraph * graph) {
    static auto real=wg_symbol<decltype(&ggml_backend_sched_reserve)>("ggml_backend_sched_reserve");
    ++reserve_calls;
    return real(sched,graph);
}

struct forced_output {
    int vocab;
    llama_token good, bad;
    std::string mode;
    unsigned writes = 0;
    static bool callback(ggml_tensor * tensor, bool ask, void * user) {
        auto & self = *static_cast<forced_output *>(user);
        if (std::strcmp(ggml_get_name(tensor), "result_output") != 0) return false;
        if (ask) return true;
        GGML_ASSERT(tensor->type == GGML_TYPE_F32 && tensor->ne[0] == self.vocab);
        std::vector<float> values(ggml_nelements(tensor), -100.0f);
        for (size_t row = 0; row < values.size()/self.vocab; ++row) {
            auto * scores = values.data()+row*self.vocab;
            scores[self.good] = 10.0f;
            scores[self.bad] = 2.0f;
            if (self.mode == "reject") scores[self.bad] = 20.0f;
            if (self.mode == "tie" || (self.mode == "default_tie" || self.mode == "across_tie")) scores[self.bad] = 10.0f;
            if (self.mode == "nan") scores[42] = std::numeric_limits<float>::quiet_NaN();
            if (self.mode == "negative_inf") scores[42] = -INFINITY;
            if (self.mode == "positive_inf") scores[self.good] = INFINITY;
        }
        ggml_backend_tensor_set(tensor, values.data(), 0, values.size()*sizeof(float));
        self.writes++;
        return true;
    }
};

static common_params_sampling parameters() {
    common_params_sampling p;
    p.temp = 0.0f;
    p.backend_sampling = false;
    p.samplers = {COMMON_SAMPLER_TYPE_TEMPERATURE};
    p.grammar = {COMMON_GRAMMAR_TYPE_USER, "root ::= [0-9]*"};
    return p;
}

static void check_exclusions(const llama_model * model) {
    for (int i = 0; i < 8; ++i) {
        auto p = parameters();
        if (i == 0) p.temp = 0.5f;
        if (i == 1) p.n_probs = 5;
        if (i == 2) { p.penalty_repeat = 1.1f; p.samplers.insert(p.samplers.begin(), COMMON_SAMPLER_TYPE_PENALTIES); }
        if (i == 3) { p.typ_p = 0.5f; p.samplers.insert(p.samplers.begin(), COMMON_SAMPLER_TYPE_TYPICAL_P); }
        if (i == 4) p.logit_bias.push_back({42,1.0f});
        if (i == 5) { p.xtc_probability = 0.5f; p.samplers.insert(p.samplers.begin(), COMMON_SAMPLER_TYPE_XTC); }
        if (i == 6) { p.reasoning_budget_start={1}; p.reasoning_budget_end={{2}}; p.reasoning_budget_tokens=1; }
        if (i == 7) p.grammar = {};
        common_sampler_ptr sampler(common_sampler_init(model,p));
        GGML_ASSERT(sampler && !p.backend_sampling);
    }
    puts("PASS eligibility exclusions");
}

static void check_case(const test_params & model, const std::string & mode, int nseq=1, bool early=false) {
    const auto * vocab = llama_model_get_vocab(model.model.get());
    forced_output forced{llama_vocab_n_tokens(vocab), -1, -1, mode};
    const bool high = mode=="high_id" || mode=="across_tie";
    GGML_ASSERT(llama_tokenize(vocab,high ? "hello" : "7",high ? 5 : 1,&forced.good,1,false,false)==1);
    if (high) GGML_ASSERT(forced.good>=128);
    GGML_ASSERT(llama_tokenize(vocab,mode=="default_tie" ? "5" : "x",1,&forced.bad,1,false,false)==1);
    auto make_params = [&] {
        auto p = parameters();
        if (high) p.grammar={COMMON_GRAMMAR_TYPE_USER,"root ::= [a-z]*"};
        if (mode == "default_tie" || mode == "default_accept" || mode == "across_tie") {
            p.samplers=common_params_sampling().samplers;
            p.penalty_repeat=1.0f;
        }
        return p;
    };
    auto expected=forced.good;
    if (mode=="default_tie" || mode=="across_tie") {
        setenv("LLAMA_WORD_GAMES_GREEDY_PREVIEW","0",1);
        auto reference_params=make_params();
        common_sampler_ptr reference(common_sampler_init(model.model.get(),reference_params));
        setenv("LLAMA_WORD_GAMES_GREEDY_PREVIEW","1",1);
        std::vector<llama_token_data> data;
        for (int id=0; id<forced.vocab; ++id)
            data.push_back({id,(id==forced.good || id==forced.bad) ? 10.0f : -100.0f,0.0f});
        llama_token_data_array candidates{data.data(),data.size(),-1,false};
        llama_sampler_apply(common_sampler_get(reference.get()),&candidates);
        expected=candidates.data[candidates.selected].id;
        GGML_ASSERT(expected==forced.good || expected==forced.bad);
    }
    std::vector<common_sampler_ptr> samplers;
    std::vector<llama_sampler_seq_config> configs;
    for (int seq=0; seq<nseq; ++seq) {
        auto p = make_params();
        common_sampler_ptr sampler(common_sampler_init(model.model.get(),p));
        GGML_ASSERT(sampler);
#ifdef TEST_CONTROL
        GGML_ASSERT(!p.backend_sampling);
#else
        GGML_ASSERT(p.backend_sampling && llama_sampler_is_greedy_preview(common_sampler_get(sampler.get())));
        // Preview ownership must survive cloning and state copying.
        common_sampler_ptr clone(common_sampler_clone(sampler.get()));
        common_sampler_copy(sampler.get(),clone.get());
        sampler = std::move(clone);
        GGML_ASSERT(llama_sampler_is_greedy_preview(common_sampler_get(sampler.get())));
#endif
        if (p.backend_sampling) configs.push_back({seq,common_sampler_get(sampler.get())});
        samplers.push_back(std::move(sampler));
    }
    auto p = llama_context_default_params();
    p.n_ctx=256; p.n_batch=64; p.n_ubatch=2; p.n_seq_max=nseq;
    p.kv_unified=true; p.n_outputs_max_per_seq=1;
    p.samplers=configs.data(); p.n_samplers=configs.size();
    p.cb_eval=forced_output::callback; p.cb_eval_user_data=&forced;
    llama_context_ptr ctx(llama_init_from_model(model.model.get(),p));
    GGML_ASSERT(ctx);
    for (int step=0; step<(mode=="toggle" ? 3 : 2); ++step) {
        reserve_calls=0;
        const bool preview_active=mode!="toggle" || step!=1;
        if (step>0 && (mode=="replace" || mode=="toggle")) {
            // Match server lifetime: destroy the old request sampler first.
            samplers[0].reset();
            setenv("LLAMA_WORD_GAMES_GREEDY_PREVIEW",preview_active ? "1" : "0",1);
            auto next=make_params();
            samplers[0].reset(common_sampler_init(model.model.get(),next));
            GGML_ASSERT(llama_set_sampler(ctx.get(),0,next.backend_sampling ? common_sampler_get(samplers[0].get()) : nullptr));
            setenv("LLAMA_WORD_GAMES_GREEDY_PREVIEW","1",1);
        }
        common_batch batch(ctx.get());
        std::vector<int> indices;
        for (int seq=0; seq<nseq; ++seq) {
            for (int token=0; token<5; ++token) {
                const bool output = token==(early ? 0 : 4);
                if (output) indices.push_back(batch.size());
                batch.add(forced.good,step*5+token,seq,output);
            }
        }
        forced.writes=0;
        wg_reset();
        GGML_ASSERT(llama_process(ctx.get(),LLAMA_PROCESS_TYPE_DECODE,batch.get())==0);
        GGML_ASSERT(forced.writes>0);
        if (mode == "explicit") {
            auto * all = llama_get_logits(ctx.get());
            GGML_ASSERT(all && all[forced.good]==10.0f);
#ifndef TEST_CONTROL
            auto bytes=wg_raw_bytes.load();
            GGML_ASSERT(llama_get_raw_logits_ith(ctx.get(),-1)[forced.good]==10.0f);
            GGML_ASSERT(wg_raw_bytes==bytes); // second read must not recopy
#endif
        }
        for (int seq=0; seq<nseq; ++seq) {
#ifndef TEST_CONTROL
            if (high) {
                float sum;
                auto view=llama_get_sampling_preview_ith(ctx.get(),indices[seq],&sum);
                fprintf(stderr,"PREVIEW %s good=%d bad=%d ids=%d,%d scores=%g,%g sum=%g bytes=%llu\n",
                    mode.c_str(),forced.good,forced.bad,view.sampled_candidates[0],view.sampled_candidates[1],
                    view.sampled_logits[0],view.sampled_logits[1],sum,wg_raw_bytes.load());
            }
#endif
            auto token=common_sampler_sample(samplers[seq].get(),ctx.get(),indices[seq],mode=="grammar_first");
            GGML_ASSERT(token==expected);
            common_sampler_accept(samplers[seq].get(),token,true);
        }
        const auto row_bytes=static_cast<unsigned long long>(forced.vocab)*sizeof(float);
#ifdef TEST_CONTROL
        GGML_ASSERT(wg_raw_bytes==row_bytes*nseq && wg_preview_bytes==0);
#else
        const bool lazy_success=(mode=="accept" || mode=="default_accept" || mode=="high_id" || mode=="replace" || (mode=="toggle" && preview_active)) && nseq==1 && !early;
        if (wg_raw_bytes!=(lazy_success ? 0 : row_bytes*nseq))
            fprintf(stderr,"DOWNLOAD mismatch mode=%s raw=%llu expected=%llu\n",mode.c_str(),wg_raw_bytes.load(),lazy_success ? 0 : row_bytes*nseq);
        GGML_ASSERT(wg_raw_bytes==(lazy_success ? 0 : row_bytes*nseq));
        GGML_ASSERT(wg_preview_bytes==(preview_active ? 20u*nseq : 0));
        if (step==1 && mode=="replace") GGML_ASSERT(reserve_calls==0);
#endif
        printf("PASS %s sequences=%d early=%d step=%d raw_bytes=%llu preview_bytes=%llu\n",
               mode.c_str(),nseq,early,step,wg_raw_bytes.load(),wg_preview_bytes.load());
    }
    samplers.clear();
    ctx.reset();
}

int main(int argc, char ** argv) {
    GGML_ASSERT(argc==2);
    setenv("LLAMA_WORD_GAMES_GREEDY_PREVIEW","1",1);
    ggml_backend_load_all_from_path("/llama");
    llama_backend_init();
    test_args args;
    args.model=argv[1]; args.device="gpu";
    test_params model{load_model(args)};
    GGML_ASSERT(model.model);
    check_exclusions(model.model.get());
    for (const char * mode : {"accept","reject","tie","nan","negative_inf","positive_inf","grammar_first","explicit","default_accept","default_tie","high_id","across_tie","replace","toggle"})
        check_case(model,mode);
    check_case(model,"accept",2);
    check_case(model,"accept",1,true);
    puts("PASS all grammar-output integration checks");
}
