// Exercise real model outputs using the pinned upstream backend-test fixtures.
#define main upstream_backend_sampler_main
#include "test-backend-sampler.cpp"
#undef main
#include <dlfcn.h>

static unsigned sync_calls = 0;
extern "C" void ggml_backend_sched_synchronize(ggml_backend_sched_t sched) {
    using function = void (*)(ggml_backend_sched_t);
    static auto real = reinterpret_cast<function>(dlsym(RTLD_NEXT, "ggml_backend_sched_synchronize"));
    GGML_ASSERT(real);
    ++sync_calls;
    real(sched);
}

#ifndef TEST_CONTROL
static llama_sampling_output check_view(llama_context * ctx, int idx) {
    sync_calls = 0;
    const auto view = llama_get_sampling_output_ith(ctx, idx);
    GGML_ASSERT(sync_calls == 1);
    GGML_ASSERT(view.sampled_probs == llama_get_sampled_probs_ith(ctx, idx));
    GGML_ASSERT(view.sampled_logits == llama_get_sampled_logits_ith(ctx, idx));
    GGML_ASSERT(view.sampled_candidates == llama_get_sampled_candidates_ith(ctx, idx));
    GGML_ASSERT(view.sampled_probs_count == llama_get_sampled_probs_count_ith(ctx, idx));
    GGML_ASSERT(view.sampled_logits_count == llama_get_sampled_logits_count_ith(ctx, idx));
    GGML_ASSERT(view.sampled_candidates_count == llama_get_sampled_candidates_count_ith(ctx, idx));
    GGML_ASSERT(view.sampled_token == llama_get_sampled_token_ith(ctx, idx));
    if (!view.sampled_probs && !view.sampled_logits) {
        GGML_ASSERT(view.logits == llama_get_logits_ith(ctx, idx));
    } else {
        GGML_ASSERT(!view.logits);
    }
    GGML_ASSERT(view.n_vocab == llama_vocab_n_tokens(llama_model_get_vocab(llama_get_model(ctx))));
    return view;
}

static void check_rows(const test_params & params) {
    // Raw logits, backend top-k logits, greedy token, and dist probabilities/token.
    for (int mode = 0; mode < 4; ++mode) {
        llama_sampler_ptr a(llama_sampler_chain_init(llama_sampler_chain_default_params()));
        llama_sampler_ptr b(llama_sampler_chain_init(llama_sampler_chain_default_params()));
        for (auto * chain : {a.get(), b.get()}) {
            if (mode) llama_sampler_chain_add(chain, llama_sampler_init_top_k(8));
            if (mode == 2) llama_sampler_chain_add(chain, llama_sampler_init_greedy());
            if (mode == 3) llama_sampler_chain_add(chain, llama_sampler_init_dist(73));
        }
        std::vector<llama_sampler_seq_config> configs;
        if (mode) configs = {{0, a.get()}, {1, b.get()}};
        test_context fixture(params, configs, 2);
        GGML_ASSERT(fixture.decode({{0, "One plus one is"}, {1, "The color of the sky is"}}));
        auto * ctx = fixture.ctx.get();
        const auto first = check_view(ctx, fixture.idx_for_seq(0));
        const auto second = check_view(ctx, fixture.idx_for_seq(1));
        const auto last = check_view(ctx, -1);
        const auto previous = check_view(ctx, -2);
        GGML_ASSERT(first.logits == previous.logits && first.sampled_logits == previous.sampled_logits);
        GGML_ASSERT(second.logits == last.logits && second.sampled_logits == last.sampled_logits);
        GGML_ASSERT(second.sampled_probs == last.sampled_probs && second.sampled_token == last.sampled_token);
        if (mode == 0) GGML_ASSERT(first.logits && second.logits && first.logits != second.logits);
        if (mode == 1) GGML_ASSERT(first.sampled_logits && first.sampled_logits_count == 8);
        if (mode == 2) GGML_ASSERT(first.sampled_token != LLAMA_TOKEN_NULL);
        if (mode == 3) GGML_ASSERT(first.sampled_probs && first.sampled_probs_count == 8);
        common_params_sampling sp;
        sp.temp = 0.0f;
        sp.samplers = {COMMON_SAMPLER_TYPE_TEMPERATURE};
        common_sampler_ptr sampler(common_sampler_init(params.model.get(), sp));
        GGML_ASSERT(sampler);
        const float * scores = first.sampled_logits ? first.sampled_logits : first.logits;
        const auto count = first.sampled_logits ? first.sampled_logits_count : first.n_vocab;
        const auto best = std::max_element(scores, scores + count) - scores;
        const auto expected = first.sampled_token != LLAMA_TOKEN_NULL ? first.sampled_token :
                              (first.sampled_logits ? first.sampled_candidates[best] : best);
        sync_calls = 0;
        GGML_ASSERT(common_sampler_sample(sampler.get(), ctx, fixture.idx_for_seq(0)) == expected);
        GGML_ASSERT(sync_calls == 1);
        GGML_ASSERT(fixture.decode_token(42, 0));
        check_view(ctx, fixture.idx_for_seq(0));  // reacquire after another decode
        printf("PASS output mode=%d: rows, negative indices, counts, reacquisition, one sync\n", mode);
    }
}
#endif

static void check_grammar(const test_params & params) {
    std::vector<llama_sampler_seq_config> configs;
    test_context fixture(params, configs);
    GGML_ASSERT(fixture.decode({{0, "Return a digit"}}));
    auto * ctx = fixture.ctx.get();
    const int idx = fixture.idx_for_seq(0);
    llama_token bad = -1, good = -1;
    GGML_ASSERT(llama_tokenize(fixture.vocab, "x", 1, &bad, 1, false, false) == 1);
    GGML_ASSERT(llama_tokenize(fixture.vocab, "7", 1, &good, 1, false, false) == 1);
    auto * logits = llama_get_logits_ith(ctx, idx);
    std::fill(logits, logits + fixture.n_vocab, -INFINITY);
    logits[bad] = 100.0f;
    logits[good] = 10.0f;
    for (int mode = 0; mode < 3; ++mode) {
        common_params_sampling sp;
        sp.temp = 0.0f;
        sp.backend_sampling = false;
        sp.samplers = {COMMON_SAMPLER_TYPE_TEMPERATURE};
        if (mode) sp.grammar = {COMMON_GRAMMAR_TYPE_USER, "root ::= \"7\""};
        common_sampler_ptr sampler(common_sampler_init(params.model.get(), sp));
        GGML_ASSERT(sampler);
        sync_calls = 0;
        const auto token = common_sampler_sample(sampler.get(), ctx, idx, mode == 2);
        GGML_ASSERT(token == (mode ? good : bad));
#ifdef TEST_CONTROL
        GGML_ASSERT(sync_calls == (mode == 1 ? 10 : 6));
#else
        GGML_ASSERT(sync_calls == 1);
#endif
        GGML_ASSERT(logits[bad] == 100.0f && logits[good] == 10.0f);
        printf("PASS grammar mode=%d: token=%d scheduler_sync_calls=%u\n", mode, token, sync_calls);
    }
}

int main(int argc, char ** argv) {
    GGML_ASSERT(argc == 2);
    ggml_backend_load_all_from_path("/llama");
    llama_backend_init();
    test_args args;
    args.model = argv[1];
    args.device = "gpu";
    test_params params{load_model(args)};
    GGML_ASSERT(params.model);
    check_grammar(params);
#ifndef TEST_CONTROL
    check_rows(params);
#endif
    puts("PASS all sampling-output integration checks");
}
