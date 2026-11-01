"""Generate the second native patch against the verified patch-1 source tree."""
from pathlib import Path
import difflib

BASE = Path('.pixi/word-games-sampling-v2/source')
DEST = Path('experiments/word_games/native_sampling/grammar-on-demand.patch')
texts = {}
originals = {}


def edit(name, old, new, count=1):
    if name not in texts:
        originals[name] = texts[name] = (BASE/name).read_text()
    if texts[name].count(old) != count:
        raise ValueError((name, old[:90], texts[name].count(old), count))
    texts[name] = texts[name].replace(old, new)


def main():
    header = 'include/llama.h'
    edit(header, '    // Get all output token embeddings.', '''    // Experimental, opt-in greedy grammar preview. The ordinary snapshot ABI
    // stays unchanged; the extra scalar rejects non-finite backend logits.
    LLAMA_API struct llama_sampling_output llama_get_sampling_preview_ith(
            struct llama_context * ctx, int32_t i, float * finite_sum);
    // Always the full, unmodified vocabulary, materializing deferred output.
    LLAMA_API float * llama_get_raw_logits_ith(struct llama_context * ctx, int32_t i);
    LLAMA_API struct llama_sampler * llama_sampler_init_greedy_preview(void);
    LLAMA_API bool llama_sampler_is_greedy_preview(const struct llama_sampler * smpl);

    // Get all output token embeddings.''')
    sampler = 'src/llama-sampler.cpp'
    edit(sampler, '// top-p\n', '''// A tagged top-2 preview; the original CPU chain remains the sampling authority.
static llama_sampler * greedy_preview_clone(const llama_sampler *) {
    return llama_sampler_init_greedy_preview();
}
static llama_sampler_i greedy_preview_i = [] {
    auto iface = llama_sampler_top_k_i;
    iface.clone = greedy_preview_clone;
    return iface;
}();

llama_sampler * llama_sampler_init_greedy_preview(void) {
    auto * result = llama_sampler_init_top_k(2);
    result->iface = &greedy_preview_i;
    return result;
}

bool llama_sampler_is_greedy_preview(const llama_sampler * smpl) {
    return smpl && llama_sampler_chain_n(smpl) == 1 &&
        static_cast<const llama_sampler_chain *>(smpl->ctx)->samplers[0].ptr->iface == &greedy_preview_i;
}

// top-p
''')
    graphh = 'src/llama-graph.h'
    edit(graphh, '    std::vector<ggml_tensor *> t_candidates;',
         '    std::vector<ggml_tensor *> t_candidates;\n    std::vector<ggml_tensor *> t_preview_sum;')
    graph = 'src/llama-graph.cpp'
    edit(graph, '    t_sampled.clear();', '    t_preview_sum.clear();\n    t_sampled.clear();')
    edit(graph, '    for (auto * tensor : t_sampled) {', '''    for (auto * tensor : t_preview_sum) {
        if (tensor) ggml_set_output(tensor);
    }
    for (auto * tensor : t_sampled) {''')
    edit(graph, '    res->t_sampled.resize(n_rows, nullptr);',
         '    res->t_preview_sum.resize(n_rows, nullptr);\n    res->t_sampled.resize(n_rows, nullptr);')
    edit(graph, '            struct llama_sampler_data data = {', '''            if (llama_sampler_is_greedy_preview(sampler)) {
                auto * guard = ggml_sum_rows(ctx0, logits_seq);
                ggml_set_name(guard, "greedy_preview_sum");
                if (active) res->t_preview_sum[row] = guard;
                outs[1] = guard;
                ggml_build_forward_select(gf, outs.data(), outs.size(), i_out);
            }

            struct llama_sampler_data data = {''')
    ctxh = 'src/llama-context.h'
    edit(ctxh, '    float * get_logits();', '''    float * get_logits();
    float get_preview_sum_ith(int32_t i);
    void materialize_deferred_logits();''')
    edit(ctxh, '    buffer_view<float> logits = {nullptr, 0};', '''    buffer_view<float> logits = {nullptr, 0};
    // Only a single-output logical batch may defer its raw logits. The graph
    // marks this tensor as output, so its storage survives until graph reuse.
    ggml_tensor * deferred_logits = nullptr;
    ggml_backend_t deferred_backend = nullptr;''')
    edit(ctxh, '        std::vector<uint32_t> logits_count;',
         '        buffer_view<float> preview_sum = {nullptr, 0};\n        std::vector<uint32_t> logits_count;')
    ctx = 'src/llama-context.cpp'
    edit(ctx, 'float * llama_context::get_logits() {\n', '''void llama_context::materialize_deferred_logits() {
    if (!deferred_logits) return;
    GGML_ASSERT(logits.data && deferred_backend);
    ggml_backend_tensor_get_async(deferred_backend, deferred_logits, logits.data,
                                 0, model.vocab.n_tokens()*sizeof(float));
    ggml_backend_synchronize(deferred_backend);
    deferred_logits = nullptr;
    deferred_backend = nullptr;
}

float llama_context::get_preview_sum_ith(int32_t i) {
    output_reorder();
    if (!sampling.preview_sum.has_data()) return NAN;
    return sampling.preview_sum.data[output_resolve_row(i)];
}

float * llama_context::get_logits() {
    materialize_deferred_logits();
''')
    edit(ctx, 'float * llama_context::get_logits_ith(int32_t i) {\n',
         'float * llama_context::get_logits_ith(int32_t i) {\n    materialize_deferred_logits();\n')
    # Discard old outputs once a validated new logical batch starts, before any
    # allocator/graph work. Other graph mutations must materialize first.
    edit(ctx, '    const uint32_t n_tokens_all  = balloc->get_n_tokens();',
         '    deferred_logits = nullptr;\n    deferred_backend = nullptr;\n\n    const uint32_t n_tokens_all  = balloc->get_n_tokens();')
    edit(ctx, '    const auto causal_attn_org = cparams.causal_attn;',
         '    deferred_logits = nullptr;\n    deferred_backend = nullptr;\n\n    const auto causal_attn_org = cparams.causal_attn;')
    edit(ctx, '    ggml_backend_sched_reset(sched.get());\n\n    // when the scheduler is reset,',
         '    materialize_deferred_logits();\n    ggml_backend_sched_reset(sched.get());\n\n    // when the scheduler is reset,')
    edit(ctx, '        // count the outputs in this ubatch',
         '        // A following microbatch may reuse the deferred tensor storage.\n        materialize_deferred_logits();\n\n        // count the outputs in this ubatch')
    edit(ctx, '        // extract logits\n        if (logits.data && t_logits && n_outputs > 0 && needs_raw_logits(ubatch, sampling.samplers)) {', '''        bool has_preview = false;
        for (const auto & entry : sampling.samplers) {
            has_preview |= llama_sampler_is_greedy_preview(entry.second);
        }
        // Preview fallback needs raw logits, retained on-device for one output.
        if (logits.data && t_logits && n_outputs > 0 &&
            (has_preview || needs_raw_logits(ubatch, sampling.samplers))) {''')
    edit(ctx, '                ggml_backend_tensor_get_async(backend_res, t_logits, logits_out, 0, n_outputs*n_vocab*sizeof(float));', '''                if (has_preview && n_outputs_all == 1 && n_outputs == 1 && n_outputs_prev == 0) {
                    deferred_logits = t_logits;
                    deferred_backend = backend_res;
                } else {
                    ggml_backend_tensor_get_async(backend_res, t_logits, logits_out, 0, n_outputs*n_vocab*sizeof(float));
                }''')
    edit(ctx, '            // async copy the sampling data from the backend to the host',
         '            copy_tensor_async_rows(res->t_preview_sum, sampling.preview_sum, 1, n_outputs_prev, sched.get());\n\n            // async copy the sampling data from the backend to the host')
    edit(ctx, 'backend_float_count = 2 * n_vocab * n_outputs_max;',
         'backend_float_count = (2 * n_vocab + 1) * n_outputs_max;')
    edit(ctx, '        sampling.sampled = {(llama_token *) (base + offset), (size_t)n_outputs_max};', '''        sampling.preview_sum = {(float *) (base + offset), (size_t)n_outputs_max};
        offset += sampling.preview_sum.size * sizeof(float);
        std::fill_n(sampling.preview_sum.data, sampling.preview_sum.size, NAN);

        sampling.sampled = {(llama_token *) (base + offset), (size_t)n_outputs_max};''')
    edit(ctx, '        sampling.logits     = {nullptr, 0};',
         '        sampling.preview_sum = {nullptr, 0};\n        sampling.logits     = {nullptr, 0};')
    edit(ctx, '        if (!sampling.samplers.empty()) {\n            assert(sampling.logits.size > 0);', '''        if (!sampling.samplers.empty()) {
            std::swap(sampling.preview_sum.data[i0], sampling.preview_sum.data[i1]);
            assert(sampling.logits.size > 0);''')
    edit(ctx, 'float * llama_get_embeddings(llama_context * ctx) {', '''llama_sampling_output llama_get_sampling_preview_ith(llama_context * ctx, int32_t i, float * finite_sum) {
    auto output = llama_get_sampling_output_ith(ctx, i);
    *finite_sum = ctx->get_preview_sum_ith(i);
    return output;
}

float * llama_get_raw_logits_ith(llama_context * ctx, int32_t i) {
    ctx->synchronize();
    return ctx->get_logits_ith(i);
}

float * llama_get_embeddings(llama_context * ctx) {''')
    common = 'common/sampling.cpp'
    edit(common, '#include <cstring>', '#include <cstring>\n#include <cstdlib>')
    edit(common, '    struct llama_sampler * chain;\n', '    struct llama_sampler * chain;\n    struct llama_sampler * preview;\n')
    edit(common, 'struct common_sampler * common_sampler_init(', '''static bool can_preview_greedy(const common_params_sampling & p, const llama_vocab * vocab) {
    int32_t suppressed = 0;
    llama_vocab_get_suppress_tokens(vocab, &suppressed);
    if (suppressed || !p.logit_bias.empty() || p.n_probs != 0 || p.mirostat != 0 ||
        p.temp != 0.0f || p.dynatemp_range != 0.0f || p.samplers.empty() ||
        p.samplers.back() != COMMON_SAMPLER_TYPE_TEMPERATURE || llama_vocab_n_tokens(vocab) < 2) return false;
    for (auto s : p.samplers) {
        switch (s) {
            case COMMON_SAMPLER_TYPE_TEMPERATURE: break;
            case COMMON_SAMPLER_TYPE_TOP_K: break;
            case COMMON_SAMPLER_TYPE_TOP_P: if (!(p.top_p > 0 && p.top_p <= 1)) return false; break;
            case COMMON_SAMPLER_TYPE_MIN_P: if (!(p.min_p >= 0 && p.min_p <= 1)) return false; break;
            case COMMON_SAMPLER_TYPE_TYPICAL_P: if (!(p.typ_p >= 1)) return false; break;
            case COMMON_SAMPLER_TYPE_TOP_N_SIGMA: if (!(p.top_n_sigma < 0)) return false; break;
            case COMMON_SAMPLER_TYPE_DRY: if (p.dry_multiplier != 0) return false; break;
            case COMMON_SAMPLER_TYPE_XTC: if (p.xtc_probability != 0) return false; break;
            case COMMON_SAMPLER_TYPE_PENALTIES:
                if (p.penalty_repeat != 1 || p.penalty_freq != 0 || p.penalty_present != 0) return false;
                break;
            default: return false;
        }
    }
    return true;
}

struct common_sampler * common_sampler_init(''')
    edit(common, '    if (grmr && params.backend_sampling) {', '''    llama_sampler * preview = nullptr;
    const char * enable_preview = std::getenv("LLAMA_WORD_GAMES_GREEDY_PREVIEW");
    if (enable_preview && std::string(enable_preview) == "1" && grmr && !rbudget &&
        can_preview_greedy(params, vocab)) {
        preview = llama_sampler_chain_init(lparams);
        llama_sampler_chain_add(preview, llama_sampler_init_greedy_preview());
        params.backend_sampling = true;
        LOG_INF("%s: greedy grammar preview enabled (full logits on demand)\\n", __func__);
    }

    if (grmr && params.backend_sampling && !preview) {''')
    edit(common, '        /* .chain   = */ chain,', '        /* .chain   = */ chain,\n        /* .preview = */ preview,')
    edit(common, '    llama_sampler_free(gsmpl->chain);',
         '    llama_sampler_free(gsmpl->chain);\n    llama_sampler_free(gsmpl->preview);')
    edit(common, '        /* .chain   = */ llama_sampler_clone(gsmpl->chain),',
         '        /* .chain   = */ llama_sampler_clone(gsmpl->chain),\n        /* .preview = */ llama_sampler_clone(gsmpl->preview),')
    edit(common, '    llama_sampler_copy(src->chain,   dst->chain);',
         '    llama_sampler_copy(src->chain,   dst->chain);\n    GGML_ASSERT((src->preview == nullptr) == (dst->preview == nullptr));\n    llama_sampler_copy(src->preview, dst->preview);')
    edit(common, '    return gsmpl->chain;\n', '    return gsmpl->preview ? gsmpl->preview : gsmpl->chain;\n')
    edit(common, '    const llama_sampling_output output = llama_get_sampling_output_ith(ctx, idx);', '''    float finite_sum = NAN;
    llama_sampling_output output = gsmpl->preview ? llama_get_sampling_preview_ith(ctx, idx, &finite_sum) :
                                                  llama_get_sampling_output_ith(ctx, idx);
    const auto tm = gsmpl->tm();
    if (gsmpl->preview) {
        if (!grammar_first && std::isfinite(finite_sum) && output.sampled_logits_count == 2 &&
            output.sampled_candidates_count == 2 && output.sampled_logits && output.sampled_candidates &&
            std::isfinite(output.sampled_logits[0]) && std::isfinite(output.sampled_logits[1]) &&
            output.sampled_logits[0] > output.sampled_logits[1]) {
            const auto id = output.sampled_candidates[0];
            llama_token_data one = {id, output.sampled_logits[0], 0.0f};
            llama_token_data_array check = {&one, 1, -1, false};
            if (grammar_should_apply(gsmpl)) llama_sampler_apply(gsmpl->grmr, &check);
            if (one.logit != -INFINITY) {
                gsmpl->cur.assign(1, one);
                gsmpl->cur_p = {gsmpl->cur.data(), 1, 0, true};
                return id;
            }
            // The CPU chain is stateless and greedy in this mode. This is the
            // same second attempt as ordinary grammar rejection sampling.
            grammar_first = true;
        }
        const auto n_vocab = output.n_vocab;
        output = {};
        output.n_vocab = n_vocab;
        output.sampled_token = LLAMA_TOKEN_NULL;
        output.logits = llama_get_raw_logits_ith(ctx, idx);
    }''')
    edit(common, '    // Exclude waiting for asynchronous work and acquiring its output view.\n    const auto tm = gsmpl->tm();\n', '')
    patch = ''.join(''.join(difflib.unified_diff(originals[name].splitlines(True), text.splitlines(True),
                    fromfile='a/'+name, tofile='b/'+name)) for name, text in texts.items())
    DEST.write_text(patch)
    print(f'{DEST}: {len(texts)} files, {len(patch)} bytes')


if __name__ == '__main__':
    main()
