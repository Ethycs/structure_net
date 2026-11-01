"""Reduce preview selection work and retain its stateless graph across requests."""
from pathlib import Path
import difflib


def main():
    root=Path('.pixi/word-games-grammar-v2/source')
    changes={}
    name='src/llama-sampler.cpp'
    old=(root/name).read_text()
    block='''// Two global winners must occur among the two winners of their 128-token
// groups. This avoids sorting/radix-selecting a whole vocabulary in one row.
// Ties and non-finite values are still rejected by the CPU preview guard.
static void greedy_preview_backend_apply(llama_sampler * smpl, ggml_context * ctx,
                                         ggml_cgraph * gf, llama_sampler_data * data) {
    const int64_t n = ggml_nelements(data->logits);
    const int64_t width = 128;
    if (n % width || n > (1 << 24) || n <= width || data->candidates) {
        llama_sampler_top_k_backend_apply(smpl, ctx, gf, data);
        return;
    }
    const int64_t groups = n/width;
    auto * rows = ggml_reshape_2d(ctx, data->logits, width, groups);
    auto * local_ids = ggml_top_k(ctx, rows, 2);
    auto * local_values = ggml_get_rows(ctx,
        ggml_reshape_3d(ctx, data->logits, 1, width, groups), local_ids);
    auto * winners = ggml_top_k(ctx, ggml_reshape_1d(ctx, local_values, 2*groups), 2);
    auto * local_winners = ggml_get_rows(ctx, ggml_reshape_2d(ctx, local_ids, 1, 2*groups), winners);
    // IDs are nonnegative and below 2^24, so these casts/scales are exact.
    auto * group_ids = ggml_cast(ctx, ggml_scale(ctx, ggml_cast(ctx, winners, GGML_TYPE_F32), 0.5f), GGML_TYPE_I32);
    auto * offsets = ggml_scale(ctx, ggml_cast(ctx, group_ids, GGML_TYPE_F32), float(width));
    data->candidates = ggml_cast(ctx, ggml_add(ctx,
        ggml_reshape_1d(ctx, ggml_cast(ctx, local_winners, GGML_TYPE_F32), 2), offsets), GGML_TYPE_I32);
    data->logits = ggml_get_rows(ctx, ggml_reshape_2d(ctx, local_values, 1, 2*groups), winners);
    ggml_set_name(data->candidates, "top_k");
    ggml_set_name(data->logits, "top_k_rows");
}

'''
    text=old.replace('static llama_sampler * greedy_preview_clone(',block+'static llama_sampler * greedy_preview_clone(')
    assert text!=old
    text=text.replace('    iface.clone = greedy_preview_clone;',
                      '    iface.clone = greedy_preview_clone;\n    iface.backend_apply = greedy_preview_backend_apply;')
    changes[name]=(old,text)
    name='common/sampling.cpp'
    old=(root/name).read_text()
    # Backend top-k promises a candidate set, not a descending order.
    text=old.replace('output.sampled_logits[0] > output.sampled_logits[1]) {',
                     'output.sampled_logits[0] != output.sampled_logits[1]) {')
    text=text.replace('''            const auto id = output.sampled_candidates[0];
            llama_token_data one = {id, output.sampled_logits[0], 0.0f};''',
                     '''            const int best = output.sampled_logits[1] > output.sampled_logits[0] ? 1 : 0;
            const auto id = output.sampled_candidates[best];
            llama_token_data one = {id, output.sampled_logits[best], 0.0f};''')
    assert text!=old
    changes[name]=(old,text)
    name='src/llama-context.h'
    old=(root/name).read_text()
    text=old.replace('        std::map<llama_seq_id, llama_sampler *> samplers;',
        '''        std::map<llama_seq_id, llama_sampler *> samplers;
        // Context-owned, fixed top-2 previews survive per-request CPU samplers.
        std::map<llama_seq_id, std::shared_ptr<llama_sampler>> previews;''')
    assert text!=old
    changes[name]=(old,text)
    name='src/llama-context.cpp'
    old=(root/name).read_text()
    text=old.replace('    const bool can_offload =\n', '''    if (llama_sampler_is_greedy_preview(sampler)) {
        auto & owned = sampling.previews[seq_id];
        if (owned) {
            const auto active = sampling.samplers.find(seq_id);
            if (active == sampling.samplers.end() || active->second != owned.get()) {
                sampling.samplers[seq_id] = owned.get();
                sched_need_reserve = true;
            }
            // Backend initialization is once per owned sampler, including
            // reactivation after an ordinary CPU-sampled request.
            return true;
        }
        owned = std::shared_ptr<llama_sampler>(
            llama_sampler_chain_init(llama_sampler_chain_default_params()), llama_sampler_free);
        llama_sampler_chain_add(owned.get(), llama_sampler_init_greedy_preview());
        sampler = owned.get();
    }

    const bool can_offload =
''')
    assert text!=old
    changes[name]=(old,text)
    output=Path('experiments/word_games/native_sampling/grammar-partitioned-preview.patch')
    output.write_text(''.join(''.join(difflib.unified_diff(old.splitlines(True),new.splitlines(True),
                     fromfile='a/'+name,tofile='b/'+name)) for name,(old,new) in changes.items()))
    print(output)


if __name__=='__main__':main()
