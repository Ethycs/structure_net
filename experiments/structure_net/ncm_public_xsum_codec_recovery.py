"""Use the verified nested codec state; preserve original selection and failure."""
from experiments.structure_net.ncm_public_xsum_codec import *
from experiments.structure_net import ncm_public_xsum_codec as original

AMENDMENT = Path('docs/07 - Status Reports/2026-10-26_ncm-public-xsum-codec-recovery.md')
RECOVERY_TEST = Path('tests/structure_net/test_ncm_public_xsum_codec_recovery.py')
INCIDENT = prior.ROOT/'20261026_public_xsum_codec_pilot_launch_failure_v1'
RUNS = {**original.RUNS, **{s: prior.ROOT/f'20261026_public_xsum_codec_{s}_v2' for s in ('pilot', 'eval', 'results')}}
PREFIX = 'concepts.codec.'


def codec_state(model_state, expected_keys):
    selected = {k[len(PREFIX):]: v for k, v in model_state.items() if k.startswith(PREFIX)}
    if not selected or set(selected) != set(expected_keys):
        raise ValueError('exact136-key nested codec extraction required')
    return selected


def freeze(root, stage, training, environment, inputs):
    return original.freeze(root, stage, training, environment, [*inputs, AMENDMENT, RECOVERY_TEST,
        *[INCIDENT/n for n in ('failed_launch_actual_tool_receipt.json', 'state_key_diagnostic.json',
                              'correct_prefix_diagnostic.json', 'actual_tool_receipt.json')]])


def measure(stage):
    training, environment = prior.runtime()
    import torch
    from experiments.structure_net.ncm_cpu_adamw import require_rtx2060
    require_rtx2060()
    diagnostic = prior.read(INCIDENT/'correct_prefix_diagnostic.json')
    if (diagnostic != {'prefix': PREFIX, 'codec_keys': 136, 'codec_sha256': CODEC_SHA}
            or prior.read(INCIDENT/'actual_tool_receipt.json')['actual_tool_exit_code'] != 0):
        raise ValueError('verified exact nested codec state required')
    public.completed(RUNS['prepare'])
    training.preparation.completed(BROAD)
    if prior.sha(BROAD/'model.pt') != CHECKPOINT_SHA: raise ValueError('fixed BROAD checkpoint required')
    all_rows = prior.read(RUNS['prepare']/'cases.json')
    pilots = prior.read(RUNS['prepare']/'selection.json')['pilot_ids']
    rows = [r for r in all_rows if r['id'] in pilots or r['split'] == 'TRAIN_control'] if stage == 'pilot' else all_rows
    controls = {(r['id'], r['arm']): r for r in prior.read(RUNS['prepare']/'control_outputs.json')}
    pilot = {}
    if stage == 'eval':
        public.completed(RUNS['pilot'])
        pilot = {(r['id'], r['arm']): r for r in prior.read(RUNS['pilot']/'outputs.json')}
    codec = training.native.load_codec()
    tokenizer = training.native.tokenizer()
    released_hash = training.native.vendor.module_hash(codec)
    released_state = {k: v.detach().cpu().clone() for k, v in codec.state_dict().items()}
    saved = torch.load(BROAD/'model.pt', map_location='cpu', weights_only=True, mmap=True)
    repaired_state = codec_state(saved['model'], released_state)
    inputs = [*RUNS['prepare'].glob('*.json'), *[BROAD/n for n in ('model.pt', 'result.json', 'actual_tool_receipt.json')],
        training.native.RELEASE/'files/lm1b/tokenizer.pt', training.native.RELEASE/'files/lm1b/small/config.yaml']
    if stage == 'eval': inputs += [RUNS['pilot']/n for n in ('outputs.json', 'result.json', 'actual_tool_receipt.json')]
    proof = freeze(RUNS[stage], stage, training, environment, inputs)
    torch.cuda.reset_peak_memory_stats()
    outputs, previous_codes, model_hashes = [], {}, {}
    with torch.no_grad():
        for arm in ARMS:
            codec.load_state_dict(released_state if arm == 'released' else repaired_state, strict=True)
            codec.float().eval().requires_grad_(False)
            before = training.native.vendor.module_hash(codec)
            if before != (released_hash if arm == 'released' else CODEC_SHA): raise ValueError('selected codec identity differs')
            model_hashes[arm] = before
            for row in rows:
                gold = torch.tensor([row['native_word_ids']], device='cuda', dtype=torch.long)
                codes = codec.encode(gold)
                as_list = codes[0].cpu().tolist()
                if len(as_list) != 641 or any(not 0 <= c < 8192 for c in as_list): raise ValueError('complete code geometry required')
                if row['id'] in previous_codes and previous_codes[row['id']] != as_list: raise ValueError('encoder/VQ differs between arms')
                previous_codes[row['id']] = as_list
                def predict():
                    logits = codec.decode(codes)
                    if logits.shape != (1, 128, 30528) or not torch.isfinite(logits).all(): raise ValueError('finite full logits required')
                    return logits.argmax(-1)[0].cpu().tolist()
                raw = predict()
                if stage == 'pilot' and (raw != predict() or not torch.equal(codes, codec.encode(gold))): raise ValueError('repeated pilot differs')
                reference = controls.get((row['id'], arm))
                if reference is not None and (raw != reference['raw_word_ids'] or as_list != reference['codes']): raise ValueError('standalone codec differs from prior control')
                decoded = training.preparation.rendering.decode(raw, tokenizer)
                value = {'id': row['id'], 'split': row['split'], 'arm': arm, 'target_sha256': row['target_sha256'],
                    'codes': as_list, 'raw_word_ids': raw, 'text': decoded['text'],
                    'invalid_outputs': int(any(v >= len(tokenizer) for v in raw)),
                    'output_native_tokens': len(tokenizer.encode(decoded['text'], add_special_tokens=False)),
                    **prior.metrics(row['native_word_ids'], raw, row['native_reference'], decoded['text'])}
                if (row['id'], arm) in pilot and pilot[row['id'], arm] != value: raise ValueError('pilot/full output differs')
                outputs.append(value)
            if training.native.vendor.module_hash(codec) != before: raise ValueError('inference changed codec tensors')
            print(json.dumps({'arm': arm, 'cases': len(rows), 'outputs': len(outputs)}), flush=True)
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated()
    if peak > 4900000000: raise ValueError('CUDA allocation cap exceeded')
    public.write(RUNS[stage]/'outputs.json', outputs)
    prior.verify(proof)
    result = {'status': 'frozen_public_TRAIN_codec_probe_complete', 'cases': len(rows), 'outputs': len(outputs),
        'codec_hashes': model_hashes, 'models_unchanged': True, 'all_control_outputs_identical': True,
        'pilot_repeat_exact': stage == 'pilot', 'peak_allocated_bytes': peak, **BOUNDARIES,
        'metrics': aggregate(outputs) if stage == 'eval' else None,
        'artifacts': {str(p): prior.sha(p) for p in RUNS[stage].glob('*.json')}}
    public.write(RUNS[stage]/'result.json', result)
    print(json.dumps({k: v for k, v in result.items() if k not in ('artifacts', 'metrics')}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('stage', choices=('pilot', 'eval'))
    stage = parser.parse_args().stage
    root = RUNS[stage]; existed = root.exists()
    try:
        measure(stage)
    except Exception as error:
        if not existed and root.exists():
            public.write(root/'failure.json', {'error_type': type(error).__name__, 'error': str(error), **BOUNDARIES})
        raise
