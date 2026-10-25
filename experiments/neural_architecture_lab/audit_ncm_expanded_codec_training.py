"""Replay every sample, precision boundary and final decoder state on CPU."""
from dataclasses import asdict
import json
import math
from pathlib import Path

import torch

from experiments.structure_net import ncm_expanded_codec_training as run

DEST = run.ROOT / '20261024_expanded9936_codec_training_audit_v1'


def expected_microbatch(raw, author, index, sampler, augmentation):
    descriptors = []
    for _ in range(4):
        if torch.rand((), generator=sampler).item() < .2:
            row = int(torch.randint(len(author), (), generator=sampler))
            frame = author[row]
            info = {'kind': 'author_TRAIN', 'row': row}
        else:
            source = int(torch.randint(len(index), (), generator=sampler))
            window = int(torch.randint(index[source]['frame_count'], (), generator=sampler))
            row = index[source]['frame_start'] + window
            frame = raw[row]
            info = {'kind': 'raw_source', 'source_index': source, 'window': window, 'row': row}
        if torch.rand((), generator=augmentation).item() < .2:
            value, mode = frame.tolist(), {'mode': 'full'}
        else:
            lexical = [v for v in frame.tolist()[1:-1] if v not in (0, 101, 102)]
            length = int(torch.randint(1, len(lexical)+1, (), generator=augmentation))
            value, mode = [101, *lexical[:length], 101]+[0]*(126-length), {'mode': 'prefix', 'length': length}
        descriptors.append({**info, **mode, 'target_sha256': run.q.digest(value)})
    return descriptors


def main():
    run.corpus.component.configure()
    run.require_corpus()
    if DEST.exists() or torch.cuda.is_initialized() or run.q.read(run.WORKFLOW / 'actual_tool_receipt.json')['actual_tool_exit_code'] != 0:
        raise ValueError('fresh CPU audit after actual whole training exit0 required')
    run.native.vendor.verify(run.q.read(run.WORKFLOW / 'producer.json'))
    whole = run.q.read(run.WORKFLOW / 'result.json')
    if [r['stage'] for r in whole['stages']] != ['pilot', 'train'] or any(r['exit_code'] != 0 for r in whole['stages']):
        raise ValueError('actual pilot and complete fixed training required')
    raw = torch.load(run.corpus.DATA / 'frames.pt', map_location='cpu', weights_only=True)
    author = torch.load(run.domain.AUTHOR_DATA / 'frames.pt', map_location='cpu', weights_only=True)
    index = run.q.read(run.corpus.DATA / 'index.json')
    results = {}
    for name, root, updates in (('pilot', run.PILOT, 2), ('full', run.TRAIN, 1024)):
        run.native.vendor.terminal(root)
        producer, result = run.q.read(root / 'producer.json'), run.q.read(root / 'result.json')
        if (result['updates'] != updates or result['initial_decoder_sha256'] != run.WARM_DECODER_SHA
                or result['initial_decoder_sha256'] == result['final_decoder_sha256']
                or not result['exact_reload'] or result['peak_allocated_bytes'] > 2_000_000_000
                or not result['component64_excluded'] or not result['permanent_DEV_disjoint']
                or result['source_only_summary_accuracy'] is not None or result['canonical_semantic_cache_changed']
                or result['final_test_opened'] or result['speed_tested']
                or producer['config']['precision'] != 'FP32weights/FP32encoder_codes/FP16decoder_training/FP32hardinference'
                or producer['config']['physical_GPU'] != 1 or producer['config']['TF32']
                or run.file_hash(root / 'model.pt') != result['checkpoint_sha256']):
            raise ValueError('decoder-only state, precision or evaluation boundary differs')
        trace = [json.loads(line) for line in (root / 'trace.jsonl').read_text().splitlines()]
        if len(trace) != updates:
            raise ValueError('every update required')
        sampler, augmentation = torch.Generator().manual_seed(137142), torch.Generator().manual_seed(138142)
        for step, value in enumerate(trace, 1):
            expected = [r for _ in range(8) for r in expected_microbatch(raw, author, index, sampler, augmentation)]
            if step <= 64:
                lr = 2e-5*step/64
            else:
                lr = 2e-6 + .5*(2e-5-2e-6)*(1+math.cos(math.pi*(step-64)/(1024-64)))
            if (value['step'] != step or value['samples'] != expected or not math.isclose(value['lr'], lr, rel_tol=1e-14, abs_tol=1e-20)
                    or not math.isfinite(value['loss']) or value['loss'] < 0
                    or not math.isfinite(value['gradient_norm']) or value['gradient_norm'] < 0
                    or value['scaler']['scale'] != 128. or value['scaler']['_growth_tracker'] != step):
                raise ValueError('actual32example update, LR or FP16 step replay differs')
        saved = torch.load(root / 'model.pt', map_location='cpu', weights_only=True, mmap=True)
        if (saved['step'] != updates or saved['config'] != asdict(run.Config())
                or saved['producer_fingerprint'] != producer['fingerprint']
                or saved['scaler'] != trace[-1]['scaler'] or saved['scaler'] != result['scaler']
                or not torch.equal(saved['sampler_rng'], sampler.get_state())
                or not torch.equal(saved['augmentation_rng'], augmentation.get_state())):
            raise ValueError('final optimizer/scaler/sampler state differs')
        for state in saved['optimizer']['state'].values():
            if int(state['step']) != updates or any(not torch.isfinite(v).all() for v in state.values() if isinstance(v, torch.Tensor)):
                raise ValueError('every Adam parameter must have exact finite update count')
        model = run.load_codec('cpu')
        protected = run.lifecycle.frozen_hash(model)
        model.decoder.load_state_dict(saved['decoder'], strict=True)
        if (run.native.vendor.module_hash(model.decoder) != result['final_decoder_sha256']
                or run.lifecycle.frozen_hash(model) != protected or saved['protected_sha256'] != protected
                or result['protected_sha256_unchanged'] != protected
                or sum(p.numel() for p in model.decoder.parameters()) != result['decoder_trainable_parameters']):
            raise ValueError('full saved decoder or protected native codec state differs')
        del model
        results[name] = {'updates': updates, 'checkpoint_sha256': result['checkpoint_sha256'],
            'decoder_sha256': result['final_decoder_sha256'], 'protected_sha256': protected,
            'samples_replayed': updates*32, 'peak_allocated_bytes': result['peak_allocated_bytes']}
    if run.q.read(run.PILOT / 'initial_probes.json') != run.q.read(run.TRAIN / 'initial_probes.json'):
        raise ValueError('full run must use original domain initialization, never pilot')
    artifacts = {str(Path(__file__)): run.file_hash(__file__)}
    for root in (run.WORKFLOW, run.PILOT, run.TRAIN):
        for path in root.rglob('*'):
            if path.is_file() and 'producer_source' not in path.parts:
                artifacts[str(path)] = run.file_hash(path)
    DEST.mkdir()
    result = {'status': 'expanded9936_FP16_decoder_samples_and_full_state_replayed_semantics_pending',
        'results': results, 'source_only_summary_accuracy': None, 'component_semantic_accuracy': None,
        'final_test_opened': False, 'speed_tested': False, 'artifacts': artifacts}
    run.write_json(DEST / 'result.json', result)
    print(json.dumps({k: v for k, v in result.items() if k != 'artifacts'}), flush=True)


if __name__ == '__main__':
    main()
