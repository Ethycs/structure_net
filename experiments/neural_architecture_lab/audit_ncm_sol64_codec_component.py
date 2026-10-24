"""CPU audit of frozen codec bindings and complete outputs, without semantic claims."""
import json
from pathlib import Path

import torch

from experiments.structure_net import ncm_sol64_codec_component as run

DEST = run.ROOT / '20261024_sol64_codec_component_audit_v1'


def main():
    run.configure()
    if DEST.exists() or torch.cuda.is_initialized():
        raise ValueError('fresh CPU codec audit required')
    if run.q.read(run.WORKFLOW / 'actual_tool_receipt.json')['actual_tool_exit_code'] != 0:
        raise ValueError('actual whole codec workflow exit0 required')
    whole = run.q.read(run.WORKFLOW / 'result.json')
    if ([r['stage'] for r in whole['stages']] != ['prepare', 'pilot', 'generate']
            or any(r['exit_code'] != 0 for r in whole['stages'])):
        raise ValueError('all numeric component stages must complete')
    for root in (run.PREP, run.PILOT, run.OUTPUTS):
        run.native.vendor.terminal(root)
    rows = run.cases()
    if (run.q.read(run.PREP / 'cases.json') != rows
            or run.q.read(run.PREP / 'split_proof.json') != run.text_control.split_guard(rows)):
        raise ValueError('all original first target/source/frame bindings differ')
    OmegaConf, VQVAE, _ = run.native.vendor.vendor_classes()
    cfg = OmegaConf.load(run.native.RELEASE / 'files/lm1b/small/config.yaml')
    codec = VQVAE(**cfg.tokenizer.vqvae).eval().requires_grad_(False)
    codec.load_state_dict(torch.load(run.native.RELEASE / 'files/lm1b/tokenizer.pt',
        map_location='cpu', weights_only=True, mmap=True)['model'], strict=True)
    hashes = {'codec': run.native.vendor.module_hash(codec)}
    codec.decoder.load_state_dict(torch.load(run.grounded.DOMAIN_CHECKPOINT,
        map_location='cpu', weights_only=True, mmap=True)['decoder'], strict=True)
    hashes['domain_decoder'] = run.native.vendor.module_hash(codec.decoder)
    del codec
    tokenizer = run.native.tokenizer()
    outputs = run.q.read(run.OUTPUTS / 'outputs.json')
    if [r['id'] for r in outputs] != [r['id'] for r in rows]:
        raise ValueError('exact ordered64 first outputs required')
    by_id = {r['id']: r for r in outputs}
    for output in run.q.read(run.PILOT / 'outputs.json'):
        if by_id[output['id']] != output:
            raise ValueError('full inference changed a pilot first output')
    for root, size in ((run.PILOT, 2), (run.OUTPUTS, 64)):
        state = run.q.read(root / 'result.json')
        if (state['cases'] != size or not state['models_unchanged'] or state['model_hashes'] != hashes
                or state['peak_allocated_bytes'] > 1_500_000_000
                or state['outputs_sha256'] != run.file_hash(root / 'outputs.json')
                or state['student_task_accuracy'] is not None or state['semantic_accuracy'] is not None):
            raise ValueError('frozen model hashes, complete outputs or stated limits differ')
    metrics = {arm: {'correct_active_word_ids': 0, 'active_word_ids': 0, 'exact_active_frames': 0,
        'missing_EOS': 0} for arm in ('released', 'domain')}
    for row, output in zip(rows, outputs):
        active = sum(v != 0 for v in row['native_word_ids'])
        if (output['source_sha256'] != row['source_sha256'] or output['lexical_positions'] != active - 2
                or len(output['codes']) != 641 or any(type(v) is not int or not 0 <= v < 8192 for v in output['codes'])
                or set(output['conditions']) != set(metrics)):
            raise ValueError('complete same-code oracle conditioning differs')
        for arm, value in output['conditions'].items():
            raw = value['raw_word_ids']
            if len(raw) != 128 or any(type(v) is not int or not 0 <= v < 30528 for v in raw):
                raise ValueError('full raw native output vocabulary required')
            text, ends, bos = run.framed.render([raw], tokenizer)
            expected = {'text': text, 'raw_word_ids': raw, 'predicted_BOS': bos[0],
                'predicted_EOS_position': ends[0], 'missing_EOS': ends[0] == 128}
            if value != expected:
                raise ValueError('native output rendering changed or repaired')
            metric = metrics[arm]
            metric['correct_active_word_ids'] += sum(a == b for a, b in zip(raw[:active], row['native_word_ids'][:active]))
            metric['active_word_ids'] += active
            metric['exact_active_frames'] += raw[:active] == row['native_word_ids'][:active]
            metric['missing_EOS'] += value['missing_EOS']
    artifacts = {str(Path(__file__)): run.file_hash(__file__)}
    for root in (run.WORKFLOW, run.PREP, run.PILOT, run.OUTPUTS):
        for path in root.rglob('*.json'):
            if 'producer_source' not in path.parts:
                artifacts[str(path)] = run.file_hash(path)
    result = {'status': 'all64_fixed_codec_output_bindings_and_rendering_replayed_semantics_pending',
        'cases': 64, 'model_hashes': hashes, 'word_ID_diagnostics_only': metrics,
        'oracle_target_codes': True, 'source_only': False, 'semantic_accuracy': None,
        'student_task_accuracy': None, 'final_test_opened': False, 'speed_tested': False, 'artifacts': artifacts}
    DEST.mkdir()
    run.write_json(DEST / 'result.json', result)
    print(json.dumps({k: v for k, v in result.items() if k != 'artifacts'}), flush=True)


if __name__ == '__main__':
    main()
