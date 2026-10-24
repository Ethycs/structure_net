"""Replay the same trained state, source payloads and every oracle output binding."""
import json
from pathlib import Path

import torch

from experiments.structure_net import ncm_sol64_grounded_oracle as run

DEST = run.ROOT / '20261024_sol64_grounded_oracle_audit_v1'


def main():
    run.configure()
    if DEST.exists() or torch.cuda.is_initialized():
        raise ValueError('fresh CPU matched-oracle audit required')
    if run.q.read(run.WORKFLOW / 'actual_tool_receipt.json')['actual_tool_exit_code'] != 0:
        raise ValueError('actual completed whole matched-oracle workflow required')
    whole = run.q.read(run.WORKFLOW / 'result.json')
    if ([r['stage'] for r in whole['stages']] != ['prepare', 'pilot', 'generate']
            or any(r['exit_code'] != 0 for r in whole['stages'])):
        raise ValueError('all matched-oracle stages must complete')
    for root in (run.PREP, run.PILOT, run.OUTPUTS):
        run.native.vendor.terminal(root)
    rows = run.inputs()
    if (run.q.read(run.PREP / 'cases.json') != rows
            or run.q.read(run.PREP / 'split_proof.json') != run.numeric.text_control.split_guard(rows)):
        raise ValueError('all original64 source/code/feature bindings must remain')
    model, state = run.load_selected('cpu')
    model_hash = run.native.vendor.module_hash(model)
    model.close()
    del model
    outputs, pilot = run.q.read(run.OUTPUTS / 'outputs.json'), run.q.read(run.PILOT / 'outputs.json')
    if [r['id'] for r in outputs] != [r['id'] for r in rows]:
        raise ValueError('every ordered original64 oracle output required')
    by_id = {r['id']: r for r in outputs}
    if any(by_id[r['id']] != r for r in pilot):
        raise ValueError('a retained pilot output changed')
    for root, size in ((run.PILOT, 2), (run.OUTPUTS, 64)):
        value = run.q.read(root / 'result.json')
        config = run.q.read(root / 'producer.json')['config']
        if (value['cases'] != size or value['model_sha256'] != model_hash or not value['model_unchanged']
                or value['checkpoint_sha256'] != state['checkpoint_sha256']
                or state['checkpoint_sha256'] != run.CHECKPOINT_SHA
                or value['outputs_sha256'] != run.file_hash(root / 'outputs.json')
                or value['peak_allocated_bytes'] > 3_000_000_000
                or value['student_task_accuracy'] is not None or value['semantic_accuracy'] is not None
                or config['precision'] != 'FP32' or config['TF32'] or config['physical_GPU'] != 0
                or config['concept_generator_called'] or config['source_only']):
            raise ValueError('same frozen checkpoint, output population or inference boundary differs')
    tokenizer = run.native.tokenizer()
    metrics = {arm: {'correct_active_word_IDs': 0, 'active_word_IDs': 0, 'exact_active_frames': 0,
        'missing_EOS': 0} for arm in run.ARMS}
    changed = 0
    for row, output in zip(rows, outputs):
        metadata = run.q.read(row['source_metadata']['path'])
        source = run.features.load_feature(row['feature'], metadata)
        if (output['source_sha256'] != row['source_sha256'] or output['oracle_codes'] != row['oracle_codes']
                or output['source_positions'] != len(source)
                or output['feature_payload_sha256'] != row['feature']['payload_sha256']
                or set(output['conditions']) != set(run.ARMS)):
            raise ValueError('source positions or identical oracle-code conditioning differs')
        active = sum(v != 0 for v in row['native_word_ids'])
        for arm, value in output['conditions'].items():
            raw = value['raw_word_ids']
            if len(raw) != 128 or any(type(v) is not int or not 0 <= v < 30528 for v in raw):
                raise ValueError('full native output vocabulary required')
            text, ends, bos = run.numeric.framed.render([raw], tokenizer)
            expected = {'text': text, 'raw_word_ids': raw, 'predicted_BOS': bos[0],
                'predicted_EOS_position': ends[0], 'missing_EOS': ends[0] == 128}
            if value != expected:
                raise ValueError('full native output rendering was changed or repaired')
            metrics[arm]['correct_active_word_IDs'] += sum(a == b for a, b in zip(raw[:active], row['native_word_ids'][:active]))
            metrics[arm]['active_word_IDs'] += active
            metrics[arm]['exact_active_frames'] += raw[:active] == row['native_word_ids'][:active]
            metrics[arm]['missing_EOS'] += value['missing_EOS']
        changed += output['conditions']['source_off']['raw_word_ids'] != output['conditions']['source_on']['raw_word_ids']
    artifacts = {str(Path(__file__)): run.file_hash(__file__)}
    for root in (run.WORKFLOW, run.PREP, run.PILOT, run.OUTPUTS):
        for path in root.rglob('*.json'):
            if 'producer_source' not in path.parts:
                artifacts[str(path)] = run.file_hash(path)
    for row in rows:
        for key in ('header_path', 'payload_path'):
            path = row['feature'][key]
            artifacts[path] = run.file_hash(path)
    result = {'status': 'all64_matched_decoder_source_code_and_rendering_bindings_replayed_semantics_pending',
        'cases': 64, 'checkpoint_sha256': state['checkpoint_sha256'], 'model_sha256': model_hash,
        'word_ID_diagnostics_only': metrics, 'source_attention_changed_raw_outputs': changed,
        'oracle_target_codes': True, 'source_only': False, 'student_task_accuracy': None,
        'semantic_accuracy': None, 'final_test_opened': False, 'speed_tested': False, 'artifacts': artifacts}
    DEST.mkdir()
    run.write_json(DEST / 'result.json', result)
    print(json.dumps({k: v for k, v in result.items() if k != 'artifacts'}), flush=True)


if __name__ == '__main__':
    main()
