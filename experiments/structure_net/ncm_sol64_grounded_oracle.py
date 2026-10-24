"""Matched oracle-code decoding with/without the same trained source attention."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

import torch

from experiments.structure_net import ncm_sol64_codec_component as numeric
from experiments.structure_net import ncm_lm1b_native_development as selected
from experiments.structure_net import ncm_additional_source_features as features
from experiments.structure_net.ncm_memory_data import file_hash, write_json

native, q, ROOT = numeric.native, numeric.q, numeric.ROOT
HYPOTHESIS = 'ncm-sol64-source-grounded-oracle-decoder-v1'
MODULE = 'experiments.structure_net.ncm_sol64_grounded_oracle'
PLAN = Path('docs/07 - Status Reports/2026-10-24_ncm-sol64-grounded-oracle-plan.md')
TEST = Path('tests/structure_net/test_ncm_sol64_grounded_oracle.py')
AUDITOR = Path('experiments/neural_architecture_lab/audit_ncm_sol64_grounded_oracle.py')
CHECKPOINT_SHA = '49a661bfe52fa677cbf3262a58616bf1b469064ae45c4606a693baf1999b2f90'
FEATURE_AUDIT = ROOT / '20261024_additional10000_source_features_audit_v1'
NUMERIC_AUDIT = ROOT / '20261024_sol64_codec_component_audit_v1'
WORKFLOW, PREP, PILOT, OUTPUTS = (ROOT / ('20261024_sol64_grounded_oracle_' + n + '_v1')
    for n in ('workflow', 'prepare', 'pilot', 'outputs'))
ARMS = ('source_off', 'source_on')


def configure():
    native.vendor.corpus.quality.dependencies()


def require_features():
    if (q.read(features.WORKFLOW / 'tool_terminal_receipt.json')['exit_code'] != 0
            or q.read(FEATURE_AUDIT / 'actual_tool_terminal_receipt.json')['exit_code'] != 0):
        raise ValueError('actual completed whole feature workflow and independent audit required')
    native.vendor.terminal(features.FEATURES)
    state, audit = q.read(features.FEATURES / 'result.json'), q.read(FEATURE_AUDIT / 'result.json')
    manifest = q.read(features.FEATURES / 'manifest.json')
    if (state['cases'] != 10000 or state['manifest_sha256'] != file_hash(features.FEATURES / 'manifest.json')
            or not state['all_FP32_byte_roundtrips_exact'] or not state['all_source_positions_retained']
            or audit['status'] != 'all10000_lossless_FP32_source_cache_payloads_and_bindings_replayed'
            or audit['cases'] != 10000 or not audit['FP32_bytes_preserved']
            or audit['original_body_hash_unchanged'] != features.base.EXPECTED_BODY):
        raise ValueError('complete audited original-body full FP32 feature corpus required')
    return manifest, {r['id']: r for r in q.read(features.metadata.DATA / 'index.json')}, audit


def checkpoint_contract():
    state, saved = selected.checkpoint_contract()
    if state['checkpoint_sha256'] != CHECKPOINT_SHA:
        raise ValueError('fixed old native655 checkpoint required; not the running FP16 candidate')
    return state, saved


def load_selected(device='cpu'):
    state, saved = checkpoint_contract()
    model = selected.training.lifecycle.mechanism.enable(selected.head.from_domain(device='cpu'))
    model.load_state_dict(saved['model'], strict=True)
    selected.model_contract(model, state)
    model.eval().requires_grad_(False)
    return model.to(device), state


def inputs():
    manifest, index, audit = require_features()
    if q.read(NUMERIC_AUDIT / 'actual_tool_receipt.json')['actual_tool_exit_code'] != 0:
        raise ValueError('actual original64 code/binding CPU audit required')
    rows, codes = numeric.cases(), {r['id']: r for r in q.read(numeric.OUTPUTS / 'outputs.json')}
    expected = q.read(NUMERIC_AUDIT / 'result.json')
    for path, value in expected['artifacts'].items():
        if file_hash(path) != value:
            raise ValueError('original oracle-code evidence changed: ' + path)
    result = []
    for row in rows:
        source, entry = index[row['id']], manifest[row['id']]
        if file_hash(source['path']) != source['sha256']:
            raise ValueError('source-only body input metadata changed')
        value = q.read(source['path'])
        if (any(value[k] != row[k] for k in ('id', 'source_id', 'group_id', 'source_sha256', 'source_component', 'source'))
                or codes[row['id']]['source_sha256'] != row['source_sha256']):
            raise ValueError('exact64 source/body-feature/oracle-code identity differs')
        for field in ('header_path', 'payload_path'):
            expected_hash = entry[field.replace('_path', '_sha256')]
            if audit['artifacts'].get(entry[field]) != expected_hash:
                raise ValueError('selected full source feature lacks independent byte audit')
        result.append({**row, 'oracle_codes': codes[row['id']]['codes'], 'feature': entry, 'source_metadata': source})
    numeric.text_control.split_guard(result)
    return result


def producing(root, stage, extra=()):
    files = [Path(__file__), PLAN, TEST, AUDITOR, Path(selected.__file__),
        Path(selected.training.__file__), selected.training.PLAN, selected.training.AUDITOR,
        Path(selected.head.__file__), Path(selected.head.concepts.__file__),
        Path(selected.training.lifecycle.mechanism.__file__), Path(features.__file__), features.PLAN,
        Path(features.codec.__file__), Path(numeric.__file__), numeric.PLAN, Path(numeric.framed.__file__),
        *sorted(native.vendor.VENDOR.rglob('*.py'))]
    common = [selected.training.TRAIN / n for n in ('model.pt', 'result.json', 'worker_terminal_receipt.json')]
    common += [selected.training.WORKFLOW / 'tool_terminal_receipt.json', selected.TRAIN_AUDIT / 'result.json',
        selected.TRAIN_AUDIT / 'actual_tool_terminal_receipt.json', features.FEATURES / 'manifest.json',
        features.FEATURES / 'result.json', features.FEATURES / 'worker_terminal_receipt.json',
        features.WORKFLOW / 'tool_terminal_receipt.json', FEATURE_AUDIT / 'result.json',
        FEATURE_AUDIT / 'actual_tool_terminal_receipt.json', features.metadata.DATA / 'index.json',
        numeric.OUTPUTS / 'outputs.json', NUMERIC_AUDIT / 'result.json', NUMERIC_AUDIT / 'actual_tool_receipt.json']
    p = native.vendor.freeze(root, stage, list(dict.fromkeys([*extra, *common, *files])), {
        'hypothesis_id': HYPOTHESIS, 'checkpoint_sha256': CHECKPOINT_SHA, 'cases': 64, 'arms': ARMS,
        'same_decoder_and_oracle_codes': True, 'oracle_target_codes': True, 'source_only': False,
        'source_prediction': False, 'student_task_accuracy': None, 'semantic_accuracy': None,
        'source_features': 'complete original28-layer BF16 body; full aligned FP32; lossless storage',
        'parent_loaded': False, 'parent_body_and_MLPs_frozen': True, 'training': False,
        'concept_generator_called': False, 'physical_GPU': 0, 'precision': 'FP32', 'TF32': False,
        'batch_size': 1, 'peak_limit_bytes': 3_000_000_000, 'raw_words': 128, 'word_vocabulary': 30528,
        'native_CLS_renderer': True, 'no_repair': True, 'external_judgments': False,
        'final_test_opened': False, 'speed_tested': False})
    p['hypothesis_id'] = HYPOTHESIS
    for i, path in enumerate(files):
        p['source_hashes'][str(path)] = file_hash(path)
        (root / 'producer_source' / f'grounded_oracle_{i}_{path.name}').write_bytes(path.read_bytes())
    p['fingerprint'] = q.digest({k: v for k, v in p.items() if k != 'fingerprint'})
    write_json(root / 'producer.json', p)
    return p


def prepare():
    configure()
    if torch.cuda.is_initialized() or os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('source/target/feature metadata preparation must remain CPU-only')
    checkpoint_contract()
    rows = inputs()
    longest = min(rows, key=lambda r: (-r['feature']['source_tokens'], r['id']))
    ids = [rows[0]['id'], longest['id']]
    if len(set(ids)) != 2:
        raise ValueError('distinct first/longest source pilot required')
    p = producing(PREP, 'exact64_oracle_codes_and_complete_source_features',
        [Path(r['source_metadata']['path']) for r in rows])
    write_json(PREP / 'cases.json', rows)
    write_json(PREP / 'selection.json', {'pilot_ids': ids})
    write_json(PREP / 'split_proof.json', numeric.text_control.split_guard(rows))
    native.vendor.verify(p)
    write_json(PREP / 'result.json', {'status': 'all64_grounded_oracle_cases_prepared', 'cases': 64,
        'cases_sha256': file_hash(PREP / 'cases.json'), 'student_task_accuracy': None})


def require_gpu():
    if (os.environ.get('CUDA_VISIBLE_DEVICES') != '0' or os.environ.get('CUDA_DEVICE_ORDER') != 'PCI_BUS_ID'
            or not torch.cuda.is_available() or '3060' not in torch.cuda.get_device_name()
            or torch.cuda.mem_get_info()[0] < 3_000_000_000):
        raise ValueError('released physicalGPU0 RTX3060 with3GBfree required')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


@torch.no_grad()
def oracle_logits(model, source, mask, codes, arm):
    if arm not in ARMS or model.training or model._word_bound is not None or model.concepts._bound is not None:
        raise ValueError('frozen matched decoder with clean source contexts and explicit arm required')
    if arm == 'source_on':
        logits = model.word_logits(source, mask, codes)
    else:
        latent = model.codec.vq.idx_to_z_q(codes)
        logits = model.codec.decoder(latent)
    if model._word_bound is not None or model.concepts._bound is not None:
        raise ValueError('source context leaked across arms or cases')
    return logits


@torch.no_grad()
def reconstruct(model, row, tokenizer):
    value = q.read(row['source_metadata']['path'])
    source = features.load_feature(row['feature'], value).unsqueeze(0).cuda()
    mask = torch.ones(source.shape[:2], dtype=torch.bool, device='cuda')
    codes = torch.tensor([row['oracle_codes']], dtype=torch.long, device='cuda')
    if codes.shape != (1, 641) or (codes < 0).any() or (codes >= 8192).any():
        raise ValueError('complete641 original oracle codes required')
    conditions = {arm: numeric.decode(oracle_logits(model, source, mask, codes, arm), tokenizer) for arm in ARMS}
    return {'id': row['id'], 'source_sha256': row['source_sha256'], 'oracle_codes': row['oracle_codes'],
        'source_positions': source.shape[1], 'feature_payload_sha256': row['feature']['payload_sha256'], 'conditions': conditions}


def generate(pilot=False):
    configure()
    require_gpu()
    native.vendor.terminal(PREP)
    rows = q.read(PREP / 'cases.json')
    if rows != inputs():
        raise ValueError('unchanged complete64 oracle cases and source features required')
    ids = q.read(PREP / 'selection.json')['pilot_ids']
    chosen = [next(r for r in rows if r['id'] == i) for i in ids] if pilot else rows
    retained, extra = {}, [PREP / n for n in ('cases.json', 'selection.json', 'split_proof.json', 'worker_terminal_receipt.json')]
    if not pilot:
        native.vendor.terminal(PILOT)
        retained = {r['id']: r for r in q.read(PILOT / 'outputs.json')}
        extra += [PILOT / n for n in ('result.json', 'outputs.json', 'worker_terminal_receipt.json')]
    for row in chosen:
        extra.extend(Path(row['feature'][key]) for key in ('header_path', 'payload_path'))
    root = PILOT if pilot else OUTPUTS
    p = producing(root, 'first_longest_source_lifecycle_only' if pilot else 'all64_matched_grounded_oracle_outputs', extra)
    model, state = load_selected('cuda')
    before = native.vendor.module_hash(model)
    tokenizer, outputs = native.tokenizer(), []
    torch.cuda.reset_peak_memory_stats()
    try:
        for row in chosen:
            output = reconstruct(model, row, tokenizer)
            if pilot and reconstruct(model, row, tokenizer) != output:
                raise ValueError('repeated matched oracle decoding differs')
            if row['id'] in retained and output != retained[row['id']]:
                raise ValueError('full oracle output changed its pilot first')
            outputs.append(output)
            write_json(root / 'outputs.json', outputs)
        peak = torch.cuda.max_memory_allocated()
        if (peak > 3_000_000_000 or native.vendor.module_hash(model) != before
                or any(v.grad is not None for v in model.parameters())):
            raise ValueError('frozen trained model or3GB memory budget differs')
    finally:
        model.close()
    native.vendor.verify(p)
    write_json(root / 'result.json', {'status': 'grounded_oracle_lifecycle_only' if pilot else 'all64_grounded_oracle_outputs_semantics_pending',
        'cases': len(outputs), 'outputs_sha256': file_hash(root / 'outputs.json'), 'model_unchanged': True,
        'model_sha256': before, 'checkpoint_sha256': state['checkpoint_sha256'], 'peak_allocated_bytes': peak,
        'oracle_target_codes': True, 'source_only': False, 'student_task_accuracy': None, 'semantic_accuracy': None,
        'final_test_opened': False, 'speed_tested': False})
    print(json.dumps(q.read(root / 'result.json')), flush=True)


def workflow():
    configure()
    checkpoint_contract()
    require_features()
    if any(root.exists() for root in (WORKFLOW, PREP, PILOT, OUTPUTS)):
        raise ValueError('fresh matched grounded-oracle roots required')
    p = producing(WORKFLOW, 'supervise_matched_source_attention_oracle_control')
    receipts = []
    for stage, root, gpu in (('prepare', PREP, ''), ('pilot', PILOT, '0'), ('generate', OUTPUTS, '0')):
        native.vendor.verify(p)
        command = [sys.executable, '-u', '-m', MODULE, stage]
        child = subprocess.Popen(command, env={**os.environ, 'CUDA_VISIBLE_DEVICES': gpu})
        write_json(WORKFLOW / 'progress.json', {'active_stage': stage, 'child_pid': child.pid, 'stages': receipts})
        code = child.wait()
        receipt = {'stage': stage, 'exit_code': code, 'command': command, 'child_pid': child.pid,
            'observer_pid': os.getpid(), 'observer': 'subprocess.wait'}
        write_json(WORKFLOW / (stage + '_receipt.json'), receipt)
        if root.exists():
            write_json(root / 'worker_terminal_receipt.json', receipt)
        receipts.append(receipt)
        if code:
            raise RuntimeError('grounded oracle stage failed: ' + stage)
    native.vendor.verify(p)
    write_json(WORKFLOW / 'result.json', {'status': 'all64_grounded_oracle_numeric_workflow_completed_semantics_pending', 'stages': receipts})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('stage', choices=('prepare', 'pilot', 'generate', 'workflow'))
    args = parser.parse_args()
    {'prepare': prepare, 'pilot': lambda: generate(True), 'generate': generate, 'workflow': workflow}[args.stage]()
