"""Collect fixed released/domain-codec reconstructions; no semantic or student claim."""
import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys

import torch

from experiments.structure_net import ncm_sol64_bert_roundtrip as text_control
from experiments.structure_net import ncm_lm1b_grounded_head as grounded
from experiments.structure_net import ncm_lm1b_framed_codec as framed
from experiments.structure_net.ncm_memory_data import file_hash, write_json

native, q, ROOT = text_control.native, text_control.q, text_control.ROOT
HYPOTHESIS = 'ncm-sol64-frozen-codec-component-v1'
MODULE = 'experiments.structure_net.ncm_sol64_codec_component'
PLAN = Path('docs/07 - Status Reports/2026-10-24_ncm-sol64-codec-component-plan.md')
TEST = Path('tests/structure_net/test_ncm_sol64_codec_component.py')
WORKFLOW, PREP, PILOT, OUTPUTS = (ROOT / ('20261024_sol64_codec_component_' + n + '_v1')
    for n in ('workflow', 'prepare', 'pilot', 'outputs'))


def configure():
    native.vendor.corpus.quality.dependencies()


def cases():
    source = text_control.teacher.require_raw()
    labels = q.read(text_control.teacher.teacher.GENERATE / 'labels.json')
    by_id = {r['id']: r for r in labels}
    tokenizer = native.tokenizer()
    result = []
    for row in source:
        raw = by_id[row['id']]
        rendered = text_control.render(raw, tokenizer)
        result.append({**row, 'native_word_ids': raw['native_word_ids'],
            'texts': {'raw': raw['target_text'], 'bert': rendered}})
    text_control.split_guard(result)
    return result


def producing(root, stage, inputs=()):
    files = [Path(__file__), PLAN, TEST, Path(text_control.__file__), text_control.PLAN,
        Path(native.__file__), Path(framed.__file__), Path(grounded.__file__),
        *sorted(native.vendor.VENDOR.rglob('*.py'))]
    common = [text_control.teacher.teacher.GENERATE / 'labels.json',
        text_control.teacher.teacher.AUDIT / 'result.json',
        text_control.teacher.teacher.AUDIT / 'actual_tool_receipt.json',
        native.RELEASE / 'files/lm1b/tokenizer.pt', native.RELEASE / 'files/lm1b/small/config.yaml',
        grounded.DOMAIN_CHECKPOINT, grounded.DOMAIN_CHECKPOINT.parent / 'result.json',
        native.CASES, text_control.development.SOURCE / 'cases.json',
        *[p for p in sorted((native.INVENTORY / 'bert-tokenizer').iterdir()) if p.is_file()]]
    p = native.vendor.freeze(root, stage, list(dict.fromkeys([*inputs, *common, *files])), {
        'hypothesis_id': HYPOTHESIS, 'cases': 64, 'oracle_target_codes': True,
        'source_only': False, 'student_task_accuracy': None, 'semantic_accuracy': None,
        'parent_loaded': False, 'NCM_generator_loaded': False, 'training': False,
        'conditions': ['released', 'domain'], 'same_encoder_VQ_and_complete_codes': True,
        'domain_checkpoint_sha256': grounded.DOMAIN_SHA, 'precision': 'FP32', 'TF32': False,
        'physical_GPU': 1, 'maximum_peak_bytes': 1_500_000_000,
        'raw_words': 128, 'full_word_vocabulary': 30528, 'codes': 641,
        'no_truncation_or_repair': True, 'external_judgments': False,
        'final_test_opened': False, 'speed_tested': False})
    p['hypothesis_id'] = HYPOTHESIS
    for i, path in enumerate(files):
        p['source_hashes'][str(path)] = file_hash(path)
        (root / 'producer_source' / f'codec_component_{i}_{path.name}').write_bytes(path.read_bytes())
    p['fingerprint'] = q.digest({k: v for k, v in p.items() if k != 'fingerprint'})
    write_json(root / 'producer.json', p)
    return p


def prepare():
    configure()
    if torch.cuda.is_initialized() or os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('codec metadata must remain CPU-only')
    rows = cases()
    longest = min(rows, key=lambda r: (-sum(v != 0 for v in r['native_word_ids']), r['id']))
    ids = [rows[0]['id'], longest['id']]
    if len(set(ids)) != 2:
        raise ValueError('distinct first/longest native frame probes required')
    p = producing(PREP, 'unchanged64_complete_native_frames')
    write_json(PREP / 'cases.json', rows)
    write_json(PREP / 'selection.json', {'pilot_ids': ids})
    write_json(PREP / 'split_proof.json', text_control.split_guard(rows))
    native.vendor.verify(p)
    write_json(PREP / 'result.json', {'status': 'all64_codec_component_frames_prepared',
        'cases_sha256': file_hash(PREP / 'cases.json'), 'cases': 64, 'student_task_accuracy': None})


def require_gpu():
    if (os.environ.get('CUDA_VISIBLE_DEVICES') != '1' or os.environ.get('CUDA_DEVICE_ORDER') != 'PCI_BUS_ID'
            or not torch.cuda.is_available() or '2060' not in torch.cuda.get_device_name()
            or 'SUPER' in torch.cuda.get_device_name() or torch.cuda.mem_get_info()[0] < 1_800_000_000):
        raise ValueError('physical1 RTX2060 with1.8GBfree required for standalone codec only')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def decode(logits, tokenizer):
    if logits.shape != (1, 128, 30528) or not torch.isfinite(logits).all():
        raise ValueError('complete finite full-vocabulary native decoder logits required')
    raw = logits.argmax(-1).cpu().tolist()[0]
    text, ends, bos = framed.render([raw], tokenizer)
    return {'text': text, 'raw_word_ids': raw, 'predicted_BOS': bos[0],
        'predicted_EOS_position': ends[0], 'missing_EOS': ends[0] == 128}


@torch.no_grad()
def reconstruct(codec, domain, row, tokenizer):
    frame = torch.tensor([row['native_word_ids']], dtype=torch.long, device='cuda')
    codes = codec.encode(frame)
    if codes.shape != (1, 641) or (codes < 0).any() or (codes >= 8192).any():
        raise ValueError('every native code and level required')
    latent = codec.vq.idx_to_z_q(codes)
    values = {name: decode(decoder(latent), tokenizer) for name, decoder in (('released', codec.decoder), ('domain', domain))}
    return {'id': row['id'], 'source_sha256': row['source_sha256'], 'codes': codes.cpu().tolist()[0],
        'conditions': values, 'lexical_positions': sum(v != 0 for v in row['native_word_ids']) - 2}


def generate(pilot=False):
    configure()
    require_gpu()
    native.vendor.terminal(PREP)
    rows = q.read(PREP / 'cases.json')
    if rows != cases():
        raise ValueError('original complete first targets changed')
    chosen = q.read(PREP / 'selection.json')['pilot_ids']
    selected = [next(r for r in rows if r['id'] == i) for i in chosen] if pilot else rows
    retained = {}
    extra = [PREP / n for n in ('cases.json', 'selection.json', 'split_proof.json', 'worker_terminal_receipt.json')]
    if not pilot:
        native.vendor.terminal(PILOT)
        retained = {r['id']: r for r in q.read(PILOT / 'outputs.json')}
        extra += [PILOT / n for n in ('result.json', 'outputs.json', 'worker_terminal_receipt.json')]
    root = PILOT if pilot else OUTPUTS
    p = producing(root, 'first_longest_CUDA_lifecycle_only' if pilot else 'all64_frozen_codec_outputs', extra)
    codec = native.load_codec()
    if file_hash(grounded.DOMAIN_CHECKPOINT) != grounded.DOMAIN_SHA:
        raise ValueError('fixed domain decoder checkpoint changed')
    domain = deepcopy(codec.decoder)
    saved = torch.load(grounded.DOMAIN_CHECKPOINT, map_location='cpu', weights_only=True, mmap=True)
    domain.load_state_dict(saved['decoder'], strict=True)
    domain.eval().requires_grad_(False)
    before = {'codec': native.vendor.module_hash(codec), 'domain_decoder': native.vendor.module_hash(domain)}
    tokenizer, outputs = native.tokenizer(), []
    torch.cuda.reset_peak_memory_stats()
    for row in selected:
        output = reconstruct(codec, domain, row, tokenizer)
        if pilot and reconstruct(codec, domain, row, tokenizer) != output:
            raise ValueError('codec CUDA repeat differs')
        if row['id'] in retained and output != retained[row['id']]:
            raise ValueError('full codec inference changed pilot outputs')
        outputs.append(output)
        write_json(root / 'outputs.json', outputs)
    after = {'codec': native.vendor.module_hash(codec), 'domain_decoder': native.vendor.module_hash(domain)}
    peak = torch.cuda.max_memory_allocated()
    if before != after or peak > 1_500_000_000 or any(v.grad is not None for m in (codec, domain) for v in m.parameters()):
        raise ValueError('frozen codec state or1.5GB memory budget differs')
    native.vendor.verify(p)
    write_json(root / 'result.json', {'status': 'codec_component_lifecycle_only' if pilot else 'all64_codec_outputs_semantics_pending',
        'cases': len(outputs), 'outputs_sha256': file_hash(root / 'outputs.json'), 'models_unchanged': True,
        'model_hashes': before, 'peak_allocated_bytes': peak, 'oracle_target_codes': True,
        'source_only': False, 'semantic_accuracy': None, 'student_task_accuracy': None,
        'final_test_opened': False, 'speed_tested': False})
    print(json.dumps(q.read(root / 'result.json')), flush=True)


def workflow():
    configure()
    if any(root.exists() for root in (WORKFLOW, PREP, PILOT, OUTPUTS)):
        raise ValueError('fresh numeric codec component roots required')
    p = producing(WORKFLOW, 'supervise_standalone_codec_numeric_control')
    receipts = []
    for stage, root, gpu in (('prepare', PREP, ''), ('pilot', PILOT, '1'), ('generate', OUTPUTS, '1')):
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
            raise RuntimeError('codec component stage failed: ' + stage)
    native.vendor.verify(p)
    write_json(WORKFLOW / 'result.json', {'status': 'all64_codec_numeric_workflow_completed_semantics_pending', 'stages': receipts})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('stage', choices=('prepare', 'pilot', 'generate', 'workflow'))
    args = parser.parse_args()
    {'prepare': prepare, 'pilot': lambda: generate(True), 'generate': generate, 'workflow': workflow}[args.stage]()
