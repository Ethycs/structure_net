"""Encode unchanged engineering TRAIN summaries with the frozen released codec."""
import json
import os
from pathlib import Path

import torch

from experiments.structure_net import ncm_engineering_summary_inventory_v2 as data
from experiments.structure_net.ncm_memory_data import file_hash, text_hash, write_json

native, ROOT = data.native, data.ROOT
DEST = ROOT / '20261025_engineering_TRAIN_target_codes_v1'
PLAN = Path('docs/07 - Status Reports/2026-10-25_ncm-engineering-target-codes-plan.md')


def main():
    native.vendor.corpus.quality.dependencies()
    native.require_gpu()
    if DEST.exists():
        raise ValueError('fresh TRAIN-only target-code root required')
    if data.old.reservation.read(data.DEST / 'actual_tool_receipt.json')['actual_tool_exit_code'] != 0:
        raise ValueError('actually completed engineering reservation required')
    native.vendor.verify(data.old.reservation.read(data.DEST / 'producer.json'))
    state = data.old.reservation.read(data.DEST / 'result.json')
    if file_hash(data.DEST / 'cases.json') != state['cases_sha256']:
        raise ValueError('unchanged original candidate targets required')
    rows = data.old.reservation.read(data.DEST / 'cases.json')['train']
    if len(rows) != 732 or len({r['id'] for r in rows}) != 732:
        raise ValueError('all and only 732 engineering TRAIN summaries required')
    first, longest = rows[0]['id'], min(rows, key=lambda r: (-r['native_lexical_tokens'], r['id']))['id']
    probes = list(dict.fromkeys((first, longest)))
    files = [Path(__file__), PLAN, Path(data.__file__), Path(native.__file__),
        *sorted(native.vendor.VENDOR.rglob('*.py'))]
    inputs = [data.DEST / n for n in ('producer.json', 'result.json', 'cases.json', 'actual_tool_receipt.json')]
    inputs += [native.RELEASE / 'files/lm1b/tokenizer.pt', native.RELEASE / 'files/lm1b/small/config.yaml']
    DEST.mkdir()
    (DEST / 'producer_source').mkdir()
    for i, path in enumerate(files):
        (DEST / 'producer_source' / f'{i}_{path.name}').write_bytes(path.read_bytes())
    p = {'schema': 'nal.ncm-engineering-TRAIN-target-codes.v1',
        'source_hashes': {str(path): file_hash(path) for path in files},
        'input_hashes': {str(path): file_hash(path) for path in inputs},
        'source_scope': 'existing unreviewed engineering TRAIN summaries only',
        'config': {'cases': 732, 'physical_GPU': 2, 'precision': 'FP32', 'TF32': False,
            'native_context': 128, 'native_levels': list(native.LEVELS), 'code_positions': 641,
            'native_encoder_and_VQ_frozen': True, 'parent_loaded': False, 'G_loaded': False,
            'heldout_targets_encoded': False, 'targets_changed': False,
            'probe_ids': probes, 'model_training': False, 'task_accuracy': None,
            'sealed_final_sources_opened_or_hashed': False}}
    write_json(DEST / 'producer.json', p)
    codec, tokenizer = native.load_codec(), native.tokenizer()
    before = native.vendor.module_hash(codec)
    torch.cuda.reset_peak_memory_stats()
    outputs = []
    with torch.no_grad():
        for i, row in enumerate(rows, 1):
            ids = tokenizer.encode(row['target_text'], add_special_tokens=False)
            if (text_hash(row['target_text']) != row['target_sha256'] or len(ids) != row['native_lexical_tokens']
                    or set(ids) & set(tokenizer.all_special_ids) or not 0 < len(ids) <= 126
                    or row['native_word_ids'] != [101, *ids, 101] + [0] * (126-len(ids))):
                raise ValueError('complete unchanged native target frame required')
            words = torch.tensor([row['native_word_ids']], dtype=torch.long, device='cuda')
            codes = codec.encode(words)
            if codes.shape != (1, 641) or codes.min() < 0 or codes.max() >= 8192:
                raise ValueError('complete released 641-position code hierarchy required')
            if row['id'] in probes and not torch.equal(codes, codec.encode(words)):
                raise ValueError('real target codec must repeat exactly')
            outputs.append({'id': row['id'], 'source_sha256': row['source_sha256'],
                'target_sha256': row['target_sha256'], 'source_component': row['source_component'],
                'native_word_ids': row['native_word_ids'], 'codes': codes.cpu().tolist()[0]})
            if i % 64 == 0 or i == len(rows):
                write_json(DEST / 'outputs.json', outputs)
                print(json.dumps({'encoded_engineering_TRAIN_targets': i, 'total': len(rows)}), flush=True)
    if native.vendor.module_hash(codec) != before or any(v.grad is not None for v in codec.parameters()):
        raise ValueError('frozen native encoder, codebooks or decoder changed')
    peak = torch.cuda.max_memory_allocated()
    if peak > 3_000_000_000:
        raise ValueError('target-code extraction exceeds3GB planned allocation')
    native.vendor.verify(p)
    write_json(DEST / 'result.json', {'status': 'all732_engineering_TRAIN_targets_encoded_with_frozen_native_codec',
        'cases': len(outputs), 'outputs_sha256': file_hash(DEST / 'outputs.json'),
        'frozen_codec_sha256': before, 'encoder_sha256': native.vendor.module_hash(codec.encoder),
        'VQ_sha256': native.vendor.module_hash(codec.vq), 'peak_allocated_bytes': peak,
        'real_probe_repeatability': True, 'heldout_targets_encoded': False,
        'model_training': False, 'task_accuracy': None, 'sealed_final_sources_opened_or_hashed': False})


if __name__ == '__main__':
    main()
