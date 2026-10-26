"""Oracle diagnostic: can later frozen residual codebooks correct prior errors?"""
import argparse
import json
from pathlib import Path
import sys

import torch

from experiments.structure_net import ncm_broad_generator as study

PLAN = Path('docs/07 - Status Reports/2026-10-25_ncm-residual-recoding-probe-plan.md')
TEST = Path('tests/structure_net/test_ncm_residual_recoding_probe.py')
PREFIX_LEVELS = (0, 2, 3, 4, 6, 9, 11, 12)
ROOTS = {s: study.runtime.ROOT / ('20261025_residual_recoding_probe_' + s + '_v1')
         for s in ('pilot', 'full')}


@torch.no_grad()
def recode(vq, target_latent, prefix, forced_levels):
    """Nearest residual codes after a fixed prefix, using native quantizer math.

    This function consumes target latents and is forbidden in source inference.
    Subtract each contribution in native order to preserve clean-code parity.
    """
    levels = vq.levels
    if (type(forced_levels) is not int or not 0 <= forced_levels <= len(levels)
            or target_latent.ndim != 3
            or target_latent.shape[1:] != (levels[-1], vq.embed_dim)
            or target_latent.dtype != torch.float32 or not torch.isfinite(target_latent).all()
            or prefix.dtype != torch.long or prefix.device != target_latent.device
            or prefix.shape != (len(target_latent), sum(levels[:forced_levels]))):
        raise ValueError('complete FP32 latent and exactly the forced prefix required')
    residual = target_latent.detach().clone()
    parts, offset = [], 0
    for i, length in enumerate(levels):
        embedding = vq.embeddings[i]
        if i < forced_levels:
            part = prefix[:, offset:offset + length].clone()
            if part.min() < 0 or part.max() >= embedding.num_embeddings:
                raise ValueError('prefix code outside its native codebook')
        else:
            value = vq._down_for_quant(residual, i, length) if i < len(levels)-1 else residual
            distances = vq._distances(value.reshape(-1, vq.embed_dim), embedding.weight)
            if not torch.isfinite(distances).all():
                raise ValueError('nonfinite residual quantization distance')
            part = distances.argmin(-1).view(len(target_latent), length)
        contribution = embedding(part)
        if i < len(levels)-1:
            contribution = vq._up_to_full(contribution, i, levels[-1])
        residual -= vq._apply_quant_resi(contribution, i)
        parts.append(part)
        offset += length
    return torch.cat(parts, dim=1)


def metrics(raw, target):
    active = target != 0
    return {'active_words': int(active.sum()),
            'active_word_matches': int(((raw == target) & active).sum()),
            'exact_active_frame': bool(torch.equal(raw[active], target[active]))}


@torch.no_grad()
def evaluate(model, row, prediction, loader, tokenizer, training):
    if row['split'] != 'TRAIN' or row['id'] != prediction['source_id_used']:
        raise ValueError('TRAIN-only correctly bound source prediction required')
    target = torch.tensor([row['native_word_ids']], device='cuda', dtype=torch.long)
    gold = torch.tensor([row['oracle_codes']], device='cuda', dtype=torch.long)
    generated = torch.tensor([prediction['codes']], device='cuda', dtype=torch.long)
    z = model.codec.encode_z(target)
    if not torch.equal(model.codec.vq.z_to_idx(z), gold):
        raise ValueError('frozen native target encoding differs')
    features = loader.get(row).unsqueeze(0).cuda()
    mask = torch.ones(features.shape[:2], dtype=torch.bool, device=features.device)
    output = []
    for count in PREFIX_LEVELS:
        end = sum(model.codec.vq.levels[:count])
        if not torch.equal(recode(model.codec.vq, z, gold[:, :end], count), gold):
            raise ValueError('clean-prefix native recoding parity failed')
        fixed = torch.cat((generated[:, :end], gold[:, end:]), dim=1)
        adaptive = recode(model.codec.vq, z, generated[:, :end], count)
        if count in (0, 12) and not torch.equal(fixed, adaptive):
            raise ValueError('zero/full-prefix endpoint control differs')
        for arm, codes in (('original_suffix', fixed), ('adaptive_suffix', adaptive)):
            latent = model.codec.vq.idx_to_z_q(codes)
            bare = model.codec.decoder(latent).argmax(-1)[0].cpu()
            bound = model.word_logits(features, mask, codes).argmax(-1)[0].cpu()
            if count == 12 and bound.tolist() != prediction['raw_word_ids']:
                raise ValueError('saved full generated-code readout differs')
            words = target[0].cpu()
            value = {'id': row['id'], 'target_sha256': row['target_sha256'],
                'source_sha256': row['source_sha256'], 'prefix_levels': count, 'arm': arm,
                'codes': codes[0].cpu().tolist(), 'suffix_changes_from_original':
                    int((codes[:, end:] != gold[:, end:]).sum()),
                'latent_MSE': float((latent - z).double().square().mean()),
                'codec_only': {'raw_word_ids': bare.tolist(), **metrics(bare, words),
                    **training.preparation.rendering.decode(bare.tolist(), tokenizer)},
                'with_source': {'raw_word_ids': bound.tolist(), **metrics(bound, words),
                    **training.preparation.rendering.decode(bound.tolist(), tokenizer)}}
            output.append(value)
    return output


def aggregate(rows):
    out = []
    for count in PREFIX_LEVELS:
        for arm in ('original_suffix', 'adaptive_suffix'):
            values = [r for r in rows if r['prefix_levels'] == count and r['arm'] == arm]
            value = {'prefix_levels': count, 'arm': arm, 'cases': len(values),
                'mean_latent_MSE': sum(r['latent_MSE'] for r in values) / len(values)}
            for mode in ('codec_only', 'with_source'):
                total = sum(r[mode]['active_words'] for r in values)
                matched = sum(r[mode]['active_word_matches'] for r in values)
                value[mode] = {'active_words': total, 'active_word_matches': matched,
                    'active_word_agreement': matched / total,
                    'exact_active_frames': sum(r[mode]['exact_active_frame'] for r in values)}
            out.append(value)
    return out


def main(stage):
    training, environment = study.runtime.configure()
    study.offload.require_rtx2060()
    training.preparation.completed(study.RUNS['train'])
    state = training.q.read(study.RUNS['train'] / 'result.json')
    if study.file_hash(study.RUNS['train'] / 'model.pt') != state['checkpoint_sha256']:
        raise ValueError('completed generator checkpoint changed')
    rows = training.q.read(study.RUNS['prepare'] / 'cases.json')
    ids = training.q.read(study.RUNS['prepare'] / 'selection.json')['diagnostic_ids']
    by_id = {r['id']: r for r in rows}
    selected = [by_id[i] for i in (ids[::15] if stage == 'pilot' else ids)]
    predictions = {r['id']: r for r in training.q.read(study.RUNS['train'] / 'outputs.json')}
    index, manifest = [training.q.read(study.catalog.DEST / n)
                       for n in ('source_index.json', 'feature_manifest.json')]
    before = {}
    if stage == 'full':
        training.preparation.completed(ROOTS['pilot'])
        before = {(r['id'], r['prefix_levels'], r['arm']): r
                  for r in training.q.read(ROOTS['pilot'] / 'outputs.json')}
    model = study.parent.load_parent(training)
    saved = torch.load(study.RUNS['train'] / 'model.pt', map_location='cpu', weights_only=True, mmap=True)
    model.load_state_dict(saved['model'], strict=True)
    model.eval().requires_grad_(False)
    initial_hash = training.native.vendor.module_hash(model)
    if initial_hash != state['final_model_sha256']:
        raise ValueError('selected full model identity differs')
    root = ROOTS[stage]
    files = {Path(__file__), PLAN, TEST}
    for module in tuple(sys.modules.values()):
        name = getattr(module, '__file__', None)
        if name:
            p = Path(name).resolve()
            if p.is_file() and p.suffix == '.py' and (p.is_relative_to(Path('experiments').resolve())
                    or p.is_relative_to(training.native.vendor.VENDOR.resolve())):
                files.add(p)
    inputs = [study.RUNS['train'] / n for n in ('producer.json', 'result.json', 'model.pt',
        'outputs.json', 'actual_tool_receipt.json')]
    inputs += [study.RUNS['prepare'] / n for n in ('cases.json', 'selection.json')]
    inputs += [study.catalog.DEST / n for n in ('source_index.json', 'feature_manifest.json')]
    inputs += [Path(manifest[r['id']]['path']) for r in selected]
    if stage == 'full':
        inputs += [ROOTS['pilot'] / n for n in ('result.json', 'outputs.json', 'actual_tool_receipt.json')]
    proof = {'schema': 'nal.ncm-residual-recoding-probe.v1', 'stage': stage,
        'config': {'TRAIN_ids': [r['id'] for r in selected], 'prefix_levels': PREFIX_LEVELS,
            'oracle_target_access': True, 'source_only_inference': False,
            'training_updates': 0, 'new_judge_requests': 0, 'final_test_opened': False,
            'precision': 'FP32', 'TF32': False, 'task_accuracy': None},
        'environment': environment, 'source_hashes': {str(p): study.file_hash(p) for p in sorted(files)},
        'input_hashes': {str(p): study.file_hash(p) for p in inputs}}
    proof['fingerprint'] = training.q.digest(proof)
    root.mkdir(); (root / 'producer_source').mkdir()
    for i, p in enumerate(sorted(files)):
        (root / 'producer_source' / f'{i}_{p.name}').write_bytes(p.read_bytes())
    study.write_json(root / 'producer.json', proof)
    loader = training.preparation.streaming.FeatureCache(index, manifest, max_bytes=256*1024**2)
    tokenizer = training.native.tokenizer()
    outputs = []
    for row in selected:
        values = evaluate(model, row, predictions[row['id']], loader, tokenizer, training)
        if stage == 'pilot' and values != evaluate(model, row, predictions[row['id']], loader, tokenizer, training):
            raise ValueError('pilot outputs do not repeat exactly')
        if any(before.get((r['id'], r['prefix_levels'], r['arm']), r) != r for r in values):
            raise ValueError('full result differs from pilot')
        outputs.extend(values)
        study.write_json(root / 'outputs.json', outputs)
        print(json.dumps({'completed_cases': len(outputs)//16, 'cases': len(selected)}), flush=True)
    if training.native.vendor.module_hash(model) != initial_hash:
        raise ValueError('diagnostic changed model tensors')
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated()
    if peak > 4_900_000_000:
        raise ValueError('diagnostic CUDA allocation budget exceeded')
    training.native.vendor.verify(proof)
    result = {'status': 'oracle_residual_recoding_diagnostic_complete', 'cases': len(selected),
        'outputs': len(outputs), 'metrics': aggregate(outputs), 'model_unchanged': True,
        'model_sha256': initial_hash, 'peak_allocated_bytes': peak, 'feature_cache': loader.metrics(),
        'task_accuracy': None, 'oracle_target_access': True, 'source_only_inference': False,
        'training_updates': 0, 'new_judge_requests': 0, 'final_test_opened': False,
        'outputs_sha256': study.file_hash(root / 'outputs.json'), 'producer_fingerprint': proof['fingerprint']}
    study.write_json(root / 'result.json', result)
    print(json.dumps({k: v for k, v in result.items() if k != 'metrics'}), flush=True)
    model.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('stage', choices=tuple(ROOTS))
    main(parser.parse_args().stage)
