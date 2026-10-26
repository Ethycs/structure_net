"""Replay sample/noise streams, raw diagnostics and frozen checkpoint tensors."""
import json
import math
from pathlib import Path

import torch

from experiments.structure_net import ncm_broad_generator as study

DEST = study.runtime.ROOT/'20261025_broad_generator_train_audit_v1'


def main():
    training, environment = study.runtime.configure()
    if DEST.exists(): raise ValueError('fresh audit root required')
    root = study.RUNS['train']
    training.preparation.completed(root)
    result, proof = [study.catalog.read(root/n) for n in ('result.json','producer.json')]
    if proof['environment'] != environment or proof['config'] != study.asdict(study.CONFIG):
        raise ValueError('same runtime and fixed training configuration required')
    if proof['fingerprint'] != training.q.digest({k:v for k,v in proof.items() if k != 'fingerprint'}):
        raise ValueError('scientific fingerprint differs')
    rows = study.catalog.read(study.RUNS['prepare']/'cases.json')
    outputs = study.catalog.read(root/'outputs.json')
    if len(rows) != 1376 or [r['id'] for r in rows] != [r['id'] for r in outputs]:
        raise ValueError('all1376 ordered TRAIN diagnostics required')
    for row, out in zip(rows, outputs):
        target, words = row['native_word_ids'], out['raw_word_ids']
        codes = out['codes']
        if (len(words) != 128 or len(codes) != 641 or out['source_id_used'] != row['id']
                or any(type(x) is not int or not 0 <= x < 8192 for x in codes)
                or out['active_words'] != sum(v != 0 for v in target)
                or out['active_word_matches'] != sum(a == b for a,b in zip(words,target) if b != 0)):
            raise ValueError('raw source/code/frame diagnostic differs')
        offset = 0
        for level in out['levels']:
            end = offset + level['positions']
            if level['rollout_correct'] != sum(a == b for a,b in zip(codes[offset:end],row['oracle_codes'][offset:end])):
                raise ValueError('raw rollout code agreement differs')
            offset = end
        if offset != 641: raise ValueError('complete hierarchy diagnostic required')
    if training.prior.summarize(outputs) != result['final_TRAIN']:
        raise ValueError('TRAIN aggregate arithmetic differs')
    tokenizer = training.native.tokenizer()
    for out in outputs:
        if any(out[k] != v for k,v in training.preparation.rendering.decode(out['raw_word_ids'],tokenizer).items()):
            raise ValueError('saved raw word rendering differs')
    trace = [json.loads(line) for line in (root/'trace.jsonl').read_text().splitlines()]
    if [r['step'] for r in trace] != list(range(1,study.CONFIG.updates+1)):
        raise ValueError('whole fixed update budget required')
    rng = torch.Generator().manual_seed(study.CONFIG.sampler_seed)
    noise = torch.Generator().manual_seed(study.CONFIG.corruption_seed)
    permutation = []; exposures = {r['id']:0 for r in rows}; changes = noisy = 0
    levels = [r['positions'] for r in outputs[0]['levels']]
    for record in trace:
        if (record['lr'] != study.rate(record['step']) or len(record['metrics']) != study.CONFIG.accumulation
                or record['mean_CE'] != sum(r['mean_CE'] for r in record['metrics'])/study.CONFIG.accumulation
                or not math.isfinite(record['gradient_norm'])):
            raise ValueError('schedule or per-update arithmetic differs')
        for i, metric in enumerate(record['metrics']):
            if not permutation: permutation = torch.randperm(len(rows), generator=rng).tolist()
            row = rows[permutation.pop()]
            if metric['id'] != row['id'] or metric['noisy_history'] != bool(i%2):
                raise ValueError('sample/noisy-history stream differs')
            if len(metric['level_CE']) != 12 or any(not math.isfinite(x) for x in metric['level_CE']):
                raise ValueError('all12 finite level losses required')
            if not math.isclose(metric['mean_CE'],sum(metric['level_CE'])/12,rel_tol=2e-6,abs_tol=1e-6):
                raise ValueError('equal-level loss arithmetic differs')
            gold = torch.tensor([row['oracle_codes']])
            _, count = study.objective.history(gold,levels,8192,noise,bool(i%2))
            if count != metric['history_corruptions']: raise ValueError('exact corruption stream differs')
            exposures[row['id']] += 1; changes += count; noisy += bool(i%2)
    if study.file_hash(root/'model.pt') != result['checkpoint_sha256']:
        raise ValueError('final checkpoint identity changed')
    original = torch.load(study.parent.PARENT/'model.pt',map_location='cpu',weights_only=True,mmap=True)
    saved = torch.load(root/'model.pt',map_location='cpu',weights_only=True,mmap=True)
    if set(original['model']) != set(saved['model']): raise ValueError('architecture changed')
    allowed = ('concepts.generator.','concepts.source_memory.','concepts.adapters.')
    changed, frozen_count = [], 0
    for name, value in saved['model'].items():
        same = torch.equal(value, original['model'][name])
        if not name.startswith(allowed):
            if not same: raise ValueError('protected tensor changed: '+name)
            frozen_count += 1
        elif not same: changed.append(name)
    if (not changed or saved['step'] != study.CONFIG.updates
            or saved['producer_fingerprint'] != proof['fingerprint']
            or study.catalog.read(root/'sampler_remaining.json') != permutation
            or not torch.equal(saved['sampler_rng'], rng.get_state())
            or not torch.equal(saved['noise_rng'], noise.get_state())
            or any(float(v['step']) != study.CONFIG.updates for v in saved['optimizer']['state'].values())
            or not result['strict_reload_passed'] or not result['oracle_unchanged']
            or result['feature_cache']['peak_resident_bytes'] > study.CONFIG.host_feature_cache_bytes):
        raise ValueError('state, optimizer, sample stream or memory integrity differs')
    memories = [json.loads(line) for line in (root/'memory.jsonl').read_text().splitlines()]
    if (max(r['peak_allocated_bytes'] for r in memories) != result['peak_allocated_bytes']
            or result['peak_allocated_bytes'] > study.CONFIG.peak_limit_bytes):
        raise ValueError('whole lifecycle memory evidence differs')
    oracles = {r['id']:r for r in study.catalog.read(study.RUNS['prepare']/'oracle_outputs.json')}
    for out in study.catalog.read(root/'oracle_outputs.json'):
        if out != oracles[out['id']]: raise ValueError('clean reconstruction changed')
    paths = [root/n for n in ('result.json','producer.json','trace.jsonl','outputs.json','model.pt',
             'actual_tool_receipt.json','memory.jsonl','sampler_remaining.json','oracle_outputs.json','checkpoint_ready.json')]
    result = {'status':'broad_generator_training_independently_replayed',
              'updates':study.CONFIG.updates,'cases':len(rows),'presentations':sum(exposures.values()),
              'noisy_presentations':noisy,'history_code_replacements':changes,
              'exposures_min':min(exposures.values()),'exposures_max':max(exposures.values()),
              'changed_trainable_tensors':changed,'frozen_tensors_identical':frozen_count,
              'first16_mean_CE':sum(r['mean_CE'] for r in trace[:16])/16,
              'last16_mean_CE':sum(r['mean_CE'] for r in trace[-16:])/16,
              'task_accuracy':None,'final_test_opened':False,'speed_tested':False,
              'artifacts':{str(p):study.file_hash(p) for p in paths},
              'auditor_sha256':study.file_hash(Path(__file__))}
    DEST.mkdir(); study.write_json(DEST/'campaign_results.json',result)
    print(json.dumps({k:v for k,v in result.items() if k not in ('artifacts','changed_trainable_tensors')}))


if __name__ == '__main__': main()
