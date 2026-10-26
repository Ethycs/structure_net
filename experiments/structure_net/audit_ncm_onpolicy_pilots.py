"""Offline checks of actual pilot updates, frozen tensors and recorded histories."""
import argparse
import json
import math
from pathlib import Path

import torch

from experiments.structure_net import ncm_onpolicy_residual_training as study

ROOTS = {a:study.broad.runtime.ROOT/f'20261025_onpolicy_residual_{a}_pilot_audit_v1'
         for a in study.objective.ARMS}


def main(arm):
    training, environment = study.broad.runtime.configure()
    dest, root = ROOTS[arm],study.ROOTS[arm,'pilot']
    if dest.exists(): raise ValueError('fresh append-only pilot audit required')
    training.preparation.completed(root)
    q = training.q
    result,proof,parity = [q.read(root/n) for n in ('result.json','producer.json','gradient_parity.json')]
    if (proof['arm'] != arm or proof['environment'] != environment
            or proof['config'] != study.asdict(study.CONFIG) or result['updates'] != 2
            or not all(result[k] for k in ('strict_reload_passed','gradient_parity_passed','oracle_unchanged'))):
        raise ValueError('same-config completed two-update pilot required')
    rows = {r['id']:r for r in q.read(study.broad.RUNS['prepare']/'cases.json')}
    selection = q.read(study.broad.RUNS['prepare']/'selection.json')
    if len(parity) != 2 or any(not p['passed'] for p in parity):
        raise ValueError('clean and generated-history gradient parity required')
    if (parity[1]['history_errors'] <= 0 or (arm == 'residual_targets' and parity[1]['target_changes'] <= 0)
            or (arm == 'fixed_targets' and parity[1]['target_changes'] != 0)):
        raise ValueError('pilot did not exercise its actual arm objective')
    trace = [json.loads(line) for line in (root/'trace.jsonl').read_text().splitlines()]
    if [r['step'] for r in trace] != [1,2]: raise ValueError('exactly two optimizer steps required')
    history_errors = target_changes = 0
    for record in trace:
        if record['arm'] != arm or record['lr'] != study.rate(record['step']) or len(record['metrics']) != 4:
            raise ValueError('pilot arm, rate or accumulation changed')
        if not math.isfinite(record['gradient_norm']): raise ValueError('nonfinite gradient')
        for micro,m in enumerate(record['metrics']):
            key = 'longest_source_id' if micro%2 else 'first_id'
            if m['id'] != selection[key] or m['onpolicy'] != bool(micro%2):
                raise ValueError('fixed pilot source/history schedule differs')
            gold = rows[m['id']]['oracle_codes']
            if not m['onpolicy']:
                if (m['history_codes'] is not None or m['target_codes'] is not None
                        or m['history_errors'] != 0 or m['target_changes'] != 0):
                    raise ValueError('clean anchor changed')
            else:
                h,t = m['history_codes'],m['target_codes']
                if len(h) != 641 or len(t) != 641 or any(type(x) is not int or not 0 <= x < 8192 for x in h+t):
                    raise ValueError('complete bounded native hierarchy required')
                if (m['history_errors'] != sum(a != b for a,b in zip(h,gold))
                        or m['target_changes'] != sum(a != b for a,b in zip(t,gold))
                        or arm == 'fixed_targets' and t != gold):
                    raise ValueError('raw history/label counts or fixed labels differ')
                if record['step'] == 1 and (h != parity[1]['history_codes'] or t != parity[1]['target_codes']):
                    raise ValueError('first update must use the checked initial policy and targets')
            if len(m['level_CE']) != 12 or any(not math.isfinite(v) for v in m['level_CE']):
                raise ValueError('complete finite level losses required')
            if not math.isclose(m['mean_CE'],sum(m['level_CE'])/12,rel_tol=2e-6,abs_tol=1e-6):
                raise ValueError('equal-level loss arithmetic differs')
            history_errors += m['history_errors']; target_changes += m['target_changes']
    original = torch.load(study.PARENT/'model.pt',map_location='cpu',weights_only=True,mmap=True)
    saved = torch.load(root/'model.pt',map_location='cpu',weights_only=True,mmap=True)
    if set(original['model']) != set(saved['model']): raise ValueError('architecture differs')
    frozen,changed = 0,[]
    allowed = ('concepts.generator.','concepts.source_memory.','concepts.adapters.')
    for name,value in saved['model'].items():
        equal = torch.equal(value,original['model'][name])
        if not name.startswith(allowed):
            if not equal: raise ValueError('protected tensor changed:'+name)
            frozen += 1
        elif not equal: changed.append(name)
    if (not changed or saved['step'] != 2 or saved['arm'] != arm
            or saved['producer_fingerprint'] != proof['fingerprint'] or saved['permutation']
            or not torch.equal(saved['sampler_rng'],torch.Generator().manual_seed(study.CONFIG.sampler_seed).get_state())
            or any(float(v['step']) != 2 for v in saved['optimizer']['state'].values())
            or study.broad.file_hash(root/'model.pt') != result['checkpoint_sha256']):
        raise ValueError('actual update/checkpoint/sampler integrity differs')
    original_oracles = {r['id']:r for r in q.read(study.broad.RUNS['prepare']/'oracle_outputs.json')}
    if q.read(root/'oracle_outputs.json') != [original_oracles[i] for i in selection['diagnostic_ids']]:
        raise ValueError('fixed clean reconstruction differs')
    paths = [Path(__file__),*[root/n for n in ('producer.json','result.json','model.pt','trace.jsonl',
        'gradient_parity.json','actual_tool_receipt.json','oracle_outputs.json')]]
    value = {'status':'pilot_updates_histories_and_protected_tensors_replayed','arm':arm,
        'updates':2,'presentations':8,'onpolicy_presentations':4,'history_errors':history_errors,
        'target_changes':target_changes,'frozen_tensors_identical':frozen,'changed_trainable_tensors':changed,
        'actual_first_update_matches_checked_policy':True,'task_accuracy':None,'training_repeated':False,
        'target_scope':'recorded labels/counts and initial gradient-parity labels; no independent encoder replay',
        'artifacts':{str(p):study.broad.file_hash(p) for p in paths}}
    dest.mkdir();study.broad.write_json(dest/'result.json',value)
    print(json.dumps({k:v for k,v in value.items() if k not in ('artifacts','changed_trainable_tensors')}),flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('arm',choices=study.objective.ARMS)
    main(parser.parse_args().arm)
