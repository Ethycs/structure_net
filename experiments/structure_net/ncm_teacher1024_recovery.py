"""Preserve the failed source check; use the original additional_train split name."""
import argparse
from collections import Counter
import json
from pathlib import Path

from experiments.structure_net import ncm_teacher1024 as original

catalog, data, base, previous = original.catalog, original.data, original.base, original.previous
configure, sources, pilot_ids, finish = original.configure, original.sources, original.pilot_ids, original.finish
PLAN = Path('docs/07 - Status Reports/2026-10-27_ncm-teacher1024-split-recovery.md')
TEST = Path('tests/structure_net/test_ncm_teacher1024_recovery.py')
SERVICE = Path('experiments/neural_architecture_lab/ncm_teacher1024_recovery_service.py')
FAILURE = data.ROOT/'20261027_teacher1024_prepare_v1_failure'
ROOTS = {s:data.ROOT/f'20261027_teacher1024_{s}_v2' for s in original.ROOTS}
OLD_FREEZE = original.freeze
INSTALLED = False


def reserved_train_split(row):
    return row['split'] == 'additional_train'


def freeze(root, stage, inputs, environment):
    receipt = data.read(FAILURE/'actual_tool_receipt.json')
    if receipt['actual_tool_exit_code'] != 1 or receipt['new_requests'] != 0:
        raise ValueError('preserved pre-request source-validator failure required')
    data.verify(receipt['artifacts'])
    return OLD_FREEZE(root, stage, [*inputs, PLAN, TEST, SERVICE, Path(__file__),
        FAILURE/'actual_tool_receipt.json', FAILURE/'producer_source.py'], environment)


def install():
    global INSTALLED
    if not INSTALLED:
        original.ROOTS, original.freeze = ROOTS, freeze
        INSTALLED = True


def prepare():
    training, environment = configure(); rows, inputs = sources()
    from experiments.structure_net import ncm_shared_summary_inventory as inventory
    from experiments.structure_net import ncm_shared_summary_source_guard as guard
    from experiments.structure_net.ncm_streaming_source_features import FeatureCache
    protected, guard_paths, counts = inventory.protected_metadata()
    cohorts, _ = guard.development.source_inputs()
    held = [r for name in guard.development.COHORTS for r in cohorts[name]] + data.read(inventory.COMPONENT)
    if len(held) != 192 or counts['reserved_components'] != 512:
        raise ValueError('all512 protected metadata components and192 open sources required')
    text_index = guard.TextIndex([r['source'] for r in held]); held_ids = {r['id'] for r in held}
    encoding = previous.data.storage_encoding()
    for row in rows:
        if (not reserved_train_split(row) or row['source_truncated']
                or catalog.text_hash(row['source']) != row['source_sha256']
                or len(encoding.encode(row['source'], disallowed_special=())) != row['source_cl100k_tokens']
                or not 192 <= row['source_cl100k_tokens'] <= 2048
                or catalog.protected_reasons(row, protected, inventory.reservation.graph.guard.source_family)
                or row['id'] in held_ids or text_index.matches(row['source'])):
            raise ValueError('complete source or held-out source guard failed')
    public = previous.qualification.reference
    public_proof = public.verify_public(rows)
    selected = {r['id'] for r in rows}
    index = [r for r in data.read(catalog.INDEX) if r['id'] in selected]
    manifest = {k:v for k,v in data.read(catalog.MANIFEST).items() if k in selected}
    if len(index) != 1024 or set(manifest) != selected:
        raise ValueError('complete selected existing source features required')
    inputs += [*guard_paths, guard.development.SOURCE/'cases.json', public.PUBLIC_FILE]
    inputs += [Path(r['path']) for r in index]
    inputs += [Path(v[k]) for v in manifest.values() for k in ('header_path', 'payload_path')]
    root = ROOTS['prepare']; proof = freeze(root, 'guard_sources_and_replay_features', inputs, environment)
    cache = FeatureCache(index, manifest, max_bytes=256*1024**2)
    total = positions = 0
    with (root/'payloads.jsonl').open('x') as log:
        for i, row in enumerate(rows, 1):
            value = cache.get(row)
            if str(value.dtype) != 'torch.float32' or not training.torch.isfinite(value).all():
                raise ValueError('complete finite original FP32 source feature required')
            size = value.numel()*value.element_size(); total += size; positions += len(value)
            log.write(json.dumps({'id': row['id'], 'raw_bytes': size, 'positions': len(value),
                'payload_sha256': manifest[row['id']]['payload_sha256']})+'\n')
            if i % 128 == 0: print(json.dumps({'features_replayed': i, 'total': 1024}), flush=True)
    for name, value in (('cases.json', rows), ('pilot_ids.json', pilot_ids(rows)),
                        ('source_index.json', index), ('feature_manifest.json', manifest),
                        ('public_proof.json', public_proof)):
        data.write(root/name, value)
    finish(root, proof, {'status': 'fresh1024_sources_guarded_features_replayed', 'cases': 1024,
        'source_bins': dict(Counter(r['bin'] for r in rows)), 'protected_metadata': counts,
        'open_protected_sources': 192, 'raw_feature_bytes_replayed': total, 'source_positions': positions,
        'feature_cache': cache.metrics(), 'new_requests': 0},
        [root/n for n in ('cases.json', 'pilot_ids.json', 'source_index.json', 'feature_manifest.json', 'public_proof.json', 'payloads.jsonl')])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__); parser.add_argument('stage',choices=tuple(ROOTS))
    stage = parser.parse_args().stage; install(); root = ROOTS[stage]; existed = root.exists()
    try:
        if stage == 'prepare': prepare()
        elif stage in ('pilot','generate'): original.generate(stage == 'pilot')
        else: original.audit(stage == 'pilot_audit')
    except Exception as error:
        if root.exists() and not existed:
            data.write(root/'failure.json',{'error_type':type(error).__name__,'error':str(error),
                'preserve_firsts':True,'retry_same_sources_prohibited':True})
        raise
