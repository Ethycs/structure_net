"""Complete, immutable inputs for the larger conversational training candidate."""
import hashlib
import json
import os
from pathlib import Path
import time

from experiments.structure_net import ncm_recovered_training_schedule as schedule
from experiments.structure_net import ncm_recovered_training_stream as stream
from experiments.structure_net import ncm_recovered_stream_pilot as qualified
from experiments.structure_net.ncm_indexed_jsonl import IndexedJSONL

BASE = qualified.BASE
FEATURES = BASE/'20261028_recovered_paraphrase_features_full_v1'
BINDINGS = BASE/'20261027_isolated_task_supervision_preflight_v1/bindings.json'


class Proofs:
    """Hash each unchanged file once within this process; never trust saved cache.

    Only large, older files can hit; recent writes and small files are rehashed.
    Every hit checks device/inode/size and modification/change times. The recent
    write guard handles filesystems that coalesce closely spaced timestamps.
    Misses bind an open descriptor and reject changes during hashing or a path
    replacement. This avoids repeatedly hashing the same completed shards via
    several receipts. Explicit proof verification starts with an empty cache;
    per-record checks in the training reader are never cached.
    """
    def __init__(self): self.cache, self.inputs = {}, {}

    @staticmethod
    def signature(stat):
        return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns

    def sha(self, path):
        path = Path(path); key = str(path.absolute())
        signature = self.signature(path.stat())
        cacheable = signature[2] >= 1024**2 and time.time_ns()-max(signature[3:]) >= 2_000_000_000
        if cacheable and key in self.cache and self.cache[key][0] == signature: return self.cache[key][1]
        h = hashlib.sha256()
        with path.open('rb') as f:
            before = self.signature(os.fstat(f.fileno()))
            if before != signature: raise ValueError('artifact replaced before hashing')
            for block in iter(lambda: f.read(8*1024**2), b''): h.update(block)
            if self.signature(os.fstat(f.fileno())) != before or self.signature(path.stat()) != before:
                raise ValueError('artifact changed while hashing')
        self.cache[key] = (signature, h.hexdigest())
        return h.hexdigest()

    def bind(self, path, expected=None):
        key = str(Path(path).absolute()); actual = self.sha(path)
        if expected is not None and actual != expected: raise ValueError('bound artifact differs: ' + key)
        if key in self.inputs and self.inputs[key] != actual: raise ValueError('artifact changed within consumer')
        self.inputs[key] = actual
        return actual

    def read(self, path):
        before = self.bind(path); value = json.loads(Path(path).read_text())
        if self.sha(path) != before: raise ValueError('JSON artifact changed while reading')
        return value

    def completed(self, root, *, result='result.json', receipt='actual_tool_receipt.json'):
        root = Path(root); observed = self.read(root/receipt)
        if observed['actual_tool_exit_code'] != 0: raise ValueError('actual successful terminal receipt required')
        bindings = {str(Path(p).absolute()): h for p, h in observed['artifacts'].items()}
        key = str((root/result).absolute())
        if bindings.get(key) != self.sha(root/result): raise ValueError('terminal receipt must bind the exact result')
        for p, h in bindings.items(): self.bind(p, h)
        value = self.read(root/result)
        for p, h in value.get('artifacts', {}).items(): self.bind(p, h)
        return value

    def verify(self, values):
        self.cache.clear()
        for p, h in values.items():
            if self.sha(p) != h: raise ValueError('immutable producer dependency changed: ' + p)


def lines(path):
    with Path(path).open() as f: return list(map(json.loads, f))


def ready(proofs):
    """Full population only: live or partial feature producers are inadmissible."""
    from experiments.structure_net import ncm_task_supervision_training as task
    config = schedule.CONFIG; config.validate()
    catalog = proofs.completed(qualified.CATALOG)
    metadata = proofs.completed(qualified.META)
    codes_result = proofs.completed(qualified.CODES)
    feature_result = proofs.completed(FEATURES)
    feature_audit = proofs.completed(FEATURES, result='audit_result.json', receipt='audit_actual_tool_receipt.json')
    io_pilot = proofs.completed(qualified.ROOT)
    balanced = proofs.completed(schedule.balancing.ROOT)
    if (catalog['coverage']['pairs'] != 43894 or catalog['coverage']['distinct_source_texts'] != 38484
            or metadata['cases'] != 40890 or codes_result['frames'] != 43894
            or codes_result['codec_sha256'] != qualified.CODEC or not codes_result['all_disk_records_replayed']
            or feature_result['cases'] != 40890 or feature_result['body_sha256'] != qualified.BODY
            or not feature_audit['all_disk_records_replayed'] or feature_audit['cases'] != 40890
            or not io_pilot['serial_and_prefetched_bytes_identical'] or not io_pilot['resumed_suffix_identical']
            or balanced['updates'] != config.updates or not balanced['all_variants_covered']):
        raise ValueError('complete fixed recovered corpus, features, codes and scheduling qualifications required')
    bindings = proofs.read(BINDINGS)
    anchor_root, anchor_catalog = Path(bindings['targets']['full']), Path(bindings['catalog']['prepare'])
    proofs.completed(anchor_root); proofs.completed(anchor_catalog)
    if (not proofs.completed(bindings['targets']['full_audit'])['training_targets_admitted']
            or not proofs.completed(bindings['catalog']['audit'])['training_data_admitted']):
        raise ValueError('complete reviewed anchor admission required')
    parent = proofs.completed(task.PARENT)
    if (parent['checkpoint_sha256'] != task.schedule.PARENT_CHECKPOINT_SHA
            or parent['final_model_sha256'] != task.PARENT_MODEL_SHA or not parent['strict_reload_passed']):
        raise ValueError('exact completed epoch2 initializer required')
    proofs.bind(task.PARENT/'model.pt', task.schedule.PARENT_CHECKPOINT_SHA)
    pair_path = qualified.CATALOG/'pairs.jsonl'
    pairs = IndexedJSONL(pair_path, lines(qualified.CATALOG/'pairs_index.jsonl'), proofs.bind(pair_path))
    code_store = stream.CodeStore(codes_result['binary'], lines(codes_result['index']),
        expected_sha256=proofs.bind(codes_result['binary']), codec_sha256=qualified.CODEC)
    meta = lines(qualified.META/'index.jsonl')
    feature_index = Path(feature_result['storage'])/'index.jsonl'
    dataset = stream.RecoveredTrainingDataset(pairs, code_store, meta, lines(feature_index),
        body_sha256=qualified.BODY, codec_sha256=qualified.CODEC)
    anchors = proofs.read(anchor_root/'cases.json')
    index, manifest = [proofs.read(anchor_catalog/n) for n in ('source_index.json', 'feature_manifest.json')]
    if len(dataset) != 43894 or len(anchors) != config.anchor_cases: raise ValueError('full populations required')
    cache = task.catalog.FeatureCache(index, manifest, max_bytes=config.cache_bytes)
    sources = stream.MixedSources(dataset, anchors, cache)
    groups = schedule.balancing.groups_for(dataset.joins, config)
    if groups != proofs.read(schedule.balancing.ROOT/'groups.json'):
        raise ValueError('source-balanced target rotation changed')
    controls = []
    for i in range(len(dataset)):
        pair = pairs[i]; code_store.get(pair)
        controls.append({'population': 'recovered', 'index': i, 'id': pair['id'],
            'source_positions': dataset.metadata[pair['source_record_id']]['source_positions'],
            'native_lexical_tokens': pair['native_lexical_tokens']})
    anchor_positions = {r['id']: r['source_positions'] for r in index}
    for i, pair in enumerate(anchors):
        controls.append({'population': 'anchor', 'index': i, 'id': pair['id'],
            'source_positions': anchor_positions[pair['id']], 'native_lexical_tokens': pair['native_lexical_tokens']})
    selected = [controls[0], min(controls, key=lambda r: (r['source_positions'], r['id'])),
        max(controls, key=lambda r: (r['source_positions'], r['id'])),
        max(controls, key=lambda r: (r['native_lexical_tokens'], r['id']))]
    unique = {r['id']: r for r in selected}
    for row in controls:
        if len(unique) == 4: break
        unique.setdefault(row['id'], row)
    for p in (qualified.CATALOG/'pairs_index.jsonl', qualified.META/'index.jsonl',
            Path(codes_result['index']), feature_index): proofs.bind(p)
    return sources, groups, list(unique.values())
