"""Check prepared TRAIN union against open held-out text and sealed metadata."""
from collections import Counter
import hashlib
import json
import os
from pathlib import Path

from experiments.structure_net import ncm_combined_pair_inventory as union
from experiments.structure_net import ncm_shared_summary_source_guard as guard
from experiments.structure_net.ncm_recovered_pair_training_pool import Components

ROOT = union.BASE/'20261028_combined_prepared_source_guard_v1'
READBACK = union.BASE/'20261028_combined_prepared_pair_inventory_readback_v1'
RECOVERED_HELD = union.BASE/'20261027_recovered_label_sample_v1'
PLAN = Path('docs/07 - Status Reports/2026-10-28_ncm-combined-source-guard.md')
TEST = Path('tests/structure_net/test_ncm_combined_source_guard.py')
CPU = union.BASE/'20261028_combined_source_guard_CPU_v1.json'
RULE = 'normalized equality or >=32 distinct common word5grams and >=80percent shorter containment'


def aliases(row):
    values = {('hash', row['source_sha256']), ('component', row['source_component'])}
    if row.get('group_id'):
        values.add(('groups', row['group_id']))
    spans = row.get('spans', [])
    source_ids = [row.get('source_id'), *[s.get('source_id') for s in spans]]
    values.update(('families', guard.integrity.source_family(s)) for s in source_ids if s)
    values.update(('turns', s[k]) for s in spans for k in ('turn_text_sha256', 'span_text_sha256') if s.get(k))
    return values


class SourceGroups:
    def __init__(self, protected):
        self.graph = Components()
        self.protected = protected
        self.directly_blocked = set()

    def add(self, row, text_conflict=False):
        key = row['source_sha256']
        self.graph.find('source:'+key)
        for kind, value in aliases(row):
            self.graph.union('source:'+key, 'alias:'+kind+':'+value)
            if value in self.protected.get(kind, ()):
                self.directly_blocked.add(key)
        if text_conflict:
            self.directly_blocked.add(key)

    def blocked_roots(self):
        return {self.graph.find('source:'+s) for s in self.directly_blocked}

    def root(self, source_hash):
        key = 'source:'+source_hash
        if key not in self.graph.parent:
            raise ValueError('every pair must have a checked original source')
        return self.graph.find(key)


def jsonl(path):
    with path.open() as stream:
        yield from map(json.loads, stream)


def run():
    if ROOT.exists() or os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('fresh CPU-only source audit required')
    checked = union.read(CPU)
    if checked['actual_tool_exit_code'] != 0 or checked['tests_passed'] != 3:
        raise ValueError('completed source alias and whole-group propagation contracts required')
    for path, digest in checked['artifacts'].items():
        if union.sha(path) != digest:
            raise ValueError('qualified source changed')
    prepared, replay = union.completed(union.ROOT), union.completed(READBACK)
    if (prepared['distinct_content_pairs'] != 208631 or replay['rows'] != 208631
            or replay['index_sha256'] != union.sha(union.ROOT/'pairs_index.jsonl')):
        raise ValueError('complete independently counted union required')
    parent_proof = union.read(union.ROOT/'producer.json')
    for path, digest in parent_proof['inputs'].items():
        if union.sha(path) != digest:
            raise ValueError('original catalog changed')
    protected, metadata_paths, counts = guard.inventory.protected_metadata()
    if counts['reserved_components'] != 512:
        raise ValueError('all existing sealed and open reserved components required')
    metadata_index, reserved, _ = guard.inventory.reservation.parent_metadata()
    del metadata_index
    protected = {**protected, 'hash': set(protected['hashes']), 'component': set(reserved)}
    source_file = guard.development.SOURCE/'cases.json'
    if union.sha(source_file) != guard.development.SOURCE_SHA:
        raise ValueError('original unchanged source-only application DEV required')
    cohorts = guard.development.validate_sources(union.read(source_file))
    held = [r for c in guard.development.COHORTS for r in cohorts[c]]
    component = union.read(guard.inventory.COMPONENT)
    public = list(jsonl(union.PUBLIC/'public_development.jsonl'))
    union.completed(RECOVERED_HELD)
    recovered = union.read(RECOVERED_HELD/'cases.json')
    if (len(held), len(component), len(public), len(recovered)) != (128, 64, 320, 128):
        raise ValueError('all existing open source cohorts required')
    held += component+public+recovered
    for row in held:
        if hashlib.sha256(row['source'].encode()).hexdigest() != row['source_sha256']:
            raise ValueError('held-out source binding differs')
        for kind, value in aliases(row):
            protected.setdefault(kind, set()).add(value)
    index = guard.TextIndex([r['source'] for r in held])
    paths = [Path(__file__).resolve(), Path(guard.__file__).resolve(),
        Path(guard.integrity.__file__).resolve(), PLAN, TEST, CPU,
        *[root/n for root in (union.ROOT, READBACK, RECOVERED_HELD) for n in ('result.json', 'actual_tool_receipt.json')],
        union.ROOT/'pairs_index.jsonl', source_file, guard.inventory.COMPONENT,
        union.PUBLIC/'public_development.jsonl', RECOVERED_HELD/'cases.json', *metadata_paths,
        union.PUBLIC/'training.jsonl', union.RECOVERED/'sources.jsonl', union.ANCHORS/'cases.json']
    proof = {'stage': 'whole_union_fixed_heldout_source_guard',
        'config': {'open_sources': 640, 'text_rule': RULE,
            'propagate_conflicts_across_source_component_group_family_turn_aliases': True,
            'new_split_assignment': False, 'sealed_final_text_read_or_hashed': False,
            'training_admitted': False, 'new_model_calls': 0, 'new_judge_calls': 0},
        'inputs': {str(p): union.sha(p) for p in dict.fromkeys(paths)}}
    ROOT.mkdir()
    union.write(ROOT/'producer.json', proof)
    groups = SourceGroups(protected)
    seen, text_blocked, context_rows = set(), set(), 0
    streams = [jsonl(union.PUBLIC/'training.jsonl'), jsonl(union.RECOVERED/'sources.jsonl'),
               iter(union.read(union.ANCHORS/'cases.json'))]
    with (ROOT/'direct_conflicts.jsonl').open('x') as conflicts:
        for stream in streams:
            for row in stream:
                key = row['source_sha256']
                if hashlib.sha256(row['source'].encode()).hexdigest() != key:
                    raise ValueError('original training source binding differs')
                matches = index.matches(row['source']) if key not in seen else []
                if matches:
                    text_blocked.add(key)
                    conflicts.write(json.dumps({'source_sha256': key,
                        'open_heldout_matches': [{'id': held[j]['id'], **detail} for j, detail in matches]})+'\n')
                groups.add(row, text_conflict=key in text_blocked)
                seen.add(key)
                context_rows += 1
                if context_rows % 10000 == 0:
                    print(json.dumps({'source_records_checked': context_rows,
                        'unique_source_texts': len(seen), 'directly_blocked_sources': len(groups.directly_blocked)}), flush=True)
    if len(seen) != prepared['distinct_source_texts'] or context_rows != 162353+40890+2384:
        raise ValueError('complete original source population required')
    blocked = groups.blocked_roots()
    eligible = excluded = 0
    domain_counts, component_ids = Counter(), set()
    with (ROOT/'pair_source_groups.jsonl').open('x') as output:
        for pair in jsonl(union.ROOT/'pairs_index.jsonl'):
            root = groups.root(pair['source_sha256'])
            component_id = hashlib.sha256(root.encode()).hexdigest()
            component_ids.add(component_id)
            admitted = root not in blocked
            eligible += admitted
            excluded += not admitted
            for origin in pair['origins']:
                domain_counts[(origin['domain'], admitted)] += 1
            output.write(json.dumps({'id': pair['id'], 'source_sha256': pair['source_sha256'],
                'union_source_component': component_id, 'passes_this_source_guard': admitted})+'\n')
    if eligible+excluded != 208631:
        raise ValueError('every combined pair must have a source decision')
    for path, digest in proof['inputs'].items():
        if union.sha(path) != digest:
            raise ValueError('audit input changed')
    value = {'status': 'combined_fixed_heldout_source_guard_completed',
        'pairs': eligible+excluded, 'passes_source_guard': eligible, 'excluded_pairs': excluded,
        'source_context_records_checked': context_rows, 'distinct_source_texts_checked': len(seen),
        'union_source_components': len(component_ids), 'direct_text_conflict_sources': len(text_blocked),
        'direct_metadata_or_text_conflict_sources': len(groups.directly_blocked), 'excluded_connected_components': len(blocked),
        'domains': {d: {'passes': domain_counts[d, True], 'excluded': domain_counts[d, False]} for d in prepared['input_rows']},
        'open_heldout_sources': len(held), 'reserved_metadata_components': counts['reserved_components'],
        'text_rule': RULE, 'whole_connected_group_exclusion': True,
        'sealed_final_text_read_or_hashed': False, 'sealed_final_full_text_overlap_checked': False,
        'all_TRAIN_to_TRAIN_near_duplicate_pairs_compared': False,
        'training_admitted': False, 'model_calls': 0, 'judge_calls': 0, 'task_accuracy': None,
        'artifacts': {str(ROOT/n): union.sha(ROOT/n) for n in ('producer.json', 'direct_conflicts.jsonl', 'pair_source_groups.jsonl')}}
    union.write(ROOT/'result.json', value)
    print(json.dumps({k: v for k, v in value.items() if k != 'artifacts'}), flush=True)


if __name__ == '__main__':
    existed = ROOT.exists()
    try:
        run()
    except Exception as error:
        if ROOT.exists() and not existed and not (ROOT/'failure.json').exists():
            union.write(ROOT/'failure.json', {'type': type(error).__name__, 'message': str(error)})
        raise
