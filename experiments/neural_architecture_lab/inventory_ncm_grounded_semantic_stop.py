"""Recover raw first judgments and preserve the stopped request frontier."""
import argparse
from collections import Counter
import json
import os
from pathlib import Path

from experiments.structure_net import ncm_sol64_grounded_oracle_semantics as run
from experiments.structure_net.ncm_memory_data import file_hash, write_json

DEST = run.ROOT / '20261025_grounded_semantic_stop_inventory_v1'
PLAN = Path('docs/07 - Status Reports/2026-10-25_ncm-grounded-semantic-stop-inventory-plan.md')
TEST = Path('tests/neural_architecture_lab/test_ncm_grounded_semantic_stop_inventory.py')
STOP = run.ROOT / '20261024_grounded_semantic_transport_stop_v1'
HISTORICAL = run.lineage.historical


def reconstruct(packet, role, root, model, rubric, parse):
    """Return complete raw firsts; partial/empty folders retain uncertainty."""
    paths = sorted((root / 'attempts').glob('*.json'))
    if len(paths) > 3 or [p.name for p in paths] != [f"{packet['case_id']}_{i}.json" for i in range(len(paths))]:
        raise ValueError('original ordered allowance of at most three attempts required')
    expected = {'model': model, 'temperature': 0, 'max_tokens': 4096,
        'messages': [{'role': 'system', 'content': rubric},
            {'role': 'user', 'content': json.dumps(packet, ensure_ascii=False)}]}
    found, seen = None, []
    for i, path in enumerate(paths):
        if found is not None:
            raise ValueError('no later attempt after first valid result')
        raw = run.q.read(path)
        if raw['attempt'] != i or raw['request'] != expected or raw['request_sha256'] != run.q.digest(expected):
            raise ValueError('complete exact original packet/model/rubric binding required')
        value = None
        try:
            if raw.get('http_status') == 200 and raw['response'].get('model') == model:
                value = parse(raw['response']['choices'][0]['message']['content'], packet)
        except (ValueError, KeyError, TypeError, IndexError):
            pass
        if raw['valid'] != (value is not None):
            raise ValueError('raw validity differs')
        if value is not None:
            found = value
        seen.append(str(path))
    if found is None and len(paths) < 3:
        # The killed process could have consumed one additional unsaved call.
        return None, {'status': 'unknown_inflight_or_unstarted', 'saved_attempts': seen,
            'possibly_consumed_unsaved_attempts': 1, 'maximum_future_attempts': max(0, 2-len(paths))}
    row = {'case_id': packet['case_id'], 'packet_sha256': run.q.digest(packet),
        'status': 'valid' if found is not None else 'invalid', 'judge': model, 'attempts': seen}
    if found is not None:
        row['evaluations'] = found
    return row, {'status': row['status'], 'saved_attempts': seen,
        'possibly_consumed_unsaved_attempts': 0, 'maximum_future_attempts': 0}


def main():
    if DEST.exists() or os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('fresh CPU-only stop inventory required')
    run.native.vendor.corpus.quality.dependencies()
    if (run.q.read(run.WORKFLOW / 'actual_tool_receipt.json')['actual_tool_exit_code'] != 1
            or run.q.read(run.WORKFLOW / 'grade_receipt.json')['exit_code'] != -15
            or run.q.read(run.GRADE / 'worker_terminal_receipt.json')['exit_code'] != -15):
        raise ValueError('actually stopped workflow and grade worker required')
    for root in (run.WORKFLOW, run.PREP, run.PILOT, run.GRADE):
        run.native.vendor.verify(run.q.read(root / 'producer.json'))
    run.native.vendor.terminal(run.PREP)
    run.native.vendor.terminal(run.PILOT)
    old = run.q.read(run.lineage.CACHE)
    if (file_hash(run.lineage.CACHE) != run.lineage.CACHE_SHA
            or file_hash(run.lineage.CAMPAIGN) != run.lineage.CAMPAIGN_SHA or len(old) != 3523):
        raise ValueError('pinned preceding completed 3523-first lineage required')
    for name in ('actual_tool_receipt.json', 'backend_actual_tool_receipt.json'):
        path = run.lineage.codec.WORKFLOW / name if name == 'actual_tool_receipt.json' else run.lineage.RESULTS / name
        if run.q.read(path)['actual_tool_exit_code'] != 0:
            raise ValueError('actual preceding completed study and backend proof required')
    partial = run.q.read(run.GRADE / 'judgment_cache.json')
    pilot = run.q.read(run.PILOT / 'judgment_cache.json')
    if (len(partial) != 3536 or any(partial.get(k) != v for k, v in old.items())
            or any(partial.get(k) != v for k, v in pilot.items())
            or run.q.read(STOP / 'cache_before_stop.json') != partial):
        raise ValueError('unchanged ancestors, pilot and exact stop cache required')
    HISTORICAL.audit_cache(partial)
    packets = run.q.read(run.PREP / 'packets.json')
    cases = run.q.read(run.PREP / 'cases.json')
    if len(cases) != 64 or any(len(packets[a]) != 64 for a in run.ARMS):
        raise ValueError('complete fixed 64-case two-arm source binding required')
    config = run.q.read(run.GRADE / 'producer.json')['config']
    if config['rubric'] != HISTORICAL.qualification.profile.RUBRIC or config['max_attempts'] != 3:
        raise ValueError('frozen qualified profile and allowance required')
    planned = {}
    for arm in run.ARMS:
        expected = [run.q.packet(c['id'], c['source'], [{'candidate_id': 'summary', 'text': c['texts'][arm]}]) for c in cases]
        if packets[arm] != expected:
            raise ValueError('source or first reconstruction packet changed')
        for role, model in config['judges'].items():
            for packet in packets[arm]:
                planned[HISTORICAL.archived_key(packet, role, model)] = (packet, role, model)
    merged, ledger, recovered = dict(partial), [], []
    all_paths = []
    for owner in (run.PILOT, run.GRADE):
        for root in sorted((owner / 'requests').glob('*/*')):
            if root.name not in planned:
                raise ValueError('request directory not bound to prepared packets')
            packet, role, model = planned[root.name]
            if root.parent.name != role:
                raise ValueError('original judge role differs')
            row, state = reconstruct(packet, role, root, model, config['rubric'], run.q.parse)
            if row is not None:
                entry = {'packet': packet, 'role': role, 'row': row, 'origin': str(root)}
                HISTORICAL.audit_entry(root.name, entry)
                if (root / 'row.json').exists() and run.q.read(root / 'row.json') != row:
                    raise ValueError('saved first row differs from raw response replay')
                if root.name in merged and merged[root.name] != entry:
                    raise ValueError('a canonical first cannot be replaced')
                if root.name not in merged:
                    merged[root.name] = entry
                    recovered.append(root.name)
            elif root.name in merged or (root / 'row.json').exists():
                raise ValueError('incomplete raw folder cannot have a complete first')
            ledger.append({'key': root.name, 'role': role, 'owner': str(owner), **state})
            all_paths.extend(p for p in root.rglob('*') if p.is_file())
    counts = Counter(r['status'] for r in ledger)
    if (len(merged) != 3539 or len(recovered) != 3 or len(ledger) != 19
            or counts != {'valid': 4, 'invalid': 12, 'unknown_inflight_or_unstarted': 3}
            or any(merged.get(k) != v for k, v in partial.items())):
        raise ValueError('all raw firsts, unknown frontier and unchanged ancestors required')
    HISTORICAL.audit_cache(merged)
    files = [Path(__file__), PLAN, TEST, Path(run.__file__), Path(HISTORICAL.__file__),
        Path(run.q.__file__), Path(HISTORICAL.qualification.profile.__file__)]
    inputs = [*files, *all_paths, run.lineage.CACHE, run.lineage.CAMPAIGN,
        run.PREP / 'cases.json', run.PREP / 'packets.json', run.PILOT / 'judgment_cache.json',
        run.GRADE / 'judgment_cache.json', STOP / 'stop_request.json', STOP / 'cache_before_stop.json',
        *[run.WORKFLOW / n for n in ('actual_tool_receipt.json', 'grade_receipt.json')],
        *[root / 'producer.json' for root in (run.WORKFLOW, run.PREP, run.PILOT, run.GRADE)]]
    p = run.native.vendor.freeze(DEST, 'recover_stopped_semantic_firsts_and_request_frontier', inputs,
        {'new_requests': 0, 'completed_producer_globals_rebound': False, 'semantic_rerolls': False,
         'task_accuracy': None, 'classification': 'transport_invalid_semantics_unestablished',
         'final_test_opened': False, 'speed_tested': False})
    write_json(DEST / 'judgment_cache.json', merged)
    write_json(DEST / 'request_frontier.json', ledger)
    write_json(DEST / 'recovered_first_keys.json', recovered)
    run.native.vendor.verify(p)
    campaign = {'classification': 'transport_invalid_semantics_unestablished',
        'canonical_first_judgments': len(merged), 'preceding_completed_firsts': len(old),
        'original_stop_cache_firsts': len(partial), 'recovered_exhausted_raw_firsts': len(recovered),
        'new_valid_firsts': 4, 'new_invalid_firsts': 12,
        'unknown_request_keys': 3, 'unknown_maximum_future_attempts_each': 2,
        'request_folders': len(ledger), 'request_folder_status_counts': dict(counts),
        'all_prior_firsts_unchanged': True, 'new_requests': 0, 'semantic_accuracy_measured': False,
        'student_task_accuracy': None, 'source_on_qualified': False,
        'final_test_opened': False, 'speed_tested': False,
        'artifacts': {str(path): file_hash(path) for path in [*inputs, DEST / 'producer.json',
            DEST / 'judgment_cache.json', DEST / 'request_frontier.json', DEST / 'recovered_first_keys.json']}}
    write_json(DEST / 'campaign_results.json', campaign)
    print(json.dumps({k: v for k, v in campaign.items() if k != 'artifacts'}), flush=True)


if __name__ == '__main__':
    main()
