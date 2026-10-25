"""Completed failure lineage: all raw firsts and uncertain allowances preserved."""
from experiments.structure_net import ncm_after_codec_judgment_lineage as previous
from experiments.structure_net.ncm_memory_data import file_hash

q, ROOT = previous.q, previous.ROOT
CACHE = ROOT / '20261025_grounded_semantic_stop_inventory_v1/judgment_cache.json'
CACHE_SHA = 'b03c85b65b5ffce650482ec7ef02e2056d1c0a4547d517cc1309b9f8a7880423'
FRONTIER = CACHE.parent / 'request_frontier.json'
RESULTS = ROOT / '20261025_interrupted_collection_results_v1'
CAMPAIGN = RESULTS / 'campaign_results.json'
CAMPAIGN_SHA = '54cf9c3f9f2285b9e8d037d8fcb81dd164b38d751a689098eb936c65f7f9cdb1'
HYPOTHESIS = 'ncm-grounded-semantic-transport-stop-v1'
COUNT = 3539
configure, require_judges, audit_cache = previous.configure, previous.require_judges, previous.audit_cache


def inputs():
    return [CACHE, FRONTIER, CAMPAIGN, RESULTS / (HYPOTHESIS + '_registry_readback.json'),
        RESULTS / 'actual_tool_receipt.json', RESULTS / 'backend_exact_readback.json',
        RESULTS / 'backend_actual_tool_receipt.json', CACHE.parent / 'actual_tool_receipt.json',
        *previous.inputs()]


def base_cache():
    if file_hash(CACHE) != CACHE_SHA or file_hash(CAMPAIGN) != CAMPAIGN_SHA:
        raise ValueError('completed 3539-first failure lineage changed')
    state = q.read(CAMPAIGN)
    record = q.read(RESULTS / (HYPOTHESIS + '_registry_readback.json'))
    if (state['classification'] != 'transport_invalid_semantics_unestablished'
            or state['canonical_first_judgments'] != COUNT or state['artifacts'].get(str(CACHE)) != CACHE_SHA
            or state['unknown_request_keys'] != 3 or state['source_on_qualified']
            or state['student_task_accuracy'] is not None
            or record['record']['campaign_sha256'] != CAMPAIGN_SHA or record['exact']['campaign_sha256'] != CAMPAIGN_SHA
            or q.read(RESULTS / 'backend_exact_readback.json')['status'] != 'independent_backend_exact_readback_passed'):
        raise ValueError('registered transport-invalid ledger and independent backend proof required')
    for path in (RESULTS / 'actual_tool_receipt.json', RESULTS / 'backend_actual_tool_receipt.json',
            CACHE.parent / 'actual_tool_receipt.json'):
        if q.read(path)['actual_tool_exit_code'] != 0:
            raise ValueError('actual failure-ledger completion and backend readback required')
    if file_hash(FRONTIER) != state['artifacts'][str(FRONTIER)]:
        raise ValueError('preserved uncertain request allowances changed')
    cache, old = q.read(CACHE), q.read(previous.CACHE)
    if file_hash(previous.CACHE) != previous.CACHE_SHA or len(old) != previous.COUNT:
        raise ValueError('unchanged preceding 3523-first lineage required')
    if len(cache) != COUNT or any(cache.get(k) != v for k, v in old.items()):
        raise ValueError('all preceding and recovered firsts required')
    audit_cache(cache)
    return cache


def uncertain_allowances():
    base_cache()
    values = [r for r in q.read(FRONTIER) if r['status'] == 'unknown_inflight_or_unstarted']
    if (len(values) != 3 or len({r['key'] for r in values}) != 3
            or any(r['saved_attempts'] or r['maximum_future_attempts'] != 2
                or r['possibly_consumed_unsaved_attempts'] != 1 for r in values)):
        raise ValueError('three original uncertain calls retain at most two future attempts each')
    return {r['key']: r['maximum_future_attempts'] for r in values}
