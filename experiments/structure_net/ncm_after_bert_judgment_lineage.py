"""Completed BERT-control lineage, to be pinned before any new semantic calls."""
from experiments.structure_net import ncm_current_judgment_lineage as preceding
from experiments.structure_net import ncm_sol64_bert_roundtrip_recovery as rendering
from experiments.structure_net.ncm_memory_data import file_hash

q, ROOT = preceding.q, preceding.ROOT
CACHE = rendering.GRADE / 'judgment_cache.json'
RESULTS = rendering.RESULTS
CAMPAIGN = RESULTS / 'campaign_results.json'
HYPOTHESIS = rendering.HYPOTHESIS
CACHE_SHA = 'cca81b0e90aa71e40197b843c6382f3c6ad3497c818f415a07886e211036d318'
CAMPAIGN_SHA = 'fc3219b54785da45db7ccdc43bf3d1810a8ba97d52e50fd0c29d44ac5fed2238'
COUNT = 3478
configure, require_judges, audit_cache = preceding.configure, preceding.require_judges, preceding.audit_cache


def inputs():
    return [CACHE, CAMPAIGN, RESULTS / (HYPOTHESIS + '_registry_readback.json'),
        RESULTS / 'backend_exact_readback.json', RESULTS / 'backend_actual_tool_receipt.json',
        rendering.WORKFLOW / 'actual_tool_receipt.json', *preceding.historical.judge_inputs()]


def base_cache():
    configure()
    if None in (CACHE_SHA, CAMPAIGN_SHA, COUNT):
        raise ValueError('pin actual completed BERT canonical lineage before launch')
    if file_hash(CACHE) != CACHE_SHA or file_hash(CAMPAIGN) != CAMPAIGN_SHA:
        raise ValueError('latest completed BERT canonical lineage changed')
    state = q.read(CAMPAIGN)
    record = q.read(RESULTS / (HYPOTHESIS + '_registry_readback.json'))
    if (state['canonical_first_judgments'] != COUNT or state['artifacts'].get(str(CACHE)) != CACHE_SHA
            or state['classification'] not in ('fixed64_BERT_roundtrip_semantically_qualified', 'fixed64_BERT_roundtrip_quality_failed')
            or record['record']['campaign_sha256'] != CAMPAIGN_SHA or record['exact']['campaign_sha256'] != CAMPAIGN_SHA
            or q.read(RESULTS / 'backend_exact_readback.json')['status'] != 'independent_backend_exact_readback_passed'
            or q.read(RESULTS / 'backend_actual_tool_receipt.json')['actual_tool_exit_code'] != 0
            or q.read(rendering.WORKFLOW / 'actual_tool_receipt.json')['actual_tool_exit_code'] != 0):
        raise ValueError('actual completed BERT registration and backend proof required')
    cache = q.read(CACHE)
    if len(cache) != COUNT:
        raise ValueError('all completed BERT firsts required')
    old = q.read(preceding.CACHE)
    if any(cache.get(k) != v for k, v in old.items()):
        raise ValueError('an earlier3350 first judgment changed')
    audit_cache(cache)
    return cache
