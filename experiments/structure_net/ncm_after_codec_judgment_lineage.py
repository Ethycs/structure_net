"""Pinned completed3523 lineage after the two failed standalone codec controls."""
from experiments.structure_net import ncm_after_bert_judgment_lineage as preceding
from experiments.structure_net import ncm_sol64_codec_semantics as codec
from experiments.structure_net.ncm_memory_data import file_hash

q, ROOT = preceding.q, preceding.ROOT
historical = preceding.preceding.historical
CACHE = codec.GRADE / 'judgment_cache.json'
CACHE_SHA = '3e4516337ef8596700c134ff9bd4af59f55984b734e82723242872b2c7d5fadc'
RESULTS = codec.RESULTS
CAMPAIGN = RESULTS / 'campaign_results.json'
CAMPAIGN_SHA = '101e31047826cc73a3a27030cd973f9f007abc85985e6ed2ac030f4e10c35e19'
HYPOTHESIS = codec.HYPOTHESIS
COUNT = 3523
configure, require_judges, audit_cache = preceding.configure, preceding.require_judges, preceding.audit_cache


def inputs():
    return [CACHE, CAMPAIGN, RESULTS / (HYPOTHESIS + '_registry_readback.json'),
        RESULTS / 'backend_exact_readback.json', RESULTS / 'backend_actual_tool_receipt.json',
        codec.WORKFLOW / 'actual_tool_receipt.json', *historical.judge_inputs()]


def base_cache():
    configure()
    if file_hash(CACHE) != CACHE_SHA or file_hash(CAMPAIGN) != CAMPAIGN_SHA:
        raise ValueError('latest completed3523 codec-control lineage changed')
    state = q.read(CAMPAIGN)
    record = q.read(RESULTS / (HYPOTHESIS + '_registry_readback.json'))
    if (state['canonical_first_judgments'] != COUNT or state['artifacts'].get(str(CACHE)) != CACHE_SHA
            or state['classification'] != 'fixed64_codec_component_quality_failed'
            or state['qualified_codecs'] or record['record']['campaign_sha256'] != CAMPAIGN_SHA
            or record['exact']['campaign_sha256'] != CAMPAIGN_SHA
            or q.read(RESULTS / 'backend_exact_readback.json')['status'] != 'independent_backend_exact_readback_passed'
            or q.read(RESULTS / 'backend_actual_tool_receipt.json')['actual_tool_exit_code'] != 0
            or q.read(codec.WORKFLOW / 'actual_tool_receipt.json')['actual_tool_exit_code'] != 0):
        raise ValueError('actual completed failed-codec study registration and backend proof required')
    cache, old = q.read(CACHE), q.read(preceding.CACHE)
    if len(cache) != COUNT or any(cache.get(k) != v for k, v in old.items()):
        raise ValueError('all3523 firsts and their3478 ancestors must remain')
    audit_cache(cache)
    return cache
