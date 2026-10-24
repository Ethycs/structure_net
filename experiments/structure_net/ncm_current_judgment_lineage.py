"""Pinned completed3350 lineage for prospective post-Sol-switch experiments."""
from experiments.structure_net import ncm_sol_judgment_lineage as historical
from experiments.structure_net.ncm_memory_data import file_hash

q, ROOT = historical.q, historical.ROOT
CACHE = ROOT / '20261024_sol_teacher64_semantic_grade_v1/judgment_cache.json'
CACHE_SHA = 'b9986a27ab2dd2589e71aea2241c35f0b994d80d9993890f8d6df4fdcc36ce18'
RESULTS = ROOT / '20261024_sol_teacher64_semantic_results_v1'
CAMPAIGN = RESULTS / 'campaign_results.json'
CAMPAIGN_SHA = '2d6ee5f2dbca777bfb0774569f2bd7e69ba426245dd0247f80a04d62f80d1b4b'
HYPOTHESIS = 'ncm-sol-evidence64-teacher-semantic-qualification-v1'
COUNT = 3350
WORKFLOW = ROOT / '20261024_sol_teacher64_semantic_workflow_v1'

configure = historical.configure
require_judges = historical.require_judges
audit_cache = historical.audit_cache


def inputs():
    return [CACHE, CAMPAIGN, RESULTS / (HYPOTHESIS + '_registry_readback.json'),
        RESULTS / 'backend_exact_readback.json', WORKFLOW / 'actual_tool_receipt.json',
        *historical.judge_inputs()]


def base_cache():
    configure()
    if file_hash(CACHE) != CACHE_SHA or file_hash(CAMPAIGN) != CAMPAIGN_SHA:
        raise ValueError('unchanged latest completed3350 first judgments required')
    campaign = q.read(CAMPAIGN)
    receipt = q.read(RESULTS / (HYPOTHESIS + '_registry_readback.json'))
    if (campaign['canonical_first_judgments'] != COUNT
            or campaign['classification'] != 'fixed64_sol_teacher_labels_semantically_qualified'
            or campaign['artifacts'].get(str(CACHE)) != CACHE_SHA
            or receipt['status'] != 'registered_and_read_back'
            or receipt['record']['campaign_sha256'] != CAMPAIGN_SHA
            or receipt['exact']['campaign_sha256'] != CAMPAIGN_SHA
            or q.read(RESULTS / 'backend_exact_readback.json')['status'] != 'independent_backend_exact_readback_passed'
            or q.read(WORKFLOW / 'actual_tool_receipt.json')['actual_tool_exit_code'] != 0):
        raise ValueError('completed registered3350 campaign and independent backend proof required')
    cache = q.read(CACHE)
    if len(cache) != COUNT:
        raise ValueError('all3350 historical first decisions required')
    audit_cache(cache)
    return cache
