"""Keep historical model identities intact when admitting the Sol replacement."""
import json
from pathlib import Path

from experiments.structure_net import ncm_sol_judge_qualification as qualification
from experiments.structure_net.ncm_memory_data import file_hash
from experiments.structure_net.ncm_paper_data import ROOT, verify

q = qualification.q
OLD_PRIMARY = 'claude_code/claude-sonnet-4-6'
SECONDARY = 'codex_sdk/gpt-5.6-terra'
CACHE = ROOT / '20261024_lm1b_native655_development_grade_v1/judgment_cache.json'
CACHE_SHA = 'ac5a5c7bda37b4acf6bcac6ba7153432fe197d887d5ef6e7a82732922644516f'
CAMPAIGN = ROOT / '20261024_lm1b_native655_development_results_v1/campaign_results.json'
CAMPAIGN_SHA = '4f8776030d08a77ae88c47bbfdf340e6da5ade083b170ba00cf0f25adc3e8d13'


def configure():
    qualification.configure()


def judge_inputs():
    secondary = ROOT / '20261023_budget128_judge_secondary_v1'
    return [qualification.DEST / n for n in ('producer.json', 'result.json', 'actual_tool_receipt.json')] + [
        qualification.AUDIT / 'result.json', qualification.AUDIT / 'actual_tool_receipt.json',
        secondary / 'producer.json', secondary / 'result.json', secondary / 'worker_terminal_receipt.json']


def require_judges():
    configure()
    primary = q.read(qualification.DEST / 'result.json')
    audit = q.read(qualification.AUDIT / 'result.json')
    secondary = ROOT / '20261023_budget128_judge_secondary_v1'
    other = q.read(secondary / 'result.json')
    if (primary['judge'] != qualification.MODEL or primary['status'] != 'passed'
            or primary['controls'] != 50 or primary['correct'] != 50
            or audit['status'] != 'replacement_judge_firsts_and_scores_replayed'
            or not audit['qualification_passed'] or audit['correct'] != 50
            or other['judge'] != SECONDARY or other['status'] != 'passed'
            or other['controls'] != 50 or other['correct'] != 50
            or q.read(qualification.DEST / 'actual_tool_receipt.json')['actual_tool_exit_code'] != 0
            or q.read(qualification.AUDIT / 'actual_tool_receipt.json')['actual_tool_exit_code'] != 0
            or q.read(secondary / 'worker_terminal_receipt.json')['exit_code'] != 0):
        raise ValueError('new Sol and original Terra50/50 qualifications with actual completion required')
    verify(q.read(qualification.DEST / 'producer.json'))
    verify(q.read(secondary / 'producer.json'))
    for path, expected in audit['artifacts'].items():
        if file_hash(path) != expected:
            raise ValueError('Sol qualification raw audit input changed: ' + path)


def archived_key(packet, role, model):
    allowed = {'primary': {OLD_PRIMARY, qualification.MODEL}, 'secondary': {SECONDARY}}
    if model not in allowed.get(role, set()):
        raise ValueError('unapproved historical judge/role')
    return q.digest({'packet': packet, 'rubric': qualification.profile.RUBRIC, 'role': role, 'model': model})


def audit_entry(key, entry):
    packet, role, row = entry['packet'], entry['role'], entry['row']
    model = row['judge']
    if (key != archived_key(packet, role, model) or row['case_id'] != packet['case_id']
            or row['packet_sha256'] != q.digest(packet)):
        raise ValueError('historical packet/rubric/model identity differs')
    if not 1 <= len(row['attempts']) <= 3:
        raise ValueError('historical attempt count differs')
    found = None
    for name in row['attempts']:
        if found is not None:
            raise ValueError('later attempt after first valid historical judgment')
        raw = q.read(name)
        request = raw['request']
        if (request['model'] != model or request['temperature'] != 0 or request['max_tokens'] != 4096
                or request['messages'][0]['content'] != qualification.profile.RUBRIC
                or json.loads(request['messages'][1]['content']) != packet):
            raise ValueError('historical raw request binding differs')
        parsed = None
        try:
            if raw.get('http_status') == 200 and raw['response'].get('model') == model:
                parsed = q.parse(raw['response']['choices'][0]['message']['content'], packet)
        except (ValueError, KeyError, TypeError, IndexError):
            pass
        if bool(raw['valid']) != (parsed is not None):
            raise ValueError('historical raw validity differs')
        if parsed is not None:
            found = parsed
    if row['status'] == 'valid':
        if found is None or row['evaluations'] != found:
            raise ValueError('historical first judgment changed')
    elif row['status'] != 'invalid' or found is not None or len(row['attempts']) != 3:
        raise ValueError('historical exhausted failure changed')


def audit_cache(cache):
    for key, entry in cache.items():
        audit_entry(key, entry)


def base_cache():
    configure()
    if file_hash(CACHE) != CACHE_SHA or file_hash(CAMPAIGN) != CAMPAIGN_SHA:
        raise ValueError('latest completed3222 canonical lineage required')
    campaign = q.read(CAMPAIGN)
    receipt = q.read(CAMPAIGN.parent / 'ncm-lm1b-native655-source-only-DEV-v1_registry_readback.json')
    if (campaign['canonical_first_judgments'] != 3222
            or campaign['classification'] != 'full128_source_only_DEV_quality_failed'
            or campaign['artifacts'].get(str(CACHE)) != CACHE_SHA
            or receipt['record']['campaign_sha256'] != CAMPAIGN_SHA
            or receipt['exact']['campaign_sha256'] != CAMPAIGN_SHA
            or q.read(CAMPAIGN.parent / 'backend_exact_readback.json')['status'] != 'independent_backend_exact_readback_passed'):
        raise ValueError('registered failed native655 campaign provenance differs')
    cache = q.read(CACHE)
    if len(cache) != 3222:
        raise ValueError('all3222 prior first judgments required')
    audit_cache(cache)
    return cache
