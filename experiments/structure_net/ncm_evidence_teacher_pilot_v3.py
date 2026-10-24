"""Recover the single leading JSON block, retaining original candidate text."""
import argparse
from pathlib import Path
import re

from experiments.structure_net import ncm_evidence_teacher_pilot_v2 as recovery

original, q, ROOT = recovery.original, recovery.q, recovery.ROOT
HYPOTHESIS = 'ncm-evidence-first-fresh64-teacher-generation-v3'
MODULE = 'experiments.structure_net.ncm_evidence_teacher_pilot_v3'
PLAN = Path('docs/07 - Status Reports/2026-10-24_ncm-evidence-teacher-pilot-v3-plan.md')
TEST = Path('tests/structure_net/test_ncm_evidence_teacher_pilot_v3.py')
AUDITOR = Path('experiments/neural_architecture_lab/audit_ncm_evidence_teacher_pilot_v3.py')
WORKFLOW = ROOT / '20261024_evidence_teacher64_workflow_v3'
PREP = ROOT / '20261024_evidence_teacher64_prepare_v3'
PILOT = ROOT / '20261024_evidence_teacher64_systems_pilot_v3'
GENERATE = ROOT / '20261024_evidence_teacher64_candidates_v3'
FAILED_WORKFLOW, FAILED_PREP, FAILED_PILOT = recovery.FAILED_WORKFLOW, recovery.FAILED_PREP, recovery.FAILED_PILOT
FAILED_V2 = recovery.WORKFLOW


def parse(content, row):
    text = content.strip()
    if text.startswith('```'):
        match = re.match(r'```(?:json)?\s*\n?([\s\S]*?)\n?```', text)
        if match is None or '```' in text[match.end():] or re.search(r'[{}]', text[match.end():]):
            raise ValueError('one leading JSON block with no additional candidate object required')
        text = match.group(1).strip()
    return recovery._parse(text, row)


def producing(root, stage, inputs=()):
    if q.read(FAILED_V2 / 'tool_terminal_receipt.json')['exit_code'] != 1:
        raise ValueError('actual failed v2 must be preserved')
    files = [Path(__file__), Path(recovery.__file__), FAILED_V2 / 'tool_terminal_receipt.json', FAILED_V2 / 'pilot_receipt.json']
    p = recovery.producing(root, stage, [*inputs, *files])
    p['hypothesis_id'] = HYPOTHESIS
    p['config'].update(hypothesis_id=HYPOTHESIS,
        formatting_amendment='one leading JSON code block; allow trailing prose only with no braces or further code blocks',
        candidate_text_edited=False, original_first_responses_reused=True,
        gateway_max_tokens_not_asserted_as_actual_usage=True)
    for i, path in enumerate(files[:2]):
        p['source_hashes'][str(path)] = original.file_hash(path)
        (root / 'producer_source' / f'evidence_recovery_v3_{i}_{path.name}').write_bytes(path.read_bytes())
    p['fingerprint'] = q.digest({k: v for k, v in p.items() if k != 'fingerprint'})
    original.write_json(root / 'producer.json', p)
    return p


def configure():
    recovery.configure()
    for name in ('MODULE', 'PLAN', 'TEST', 'AUDITOR', 'WORKFLOW', 'PREP', 'PILOT', 'GENERATE'):
        setattr(original, name, globals()[name])
    recovery.parse = parse
    original.parse, original.producing = parse, producing


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('stage', choices=('prepare', 'pilot', 'generate', 'workflow'))
    args = parser.parse_args()
    configure()
    {'prepare': original.prepare, 'pilot': lambda: original.generate(True), 'generate': original.generate,
        'workflow': original.workflow}[args.stage]()
