"""Formatting-only recovery of the failed draft/review systems pilot."""
import argparse
import copy
from pathlib import Path
import re

from experiments.structure_net import ncm_evidence_teacher_pilot as original

ROOT, q, native = original.ROOT, original.q, original.native
HYPOTHESIS = "ncm-evidence-first-fresh64-teacher-generation-v2"
MODULE = "experiments.structure_net.ncm_evidence_teacher_pilot_v2"
PLAN = Path("docs/07 - Status Reports/2026-10-24_ncm-evidence-teacher-pilot-v2-plan.md")
TEST = Path("tests/structure_net/test_ncm_evidence_teacher_pilot_v2.py")
AUDITOR = Path("experiments/neural_architecture_lab/audit_ncm_evidence_teacher_pilot_v2.py")
WORKFLOW = ROOT / "20261024_evidence_teacher64_workflow_v2"
PREP = ROOT / "20261024_evidence_teacher64_prepare_v2"
PILOT = ROOT / "20261024_evidence_teacher64_systems_pilot_v2"
GENERATE = ROOT / "20261024_evidence_teacher64_candidates_v2"
FAILED_WORKFLOW, FAILED_PREP, FAILED_PILOT = original.WORKFLOW, original.PREP, original.PILOT
_parse, _request, _producing = original.parse, original.request, original.producing


def parse(content, row):
    text = content.strip()
    if text.startswith('```'):
        match = re.fullmatch(r'```(?:json)?\s*\n?([\s\S]*?)\n?```', text)
        if match is None:
            raise ValueError('one whole-response JSON fence required')
        text = match.group(1).strip()
    return _parse(text, row)


def producing(root, stage, inputs=()):
    if q.read(FAILED_WORKFLOW / 'tool_terminal_receipt.json')['exit_code'] != 1:
        raise ValueError('actual failed v1 must be preserved')
    recovery_inputs = [FAILED_WORKFLOW / 'tool_terminal_receipt.json', FAILED_WORKFLOW / 'pilot_receipt.json',
        FAILED_PREP / 'cases.json', FAILED_PREP / 'pilot_ids.json', FAILED_PILOT / 'producer.json']
    recovery_inputs += sorted((FAILED_PILOT / 'requests').rglob('attempt_*.json'))
    files = [Path(__file__), Path('experiments/neural_architecture_lab/audit_ncm_evidence_teacher_pilot.py')]
    p = _producing(root, stage, [*inputs, *recovery_inputs, *files])
    p['hypothesis_id'] = HYPOTHESIS
    p['config'].update(hypothesis_id=HYPOTHESIS, formatting_amendment='strip one enclosing optional-json code fence only',
        source_selection_namespace='ncm-evidence-first-fresh64-teacher-generation-v1',
        pilot_recovery='first original schema-valid after fence normalization; ignore every later original attempt',
        missing_cases_directory_fixed=True)
    for i, path in enumerate(files):
        p['source_hashes'][str(path)] = original.file_hash(path)
        (root / 'producer_source' / f'evidence_recovery_{i}_{path.name}').write_bytes(path.read_bytes())
    p['fingerprint'] = q.digest({k: v for k, v in p.items() if k != 'fingerprint'})
    original.write_json(root / 'producer.json', p)
    (root / 'cases').mkdir()
    return p


def request(row, stage, root, encoding, draft=None):
    old_root = FAILED_PILOT / 'requests' / row['id'] / stage
    if not old_root.exists():
        return _request(row, stage, root, encoding, draft)
    body = original.payload(row, stage, draft)
    # Recover the FIRST eligible recorded response; no network call or text edit.
    for path in sorted(old_root.glob('attempt_*.json')):
        raw = q.read(path)
        if raw['request'] != body:
            raise ValueError('original source/draft/model/prompt changed in recovery')
        value = None
        try:
            if (raw.get('http_status') != 200 or raw['response'].get('model') != original.MODELS[stage]
                    or raw['response']['choices'][0]['finish_reason'] != 'stop'):
                raise ValueError('invalid original transport/model/termination')
            value = parse(raw['response']['choices'][0]['message']['content'], row)
        except (ValueError, KeyError, TypeError, IndexError):
            continue
        if path.name != 'attempt_0.json':
            raise ValueError('this amendment requires both original first responses to be reusable')
        destination = root / 'requests' / row['id'] / stage
        destination.mkdir(parents=True, exist_ok=False)
        corrected = copy.deepcopy(raw)
        corrected.update(valid=True, original_valid=raw['valid'], recovered_from=str(path),
            recovered_sha256=original.file_hash(path),
            ignored_later_formatting_attempts=[str(p) for p in sorted(old_root.glob('attempt_*.json')) if p != path])
        target = destination / 'attempt_0.json'
        original.write_json(target, corrected)
        return {'status': 'valid', 'value': value, 'attempts': [str(target)]}
    raise ValueError('an original first cannot be recovered; new pilot sampling prohibited')


def configure():
    # Patches affect this wrapper process only; original producing source stays immutable.
    for name in ('MODULE', 'PLAN', 'TEST', 'AUDITOR', 'WORKFLOW', 'PREP', 'PILOT', 'GENERATE'):
        setattr(original, name, globals()[name])
    original.parse, original.producing, original.request = parse, producing, request


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('stage', choices=('prepare', 'pilot', 'generate', 'workflow'))
    args = parser.parse_args()
    configure()
    {'prepare': original.prepare, 'pilot': lambda: original.generate(True), 'generate': original.generate,
        'workflow': original.workflow}[args.stage]()
