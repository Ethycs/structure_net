"""Replay v2 candidate pipeline and exact formatting-only reuse of all pilot firsts."""
from pathlib import Path

from experiments.structure_net import ncm_evidence_teacher_pilot_v2 as run
from experiments.neural_architecture_lab import audit_ncm_evidence_teacher_pilot as audit


def main():
    run.configure()
    pilot_ids = run.q.read(run.FAILED_PREP / 'pilot_ids.json')
    rows = {r['id']: r for r in run.q.read(run.FAILED_PREP / 'cases.json')}
    count = 0
    for identity in pilot_ids:
        for stage in ('draft', 'review'):
            target = run.PILOT / 'requests' / identity / stage / 'attempt_0.json'
            corrected = run.q.read(target)
            old = Path(corrected['recovered_from'])
            expected_old = run.FAILED_PILOT / 'requests' / identity / stage / 'attempt_0.json'
            original = run.q.read(old)
            annotations = {'original_valid', 'recovered_from', 'recovered_sha256', 'ignored_later_formatting_attempts'}
            base = {k: v for k, v in corrected.items() if k not in annotations}
            base['valid'] = original['valid']
            if (old != expected_old or base != original or corrected['original_valid'] != original['valid']
                    or corrected['recovered_sha256'] != run.original.file_hash(old)
                    or corrected['ignored_later_formatting_attempts'] != [str(p) for p in sorted(old.parent.glob('attempt_*.json')) if p != old]
                    or not corrected['valid']):
                raise ValueError('all original pilot response bytes/first chronology must remain exact')
            run.parse(original['response']['choices'][0]['message']['content'], rows[identity])
            count += 1
    if count != 4:
        raise ValueError('all four pilot first responses required')
    audit.DEST = run.ROOT / '20261024_evidence_teacher64_raw_audit_v2'
    audit.main()


if __name__ == '__main__':
    main()
