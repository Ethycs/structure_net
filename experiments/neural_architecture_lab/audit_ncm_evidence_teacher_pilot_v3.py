"""Replay all64 first candidates and all four original first pilot responses."""
from pathlib import Path

from experiments.structure_net import ncm_evidence_teacher_pilot_v3 as run
from experiments.neural_architecture_lab import audit_ncm_evidence_teacher_pilot as audit


def main():
    run.configure()
    identities = run.q.read(run.FAILED_PREP / 'pilot_ids.json')
    rows = {r['id']: r for r in run.q.read(run.FAILED_PREP / 'cases.json')}
    count = 0
    for identity in identities:
        for stage in ('draft', 'review'):
            new = run.q.read(run.PILOT / 'requests' / identity / stage / 'attempt_0.json')
            old = Path(new['recovered_from'])
            expected_old = run.FAILED_PILOT / 'requests' / identity / stage / 'attempt_0.json'
            original = run.q.read(old)
            annotations = {'original_valid', 'recovered_from', 'recovered_sha256', 'ignored_later_formatting_attempts'}
            base = {k: v for k, v in new.items() if k not in annotations}
            base['valid'] = original['valid']
            if (old != expected_old or base != original or new['original_valid'] != original['valid']
                    or new['recovered_sha256'] != run.original.file_hash(old)
                    or new['ignored_later_formatting_attempts'] != [str(p) for p in sorted(old.parent.glob('attempt_*.json')) if p != old]
                    or not new['valid']):
                raise ValueError('original first response bytes/chronology differ')
            run.parse(original['response']['choices'][0]['message']['content'], rows[identity])
            count += 1
    if count != 4:
        raise ValueError('all four original first pilot responses required')
    audit.DEST = run.ROOT / '20261024_evidence_teacher64_raw_audit_v3'
    audit.main()


if __name__ == '__main__':
    main()
