"""Reuse guarded hashes within one consumer process; never across processes."""
import argparse
from pathlib import Path

from experiments.structure_net import ncm_recovered_paraphrase_training as study

OriginalProofs = study.inputs.Proofs
PLAN = Path('docs/07 - Status Reports/2026-10-28_ncm-recovered-training-validation-cache.md')
TEST = Path('tests/structure_net/test_ncm_recovered_training_verified_cache.py')
CPU = study.inputs.BASE/'20261028_recovered_training_verified_cache_CPU_v1.json'
POLICY = {'identity': 'process-local-stat-guarded-hash-reuse-v1',
          'cross_process_cache': False, 'first_access_full_SHA256': True,
          'stat_guards': ['device', 'inode', 'size', 'mtime_ns', 'ctime_ns'],
          'minimum_cacheable_bytes': 1024**2, 'recent_write_guard_seconds': 2,
          'per_record_payload_and_reconstruction_checks': 'unchanged'}


class ProcessProofs(OriginalProofs):
    """Keep the existing sha() guards when verifying an entire dependency map.

    The old verify() clears all previously computed hashes, forcing another
    whole-corpus read within the same stage. sha() already rejects changed file
    identities/timestamps and only caches large, sufficiently old files. Each
    independently launched stage starts empty, including every audit. No declared
    or saved digest is ever installed as if it had been measured locally.
    """
    def verify(self, values):
        for path, expected in values.items():
            if self.sha(path) != expected:
                raise ValueError('immutable producer dependency changed: '+str(path))


def install():
    proof = OriginalProofs(); checked = proof.read(CPU)
    if checked['actual_tool_exit_code'] != 0 or checked['tests_passed'] != 3:
        raise ValueError('guarded cache reuse requires successful CPU contracts')
    proof.verify(checked['artifacts'])
    # The already running original preparation remains untouched and mandatory.
    study.ROOTS = {name: root if name == 'prepare' else root.with_name(root.name.removesuffix('_v1')+'_v2')
                   for name, root in study.ROOTS.items()}
    study.STORAGE = Path('/data2/structure_net/ncm_recovered_paraphrase_20261028_v1/training_v2')
    study.inputs.Proofs = ProcessProofs
    study.TESTS = [*study.TESTS, TEST]
    original_freeze = study.freeze
    def freeze(root, stage, proofs, training, environment):
        for path in (PLAN, TEST, CPU): proofs.bind(path)
        return original_freeze(root, stage, proofs, training,
                               {**environment, 'input_validation_policy': POLICY})
    study.freeze = freeze


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('stage', choices=('pilot', 'train', 'audit_pilot', 'audit_train'))
    args = parser.parse_args(); install()
    stage = {'audit_pilot': 'pilot_audit', 'audit_train': 'audit'}.get(args.stage, args.stage)
    root = study.ROOTS[stage]; existed = root.exists()
    try:
        if args.stage.startswith('audit_'):
            from experiments.neural_architecture_lab import audit_ncm_recovered_paraphrase_training as audit
            audit.audit(args.stage.removeprefix('audit_'))
        else: study.run(args.stage)
    except Exception as error:
        if root.exists() and not existed:
            study.data.write(root/'failure.json', {'type': type(error).__name__, 'message': str(error),
                                                  'task_accuracy': None})
        raise


if __name__ == '__main__': main()
