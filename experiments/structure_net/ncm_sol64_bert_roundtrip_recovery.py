"""Recover the CPU dependency setup before any original grading request existed."""
import argparse
from pathlib import Path

from experiments.structure_net import ncm_sol64_bert_roundtrip as base

MODULE = 'experiments.structure_net.ncm_sol64_bert_roundtrip_recovery'
AUDITOR_MODULE = 'experiments.neural_architecture_lab.finalize_ncm_sol64_bert_roundtrip_recovery'
AUDITOR = Path('experiments/neural_architecture_lab/finalize_ncm_sol64_bert_roundtrip_recovery.py')
PLAN = Path('docs/07 - Status Reports/2026-10-24_ncm-sol64-bert-roundtrip-recovery.md')
TEST = Path('tests/structure_net/test_ncm_sol64_bert_roundtrip_recovery.py')
WORKFLOW, PREP, PILOT, GRADE, RESULTS = (base.ROOT / ('20261024_sol64_bert_roundtrip_' + n + '_v2')
    for n in ('workflow', 'prepare', 'pilot', 'grade', 'results'))
REPORT = Path('docs/08 - Analysis/2026-10-24_ncm-sol64-bert-roundtrip-recovered.md')
OLD_WORKFLOW, OLD_PREP, OLD_PILOT = base.WORKFLOW, base.PREP, base.PILOT
OLD_PLAN, OLD_TEST, OLD_AUDITOR = base.PLAN, base.TEST, base.AUDITOR
_producing, _prepare = base.producing, base.prepare


def configure():
    base.native.vendor.corpus.quality.dependencies()
    import tiktoken
    import requests
    if tiktoken.get_encoding('cl100k_base') is None or not callable(requests.post):
        raise ValueError('semantic worker dependency readiness failed')
    base.lineage.configure()
    if base.torch.cuda.is_initialized() or base.os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('recovered control remains CPU-only')
    for name in ('WORKFLOW', 'PREP', 'PILOT', 'GRADE', 'RESULTS', 'REPORT', 'MODULE', 'AUDITOR_MODULE', 'AUDITOR', 'PLAN', 'TEST'):
        setattr(base, name, globals()[name])
    base.configure, base.producing, base.prepare = configure, producing, prepare


def require_unstarted():
    if (base.q.read(OLD_WORKFLOW / 'actual_tool_receipt.json')['actual_tool_exit_code'] != 1
            or base.q.read(OLD_WORKFLOW / 'pilot_receipt.json')['exit_code'] != 1
            or (OLD_PILOT / 'requests').exists()):
        raise ValueError('actual pre-request v1 dependency failure required')
    if base.q.read(OLD_PILOT / 'judgment_cache.json') != base.lineage.base_cache():
        raise ValueError('no new first judgment may be reset during dependency recovery')
    base.native.vendor.terminal(OLD_PREP)


def producing(root, stage, inputs=()):
    extra = [Path(__file__), PLAN, TEST, AUDITOR, OLD_PLAN, OLD_TEST, OLD_AUDITOR,
        OLD_WORKFLOW / 'actual_tool_receipt.json', OLD_WORKFLOW / 'pilot_receipt.json',
        OLD_PILOT / 'producer.json', OLD_PILOT / 'judgment_cache.json',
        *[OLD_PREP / n for n in ('cases.json', 'packets.json', 'split_proof.json', 'public_proof.json', 'worker_terminal_receipt.json')]]
    p = _producing(root, stage, [*inputs, *extra])
    for i, path in enumerate((Path(__file__), PLAN, TEST, AUDITOR, OLD_AUDITOR)):
        p['source_hashes'][str(path)] = base.file_hash(path)
        (root / 'producer_source' / f'recovery_{i}_{path.name}').write_bytes(path.read_bytes())
    p['config'].update(dependency_recovery=True, new_judgments_reset=0, same_original64=True)
    p['fingerprint'] = base.q.digest({k: v for k, v in p.items() if k != 'fingerprint'})
    base.write_json(root / 'producer.json', p)
    return p


def prepare():
    configure()
    require_unstarted()
    _prepare()
    for name in ('cases.json', 'packets.json', 'split_proof.json', 'public_proof.json'):
        if base.q.read(PREP / name) != base.q.read(OLD_PREP / name):
            raise ValueError('recovery changed original prepared64 cases or packets')


def __getattr__(name):
    return getattr(base, name)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('stage', choices=('prepare', 'pilot', 'grade', 'workflow'))
    args = parser.parse_args()
    configure()
    require_unstarted()
    {'prepare': prepare, 'pilot': lambda: base.grade(True), 'grade': base.grade, 'workflow': base.workflow}[args.stage]()
