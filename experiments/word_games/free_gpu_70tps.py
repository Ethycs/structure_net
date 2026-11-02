"""Use the user-authorized expanded GPU budget; restore resident services."""
import argparse
from dataclasses import asdict
from pathlib import Path
import signal

from . import target_70tps as target
from .common import PACKAGE, digest, read, sha, write
from .gpu_services import ModelLease
from .check_qwen_tokenizers import gguf_metadata
from .prepare_qwen38_moe import WEIGHTS, SHA256
from .qwen38_moe_runtime import validate_metadata
from .qwen_runtime import REQUESTS, REQUESTS_SHA

PLAN = Path('docs/07 - Status Reports/word-games/2026-10-31_word-games-70tps-free-gpus.md')
PREVIOUS = Path('data/experiments/word_games/20261031_70tps_batch_mtp_v1/campaign_results.json')
C = target.Config
CONFIGS = {c.name: c for c in (
    C('control'), C('free_whole512', True, 512, True, batch=1024, no_checkpoints=True, free_layout=True),
    C('free_whole1024', True, 1024, True, batch=2048, no_checkpoints=True, free_layout=True),
    C('free_mtp1', True, 512, True, batch=1024, mtp=True, mtp_gpu=True, free_layout=True),
    C('free_mtp2', True, 512, True, batch=1024, mtp=True, mtp_gpu=True, mtp_tokens=2, free_layout=True))}


def run(args, *, configs=None, plan=PLAN, previous=PREVIOUS, stage='free-gpus', extra_sources=()):
    configs = CONFIGS if configs is None else configs
    out = args.output
    out.mkdir(parents=True, exist_ok=False)
    print('Checking original checkpoint before releasing both resident models', flush=True)
    if sha(WEIGHTS) != SHA256 or sha(REQUESTS) != REQUESTS_SHA:
        raise ValueError('weight or input identity mismatch')
    for name, expected in read(target.cpu.BUILD)['runtimes']['candidate']['hashes'].items():
        if sha(target.cpu.RUNTIME / name) != expected:
            raise ValueError('native runtime changed')
    prior = read(previous)
    if prior['error'] is not None or not prior['service_restored']:
        raise ValueError('previous campaign must be closed')
    metadata = gguf_metadata(WEIGHTS)
    validate_metadata(metadata)
    sources = sorted(p for p in PACKAGE.rglob('*') if p.is_file() and '__pycache__' not in str(p))
    sources += [plan, Path('pixi.lock'), Path('pyproject.toml'), *extra_sources]
    bindings = {str(p): sha(p) for p in sources}
    producer = {'hypothesis_id': target.HYPOTHESIS, 'schema': f'word-games.70tps.{stage}.v1',
        'weights_sha256': SHA256, 'requests_sha256': REQUESTS_SHA, 'source_hashes': bindings,
        'runtime_manifest_sha256': sha(target.cpu.BUILD), 'stage': stage,
        'first_stage': str(previous), 'first_stage_sha256': sha(previous),
        'resident_models_unloaded': ['qwen3-8b-gguf', 'glm-ocr'],
        'configs': {k: asdict(v) for k, v in configs.items()}, 'target_tok_s': 70,
        'selection_gate': .98, 'max_candidates': 2}
    producer['fingerprint'] = digest(producer)
    write(out / 'producer.json', producer)
    for i, p in enumerate(sources):
        dest = out / 'producer_source' / f'{i}_{p.name}'
        dest.parent.mkdir(exist_ok=True)
        dest.write_bytes(p.read_bytes())
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
    lease = ModelLease(out / 'services')
    results, advanced, error = [], [], None
    try:
        lease.acquire()
        print('Qwen3-8B and OCR gracefully unloaded', flush=True)
        for name, config in configs.items():
            r = target.cell(out / ('pilot_' + name), config, 'pilot', producer, metadata, args.port)
            results.append({'repeat': 0, **r})
            if not r['completed'] and ('readiness' in str(r['error']) or name == 'control'):
                raise RuntimeError('baseline or resident-server health failure')
        baseline = results[0]['totals']['output_tokens_per_second']
        eligible = [r for r in results[1:] if r['completed'] and
                    r['totals']['output_tokens_per_second'] >= .98 * baseline and
                    (not configs[r['arm']].mtp or r['totals']['drafted'] > 0)]
        advanced = [r['arm'] for r in sorted(eligible, key=lambda r: r['totals']['output_tokens_per_second'], reverse=True)[:2]]
        write(out / 'selection.json', {'control_rate': baseline, 'advanced': advanced,
              'pilots': [{'arm': r['arm'], 'completed': r['completed'], 'totals': r['totals'], 'error': r['error']} for r in results]})
        order = ['control', *advanced]
        for repeat, names in enumerate((order, list(reversed(order))), 1):
            for name in names:
                path = out / f'full_{repeat}_{name}'
                r = target.cell(path, configs[name], 'full', producer, metadata, args.port)
                results.append({'repeat': repeat, **r})
                if not r['completed']:
                    raise RuntimeError('full replay failed: ' + name)
                quality = target.cpu.receiver_accuracy(path)
                if quality['receiver_cases'] != 12:
                    raise ValueError('incomplete receiver population')
                write(path / 'quality.json', quality)
    except BaseException as exc:
        error = repr(exc)
        raise
    finally:
        try:
            lease.restore()
        finally:
            write(out / 'campaign_results.json', {'hypothesis_id': target.HYPOTHESIS,
                'fingerprint': producer['fingerprint'], 'results': results, 'advanced': advanced,
                'error': error, 'service_restored': lease.restored, 'confirmed': False,
                'classification': 'exploratory_systems_comparison'})
        print('Both resident models restored:', lease.restored, flush=True)
    if any(sha(p) != v for p, v in bindings.items()):
        raise ValueError('source changed during campaign')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--port', type=int, default=18088)
    run(p.parse_args())
