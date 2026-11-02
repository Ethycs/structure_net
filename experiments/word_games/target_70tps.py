"""Explore contiguous whole-layer placement at unchanged model precision."""
import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import re
import signal
import time

import requests

from . import cpu_expert_elimination as cpu
from .common import PACKAGE, digest, read, sha, write
from .check_qwen_tokenizers import gguf_metadata
from .prepare_qwen38_moe import WEIGHTS, SHA256
from .qwen38_moe_runtime import validate_metadata
from .qwen_runtime import REQUESTS, REQUESTS_SHA, memory

HYPOTHESIS = 'word-games-70tps-v1'
PLAN = Path('docs/07 - Status Reports/word-games/2026-10-31_word-games-70tps-plan.md')
FOLLOWUP_PLAN = Path('docs/07 - Status Reports/word-games/2026-10-31_word-games-70tps-followup.md')
BATCH_MTP_PLAN = Path('docs/07 - Status Reports/word-games/2026-10-31_word-games-70tps-batching-mtp.md')
FIRST_STAGE = Path('data/experiments/word_games/20261031_70tps_replay_v1/campaign_results.json')
SECOND_STAGE = Path('data/experiments/word_games/20261031_70tps_followup_v1/campaign_results.json')
CPU_COMMAND, CPU_VERIFY = cpu.command, cpu.verify
PARENT_BUFFERS = {'CUDA0': 10668.90, 'CUDA1': 4284.00, 'CUDA2': 4950.00, 'CUDA_Host': 272.81}


@dataclass(frozen=True)
class Config:
    name: str
    whole: bool = False
    microbatch: int = 512
    pipeline: bool = False
    verbosity: int = 5
    no_checkpoints: bool = False
    graph_opt: bool = False
    batch: int = 512
    mtp: bool = False
    mtp_gpu: bool = False
    mtp_tokens: int = 1
    free_layout: bool = False

    def __post_init__(self):
        if self.microbatch not in (256, 384, 512, 1024) or self.verbosity not in (3, 4, 5):
            raise ValueError('unsupported measurement setting')
        if self.pipeline and not self.whole:
            raise ValueError('pipeline requires whole-layer placement')
        if self.batch not in (512, 1024, 2048) or self.microbatch > self.batch:
            raise ValueError('unsupported logical batch')
        if self.mtp and not self.mtp_gpu and (self.whole or self.pipeline):
            raise ValueError('CPU MTP probe requires supported expert placement')
        if self.mtp_gpu and (not self.mtp or not self.whole):
            raise ValueError('GPU MTP requires whole-layer MTP configuration')
        if self.mtp_tokens not in (1, 2) or (self.mtp_tokens != 1 and not self.mtp):
            raise ValueError('unsupported draft length')
        if self.free_layout and not self.whole:
            raise ValueError('expanded layout requires whole layers')


CONFIGS = {c.name: c for c in (
    Config('control'), Config('whole256', True, 256), Config('whole512', True),
    Config('whole_pipe256', True, 256, True), Config('expert256', microbatch=256),
    Config('quiet_control', verbosity=3))}

FOLLOWUPS = {c.name: c for c in (
    Config('control'), Config('quiet4', verbosity=4),
    Config('no_checkpoints', verbosity=4, no_checkpoints=True),
    Config('whole_no_cp256', True, 256, True, no_checkpoints=True),
    Config('whole_no_cp384', True, 384, True, no_checkpoints=True),
    Config('whole_no_cp512', True, 512, True, no_checkpoints=True),
    Config('whole_no_cp384_graph', True, 384, True, no_checkpoints=True, graph_opt=True))}

BATCH_MTP = {c.name: c for c in (
    Config('control'),
    Config('whole_b1024', True, 256, True, no_checkpoints=True, batch=1024),
    Config('whole_b2048', True, 256, True, no_checkpoints=True, batch=2048),
    Config('mtp1_cpu12', mtp=True))}


def command(config, unused, port, name):
    args = CPU_COMMAND(cpu.ARMS['gpu40_cpu1'], WEIGHTS, port, name)
    def set_arg(flag, value):
        args[args.index(flag) + 1] = str(value)
    set_arg('--ubatch-size', config.microbatch)
    set_arg('--batch-size', config.batch)
    set_arg('--verbosity', config.verbosity)
    if config.no_checkpoints:
        args += ['--ctx-checkpoints', '0']
    if config.graph_opt:
        args[args.index(cpu.base.IMAGE):args.index(cpu.base.IMAGE)] = ['-e', 'GGML_CUDA_GRAPH_OPT=1']
    if config.mtp:
        set_arg('--spec-type', 'draft-mtp')
        threads = '1' if config.mtp_gpu else '12'
        set_arg('--threads', threads)
        set_arg('--threads-batch', threads)
        if not config.mtp_gpu:
            pos = args.index('--override-tensor') + 1
            args[pos] += ',^blk\\.40\\.ffn_.*_exps\\.weight$=CPU'
        args += ['--spec-draft-n-max', str(config.mtp_tokens), '--spec-draft-n-min', '1', '--spec-draft-p-min', '0',
                 '--spec-draft-threads', threads, '--spec-draft-threads-batch', threads,
                 '--spec-draft-type-k', 'q8_0', '--spec-draft-type-v', 'q8_0']
    if config.whole:
        set_arg('--device', 'CUDA2,CUDA1,CUDA0')
        # 42 assigned indices include the unused MTP layer and output.
        set_arg('--tensor-split', '7.5,14,20.5' if config.free_layout else '9.5,9,23.5')
        if config.pipeline:
            pos = args.index('--override-tensor')
            del args[pos:pos + 2]
        else:
            set_arg('--override-tensor', '^blk\\.0\\.attn_norm\\.weight$=CUDA2')
    return args


def verify(log, config):
    buffers = {dev: float(size) for dev, size in re.findall(
        r'load_tensors:\s+(\S+) model buffer size =\s+([\d.]+) MiB', log)}
    if not config.whole:
        expected_buffers = dict(PARENT_BUFFERS)
        trunk_log = log
        if config.mtp:
            expected_buffers['CUDA0'] += 23.3828125
            expected_buffers['CUDA_Host'] += 498
            projections = re.findall(r'tensor blk\.40\.ffn_(\w+)_exps\.weight \([^)]*\) buffer type overridden to (\S+)', log)
            if len(projections) != 3 or {p for p, _ in projections} != {'gate', 'up', 'down'} or any(
                    not (d.startswith('CPU') or d == 'CUDA_Host') for _, d in projections):
                raise ValueError('MTP CPU expert assignment mismatch')
            if 'creating MTP draft context' not in log or not re.search(r'n_rs_seq\s+= 1\b', log):
                raise ValueError('MTP context or rollback state missing')
            trunk_log = '\n'.join(line for line in log.splitlines() if 'tensor blk.40.' not in line)
        if set(buffers) != set(expected_buffers) or any(abs(buffers[k] - v) > .02 for k, v in expected_buffers.items()):
            raise ValueError('expert placement model buffers changed')
        if config.verbosity == 5:
            CPU_VERIFY(trunk_log, cpu.ARMS['gpu40_cpu1'])
    else:
        actual = {int(i): dev for i, dev in re.findall(r'layer\s+(\d+) assigned to device ([^,\s]+)', log)}
        first, second = (8, 22) if config.free_layout else (10, 19)
        expected = {i: 'CUDA2' if i < first else 'CUDA1' if i < second else 'CUDA0' for i in range(42)}
        if actual != expected:
            raise ValueError('whole-layer assignment mismatch')
        if set(buffers) != set(PARENT_BUFFERS) or abs(buffers.get('CUDA_Host', 0) - 272.81) > .02:
            raise ValueError('unexpected host expert weights or device buffers')
        if 'buffer type overridden to CPU' in log or 'buffer type overridden to CUDA_Host' in log:
            raise ValueError('CPU weight override')
        if config.free_layout:
            expected_buffers = {'CUDA0': 9242.57 + (521.38 if config.mtp else 0),
                                'CUDA1': 6621.65, 'CUDA2': 4038.68, 'CUDA_Host': 272.81}
            if any(abs(buffers[k] - v) > .03 for k, v in expected_buffers.items()):
                raise ValueError('expanded layout model buffers changed')
        if config.mtp:
            if 'creating MTP draft context' not in log or not re.search(r'n_rs_seq\s+= ' + str(config.mtp_tokens) + r'\b', log):
                raise ValueError('MTP context or rollback state missing')
    offloads = re.findall(r'offloaded (\d+)/(\d+) layers to GPU', log)
    if not offloads or any(a != b for a, b in offloads):
        raise ValueError('incomplete layer offload')
    pipeline = 'pipeline parallelism enabled' in log
    if pipeline != config.pipeline or 'retrying without pipeline parallelism' in log:
        raise ValueError('pipeline status differs from configured arm')
    graphs = [{'nodes': int(n), 'splits': int(s)} for n, s in
              re.findall(r'graph: nodes = (\d+), splits = (\d+)', log)]
    return {'verified': True, 'config': asdict(config), 'model_buffers_mib': buffers,
            'pipeline': pipeline, 'graphs': graphs}


def cell(out, config, phase, producer, metadata, port):
    original = cpu.command, cpu.verify, cpu.HYPOTHESIS
    cpu.command, cpu.verify, cpu.HYPOTHESIS = command, verify, HYPOTHESIS
    try:
        result = cpu.cell(out, config, phase, producer, metadata, WEIGHTS, port)
        log = (out / 'server.log').read_text()
        checkpoints = len(re.findall(r'created context checkpoint \d+', log))
        if result['completed'] and config.no_checkpoints and checkpoints:
            raise ValueError('checkpoint suppression failed')
        write(out / 'checkpoint_check.json', {'disabled_requested': config.no_checkpoints,
              'created_including_warmup': checkpoints, 'completed': result['completed']})
        return {**result, 'checkpoint_events_including_warmup': checkpoints}
    finally:
        cpu.command, cpu.verify, cpu.HYPOTHESIS = original


def run(args):
    out = args.output
    out.mkdir(parents=True, exist_ok=False)
    print('Hash-checking original checkpoint before service pause', flush=True)
    if sha(WEIGHTS) != SHA256 or sha(REQUESTS) != REQUESTS_SHA:
        raise ValueError('model or request identity changed')
    for name, expected in read(cpu.BUILD)['runtimes']['candidate']['hashes'].items():
        if sha(cpu.RUNTIME / name) != expected:
            raise ValueError('runtime changed')
    metadata = gguf_metadata(WEIGHTS)
    validate_metadata(metadata)
    configs = {'initial': CONFIGS, 'followup': FOLLOWUPS, 'batch-mtp': BATCH_MTP}[args.stage]
    previous = FIRST_STAGE if args.stage == 'followup' else SECOND_STAGE if args.stage == 'batch-mtp' else None
    if previous and (read(previous)['error'] is not None or not read(previous)['service_restored']):
        raise ValueError('preceding stage must be closed with services restored')
    sources = sorted(p for p in PACKAGE.rglob('*') if p.is_file() and '__pycache__' not in str(p))
    sources += [PLAN, Path('pixi.lock'), Path('pyproject.toml')]
    if args.stage != 'initial':
        sources += [FOLLOWUP_PLAN]
    if args.stage == 'batch-mtp':
        sources += [BATCH_MTP_PLAN]
    bindings = {str(p): sha(p) for p in sources}
    producer = {'hypothesis_id': HYPOTHESIS, 'schema': 'word-games.70tps.v1',
                'weights_sha256': SHA256, 'requests_sha256': REQUESTS_SHA,
                'runtime_manifest_sha256': sha(cpu.BUILD), 'source_hashes': bindings,
                'configs': {k: asdict(c) for k, c in configs.items()}, 'stage': args.stage,
                'first_stage': str(previous) if previous else None,
                'first_stage_sha256': sha(previous) if previous else None,
                'target_tok_s': 70, 'selection_gate': .98, 'max_candidates': 2}
    producer['fingerprint'] = digest(producer)
    write(out / 'producer.json', producer)
    for i, path in enumerate(sources):
        dest = out / 'producer_source' / f'{i}_{path.name}'
        dest.parent.mkdir(exist_ok=True)
        dest.write_bytes(path.read_bytes())
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
    base = cpu.base
    url, service = base.TRITON, base.SERVICE
    before = requests.get(url + f'/v2/models/{service}/config', timeout=10)
    before.raise_for_status()
    requests.get('http://127.0.0.1:8010/v2/health/ready', timeout=5).raise_for_status()
    write(out / 'service_before.json', {'config': before.json(), 'memory': memory()})
    attempted = restored = False
    results, advanced = [], []
    error = None
    try:
        attempted = True
        response = requests.post(url + f'/v2/repository/models/{service}/unload', json={}, timeout=120)
        write(out / 'unload.json', {'status': response.status_code, 'body': response.text})
        response.raise_for_status()
        base.wait_for_release(out)
        print('Qwen3-8B temporarily unloaded; OCR remains resident', flush=True)
        for name, config in configs.items():
            r = cell(out / ('pilot_' + name), config, 'pilot', producer, metadata, args.port)
            results.append({'repeat': 0, **r})
            if not r['completed'] and 'readiness' in str(r['error']):
                raise RuntimeError('resident service health failure')
            if name == 'control' and not r['completed']:
                raise RuntimeError('control pilot failed')
        control_rate = results[0]['totals']['output_tokens_per_second']
        eligible = [r for r in results[1:] if r['completed'] and
                    r['totals']['output_tokens_per_second'] >= .98 * control_rate and
                    (not configs[r['arm']].mtp or r['totals']['drafted'] > 0)]
        advanced = [r['arm'] for r in sorted(eligible,
                    key=lambda r: r['totals']['output_tokens_per_second'], reverse=True)[:2]]
        write(out / 'selection.json', {'control_rate': control_rate, 'advanced': advanced,
              'pilots': [{'arm': r['arm'], 'completed': r['completed'], 'totals': r['totals'], 'error': r['error']} for r in results]})
        order = ['control', *advanced]
        for repeat, names in enumerate((order, list(reversed(order))), 1):
            for name in names:
                path = out / f'full_{repeat}_{name}'
                r = cell(path, configs[name], 'full', producer, metadata, args.port)
                results.append({'repeat': repeat, **r})
                if not r['completed']:
                    raise RuntimeError('full replay failed: ' + name)
                quality = cpu.receiver_accuracy(path)
                if quality['receiver_cases'] != 12:
                    raise ValueError('receiver population incomplete')
                write(path / 'quality.json', quality)
    except BaseException as exc:
        error = repr(exc)
        raise
    finally:
        try:
            if attempted:
                response = requests.post(url + f'/v2/repository/models/{service}/load', json={}, timeout=240)
                write(out / 'reload.json', {'status': response.status_code, 'body': response.text})
                response.raise_for_status()
                deadline = time.monotonic() + 180
                while time.monotonic() < deadline:
                    if requests.get(url + f'/v2/models/{service}/ready', timeout=5).status_code == 200:
                        restored = True
                        break
                    time.sleep(1)
                after = requests.get(url + f'/v2/models/{service}/config', timeout=10)
                after.raise_for_status()
                restored = restored and after.json() == before.json()
                ocr_ready = requests.get('http://127.0.0.1:8010/v2/health/ready', timeout=5).status_code == 200
                write(out / 'service_restored.json', {'ready_and_config_equal': restored,
                      'ocr_ready': ocr_ready, 'memory': memory()})
                if not ocr_ready:
                    raise RuntimeError('OCR readiness failed')
                print('Qwen3-8B restored:', restored, flush=True)
        finally:
            write(out / 'campaign_results.json', {'hypothesis_id': HYPOTHESIS,
                  'fingerprint': producer['fingerprint'], 'results': results, 'advanced': advanced,
                  'error': error, 'service_restored': restored, 'confirmed': False,
                  'classification': 'exploratory_systems_comparison'})
    if not restored:
        raise RuntimeError('Qwen3-8B restoration failed')
    if any(sha(p) != value for p, value in bindings.items()):
        raise ValueError('source changed during campaign')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--port', type=int, default=18088)
    parser.add_argument('--stage', choices=('initial', 'followup', 'batch-mtp'), default='initial')
    run(parser.parse_args())
