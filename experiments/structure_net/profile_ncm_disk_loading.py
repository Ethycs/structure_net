"""Profile the actual bounded loader on 32 fixed TRAIN presentations, CPU only."""
import cProfile
import json
from pathlib import Path
import pstats
import time

from experiments.structure_net import ncm_pixi as runtime


def io_counters():
    return {key: int(value) for key, value in
            (line.split(':', 1) for line in Path('/proc/self/io').read_text().splitlines())}


def main():
    training, environment = runtime.configure()
    root = runtime.ROOT / '20261025_ncm_disk_loading_profile_v1'
    if root.exists() or training.torch.cuda.is_initialized():
        raise ValueError('Fresh CPU-only profiling root required')
    rows = {r['id']: r for r in training.q.read(training.base.DATA / 'cases.json')}
    trace = [json.loads(line) for line in (runtime.RUNS['train'] / 'trace.jsonl').read_text().splitlines()]
    # A fixed eight-update window after the first traversal, chosen before
    # measuring. This preserves the real variable-length sampling order.
    chosen = [rows[m['id']] for update in trace[256:264] for m in update['metrics']]
    if len(chosen) != 32:
        raise ValueError('The fixed completed eight-update window is required')
    loader = training.preparation.streaming.FeatureCache(
        training.q.read(training.base.DATA / 'source_index.json'),
        training.q.read(training.base.DATA / 'feature_manifest.json'), max_bytes=256 * 1024**2)
    root.mkdir()
    profiler = cProfile.Profile()
    before = io_counters()
    started = time.perf_counter()
    timings = []
    profiler.enable()
    for row in chosen:
        start = time.perf_counter()
        value = loader.get(row)
        timings.append({'id': row['id'], 'seconds': time.perf_counter() - start,
                        'raw_bytes': value.numel() * value.element_size()})
    profiler.disable()
    elapsed = time.perf_counter() - started
    after = io_counters()
    profiler.dump_stats(str(root / 'loader.prof'))
    stats = pstats.Stats(profiler)
    functions = [{'file': k[0], 'line': k[1], 'function': k[2], 'calls': v[1],
                  'self_seconds': v[2], 'cumulative_seconds': v[3]}
                 for k, v in stats.stats.items()]
    functions.sort(key=lambda r: -r['self_seconds'])
    result = {'status': 'CPU_loader_profile_completed', 'cases': len(chosen),
        'seconds': elapsed, 'mean_seconds_per_example': elapsed / len(chosen),
        'seconds_per_four_example_update': elapsed / 8, 'loader': loader.metrics(),
        'io_delta': {k: after[k] - before[k] for k in before},
        'profile_self_time_top20': functions[:20],
        'concurrent_training': True, 'controlled_end_to_end_speed_comparison': False,
        'OS_cache_state_uncontrolled': True, 'runtime': environment,
        'profiler_sha256': training.file_hash(Path(__file__)),
        'task_accuracy': None, 'final_test_opened': False}
    training.write_json(root / 'timings.json', timings)
    training.write_json(root / 'functions.json', functions)
    training.write_json(root / 'result.json', result)
    print(json.dumps({k: v for k, v in result.items() if k != 'runtime'}), flush=True)


if __name__ == '__main__':
    main()
