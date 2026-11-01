"""Measure process reuse with exact reference-token parity on public packets."""
import argparse
import json
from pathlib import Path
import statistics
import time

from .common import read, write, sha
from .golf_poker_proxy import reference_counts
from .worker_environment import ReferenceTokenizer


def run(root):
    root.mkdir(parents=True, exist_ok=False)
    source = Path('data/experiments/word_games/20261101_claude_research_bridge_v1/public/turns')
    paths = sorted(source.glob('Q*.json'))
    texts = [json.dumps(read(p), separators=(',', ':')) for p in paths]+['', 'é 中文 🧪\nJSON: {"x": 1}', '<|im_start|>']
    write(root/'producer.json', {'source_hashes': {str(p): sha(p) for p in
        (Path(__file__), Path('experiments/word_games/worker_environment.py'), Path('experiments/word_games/golf_poker_proxy.py'))},
        'packet_hashes': {str(p): sha(p) for p in paths}, 'texts': texts})
    counter = ReferenceTokenizer(root/'stderr.txt')
    rows = []
    try:
        for repeat in range(2):
            for index, text in enumerate(texts):
                pair = {}
                order = ('baseline', 'persistent') if (repeat+index) % 2 else ('persistent', 'baseline')
                for arm in order:
                    start = time.perf_counter()
                    count = (reference_counts if arm == 'baseline' else counter.counts)([text])[0]
                    pair[arm] = {'count': count, 'seconds': time.perf_counter()-start}
                assert pair['baseline']['count'] == pair['persistent']['count']
                rows.append({'repeat': repeat, 'index': index, **pair})
    finally:
        counter.close()
    totals = {arm: sum(r[arm]['seconds'] for r in rows) for arm in ('baseline', 'persistent')}
    result = {'completed': True, 'counts_equal': True, 'requests_per_arm': len(rows), 'rows': rows,
        'total_seconds_including_cold_start': totals,
        'warm_median_seconds': {arm: statistics.median(r[arm]['seconds'] for r in rows[1:]) for arm in totals},
        'speedup_including_cold_start': totals['baseline']/totals['persistent'],
        'producer_sha256': sha(root/'producer.json')}
    write(root/'result.json', result)
    print(json.dumps({k: v for k, v in result.items() if k != 'rows'}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    run(parser.parse_args().output)
