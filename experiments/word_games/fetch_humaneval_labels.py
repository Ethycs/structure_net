"""Fetch the pinned public dataset for the paired HumanEval label study."""
import argparse
import gzip
import json
from pathlib import Path
import random
import urllib.request

from .common import sha, write

REVISION = 'b9453e8f1865df0a0057a1383767caa400acfbcf'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('data/experiments/word_games/20261112_humaneval_data_v1'))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    base = f'https://raw.githubusercontent.com/openai/human-eval/{REVISION}/'
    for source, target in [('data/HumanEval.jsonl.gz', 'HumanEval.jsonl.gz'),
                           ('LICENSE', 'LICENSE'), ('human_eval/execution.py', 'upstream_execution.py.txt')]:
        request = urllib.request.Request(base + source, headers={'User-Agent': 'StructureNet-research'})
        with urllib.request.urlopen(request, timeout=45) as response:
            (args.output / target).write_bytes(response.read())
    rows = [json.loads(line) for line in gzip.decompress((args.output / 'HumanEval.jsonl.gz').read_bytes()).decode().splitlines()]
    if len(rows) != 164: raise ValueError('Unexpected pinned dataset size')
    write(args.output / 'source.json', {'repository': 'https://github.com/openai/human-eval',
        'revision': REVISION, 'data_sha256': sha(args.output / 'HumanEval.jsonl.gz'),
        'sample_seed': 731113, 'selected_ids': sorted(random.Random(731113).sample(range(164), 32)),
        'n_problems': len(rows)})
    print('Fetched pinned dataset', REVISION)


if __name__ == '__main__':
    main()
