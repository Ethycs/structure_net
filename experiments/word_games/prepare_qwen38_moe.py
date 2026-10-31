"""Verify and download Empero's Qwen3.8-35B-A3B distillation Q4 GGUF."""
import argparse
import hashlib
import json
from pathlib import Path

import requests

from .common import sha, write

REPO = 'empero-ai/Qwen3.8-35B-A3B-Distill-GGUF'
REVISION = 'b1f9d1dcc3de8aa867669b0ab919384aeeb9b8d5'
FILENAME = 'Qwen3.8-35B-A3B-Q4_K_M.gguf'
SIZE = 21713462944
SHA256 = '196103269085bc54c9b8f49ed21e9f53e1b56b465e8b796c6d8e31e06f63cfa5'
WEIGHTS = Path('/home/rabbit/models/Empero-Qwen3.8-35B-A3B-Distill-GGUF') / FILENAME


def prepare(out):
    out.mkdir(parents=True, exist_ok=False)
    response = requests.get(f'https://huggingface.co/api/models/{REPO}/revision/{REVISION}?blobs=true', timeout=60)
    response.raise_for_status()
    metadata = response.json()
    entry = next(x for x in metadata['siblings'] if x['rfilename'] == FILENAME)
    if metadata['sha'] != REVISION or entry['size'] != SIZE or entry['lfs']['sha256'] != SHA256:
        raise ValueError('published checkpoint identity differs')
    write(out/'source.json', {'repository': REPO, 'revision': REVISION, 'file': entry,
                             'producer_sha256': sha(__file__)})
    card = requests.get(f'https://huggingface.co/{REPO}/raw/{REVISION}/README.md', timeout=30)
    card.raise_for_status()
    (out/'model_card.md').write_text(card.text)
    WEIGHTS.parent.mkdir(parents=True, exist_ok=True)
    try:
        if not WEIGHTS.exists():
            partial = WEIGHTS.with_suffix('.gguf.part')
            with requests.get(f'https://huggingface.co/{REPO}/resolve/{REVISION}/{FILENAME}',
                              stream=True, timeout=(30,120)) as download:
                download.raise_for_status()
                count, digest, report_at = 0, hashlib.sha256(), 1024**3
                with partial.open('xb') as target:
                    for block in download.iter_content(8*1024**2):
                        target.write(block); digest.update(block); count += len(block)
                        if count >= report_at:
                            print(json.dumps({'download_bytes':count, 'total_bytes':SIZE}), flush=True)
                            report_at += 1024**3
            if count != SIZE or digest.hexdigest() != SHA256:
                raise ValueError('download integrity failure')
            partial.rename(WEIGHTS)
        elif WEIGHTS.stat().st_size != SIZE or sha(WEIGHTS) != SHA256:
            raise ValueError('cached checkpoint differs')
        write(out/'prepared.json', {'weights':str(WEIGHTS), 'bytes':SIZE, 'sha256':SHA256, 'verified':True})
        print('Empero Qwen3.8-35B-A3B checkpoint verified', flush=True)
    except Exception as exc:
        write(out/'interruption.json', {'error':repr(exc)})
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    prepare(parser.parse_args().output)
