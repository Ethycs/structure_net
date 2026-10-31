"""Download and verify the official Bonsai 2 27B Q2_0 checkpoint."""
import argparse
import hashlib
import json
from pathlib import Path

import requests

from .common import sha, write

REPO = 'prism-ml/Ternary-Bonsai-2-27B-gguf'
REVISION = '94dd51cdbbec5309ed8b76a12f3268bc4fdceb93'
FILENAME = 'Ternary-Bonsai-2-27B-Q2_0.gguf'
SIZE = 7626008928
SHA256 = '4f99aed01b8a877e153f9aa6569a4440fe17c59701cc0953e36d7f460549e70e'
WEIGHTS = Path('/home/rabbit/models/Ternary-Bonsai-2-27B-GGUF') / FILENAME


def prepare(out):
    out.mkdir(parents=True, exist_ok=False)
    r = requests.get(f'https://huggingface.co/api/models/{REPO}/revision/{REVISION}?blobs=true',timeout=60)
    r.raise_for_status()
    metadata = r.json()
    entry = next(x for x in metadata['siblings'] if x['rfilename']==FILENAME)
    if metadata['sha'] != REVISION or entry['size'] != SIZE or entry['lfs']['sha256'] != SHA256:
        raise ValueError('published checkpoint identity differs')
    write(out/'source.json',{'repository':REPO,'revision':REVISION,'file':entry,'producer_sha256':sha(__file__)})
    WEIGHTS.parent.mkdir(parents=True,exist_ok=True)
    try:
        if not WEIGHTS.exists():
            partial = WEIGHTS.with_suffix('.gguf.part')
            response=requests.get(f'https://huggingface.co/{REPO}/resolve/{REVISION}/{FILENAME}',stream=True,timeout=(30,120))
            response.raise_for_status()
            count,h,report_at=0,hashlib.sha256(),512*1024**2
            with partial.open('xb') as f:
                for block in response.iter_content(8*1024**2):
                    f.write(block);h.update(block);count+=len(block)
                    if count>=report_at:
                        print(json.dumps({'download_bytes':count,'total_bytes':SIZE}),flush=True)
                        report_at+=512*1024**2
            if count != SIZE or h.hexdigest()!=SHA256: raise ValueError('download integrity failure')
            partial.rename(WEIGHTS)
        elif WEIGHTS.stat().st_size != SIZE or sha(WEIGHTS)!=SHA256:
            raise ValueError('cached checkpoint differs')
        write(out/'prepared.json',{'weights':str(WEIGHTS),'bytes':SIZE,'sha256':SHA256,'verified':True})
        print('Bonsai checkpoint verified',flush=True)
    except Exception as exc:
        write(out/'interruption.json',{'error':repr(exc)})
        raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True,type=Path)
    prepare(p.parse_args().output)
