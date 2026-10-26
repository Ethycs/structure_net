"""Fetch only pinned public TRAIN bytes; this does not admit training examples."""
import argparse
import hashlib
import json
from pathlib import Path
from urllib.request import Request, urlopen

INVENTORY = Path('data/experiments/ncm_paper_regime/20261025_public_xsum_training_inventory_v1')
DEST = Path('data/experiments/ncm_paper_regime/20261026_public_xsum_train_download_v1')


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(4*1024**2),b''):
            digest.update(block)
    return digest.hexdigest()


def write(path, value):
    with path.open('x') as stream:
        json.dump(value,stream,indent=2,allow_nan=False)
        stream.write('\n')


def download():
    receipt = json.loads((INVENTORY/'actual_tool_receipt.json').read_text())
    if receipt['actual_tool_exit_code'] != 0:
        raise ValueError('completed public metadata inventory required')
    for name,digest in receipt['artifacts'].items():
        if sha(name) != digest:
            raise ValueError('public inventory changed: '+name)
    inventory = json.loads((INVENTORY/'result.json').read_text())
    if inventory['dataset'] != 'EdinburghNLP/xsum':
        raise ValueError('fixed public dataset required')
    selected = [r for r in inventory['files'] if r['path'] == 'data/train-00000-of-00001.parquet']
    if len(selected) != 1:
        raise ValueError('exactly one pinned TRAIN shard required')
    entry = selected[0]
    if not 0 < entry['size'] < 400_000_000 or entry['size'] != entry['lfs']['size']:
        raise ValueError('bounded expected LFS size required')
    revision = inventory['revision']
    if len(revision) != 40 or any(c not in '0123456789abcdef' for c in revision):
        raise ValueError('immutable repository revision required')
    url = f'https://huggingface.co/datasets/EdinburghNLP/xsum/resolve/{revision}/{entry["path"]}'
    inputs = {str(INVENTORY/'actual_tool_receipt.json'):sha(INVENTORY/'actual_tool_receipt.json'),
              **receipt['artifacts'],str(Path(__file__)):sha(__file__)}
    DEST.mkdir()
    (DEST/'producer_source.py').write_bytes(Path(__file__).read_bytes())
    write(DEST/'producer.json',{'url':url,'revision':revision,'expected_bytes':entry['size'],
        'expected_sha256':entry['lfs']['oid'],'input_hashes':inputs,
        'training_data_admitted':False,'new_model_training':False,'final_test_opened':False})
    temporary = DEST/'train.parquet.partial'
    try:
        digest,total,reported = hashlib.sha256(),0,0
        with urlopen(Request(url,headers={'User-Agent':'structure-net-public-data-preparation'}),timeout=30) as response:
            with temporary.open('xb') as stream:
                while block := response.read(4*1024**2):
                    total += len(block)
                    if total > entry['size']:
                        raise ValueError('download exceeds pinned byte count')
                    stream.write(block);digest.update(block)
                    if total-reported >= 64*1024**2:
                        print(json.dumps({'downloaded_bytes':total,'expected_bytes':entry['size']}),flush=True)
                        reported = total
        if total != entry['size'] or digest.hexdigest() != entry['lfs']['oid']:
            raise ValueError('public TRAIN bytes differ from pinned LFS identity')
        for name,expected in inputs.items():
            if sha(name) != expected:
                raise ValueError('download producer or metadata changed: '+name)
        target = DEST/'train.parquet'
        if target.exists():
            raise ValueError('append-only destination required')
        temporary.rename(target)
        result = {'status':'pinned_public_TRAIN_bytes_verified','dataset':inventory['dataset'],
            'revision':revision,'url':url,'path':str(target),'sha256':digest.hexdigest(),'bytes':total,
            'published_expected_rows':inventory['published_split_counts']['train'],
            'parsed_rows_verified':False,'training_data_admitted':False,
            'source_disjointness_or_factual_quality_established':False,
            'new_model_training':False,'new_judge_requests':0,'final_test_opened':False}
        write(DEST/'result.json',result)
        print(json.dumps(result),flush=True)
    except Exception as error:
        write(DEST/'failure.json',{'status':'public_TRAIN_download_failed',
            'error_type':type(error).__name__,'error':str(error),'training_data_admitted':False})
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('stage',choices=('download',))
    parser.parse_args()
    download()
