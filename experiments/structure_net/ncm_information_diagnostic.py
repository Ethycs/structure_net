"""Describe saved NCM hard-code usage without estimating mutual information."""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path

ROOT=Path('data/experiments/ncm_paper_regime')
LEVELS=(1,8,16,24,32,40,52,64,76,92,108,128)
FILES={'target':ROOT/'20261025_engineering_attention_repair732_data_v1/cases.json',
       'train':ROOT/'20261025_sequential_full732_pixi_train_v1/outputs.json',
       'DEV':ROOT/'20261025_sequential_full732_pixi_DEV_outputs_v1/outputs.json'}


def entropy(counts):
    n=sum(counts.values())
    return -sum(v/n*math.log2(v/n) for v in counts.values())


def measure():
    arrays={key:[r['oracle_codes'] if key=='target' else r['codes']
        for r in json.loads(path.read_text())] for key,path in FILES.items()}
    if any(len(r)!=641 or any(not 0<=v<8192 for v in r) for rows in arrays.values() for r in rows):
        raise ValueError('complete native code hierarchy required')
    out={}
    for name,rows in arrays.items():
        offset=0;stats=[]
        for length in LEVELS:
            marginals=[Counter(r[i] for r in rows) for i in range(offset,offset+length)]
            pool=Counter(v for r in rows for v in r[offset:offset+length])
            stats.append({'positions':length,'observed_codes':len(pool),
                'mean_position_entropy_bits':sum(entropy(c) for c in marginals)/length,
                'constant_positions':sum(len(c)==1 for c in marginals),
                'mean_majority_fraction':sum(max(c.values())/len(rows) for c in marginals)/length})
            offset+=length
        out[name]={'cases':len(rows),'distinct_complete_hierarchies':len(set(map(tuple,rows))),
            'root_top_codes':[[k,v] for k,v in Counter(r[0] for r in rows).most_common(5)],'by_level':stats}
    return out


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--replay',type=Path,required=True)
    args=p.parse_args()
    result=json.loads(args.replay.read_text())
    if result['cohorts']!=measure():raise ValueError('saved descriptive code statistics differ')
    for path in FILES.values():
        if hashlib.sha256(path.read_bytes()).hexdigest()!=result['artifacts'][str(path)]:
            raise ValueError('saved code source changed')
    print(json.dumps({'replayed':str(args.replay),'task_accuracy_measured':False,
        'mutual_information_estimated':False}))
