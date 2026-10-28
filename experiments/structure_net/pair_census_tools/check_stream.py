"""Focused stdlib checks run in the isolated locked streaming-tool environment."""
from io import BytesIO
import json
from collections import Counter

from experiments.structure_net import ncm_available_pair_deep_scan as scan


def main():
    label={'summary':'A short target','text_sha256':'a'*64}
    value={'large':[{'vector':[1.0]*10000} for _ in range(5)],'rows':[label],
           'index_json':json.dumps({'sections':[{'summary':'Nested','spans':[{'start_char':0}]}]})}
    found=[];counts=Counter();scan.stream_objects(BytesIO(json.dumps(value).encode()),lambda p,v:found.append(v),counts)
    assert label in found
    assert any(v.get('summary')=='Nested' for v in found)
    assert not scan.relevant({'summary':{'passed':10}})
    old_cap=scan.CAP;scan.CAP=128
    second=[]
    scan.stream_objects(BytesIO(json.dumps({'padding':'x'*1000,'index_json':json.dumps(label)}).encode()),lambda p,v:second.append(v),Counter())
    scan.CAP=old_cap
    assert label in second
    try:scan.stream_objects(BytesIO(b'{"broken":['),lambda p,v:None,Counter())
    except Exception:pass
    else:raise AssertionError('invalid JSON accepted')
    print('streamed nested labels, skipped numeric arrays, and malformed JSON checks passed')


if __name__=='__main__':main()
