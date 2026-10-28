"""Check the streaming metadata filter against protected input containers."""
import hashlib
import json
from pathlib import Path
import subprocess

BINARY=Path('data/experiments/ncm_paper_regime/20261027_available_pair_numeric_filter_build_v1/numeric_arrays')


def run(value):
    result=subprocess.run([str(BINARY)],input=json.dumps(value,ensure_ascii=False).encode(),capture_output=True)
    if result.returncode:raise ValueError(result.stderr.decode())
    return json.loads(result.stdout),json.loads(result.stderr)


def main():
    source='Exact Unicode source: π, 雨, 3.14 and 00123.'
    value={'matrix':[[1.1,2.2],[3.3,None]],'spans':[{'start_char':0,'end_char':len(source),'numeric_annotation':[1,2]}],
           'messages':[{'role':'user','content':source,'numeric_annotation':[3,4]}],
           'child_section_ids':[1,2],'fragments':[1,2],'support':[1,2],'turns':[1,2],
           'summary':'Target','source_text':source,'mixed':[1,2,{'role':'user','text':source},'tail']}
    filtered,stats=run(value)
    assert filtered=={**value,'matrix':[]}
    assert stats['number_values_elided']==3
    # More than the in-memory cap, followed by an object: spill and replay
    # must preserve the entire mixed array instead of silently dropping it.
    large={'mixed':[123456789012345]*600000+[{'summary':'kept','source_text':source}]}
    filtered,stats2=run(large)
    assert filtered==large
    invalid=subprocess.run([str(BINARY)],input=b'{"broken":[',capture_output=True)
    assert invalid.returncode!=0
    proof={'status':'numeric_filter_source_preservation_and_spill_checks_passed',
        'binary_sha256':hashlib.sha256(BINARY.read_bytes()).hexdigest(),
        'producer_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'protected_input_containers_preserved':True,'unicode_source_exact':True,
        'large_mixed_array_replayed_exactly':True,'malformed_JSON_rejected':True}
    path=BINARY.parent/'checks.json'
    with path.open('x') as f:json.dump(proof,f,indent=2);f.write('\n')
    print(json.dumps(proof))


if __name__=='__main__':main()
