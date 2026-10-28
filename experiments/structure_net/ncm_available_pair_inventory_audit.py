"""Independent readback of the exported inventory, including trivial merges."""
import hashlib
import json
from collections import Counter
from pathlib import Path
import sqlite3

from experiments.structure_net import ncm_available_pair_total as total


def main():
    root=total.ROOT;result=total.archive.read(root/'result.json')
    manifest=result['candidate_export'];path=Path(manifest['path'])
    digest=hashlib.sha256();counts=Counter();merge=Counter()
    with path.open('rb') as source:
        for line in source:
            digest.update(line);row=json.loads(line);kind=row['input_kind']
            if row['training_admitted'] is not False or row['split']!='unassigned':raise ValueError('unexpected training admission')
            if row['source']==row['target']:raise ValueError('exact copy included in fitted export')
            if not (1<=row['target_tokens']<=128 and row['input_tokens']<=2048):raise ValueError('length bound violated')
            if kind=='raw' and row['input_tokens']<192:raise ValueError('raw bypass included')
            counts[kind]+=1
            if kind=='children':
                children=json.loads(row['source'])
                if not isinstance(children,list) or not children or any(not isinstance(c,str) for c in children):
                    raise ValueError('merge input is not a nonempty string list')
                if len(children)==1 and row['target']==children[0]:merge['one_child_exact_identity']+=1
                elif row['target'] in children:merge['copies_one_of_multiple_children']+=1
                elif any(row['target']==sep.join(children) for sep in ('',' ','\n','\n\n')):merge['exact_child_concatenation']+=1
                else:merge['other_target_not_semantically_reviewed']+=1
    if sum(counts.values())!=manifest['pairs'] or digest.hexdigest()!=manifest['sha256']:
        raise ValueError('export row count or content hash changed')
    db=sqlite3.connect((root/'pairs.sqlite').resolve().as_uri()+'?mode=ro',uri=True)
    cross_kind=db.execute('''SELECT COUNT(*) FROM (SELECT input_sha,target_sha FROM pairs
      WHERE input_kind IN ('raw','children') GROUP BY input_sha,target_sha HAVING COUNT(*)>1)''').fetchone()[0]
    expected={r['kind']:r['application_length_fit_pairs'] for r in result['by_input_kind'] if r['kind'] in ('raw','children')}
    if dict(counts)!=expected:raise ValueError('export kinds do not reconcile with SQL count')
    proof={'status':'export_readback_and_length_partitions_verified','counts':dict(counts),'merge_copy_categories':dict(merge),
        'same_source_target_content_in_both_ordinary_input_kinds':cross_kind,
        'fit_pairs_excluding_one_child_identity_merges':sum(counts.values())-merge['one_child_exact_identity'],
        'export_sha256':digest.hexdigest(),'result_sha256':total.archive.sha(root/'result.json'),
        'producer_sha256':total.archive.sha(Path(__file__)),'semantic_quality_reviewed':False,'training_admitted':False}
    total.archive.write(root/'inventory_audit.json',proof);db.close();print(json.dumps(proof),flush=True)


if __name__=='__main__':main()
