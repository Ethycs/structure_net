from experiments.structure_net.ncm_word_final_long_diagnostic import select_additions


def test_scarce_band_is_kept_when_groups_span_multiple_bands():
    rows=[]
    for b in range(4):
        for i in range(2):
            group=('shared'+str(i)) if b==2 else str(b)+'_'+str(i)
            rows.append({'id':str(b)+'_'+str(i),'source_id':group,'group_id':group,'source_sha256':group,'bin':b})
    rows += [{**r,'id':'other_'+r['id'],'bin':3} for r in rows if r['bin']==2]
    blocked={k:set() for k in ('id','source_id','group_id','source_sha256')}
    result=select_additions(rows,blocked,per_bin=2)
    assert [r['bin'] for r in result]==[2,2,3,3,1,1,0,0]
    assert len({r['group_id'] for r in result})==8
    assert select_additions(list(reversed(rows)),blocked,per_bin=2)==result
