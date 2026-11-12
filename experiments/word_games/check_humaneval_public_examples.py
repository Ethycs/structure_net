"""Post-hoc check of public examples against unchanged original programs."""
import argparse
from pathlib import Path

from .common import read, sha, write, canonical
from .diagnose_humaneval_failures import PRIOR, EXPECTED_PRIOR
from .humaneval_labels import grade

PUBLIC_TESTS = {
    32: '''def check(candidate):
    assert abs(candidate([1,2])+0.5)<1e-6
    x=candidate([-6,11,-6,1])
    assert abs(-6+11*x-6*x*x+x*x*x)<1e-5
''',
    113: '''def check(candidate):
    assert candidate(['1234567'])==['the number of odd elements 4n the str4ng 4 of the 4nput.']
    assert candidate(['3','11111111'])==['the number of odd elements 1n the str1ng 1 of the 1nput.','the number of odd elements 8n the str8ng 8 of the 8nput.']
''',
    160: '''def check(candidate):
    assert candidate(['+','*','-'],[2,3,4,5])==9
''',
}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();root=args.output;root.mkdir(parents=True,exist_ok=False)
    assert sha(PRIOR/'result.json')==EXPECTED_PRIOR
    write(root/'plan.json',{'classification':'posthoc_public_example_check','tests':PUBLIC_TESTS,
        'prior_result_sha256':EXPECTED_PRIOR,'source_sha256':sha(__file__),
        'known_findings':'Five original defects already isolated; test which are detectable from public examples alone.',
        'root_grading':'For the multiple-root example accept any root, as the prose contract permits.'})
    (root/'producer.py').write_bytes(Path(__file__).read_bytes())
    workroot=Path('.pixi/word-games-public-examples')/root.name;workroot.mkdir(parents=True,exist_ok=False)
    problems={int(p['task_id'].split('/')[1]):p for p in read(PRIOR/'suite.json')}
    rows=[]
    for task,test in PUBLIC_TESTS.items():
        original=problems[task];problem={**original,'test':test}
        for variant in ('canonical','native','relabeled'):
            workspace=workroot/f'{task}_{variant}';workspace.mkdir()
            source=original['prompt']+original['canonical_solution'] if variant=='canonical' else (PRIOR/f'HumanEval_{task}'/variant/'solution.py').read_text()
            (workspace/'solution.py').write_text(source)
            scored=grade(workspace,problem,'relabeled' if variant=='relabeled' else 'native',root/str(task)/variant)
            row={'task':task,'variant':variant,'solution_sha256':sha(workspace/'solution.py'),'grade':scored}
            rows.append(row)
            if variant=='canonical' and not scored['passed']:raise RuntimeError('Public-example reference failed')
    write(root/'result.json',{'classification':'posthoc_public_example_check','rows':rows,'primary_accuracy_unchanged':True})
    print(canonical([{'task':r['task'],'variant':r['variant'],'passed':r['grade']['passed']} for r in rows]))


if __name__=='__main__':main()
