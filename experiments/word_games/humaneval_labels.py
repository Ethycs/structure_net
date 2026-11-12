"""HumanEval interface relabeling, official-test grading and fixed sample."""
import ast
import copy
import gzip
import json
import math
from pathlib import Path
import random
import subprocess

from .common import canonical, read, sha, write
from .opencode_harness import CONFIG
from .opencode_sandbox import launch

SEED = 731113
COUNT = 32
DATA = Path('data/experiments/word_games/20261112_humaneval_data_v1')


class Rename(ast.NodeTransformer):
    """Rename Python symbols, never string constants or type annotations."""
    def __init__(self, table):
        self.table = table

    def visit_Name(self, node):
        node.id = self.table.get(node.id, node.id)
        return node

    def visit_arg(self, node):
        node.arg = self.table.get(node.arg, node.arg)
        return node

    def visit_keyword(self, node):
        node.arg = self.table.get(node.arg, node.arg)
        node.value = self.visit(node.value)
        return node

    def visit_FunctionDef(self, node):
        node.name = self.table.get(node.name, node.name)
        node.args = self.visit(node.args)
        node.body = [self.visit(n) for n in node.body]
        return node


def interface(problem):
    tree = ast.parse(problem['prompt'])
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                and n.name == problem['entry_point'])
    args = node.args.posonlyargs + node.args.args + node.args.kwonlyargs
    if node.args.vararg: args.append(node.args.vararg)
    if node.args.kwarg: args.append(node.args.kwarg)
    table = {node.name: 'bridge_operation',
             **{a.arg: 'parcel_' + str(i) for i, a in enumerate(args)}}
    if len(table) != len(args) + 1 or len(set(table.values())) != len(table):
        raise ValueError('Identifier collision')
    occupied = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)} | {
        n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
    if set(table.values()) & occupied: raise ValueError('Target name already used')
    node.body = [ast.Raise(exc=ast.Call(func=ast.Name(id='NotImplementedError', ctx=ast.Load()),
                                      args=[], keywords=[]))]
    ast.fix_missing_locations(tree)
    native = ast.unparse(tree) + '\n'
    relabeled = ast.unparse(Rename(table).visit(copy.deepcopy(tree))) + '\n'
    return table, native, relabeled


def selected():
    source = read(DATA / 'source.json')
    assert sha(DATA / 'HumanEval.jsonl.gz') == source['data_sha256']
    problems = [json.loads(line) for line in gzip.decompress(
        (DATA / 'HumanEval.jsonl.gz').read_bytes()).decode().splitlines()]
    ids = sorted(random.Random(SEED).sample(range(len(problems)), COUNT))
    assert len(problems) == 164 and ids == source['selected_ids']
    by_id = {int(p['task_id'].split('/')[1]): p for p in problems}
    return [by_id[i] for i in ids]


EVALUATOR = '''
import contextlib, io, json, resource, sys
resource.setrlimit(resource.RLIMIT_CPU,(3,3))
resource.setrlimit(resource.RLIMIT_AS,(512*1024*1024,512*1024*1024))
resource.setrlimit(resource.RLIMIT_FSIZE,(262144,262144))
payload=json.load(sys.stdin)
ns={}
captured=io.StringIO()
try:
    with contextlib.redirect_stdout(captured), contextlib.redirect_stderr(captured):
        exec(compile(open('solution.py').read(),'solution.py','exec'),ns)
        fn=ns[payload['entry_point']]
        mapping=payload['parameter_map']
        def candidate(*args,**kwargs):
            return fn(*args,**{mapping.get(k,k):v for k,v in kwargs.items()})
        exec(compile(payload['test'],'official_test.py','exec'),ns)
        ns['check'](candidate)
    result={'passed':True,'error':None}
except BaseException as exc:
    result={'passed':False,'error':type(exc).__name__,'message':str(exc)[:2000]}
result['captured_output']=captured.getvalue()[:4000]
print(json.dumps(result))
'''


def grade(workspace, problem, arm, output, delivered=True, timeout=8):
    output.mkdir(parents=True, exist_ok=False)
    table, _, _ = interface(problem)
    payload = {'test': problem['test'],
               'entry_point': problem['entry_point'] if arm == 'native' else table[problem['entry_point']],
               'parameter_map': {} if arm == 'native' else {k: v for k, v in table.items() if k != problem['entry_point']}}
    write(output / 'evaluation_input.json', payload)
    prefix, env = launch(workspace, CONFIG, 'no-inference-credential-in-evaluator', read_only=True)
    prefix.remove('--share-net')
    error = None
    with (output / 'stdout').open('xb') as stdout, (output / 'stderr').open('xb') as stderr:
        try:
            process = subprocess.run(prefix + ['/usr/bin/python3', '-c', EVALUATOR],
                input=canonical(payload).encode(), env=env, stdout=stdout, stderr=stderr, timeout=timeout)
            if process.returncode: error = 'exit:' + str(process.returncode)
        except subprocess.TimeoutExpired:
            error = 'timeout'
    response = None
    if error is None:
        try:
            response = read(output / 'stdout')
            if type(response.get('passed')) is not bool: raise ValueError('invalid_grade')
        except (ValueError, AttributeError):
            error = 'invalid_grader_response'
    artifact = error is None and response['passed']
    result = {'passed': bool(artifact and delivered), 'artifact_passed': bool(artifact),
              'delivery_complete': delivered, 'error': error, 'response': response,
              'network': 'unshared', 'grader': 'official HumanEval check(candidate)'}
    write(output / 'grade.json', result)
    return result


def wilson(passed, total):
    if total == 0: return None
    z = 1.959963984540054
    p = passed / total
    centre = (p + z*z/(2*total)) / (1 + z*z/total)
    margin = z * math.sqrt(p*(1-p)/total + z*z/(4*total*total)) / (1 + z*z/total)
    return [max(0., centre-margin), min(1., centre+margin)]
